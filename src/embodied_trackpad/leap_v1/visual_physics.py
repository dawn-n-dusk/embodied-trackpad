"""LEAP v1 Full visual-episode physics core (P1 pad profile, teacher truth).

Physics/controller core for a future bounded visual runner: NO native loop,
CLI, GUI/app coupling, episode driver or manifest machinery here; all output
is physics-side teacher/device truth, separate from actor observations.

* Scene: the accepted pad scene with ONLY the reviewed P1 pad selection
  applied (priority=1, solref=[0.02, 1], solimp=[0.9, 0.95, 0.001, 0.5, 2]);
  friction, 2 ms timestep and all other hand parameters stay as loaded.
* :func:`physics_step`: validated full-16 ctrl -> mj_step -> mj_forward -> raw
  contacts -> raw validation, one complete record per substep (empty contacts
  included). The FIRST failed safety check raises :class:`PhysicsStepError`
  with that substep's evidence, so a later good sample can never hide it.
* :func:`plan_index_correction`: pure teacher helper; pixel error / fixed 4000
  px/m -> bounded pad-frame x/y displacement plus z-hold, 3x4 damped least
  squares over if_mcp/if_rot/if_pip/if_dip with a caller-supplied privileged
  Jacobian, scattered by name into the 16 channels. Never reads native
  state/reward/layout, never synthesizes touch coordinates; a teacher label,
  not a learned actor.
* :func:`reset_open` is the ONLY qpos assignment; :func:`open_release_ctrl`
  commands the open pose but is NOT proof of an actual lift (only a measured
  empty-contact record is).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import mujoco
import numpy as np

from embodied_trackpad.leap_sim.pad_env import apply_ctrl_checked, validate_raw_contacts
from embodied_trackpad.leap_v1 import pad_env as _pad
from embodied_trackpad.leap_v1.model import MOTOR_JOINTS, open_pose_qpos
from embodied_trackpad.leap_v1.visual_servo import CONTACT_PROFILE, DEVICE_GAIN_PX_PER_M

NU = 16  # accepted v1 Full contract: 16 independent position actuators
INDEX_JOINTS = ("if_mcp", "if_rot", "if_pip", "if_dip")
P1_SCENE_CONTACT = CONTACT_PROFILE["scene_contact"]  # reviewed P1 selection


# --- 1. scene loading + loaded-profile verification --------------------------

def apply_p1_profile(model: mujoco.MjModel, pad_geom_id: int) -> None:
    """Apply ONLY the P1 pad priority/solref/solimp; contact params are runtime
    model fields read by the solver each step, so post-compile writes are
    equivalent to spec baking; friction and all else stay as loaded."""
    model.geom_priority[pad_geom_id] = 1
    model.geom_solref[pad_geom_id] = P1_SCENE_CONTACT["solref"]
    model.geom_solimp[pad_geom_id] = P1_SCENE_CONTACT["solimp"]


@dataclass(frozen=True)
class NamedJointMap:
    """Explicit joint-name -> qpos/dof/actuator-channel mapping (no list guesses)."""

    joint_names: tuple[str, ...]
    qposadr: tuple[int, ...]
    dofadr: tuple[int, ...]
    act_index: tuple[int, ...]
    ranges: tuple[tuple[float, float], ...]


def build_named_mapping(
    model: mujoco.MjModel, joint_names: Sequence[str] = INDEX_JOINTS
) -> NamedJointMap:
    """Resolve each named joint's qpos/dof address and driving actuator by id."""
    qposadr, dofadr, act_index, ranges = [], [], [], []
    act_joint_id = {int(model.actuator_trnid[i, 0]): i for i in range(model.nu)}
    for name in joint_names:
        jid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise ValueError(f"joint missing in model: {name}")
        if jid not in act_joint_id:
            raise ValueError(f"no actuator drives joint {name}")
        qposadr.append(int(model.jnt_qposadr[jid]))
        dofadr.append(int(model.jnt_dofadr[jid]))
        act_index.append(act_joint_id[jid])
        ranges.append((float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])))
    return NamedJointMap(tuple(joint_names), tuple(qposadr), tuple(dofadr),
                         tuple(act_index), tuple(ranges))


@dataclass
class VisualPhysicsEnv:
    """Loaded P1 scene plus the explicit index mapping and warning baseline."""

    scene: _pad.PadScene
    index_map: NamedJointMap
    data: mujoco.MjData
    warn_counts: np.ndarray | None = None

    @property
    def model(self) -> mujoco.MjModel:
        return self.scene.model


def load_p1_scene(**pad_kw) -> _pad.PadScene:
    """Load the accepted pad scene, then apply ONLY the P1 pad profile."""
    scene = _pad.load_pad_scene(**pad_kw)
    apply_p1_profile(scene.model, scene.pad_geom_id)
    return scene


def make_env(**pad_kw) -> VisualPhysicsEnv:
    """Load the P1 scene, build the explicit index mapping, verify the model."""
    scene = load_p1_scene(**pad_kw)
    env = VisualPhysicsEnv(scene=scene, index_map=build_named_mapping(scene.model),
                           data=mujoco.MjData(scene.model))
    verify_scene_profile(env)
    return env


def verify_scene_profile(env: VisualPhysicsEnv) -> dict[str, Any]:
    """Re-read the compiled model; raise ValueError on any profile mismatch."""
    m = env.model
    g = env.scene.pad_geom_id
    observed = {
        "profile_id": CONTACT_PROFILE["profile_id"],
        "timestep_s": float(m.opt.timestep),
        "pad_priority": int(m.geom_priority[g]),
        "pad_solref": [float(v) for v in m.geom_solref[g]],
        "pad_solimp": [float(v) for v in m.geom_solimp[g]],
        "pad_friction": [float(v) for v in m.geom_friction[g]],
        "nu": int(m.nu),
        "motor_order_preserved": [
            m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)
        ] == MOTOR_JOINTS,
    }
    want = P1_SCENE_CONTACT
    problems = []
    if observed["timestep_s"] != float(want["timestep_s"]):
        problems.append("timestep_s")
    if observed["pad_priority"] != 1:
        problems.append("pad_priority")
    for key, wkey in (("pad_solref", "solref"), ("pad_solimp", "solimp"),
                      ("pad_friction", "friction")):
        if not np.allclose(observed[key], want[wkey], rtol=0, atol=1e-12):
            problems.append(key)
    if observed["nu"] != NU or not observed["motor_order_preserved"]:
        problems.append("actuator_contract")
    if problems:
        raise ValueError(f"P1 scene profile mismatch {problems}: {observed}")
    return observed


# --- 2. single validated physics step ----------------------------------------

@dataclass(frozen=True)
class StepSafety:
    max_pad_force_n: float = _pad.PAD_FORCE_MAX_N
    penetration_tol_m: float = _pad.PENETRATION_TOL
    limit_excursion_tol_rad: float = _pad.LIMIT_EXCURSION_TOL


@dataclass
class StepRecord:
    """One complete raw substep record (teacher/device truth, never actor data).

    All numeric fields are JSON-strict safe: any non-finite value is stored as
    the STRING "nan"/"inf"/"-inf" (documented representation, never a bare
    float), so ``json.dumps(record.to_dict(), allow_nan=False)`` always works,
    including on first-failure evidence records.
    """

    step_index: int
    sim_time_s: float | str
    frame_id: Any
    ctrl: list  # requested values, sanitized flat; see ctrl_shape
    qpos: list
    qvel: list
    contacts: list[dict]  # full raw contacts, empty list preserved
    aggregate: dict[str, dict]
    warning_delta: list[int]
    checks: dict[str, bool]
    ok: bool
    failure_reasons: list[str] = field(default_factory=list)
    ctrl_shape: list[int] = field(default_factory=list)  # requested ctrl shape

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class PhysicsStepError(RuntimeError):
    """Raised on the FIRST failed substep; carries that substep's evidence."""

    def __init__(self, record: StepRecord):
        super().__init__(f"physics step {record.step_index} failed: {record.failure_reasons}")
        self.record = record


def reset_open(env: VisualPhysicsEnv) -> None:
    """Reset to the open pose; the ONLY qpos assignment in this module."""
    m, d = env.model, env.data
    mujoco.mj_resetData(m, d)
    d.qpos[:] = open_pose_qpos(m)
    apply_ctrl_checked(m, d, np.zeros(m.nu))
    mujoco.mj_forward(m, d)
    env.warn_counts = np.array([w.number for w in d.warning], dtype=int)


def _safe_num(v: Any) -> float | str:
    """Finite float, else the STRING "nan"/"inf"/"-inf" (JSON-strict safe)."""
    f = float(v)
    return f if math.isfinite(f) else str(f)


def _safe_seq(vals: Any) -> list:
    """Sanitized flat numeric list; repr fallback for non-numeric input."""
    try:
        arr = np.asarray(vals, dtype=float)
    except (TypeError, ValueError):
        return [repr(vals)]
    return [_safe_num(v) for v in arr.ravel()]


def _raw_contact_dict(c: "_pad.RawContact") -> dict[str, Any]:
    return {"geom1": c.geom1, "geom2": c.geom2, "body1": c.body1, "body2": c.body2,
            "kind": c.kind, "normal_force_n": _safe_num(c.normal_force_n),
            "force6": _safe_seq(c.force6), "frame": _safe_seq(c.frame),
            "dist_m": _safe_num(c.dist_m), "pos_world": _safe_seq(c.pos_world),
            "pad_local": None if c.pad_local is None else _safe_seq(c.pad_local),
            "top_face_cos": None if c.top_face_cos is None else _safe_num(c.top_face_cos)}


def physics_step(
    env: VisualPhysicsEnv,
    ctrl: np.ndarray,
    *,
    step_index: int,
    frame_id: Any = None,
    safety: StepSafety = StepSafety(),
    step_fn: Callable = mujoco.mj_step,
    forward_fn: Callable = mujoco.mj_forward,
    contact_reader: Callable = _pad.read_step_contacts,
    aggregator: Callable = _pad.aggregate_fingers,
) -> StepRecord:
    """Run one validated substep and return its complete raw record. The
    step/forward/contact callables are injectable so tests can stub the
    solver; production defaults are the real MuJoCo functions. Raw contacts
    are validated BEFORE aggregation: on invalid raw the aggregator and all
    force/penetration summaries are skipped and the raw failure evidence is
    preserved in the record."""
    m, d = env.model, env.data
    try:
        ctrl_arr = np.asarray(ctrl, dtype=float)
        ctrl_shape = [int(s) for s in ctrl_arr.shape]
    except (TypeError, ValueError):
        ctrl_arr, ctrl_shape = np.zeros(0), [-1]
    if env.warn_counts is None:
        env.warn_counts = np.array([w.number for w in d.warning], dtype=int)
    delta: list[int] = [0] * len(env.warn_counts)
    raw: list = []
    aggregate: dict = {}
    try:
        apply_ctrl_checked(m, d, ctrl_arr)  # validates full-16 BEFORE any write
    except ValueError as exc:
        checks = {"ctrl_valid": False}
        reasons = [f"ctrl_invalid: {exc}"]
    else:
        pre_finite = bool(
            np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
            and np.isfinite(float(d.time))
        )
        if not pre_finite:
            # Never let the solver silently recover an invalid prestate.
            checks = {"pre_step_state_finite": False}
            reasons = ["non_finite_pre_step_state"]
        else:
            step_fn(m, d)
            forward_fn(m, d)  # observations below belong to this recorded state
            now = np.array([w.number for w in d.warning], dtype=int)
            delta = (now - env.warn_counts).tolist()
            env.warn_counts = now
            finite = bool(np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))
                          and np.all(np.isfinite(d.ctrl)) and np.isfinite(float(d.time)))
            raw = contact_reader(env.scene, d)
            try:
                validate_raw_contacts(raw)  # NaN never reaches the >0 filters
            except ValueError as exc:
                checks = {"pre_step_state_finite": True, "finite_state": finite,
                          "raw_contacts_valid": False}
                reasons = [f"raw_contacts_invalid: {exc}"]
                if not finite:
                    reasons.append("non_finite_state")
            else:
                aggregate, unintended = aggregator(env.scene, raw)
                pad_force_all = float(sum(
                    c.normal_force_n for c in raw
                    if c.kind in _pad.PAD_CONTACT_KINDS and c.normal_force_n > 0.0))
                penetration = float(max((-c.dist_m for c in raw if c.dist_m < 0),
                                        default=0.0))
                rng = np.asarray(m.jnt_range, dtype=float)
                q = np.asarray(d.qpos, dtype=float)[np.asarray(m.jnt_qposadr)[: m.njnt]]
                excursion = float(max(
                    0.0, float(np.maximum(rng[:, 0] - q, q - rng[:, 1]).max())))
                checks = {
                    "pre_step_state_finite": True,
                    "finite_state": finite,
                    "raw_contacts_valid": True,
                    "pad_force<=5N": pad_force_all <= safety.max_pad_force_n,
                    "penetration<=tol": penetration <= safety.penetration_tol_m,
                    "no_unintended_contact": not unintended,
                    "joint_excursion<=tol": excursion <= safety.limit_excursion_tol_rad,
                    "no_new_warnings": not any(delta),
                }
                detail = {
                    "pre_step_state_finite": "non_finite_pre_step_state",
                    "finite_state": "non_finite_state",
                    "raw_contacts_valid": "raw_contacts_invalid",
                    "pad_force<=5N": f"pad_force_all={pad_force_all:.4f}N > {safety.max_pad_force_n}N",
                    "penetration<=tol": f"penetration={penetration:.6f}m > {safety.penetration_tol_m}m",
                    "no_unintended_contact": f"unintended_contacts={len(unintended)}",
                    "joint_excursion<=tol": f"joint_excursion={excursion:.5f}rad",
                    "no_new_warnings": f"new_warnings={delta}",
                }
                reasons = [detail[k] for k, ok in checks.items() if not ok]
    rec = StepRecord(
        step_index=int(step_index), sim_time_s=_safe_num(d.time), frame_id=frame_id,
        ctrl=_safe_seq(ctrl_arr), ctrl_shape=ctrl_shape,
        qpos=_safe_seq(d.qpos), qvel=_safe_seq(d.qvel),
        contacts=[_raw_contact_dict(c) for c in raw],
        aggregate=aggregate, warning_delta=delta, checks=checks,
        ok=not reasons, failure_reasons=reasons)
    if reasons:
        raise PhysicsStepError(rec)
    return rec


# --- 3. pure teacher planner (pixel error -> bounded index correction) --------

@dataclass(frozen=True)
class PlannerConfig:
    px_per_m: float = DEVICE_GAIN_PX_PER_M  # fixed 4000 px/m device config gain
    max_planar_step_m: float = 0.010  # per-cycle bounded x/y displacement
    max_z_step_m: float = 0.005  # bounded z-hold correction
    damping: float = 0.05  # damped least-squares lambda (rad vs m)
    max_joint_delta_rad: float = 0.10  # per-joint per-cycle delta cap

    def __post_init__(self) -> None:
        for name in ("px_per_m", "max_planar_step_m", "max_z_step_m",
                     "damping", "max_joint_delta_rad"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0):
                raise ValueError(f"PlannerConfig.{name} must be a finite positive number, got {v!r}")


def damped_least_squares(jac: np.ndarray, task: np.ndarray, damping: float) -> np.ndarray:
    """dq = J^T (J J^T + lambda^2 I)^-1 task for a 3x4 Jacobian."""
    jac = np.asarray(jac, dtype=float)
    task = np.asarray(task, dtype=float)
    if jac.shape != (3, 4) or task.shape != (3,):
        raise ValueError(f"bad shapes: jac {jac.shape}, task {task.shape}")
    if not (np.all(np.isfinite(jac)) and np.all(np.isfinite(task))):
        raise ValueError("non-finite Jacobian or task")
    return jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), task)


def plan_index_correction(
    error_px: tuple[float, float],
    z_error_m: float,
    jac_3x4: np.ndarray,
    current_ctrl: np.ndarray,
    mapping: NamedJointMap,
    cfg: PlannerConfig = PlannerConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map a pixel error to a bounded 16-channel teacher correction.
    ``error_px`` is (target - cursor) in logical px from the CURRENT decoded
    capture only; ``z_error_m`` is the teacher-FK z-hold error. The pad frame
    is axis-aligned with the world frame here, so the pad-frame task feeds the
    world Jacobian directly. JACOBIAN COLUMN-ORDER CONTRACT: column k of
    ``jac_3x4`` MUST be the translational column of ``mapping.joint_names[k]``
    (column permutation permutes the solved dq accordingly). The returned
    increment is explicitly capped per joint and clipped to joint ranges, and
    the full-16 output is verified finite before returning; other channels
    pass through unchanged."""
    ex, ey = float(error_px[0]), float(error_px[1])
    if not (np.isfinite(ex) and np.isfinite(ey) and np.isfinite(z_error_m)):
        raise ValueError("non-finite planner input")
    ctrl = np.asarray(current_ctrl, dtype=float)
    if ctrl.shape != (NU,) or not np.all(np.isfinite(ctrl)):
        raise ValueError(f"current_ctrl must be a finite ({NU},) vector")
    task = np.array([
        float(np.clip(ex / cfg.px_per_m, -cfg.max_planar_step_m, cfg.max_planar_step_m)),
        float(np.clip(ey / cfg.px_per_m, -cfg.max_planar_step_m, cfg.max_planar_step_m)),
        float(np.clip(z_error_m, -cfg.max_z_step_m, cfg.max_z_step_m)),
    ])
    dq = damped_least_squares(jac_3x4, task, cfg.damping)
    dq = np.clip(dq, -cfg.max_joint_delta_rad, cfg.max_joint_delta_rad)
    out = ctrl.copy()
    for k, act in enumerate(mapping.act_index):
        lo, hi = mapping.ranges[k]
        out[act] = float(np.clip(out[act] + dq[k], lo, hi))
    if not np.all(np.isfinite(out)):
        raise ValueError("planner produced a non-finite command")
    return out, {"task_m": task.tolist(), "dq_rad": dq.tolist(),
                 "teacher_only": True, "joints": list(mapping.joint_names)}


def open_release_ctrl(model: mujoco.MjModel) -> np.ndarray:
    """Open-pose command reference; NOT proof of actual lift (only a measured
    empty-contact record evidences release)."""
    return np.zeros(model.nu)
