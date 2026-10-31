"""Measured motion diagnostics for the LEAP v1 Full baseline (hand-only).

Executes three scripted phases purely through ``data.ctrl`` + ``mujoco.mj_step``
(the only live qpos write is the reset to the all-zero open pose):

1. ``sweep``: each of the 16 channels is individually ramped to a flexion
   target, held, and returned while all other channels hold the open pose.
2. ``spread``: index/middle ``*_rot`` channels open and close the fingertip
   spacing; the index-middle tip distance is measured, never assumed.
3. ``flex_return``: all 16 channels flex together, hold, and return to the
   open pose with a >=0.7 s settle; the recovered-open error is measured.

Recorded per step: time, ctrl, actuated qpos, tracking error, joint-limit
excursion, MuJoCo solver warnings, contact count and every self-contact pair
with forces, plus sampled full state (qpos/qvel/qacc/ctrl/actuator_force) and
fingertip site positions. Acceptance reference gates: limit excursion
<= 0.01 rad, zero warnings, finite state, episode completeness. Tracking,
spread and recovered-open errors are REPORTED measurements (with explicit
reference tolerances), never hidden and never relaxed to pass.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from embodied_trackpad.leap_sim.pad_env import (  # v2 helpers reused unmodified
    apply_ctrl_checked,
    dump_json,
    write_png,
)
from embodied_trackpad.leap_v1.model import (
    FINGER_JOINTS,
    FINGERS,
    MOTOR_JOINTS,
    TIP_SITES,
    LeapV1Hand,
    open_pose_qpos,
)

# Bodies of each finger chain (used to distinguish contact-blocked tracking
# from free-space tracking, mirroring the v2 free/blocked split).
FINGER_BODIES = {
    "index": ("if_bs", "if_px", "if_md", "if_ds"),
    "middle": ("mf_bs", "mf_px", "mf_md", "mf_ds"),
    "ring": ("rf_bs", "rf_px", "rf_md", "rf_ds"),
    "thumb": ("th_mp", "th_bs", "th_px", "th_ds"),
}

LIMIT_EXCURSION_TOL = 0.01  # rad; acceptance gate
HELD_TRACKING_TOL = 0.10  # rad; reported reference tolerance
FINAL_OPEN_TOL = 0.05  # rad; reported reference tolerance
MIN_OPEN_HOLD_S = 0.7  # s settle before the recovered-open measurement

SWEEP_FRACTION = 0.6  # fraction of the upper joint range commanded
RAMP_S = 0.4
HOLD_S = 0.3
RETURN_S = 0.4
SETTLE_S = 0.2

SPREAD_TARGET_RAD = 0.5  # per-side if_rot/mf_rot command magnitude
SPREAD_RAMP_S = 0.5
SPREAD_HOLD_S = 0.4

FLEX_TARGET = {
    "if_mcp": 1.2, "if_pip": 1.0, "if_dip": 0.8, "if_rot": 0.0,
    "mf_mcp": 1.2, "mf_pip": 1.0, "mf_dip": 0.8, "mf_rot": 0.0,
    "rf_mcp": 1.2, "rf_pip": 1.0, "rf_dip": 0.8, "rf_rot": 0.0,
    "th_cmc": 0.6, "th_axl": 0.3, "th_mcp": 0.9, "th_ipl": 0.8,
}
FLEX_RAMP_S = 0.8
FLEX_HOLD_S = 0.5
FLEX_RETURN_S = 0.8


@dataclass
class Phase:
    name: str
    t0: float
    t1: float
    kind: str  # hold | ramp | settle
    base: np.ndarray | None = None  # start vector
    target: np.ndarray | None = None  # end vector (ramp)
    hold_channels: tuple = ()  # channels whose tracking is scored (holds)
    render_label: str | None = None


def _phase_vector(m: mujoco.MjModel, overrides: dict[str, float]) -> np.ndarray:
    vec = np.zeros(m.nu)
    names = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    for jname, value in overrides.items():
        vec[names.index(jname)] = value
    return vec


def build_script(m: mujoco.MjModel) -> list[Phase]:
    """Full phase list: per-channel sweep, spread, flex/return."""
    phases: list[Phase] = []
    t = 0.0
    open_vec = np.zeros(m.nu)
    phases.append(Phase("open_settle", t, t + SETTLE_S, "hold", base=open_vec))
    t += SETTLE_S
    for jname in MOTOR_JOINTS:
        jid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname))
        hi = float(m.jnt_range[jid][1])
        target = _phase_vector(m, {jname: SWEEP_FRACTION * hi})
        phases.append(
            Phase(f"sweep_{jname}_ramp", t, t + RAMP_S, "ramp", base=open_vec, target=target)
        )
        t += RAMP_S
        phases.append(
            Phase(
                f"sweep_{jname}_hold", t, t + HOLD_S, "hold", base=target,
                hold_channels=(jname,), render_label=f"sweep_{jname}",
            )
        )
        t += HOLD_S
        phases.append(
            Phase(f"sweep_{jname}_return", t, t + RETURN_S, "ramp", base=target, target=open_vec)
        )
        t += RETURN_S
    # spread: rot channels of index/middle in opposite directions. Measured on
    # the pinned model: if_rot negative / mf_rot positive INCREASES the
    # index-middle tip distance (0.045 m rest -> 0.127 m at +/-0.5 rad); the
    # opposite sign closes it.
    spread_vec = _phase_vector(
        m, {"if_rot": -SPREAD_TARGET_RAD, "mf_rot": SPREAD_TARGET_RAD}
    )
    phases.append(Phase("spread_ramp", t, t + SPREAD_RAMP_S, "ramp", base=open_vec, target=spread_vec))
    t += SPREAD_RAMP_S
    phases.append(
        Phase("spread_hold", t, t + SPREAD_HOLD_S, "hold", base=spread_vec,
              hold_channels=("if_rot", "mf_rot"), render_label="spread")
    )
    t += SPREAD_HOLD_S
    phases.append(Phase("spread_close", t, t + SPREAD_RAMP_S, "ramp", base=spread_vec, target=open_vec))
    t += SPREAD_RAMP_S
    # flex/return (lift/return): all channels flex together, then reopen
    flex_vec = _phase_vector(m, FLEX_TARGET)
    phases.append(Phase("flex_ramp", t, t + FLEX_RAMP_S, "ramp", base=open_vec, target=flex_vec))
    t += FLEX_RAMP_S
    phases.append(Phase("flex_hold", t, t + FLEX_HOLD_S, "hold", base=flex_vec,
                        hold_channels=tuple(MOTOR_JOINTS), render_label="flex"))
    t += FLEX_HOLD_S
    phases.append(Phase("flex_return", t, t + FLEX_RETURN_S, "ramp", base=flex_vec, target=open_vec))
    t += FLEX_RETURN_S
    phases.append(Phase("open_final", t, t + MIN_OPEN_HOLD_S, "hold", base=open_vec,
                        render_label="returned"))
    t += MIN_OPEN_HOLD_S
    return phases


@dataclass
class DiagResult:
    metrics: dict
    physics_seconds: float
    failures: list[str] = field(default_factory=list)


def _ctrl_at(phases: list[Phase], t: float) -> tuple[np.ndarray, Phase]:
    for ph in phases:
        if ph.t0 - 1e-12 <= t < ph.t1 or (t >= phases[-1].t1 and ph is phases[-1]):
            if ph.kind == "ramp":
                a = min(max((t - ph.t0) / (ph.t1 - ph.t0), 0.0), 1.0)
                return (1 - a) * ph.base + a * ph.target, ph
            return ph.base.copy(), ph
    return phases[-1].base.copy(), phases[-1]


def run_motion_diagnostics(
    hand: LeapV1Hand,
    out_dir: Path | None = None,
    render: bool = True,
    sample_every: int = 25,
    channels: tuple[str, ...] = tuple(MOTOR_JOINTS),
) -> DiagResult:
    """Execute the scripted diagnostics; ctrl + mj_step only after the reset."""
    m = hand.model
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if (out_dir / "trace.npz").exists() or (out_dir / "motion-metrics.json").exists():
            raise FileExistsError(
                f"output directory already contains diagnostics results: {out_dir}"
            )
    phases = [p for p in build_script(m) if _phase_in_scope(p, channels)]
    t_end = phases[-1].t1
    steps = int(round(t_end / m.opt.timestep))

    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = open_pose_qpos(m)  # the only live qpos write of the whole run
    apply_ctrl_checked(m, d, np.zeros(m.nu))
    mujoco.mj_forward(m, d)

    trnid = m.actuator_trnid[:, 0]
    act_qposadr = m.jnt_qposadr[trnid]
    act_jnames = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    tip_site_ids = {
        f: int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TIP_SITES[f])) for f in FINGERS
    }

    renderer = None
    cam = None
    if render and out_dir is not None:
        renderer = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        tips0 = np.stack([d.site(sid).xpos for sid in tip_site_ids.values()])
        palm_pos = d.body("palm").xpos
        centre = 0.5 * (tips0.mean(axis=0) + palm_pos)
        cam.lookat = [float(v) for v in centre]
        cam.distance = 0.38
        cam.azimuth = 115.0
        cam.elevation = -25.0
        renderer.update_scene(d, cam)
        png = out_dir / "overview_open.png"
        write_png(png, renderer.render())

    series: dict[str, list] = {
        "t": [], "t_ctrl": [], "phase": [], "limit_excursion_rad": [], "ncon": [],
        "max_tracking_rad": [],
    }
    trace: dict[str, list] = {"t": [], "qpos": [], "qvel": [], "qacc": [], "ctrl": [],
                              "actfrc": [], "tips": [], "phase": []}
    self_contact_log: list[dict] = []
    hold_tracking: dict[str, float] = {}
    hold_tracking_free: dict[str, float] = {}
    hold_tracking_blocked: dict[str, float] = {}
    channel_tip_disp: dict[str, float] = {}
    rendered: set[str] = set()
    failures: list[str] = []
    max_limit_excursion = 0.0
    max_negative_normal = 0.0
    warn_start = np.array([w.number for w in d.warning], dtype=int)
    tips_initial: dict[str, np.ndarray] | None = None
    spread_dist_initial = spread_dist_final = None
    final_open_error = None
    steps_executed = 0
    physics_t0 = time.perf_counter()

    for k in range(steps):
        t_ctrl = k * m.opt.timestep
        desired, phase = _ctrl_at(phases, t_ctrl)
        apply_ctrl_checked(m, d, desired)  # raises before mj_step on invalid input
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)  # observation instant matches recorded state
        t = float(d.time)
        steps_executed = k + 1

        if not (
            np.all(np.isfinite(d.qpos))
            and np.all(np.isfinite(d.qvel))
            and np.all(np.isfinite(d.qacc))
            and np.all(np.isfinite(d.ctrl))
            and np.all(np.isfinite(d.actuator_force))
        ):
            failures.append(f"non-finite state at step {k} (t_ctrl={t_ctrl:.3f}, phase={phase.name})")
            break

        err = np.abs(d.ctrl - d.qpos[act_qposadr])
        lim_exc = 0.0
        for j in range(m.njnt):
            lo = float(m.jnt_range[j][0])
            hi = float(m.jnt_range[j][1])
            q = float(d.qpos[m.jnt_qposadr[j]])
            lim_exc = float(max(lim_exc, max(lo - q, q - hi, 0.0)))
        max_limit_excursion = max(max_limit_excursion, lim_exc)

        if phase.kind == "hold":
            scored = phase.hold_channels or tuple(act_jnames)
            contact_bodies = set()
            for i in range(d.ncon):
                c = d.contact[i]
                contact_bodies.add(m.body(int(m.geom_bodyid[int(c.geom1)])).name)
                contact_bodies.add(m.body(int(m.geom_bodyid[int(c.geom2)])).name)
            for i, jname in enumerate(act_jnames):
                if jname not in scored:
                    continue
                finger = next(f for f, js in FINGER_JOINTS.items() if jname in js)
                bodies = FINGER_BODIES[finger]
                blocked = any(b in contact_bodies for b in bodies)
                store = hold_tracking_blocked if blocked else hold_tracking_free
                store[jname] = max(store.get(jname, 0.0), float(err[i]))
                hold_tracking[jname] = max(hold_tracking.get(jname, 0.0), float(err[i]))

        tips_now = {f: d.site(sid).xpos.copy() for f, sid in tip_site_ids.items()}
        if tips_initial is None:
            tips_initial = {f: v.copy() for f, v in tips_now.items()}
        if phase.name == "spread_ramp" and spread_dist_initial is None:
            spread_dist_initial = float(
                np.linalg.norm(tips_now["index"] - tips_now["middle"])
            )
        if phase.name == "spread_hold":
            spread_dist_final = float(np.linalg.norm(tips_now["index"] - tips_now["middle"]))
        if phase.render_label and phase.render_label not in rendered and t >= phase.t1 - m.opt.timestep * 1.5:
            rendered.add(phase.render_label)
            if renderer is not None:
                renderer.update_scene(d, cam)
                write_png(out_dir / f"{phase.render_label}.png", renderer.render())
            if phase.render_label.startswith("sweep_"):
                jname = phase.render_label[len("sweep_"):]
                finger = next(f for f, js in FINGER_JOINTS.items() if jname in js)
                channel_tip_disp[jname] = float(
                    np.linalg.norm(tips_now[finger] - tips_initial[finger])
                )
        if phase.name == "open_final" and t >= phase.t1 - 1.5 * m.opt.timestep:
            final_open_error = float(np.max(np.abs(d.qpos[act_qposadr])))

        for i in range(d.ncon):
            c = d.contact[i]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, f6)
            if not np.all(np.isfinite(f6)) or not np.all(np.isfinite(c.frame)):
                failures.append(f"non-finite contact at step {k} (t_ctrl={t_ctrl:.3f})")
                break
            fn = float(f6[0])
            if fn < 0:
                max_negative_normal = max(max_negative_normal, -fn)
            self_contact_log.append(
                {
                    "t": t,
                    "phase": phase.name,
                    "body1": m.body(int(m.geom_bodyid[int(c.geom1)])).name or "world",
                    "body2": m.body(int(m.geom_bodyid[int(c.geom2)])).name or "world",
                    "normal_force_n": fn,
                    "force6": [float(v) for v in f6],
                    "dist_m": float(c.dist),
                    "pos_world": [float(v) for v in c.pos],
                }
            )

        series["t"].append(t)
        series["t_ctrl"].append(t_ctrl)
        series["phase"].append(phase.name)
        series["limit_excursion_rad"].append(lim_exc)
        series["ncon"].append(int(d.ncon))
        series["max_tracking_rad"].append(float(err.max()))
        if k % sample_every == 0 or k == steps - 1:
            trace["t"].append(t)
            trace["qpos"].append(d.qpos.copy())
            trace["qvel"].append(d.qvel.copy())
            trace["qacc"].append(d.qacc.copy())
            trace["ctrl"].append(d.ctrl.copy())
            trace["actfrc"].append(d.actuator_force.copy())
            trace["tips"].append(np.stack([tips_now[f] for f in FINGERS]))
            trace["phase"].append(phase.name)

    physics_seconds = time.perf_counter() - physics_t0
    if renderer is not None:
        renderer.close()

    warn_end = np.array([w.number for w in d.warning], dtype=int)
    warn_delta = (warn_end - warn_start).tolist()
    warn_names = [str(w).split(".")[-1] for w in mujoco.mjtWarning.__members__.values()]
    warnings = {warn_names[i]: int(warn_delta[i]) for i in range(len(warn_delta)) if warn_delta[i]}

    episode_complete = steps_executed == steps and not failures
    held_max = max(hold_tracking.values()) if hold_tracking else 0.0
    spread_delta = (
        None
        if spread_dist_initial is None or spread_dist_final is None
        else spread_dist_final - spread_dist_initial
    )
    contact_steps = sum(1 for v in series["ncon"] if v > 0)
    contact_pairs: dict[str, int] = {}
    for c in self_contact_log:
        key = f"{c['body1']}|{c['body2']}"
        contact_pairs[key] = contact_pairs.get(key, 0) + 1

    gates = {
        "finite": not failures,
        "episode_complete": episode_complete,
        "no_warnings": not warnings,
        "limit_excursion<=0.01": max_limit_excursion <= LIMIT_EXCURSION_TOL,
        "ctrl_in_bounds": True,  # apply_ctrl_checked raises before any invalid write
        "no_negative_normal_force": max_negative_normal == 0.0,
    }
    free_max = max(hold_tracking_free.values()) if hold_tracking_free else 0.0
    reported = {
        "held_tracking_max_rad": held_max,
        "held_tracking_per_channel_rad": hold_tracking,
        "held_tracking_free_rad": hold_tracking_free,
        "held_tracking_blocked_rad": hold_tracking_blocked,
        "held_tracking_free_max_rad": free_max,
        "held_tracking_reference_tol_rad": HELD_TRACKING_TOL,
        "held_tracking_within_reference": held_max <= HELD_TRACKING_TOL,
        "held_tracking_free_within_reference": free_max <= HELD_TRACKING_TOL,
        "spread_index_middle_initial_m": spread_dist_initial,
        "spread_index_middle_final_m": spread_dist_final,
        "spread_delta_m": spread_delta,
        "recovered_open_error_rad": final_open_error,
        "recovered_open_reference_tol_rad": FINAL_OPEN_TOL,
        "recovered_open_within_reference": (
            final_open_error is not None and final_open_error <= FINAL_OPEN_TOL
        ),
        "self_contact_steps": contact_steps,
        "self_contact_events": len(self_contact_log),
        "self_contact_pairs": contact_pairs,
        "channel_tip_displacement_m": channel_tip_disp,
        "max_negative_normal_force_n": max_negative_normal,
    }
    verdict = "PASS" if all(gates.values()) else "FAIL"
    metrics = {
        "verdict": verdict,
        "failure_reasons": [n for n, ok in gates.items() if not ok] + failures,
        "steps_planned": steps,
        "steps_executed": steps_executed,
        "sim_seconds": t_end,
        "timestep": float(m.opt.timestep),
        "physics_seconds_wall": physics_seconds,
        "gates": gates,
        "reported": reported,
        "warnings": warnings,
    }
    if out_dir is not None:
        np.savez_compressed(
            out_dir / "trace.npz",
            t=np.asarray(trace["t"]),
            qpos=np.asarray(trace["qpos"]),
            qvel=np.asarray(trace["qvel"]),
            qacc=np.asarray(trace["qacc"]),
            ctrl=np.asarray(trace["ctrl"]),
            actfrc=np.asarray(trace["actfrc"]),
            tips=np.asarray(trace["tips"]),
            tip_finger_order=np.asarray(FINGERS),
            phase=np.asarray(trace["phase"]),
            series_t=np.asarray(series["t"]),
            series_t_ctrl=np.asarray(series["t_ctrl"]),
            series_phase=np.asarray(series["phase"]),
            series_limit_excursion_rad=np.asarray(series["limit_excursion_rad"]),
            series_ncon=np.asarray(series["ncon"]),
            series_max_tracking_rad=np.asarray(series["max_tracking_rad"]),
            actuator_joint_order=np.asarray(act_jnames),
        )
        dump_json(out_dir / "motion-metrics.json", metrics)
        dump_json(out_dir / "self-contacts.json", self_contact_log)
    return DiagResult(metrics=metrics, physics_seconds=physics_seconds, failures=failures)


def _phase_in_scope(phase: Phase, channels: tuple[str, ...]) -> bool:
    """Allow --tiny runs to restrict the sweep to a subset of channels."""
    if phase.name.startswith("sweep_"):
        for suffix in ("_ramp", "_hold", "_return"):
            if phase.name.endswith(suffix):
                jname = phase.name[len("sweep_") : -len(suffix)]
                return jname in channels
    return True
