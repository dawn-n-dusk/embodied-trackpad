"""Bounded CPU probe for the imported LEAP Hand v2 Advanced model.

Executes a scripted flex / side-spread / pinch / release motion purely through
``data.ctrl`` + ``mujoco.mj_step`` (no qpos teleports after the reset pose),
records actual q/qvel/qacc/ctrl/force trajectories and measured fingertip
distances, and enforces a numerical contract with an explicit verdict and a
nonzero exit code on failure:

  * finite qpos/qvel/qacc/ctrl/actuator-force at every step
  * joint-limit soft excursion <= 0.01 rad
  * dip/pip coupling residual <= 0.01 rad
  * zero MuJoCo solver warnings
  * any wrong-shape/non-finite/out-of-range command raises ValueError before
    any ctrl assignment (never clipped, never NaN-patched), so mj_step never
    sees an invalid command
  * candidate-B motion gates (full runs only; reported as not evaluated for
    --tiny): max contact penetration <= 0.001 m, final open-pose error
    <= 0.05 rad after a >=0.5 s hold, held-phase actuator tracking
    <= 0.10 rad over ALL steps in 0.3-0.5 / 2.2-2.5 / 6.6-7.0 s

After every mj_step the forward state is refreshed at the observation instant
(mj_forward) so recorded site positions/contacts/forces match the recorded
qpos and the actual data.time; the control start time is stored alongside.
The output directory must not already exist.

Renders >=640x480 PNG frames (initial/flex/spread/release) from side and palm
cameras for coordinator visual review. Free-space hand-only motion; any
remaining self-contacts are reported, not hidden; this is NOT touch readiness.

Usage:
    python tools/leap_model_probe.py [--out .local/leap-probe-r3] [--tiny]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_trackpad.leap_sim import (  # noqa: E402
    DERIVED_URDF,
    MANIFEST_JSON,
    TIP_SITES,
    load_hand,
    model_inventory,
    open_pose_qpos,
)

RENDER_W, RENDER_H = 640, 480
FINGERS = ("index", "middle", "ring", "pinky")

LIMIT_EXCURSION_TOL = 0.01  # rad; contract gate
COUPLING_TOL = 0.01  # rad; contract gate
PENETRATION_TOL = 0.001  # m; candidate-B gate
FINAL_OPEN_TOL = 0.05  # rad; candidate-B gate (after >=0.5 s open hold)
HELD_TRACKING_TOL = 0.10  # rad; candidate-B gate over HELD_WINDOWS
MIN_OPEN_HOLD_S = 0.5  # s; required hold before the final open-pose error gate
HELD_WINDOWS = ((0.3, 0.5), (2.2, 2.5), (6.6, 7.0))  # script hold phases, seconds

# Script keyframes: (time_s, joint->target). OPEN resets the full vector to the
# open pose; every other keyframe INHERITS the previous keyframe's complete
# 17-motor vector and overrides only its listed keys, so no commanded channel
# is ever dropped mid-script (the old flex->spread transition silently reset
# unlisted joints to open defaults). Interpolation is piecewise-linear between
# consecutive full vectors, hence continuous.
OPEN = {}  # sentinel: expanded to the full open-pose vector below
FLEX = {
    **{f"{f}_dip": 1.4 for f in FINGERS},
    **{f"{f}_mcpf": 1.2 for f in FINGERS},
    "thumb_ip": 1.4,
    "thumb_mcpf": 1.2,
    "thumb_mcps": -0.3,
}
SPREAD = {
    **{f"{f}_dip": 0.6 for f in FINGERS},
    "index_mcps": 0.5,
    "pinky_mcps": -0.5,
}
PINCH = {
    "index_dip": 1.6,
    "index_mcpf": 1.3,
    "thumb_ip": 1.6,
    "thumb_mcpf": 1.5,
    "thumb_mcps": -0.45,
}
KEYFRAMES = [
    (0.0, OPEN),
    (0.5, OPEN),
    (2.0, FLEX),
    (2.5, FLEX),
    (3.5, SPREAD),
    (3.8, OPEN),
    (5.2, PINCH),
    (6.4, OPEN),
    (7.0, OPEN),
]
T_END = KEYFRAMES[-1][0]


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal stdlib PNG writer (8-bit RGB)."""
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _full_targets(m: mujoco.MjModel, partial: dict[str, float], base: np.ndarray) -> np.ndarray:
    """Overlay a partial joint->target dict on a complete base ctrl vector."""
    ctrl = base.copy()
    for i in range(m.nu):
        jname = m.joint(m.actuator_trnid[i, 0]).name
        if jname in partial:
            ctrl[i] = partial[jname]
    return ctrl


def _open_vector(m: mujoco.MjModel) -> np.ndarray:
    from embodied_trackpad.leap_sim import OPEN_POSE

    base = np.zeros(m.nu)
    for i in range(m.nu):
        jname = m.joint(m.actuator_trnid[i, 0]).name
        base[i] = OPEN_POSE.get(jname, 0.0)
    return base


def ctrl_at(m: mujoco.MjModel, t: float) -> np.ndarray:
    vecs = []
    prev = _open_vector(m)
    for tk, partial in KEYFRAMES:
        # OPEN resets to the open pose; other keyframes inherit previous keys.
        prev = prev if partial is not OPEN else _open_vector(m)
        prev = _full_targets(m, partial, prev)
        vecs.append((tk, prev))
    if t <= vecs[0][0]:
        return vecs[0][1]
    for (t0, v0), (t1, v1) in zip(vecs, vecs[1:]):
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0)
            return (1 - a) * v0 + a * v1
    return vecs[-1][1]


def apply_ctrl_checked(
    m: mujoco.MjModel, d: mujoco.MjData, desired: np.ndarray
) -> None:
    """Validate the complete (nu,) command and assign it atomically.

    Shape, finiteness, and every actuator's ctrlrange are checked BEFORE any
    element of ``d.ctrl`` is written. Any violation raises ValueError and
    leaves ``d.ctrl`` completely unchanged: commands are never clipped and
    NaNs are never replaced, so an invalid command can never reach mj_step.
    """
    desired = np.asarray(desired, dtype=float)
    if desired.shape != (m.nu,):
        raise ValueError(f"ctrl shape {desired.shape} != ({m.nu},)")
    if not np.all(np.isfinite(desired)):
        raise ValueError("non-finite ctrl command")
    lo = m.actuator_ctrlrange[:, 0]
    hi = m.actuator_ctrlrange[:, 1]
    bad = np.nonzero((desired < lo - 1e-9) | (desired > hi + 1e-9))[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"ctrl[{i}]={float(desired[i])} outside ctrlrange "
            f"[{float(lo[i])}, {float(hi[i])}] for actuator {m.actuator(i).name}"
        )
    d.ctrl[:] = desired


def make_cameras(m: mujoco.MjModel, d: mujoco.MjData) -> dict[str, mujoco.MjvCamera]:
    xs = d.geom_xpos
    lo, hi = xs.min(0), xs.max(0)
    center = ((lo + hi) / 2).tolist()
    dist = max(0.25, float(np.linalg.norm(hi - lo)) * 1.4)
    side = mujoco.MjvCamera()
    side.lookat = center
    side.distance = dist
    side.azimuth = 130.0
    side.elevation = -25.0
    palm = mujoco.MjvCamera()
    palm.lookat = center
    palm.distance = dist * 0.9
    palm.azimuth = -45.0
    palm.elevation = -60.0
    return {"side": side, "palm": palm}


def coupling_error(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    err = 0.0
    for f in FINGERS:
        jd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_dip")
        jp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_pip")
        err = max(err, abs(float(d.qpos[m.jnt_qposadr[jd]] - d.qpos[m.jnt_qposadr[jp]])))
    return err


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def contact_snapshot(m: mujoco.MjModel, d: mujoco.MjData) -> dict:
    pairs = {}
    min_dist = 0.0
    worst = None
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body(int(m.geom_bodyid[int(c.geom1)])).name or "world"
        b2 = m.body(int(m.geom_bodyid[int(c.geom2)])).name or "world"
        key = (b1, b2)
        pairs[key] = min(pairs.get(key, 0.0), float(c.dist))
        if c.dist < min_dist:
            min_dist = float(c.dist)
            worst = [b1, b2]
    return {
        "ncon": int(d.ncon),
        "pairs": {f"{a}|{b}": v for a, b, v in ((k[0], k[1], v) for k, v in pairs.items())},
        "max_penetration_m": -min_dist if min_dist < 0 else 0.0,
        "worst_pair": worst,
    }


def run_probe(out_dir: Path, tiny: bool = False) -> dict:
    started = datetime.now(timezone.utc)
    if out_dir.exists():
        raise FileExistsError(
            f"output directory already exists, refusing to overwrite: {out_dir}"
        )
    out_dir.mkdir(parents=True)
    hand = load_hand()
    m = hand.model
    inventory = model_inventory(hand)
    (out_dir / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")

    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = open_pose_qpos(m)  # the only live qpos write; all later motion is ctrl+mj_step
    apply_ctrl_checked(m, d, _open_vector(m))  # holding ctrl in place before the initial forward
    mujoco.mj_forward(m, d)
    initial_contacts = contact_snapshot(m, d)
    (out_dir / "initial-contacts.json").write_text(
        json.dumps({"state": "reset+OPEN_POSE+holding-ctrl+forward, before first mj_step", **initial_contacts}, indent=2) + "\n"
    )

    cams = make_cameras(m, d)
    renderer = mujoco.Renderer(m, height=RENDER_H, width=RENDER_W)

    steps = 100 if tiny else int(round(T_END / m.opt.timestep))
    render_steps = {} if tiny else {
        0: "initial",
        int(2.4 / m.opt.timestep): "flex",
        int(3.49 / m.opt.timestep): "spread",
        int(6.55 / m.opt.timestep): "release",
    }

    traj = {"t": [], "t_ctrl": [], "qpos": [], "qvel": [], "qacc": [], "ctrl": [], "actfrc": [],
            "tips": [], "dist_index_thumb": []}
    failures: list[str] = []
    max_eq_err = 0.0
    max_abs_qvel = 0.0
    max_limit_excursion = 0.0
    max_penetration = 0.0
    n_contacts_max = 0
    ctrl_violations: list[dict] = []  # stays empty: invalid commands raise before any step
    warn_start = np.array([w.number for w in d.warning], dtype=int)
    contact_pairs_seen: dict[str, float] = {}
    final_open_err = 0.0
    held_tracking = {f"{a}-{b}s": 0.0 for a, b in HELD_WINDOWS}
    trnid = m.actuator_trnid[:, 0]
    act_qposadr = m.jnt_qposadr[trnid]

    for k in range(steps):
        t_ctrl = k * m.opt.timestep  # control start time of this step
        desired = ctrl_at(m, t_ctrl)
        apply_ctrl_checked(m, d, desired)  # raises before mj_step on any invalid command
        mujoco.mj_step(m, d)
        # Refresh the forward state at the observation instant so recorded site
        # positions, contacts, and forces belong to the recorded qpos/d.time,
        # not to the pre-integration state left by mj_step.
        mujoco.mj_forward(m, d)
        t = float(d.time)
        if not (
            np.all(np.isfinite(d.qpos))
            and np.all(np.isfinite(d.qvel))
            and np.all(np.isfinite(d.qacc))
            and np.all(np.isfinite(d.ctrl))
            and np.all(np.isfinite(d.actuator_force))
        ):
            failures.append(f"non-finite state at step {k} (t_ctrl={t_ctrl:.3f})")
            break
        max_eq_err = max(max_eq_err, coupling_error(m, d))
        max_abs_qvel = max(max_abs_qvel, float(np.abs(d.qvel).max()))
        step_tracking = float(np.abs(d.ctrl - d.qpos[act_qposadr]).max())
        for a, b in HELD_WINDOWS:
            if a - 1e-9 <= t <= b + 1e-9:
                key = f"{a}-{b}s"
                held_tracking[key] = max(held_tracking[key], step_tracking)
        snap = contact_snapshot(m, d)
        n_contacts_max = max(n_contacts_max, snap["ncon"])
        max_penetration = max(max_penetration, snap["max_penetration_m"])
        for pair, dist in snap["pairs"].items():
            contact_pairs_seen[pair] = min(contact_pairs_seen.get(pair, 0.0), dist)
        for j in range(m.njnt):
            lo, hi = m.jnt_range[j]
            q = float(d.qpos[m.jnt_qposadr[j]])
            max_limit_excursion = max(max_limit_excursion, max(lo - q, q - hi, 0.0))
        if k % 10 == 0 or k == steps - 1 or tiny:
            traj["t"].append(t)
            traj["t_ctrl"].append(t_ctrl)
            traj["qpos"].append(d.qpos.copy())
            traj["qvel"].append(d.qvel.copy())
            traj["qacc"].append(d.qacc.copy())
            traj["ctrl"].append(d.ctrl.copy())
            traj["actfrc"].append(d.actuator_force.copy())
            traj["tips"].append(np.stack([d.site(s).xpos for s in TIP_SITES]))
            traj["dist_index_thumb"].append(hand.tip_distance(d))
        if k in render_steps:
            for cam_name, cam in cams.items():
                renderer.update_scene(d, cam)
                png = out_dir / f"{render_steps[k]}_{cam_name}.png"
                write_png(png, renderer.render())
    renderer.close()

    # recovered open-pose error at the end (release quality, measured)
    if not tiny and not failures:
        open_q = open_pose_qpos(m)
        final_open_err = float(np.abs(d.qpos - open_q).max())

    warn_end = np.array([w.number for w in d.warning], dtype=int)
    warn_delta = (warn_end - warn_start).tolist()
    warn_names = [str(w).split(".")[-1] for w in mujoco.mjtWarning.__members__.values()]
    warnings = {warn_names[i]: int(warn_delta[i]) for i in range(len(warn_delta)) if warn_delta[i]}

    dists = np.asarray(traj["dist_index_thumb"])
    tips_arr = np.asarray(traj["tips"])
    qpos_arr = np.asarray(traj["qpos"])
    ctrl_arr = np.asarray(traj["ctrl"])

    # command tracking: worst |ctrl - q_actuated| over the run (servo lag measure)
    if len(qpos_arr):
        trnid = m.actuator_trnid[:, 0]
        q_act = qpos_arr[:, m.jnt_qposadr[trnid]]
        tracking_err = float(np.abs(ctrl_arr - q_act).max())
    else:
        tracking_err = None

    open_hold_s = KEYFRAMES[-1][0] - KEYFRAMES[-2][0]
    held_tracking_max = max(held_tracking.values()) if held_tracking else 0.0
    checks = {
        "finite_states": not failures,
        "max_limit_excursion_rad": max_limit_excursion,
        "max_coupling_error_rad": max_eq_err,
        "warnings": warnings,
        "ctrl_out_of_bounds_count": len(ctrl_violations),
        "max_abs_qvel_rad_s": max_abs_qvel,
        "max_contact_penetration_m": max_penetration,
        "max_command_tracking_error_rad": tracking_err,
        "final_open_pose_error_rad": None if tiny else final_open_err,
        "open_hold_duration_s": None if tiny else open_hold_s,
        "held_phase_tracking_rad": (
            "not_evaluated" if tiny else {**held_tracking, "max": held_tracking_max}
        ),
    }
    gates = {
        "finite": not failures,
        "limit_excursion<=0.01": max_limit_excursion <= LIMIT_EXCURSION_TOL,
        "coupling<=0.01": max_eq_err <= COUPLING_TOL,
        "no_warnings": not warnings,
        "ctrl_in_bounds": not ctrl_violations,
    }
    # Candidate-B motion gates are only evaluated on full runs; --tiny reports
    # them as not evaluated and never implies general motion readiness.
    if not tiny:
        gates.update(
            {
                "penetration<=0.001": max_penetration <= PENETRATION_TOL,
                "final_open<=0.05_after>=0.5s_hold": (
                    open_hold_s >= MIN_OPEN_HOLD_S and final_open_err <= FINAL_OPEN_TOL
                ),
                "held_phase_tracking<=0.10": held_tracking_max <= HELD_TRACKING_TOL,
            }
        )
    verdict = "PASS" if all(gates.values()) else "FAIL"
    reasons = [name for name, ok in gates.items() if not ok] + failures

    metrics = {
        "probe": "leap-v2-adv-fixed-palm-cpu",
        "verdict": verdict,
        "failure_reasons": reasons,
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tiny": tiny,
        "motion_gates_evaluated": not tiny,
        "motion_gates_note": (
            None
            if not tiny
            else "tiny run: penetration/final-open/held-tracking gates not evaluated; NOT general motion-ready"
        ),
        "steps": steps,
        "timestep": float(m.opt.timestep),
        "hashes": {
            "model_py_sha256": _sha256(REPO_ROOT / "src/embodied_trackpad/leap_sim/model.py"),
            "probe_py_sha256": _sha256(Path(__file__).resolve()),
            "manifest_sha256": _sha256(MANIFEST_JSON),
            "upstream_urdf_sha256": _sha256(
                REPO_ROOT / "assets/downloads/leap_hand_v2_adv/leap_v2_right/robot.urdf"
            ),
            "derived_urdf_sha256": _sha256(DERIVED_URDF),
        },
        "checks": checks,
        "gates": gates,
        "contacts": {
            "initial": initial_contacts,
            "max_simultaneous_contacts": n_contacts_max,
            "pairs_min_signed_distance_m": contact_pairs_seen,
            "note": "free-space hand-only motion; no pad/GUI; NOT touch readiness",
        },
        "ctrl_violations": ctrl_violations[:50],
        "motion": {
            "index_thumb_tip_distance_m": {
                "initial": float(dists[0]) if len(dists) else None,
                "min": float(dists.min()) if len(dists) else None,
                "final": float(dists[-1]) if len(dists) else None,
            },
            "tip_site_order": list(TIP_SITES),
            "tip_travel_m_initial_to_final": (
                np.linalg.norm(tips_arr[-1] - tips_arr[0], axis=1).tolist() if len(tips_arr) else []
            ),
        },
    }
    np.savez_compressed(
        out_dir / "trajectories.npz",
        t=np.asarray(traj["t"]),
        t_ctrl=np.asarray(traj["t_ctrl"]),
        qpos=qpos_arr,
        qvel=np.asarray(traj["qvel"]),
        qacc=np.asarray(traj["qacc"]),
        ctrl=ctrl_arr,
        actfrc=np.asarray(traj["actfrc"]),
        tips=tips_arr,
        dist_index_thumb=dists,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / ".local" / "leap-probe-r3")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    try:
        metrics = run_probe(args.out, tiny=args.tiny)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: metrics[k] for k in ("verdict", "failure_reasons", "checks", "motion")}, indent=2))
    return 0 if metrics["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
