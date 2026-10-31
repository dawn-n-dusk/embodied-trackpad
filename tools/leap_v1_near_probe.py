#!/usr/bin/env python3
"""Bounded v1 Full near-pointing VISUAL diagnostic probe (privileged teacher).

ONE seed -> ONE fresh output directory. Fixed prep (0.6 s open, 1.8 s
minimum-jerk to if_mcp=0.6/if_pip=0.3/if_dip=0.3, 0.5 s hold; physics only, no
visual claim), then a pixel-only visual loop: decode the confined
TrackpadSandbox's own-window PNG, gate it (freshness/uncertainty/confidence),
map the cursor->target pixel error through the 3x4 damped least-squares index
planner (bounds frozen in PLAN_CFG), apply 25 substeps (0.05 s) and stream the
ACTUAL aggregated contacts to the app as touch.v1. Success is <=3 px for 3 new
frames plus >=50% error reduction, initial error >=10 px; evaluator/button
flags are never consulted. Ends with a bounded cleanup (flex->0 over 0.8 s,
rot->0 over 0.7 s, open hold 0.7 s); release_measured requires actual trailing
>=0.5 s records with no pad contact kinds. First unsafe physics step stops ALL
physics (cleanup included); the app is always software-stopped and quit.

Clocks: the touch wire ``simulation_time_s`` is the physics ``env.data.time`` —
seconds since this episode's ``mj_resetData`` (a relative sim epoch, NOT wall
time and NOT the app capture clock). Frame ages come from the measured
ClockReconciler offset; its uncertainty feeds the FrameGate. Budgets: 120 s
total wall, 20 s total sim, 60 MiB output directory; at most 120 visual cycles,
each paced to >=0.1 s wall (measured, NOT a 20 Hz claim).

Privileged visual teacher diagnostic only: controller_kind is
``privileged_visual_teacher`` and every output is training_eligible=false.
No native GUI/AX automation, no evaluator/state()/seed-to-layout lookups;
pixel PNGs are the only observation. Coordinator source/hash acceptance is the
execution gate and is intentionally NOT re-validated here.

Usage:
    PYTHONPATH=src:. python tools/leap_v1_near_probe.py --seed 1 --out .local/v1-visual/near-SEED1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Numerical-thread cap (hard assignment) BEFORE numpy/mujoco are imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "2"

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import mujoco  # noqa: E402
from embodied_trackpad.leap_v1 import pad_env as _pad  # noqa: E402
from embodied_trackpad.leap_v1 import visual_physics as vp  # noqa: E402
from embodied_trackpad.leap_v1.model import TIP_SITES  # noqa: E402
from embodied_trackpad.leap_v1.visual_servo import (  # noqa: E402
    ClockReconciler, FrameGate, FrameGateConfig, decode_png_bytes,
    extract_visual_estimate,
)

# --- frozen constants (no configurable physics/safety) -----------------------
TIMESTEP_S = 0.002  # reviewed P1 scene timestep (verified by make_env)
WALL_BUDGET_S, SIM_BUDGET_S = 120.0, 20.0
OPERATION_WALL_S = WALL_BUDGET_S - 10.0  # last 10 s of wall reserved for teardown
DIR_CAP_BYTES = 60 * 1024 * 1024
DIR_HEADROOM_BYTES = 128 * 1024  # reserved for final records + summary
PREP_OPEN_S, PREP_MOVE_S, PREP_HOLD_S = 0.6, 1.8, 0.5
PRESS_POSE = {"if_mcp": 0.6, "if_pip": 0.3, "if_dip": 0.3}
CLEAN_FLEX_S, CLEAN_ROT_S, CLEAN_HOLD_S = 0.8, 0.7, 0.7
TOUCH_EVERY = 25  # substeps (0.05 s)
VISUAL_SUBSTEPS = 25  # 0.05 s sim per accepted frame
MAX_CYCLES, CYCLE_MIN_WALL_S = 120, 0.1
INITIAL_ERROR_MIN_PX, SUCCESS_PX, SUCCESS_FRAMES, SUCCESS_REDUCTION = 10.0, 3.0, 3, 0.5
RELEASE_WINDOW_S = 0.5
PLAN_CFG = vp.PlannerConfig(max_joint_delta_rad=0.005, max_planar_step_m=0.0005,
                            max_z_step_m=0.0002, damping=0.05, px_per_m=4000.0)


class ResourceStop(RuntimeError):
    """Wall/sim/bytes budget exhausted; the episode still ends safely."""


class TouchRejected(RuntimeError):
    """The device rejected a touch frame (non-touch_ok ack or bad sequence)."""


def min_jerk_blend(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def check_budgets(R: dict) -> None:
    """Operation budgets (wall keeps a 10 s teardown reserve; the projected
    NEXT sim step must fit; the dir cap keeps 128 KiB of final-record headroom)."""
    if time.monotonic() - R["t0"] > OPERATION_WALL_S:
        raise ResourceStop("operation_wall_budget_exceeded")
    if float(R["env"].data.time) + TIMESTEP_S > SIM_BUDGET_S:
        raise ResourceStop("sim_budget_exceeded")
    if dir_bytes(R["out"]) > DIR_CAP_BYTES - DIR_HEADROOM_BYTES:
        raise ResourceStop("dir_cap_exceeded")


def evaluate_success(initial_error_px: float, accepted_errors: list[float]) -> bool:
    if len(accepted_errors) < SUCCESS_FRAMES:
        return False
    tail = accepted_errors[-SUCCESS_FRAMES:]
    return all(e <= SUCCESS_PX for e in tail) and \
        tail[-1] <= SUCCESS_REDUCTION * initial_error_px


def release_from_records(records: list[dict]) -> bool:
    """True only on an actual measured release: timestamps must be finite,
    strictly increasing and contiguous (~one TIMESTEP_S apart), and the
    trailing window of records must SPAN >= RELEASE_WINDOW_S of sim time (250
    samples at 2 ms span only 0.498 s, so 251 are needed) with NO
    PAD_CONTACT_KINDS contact anywhere in it (hand_self does not block)."""
    if not records:
        return False
    tol = 1e-9
    times = []
    for r in records:
        t = r.get("sim_time_s")
        if not isinstance(t, float) or isinstance(t, bool) or not math.isfinite(t):
            return False
        times.append(t)
    for a, b in zip(times, times[1:]):
        gap = b - a
        if gap <= tol or gap > 1.5 * TIMESTEP_S + tol:  # monotonic + contiguous
            return False
    t_end = times[-1]
    start = len(times) - 1
    while start > 0 and t_end - times[start - 1] <= RELEASE_WINDOW_S + tol:
        start -= 1
    if t_end - times[start] < RELEASE_WINDOW_S - tol:
        return False  # trailing records do not span the full release window
    return not any(c.get("kind") in _pad.PAD_CONTACT_KINDS
                   for r in records[start:] for c in r.get("contacts", []))


def tip_z(env) -> float:
    return float(env.data.site(TIP_SITES["index"]).xpos[2])


def site_jacobian(env) -> np.ndarray:
    """3x4 translational mj_jacSite; column k = env.index_map.joint_names[k]."""
    jacp = np.zeros((3, env.model.nv))
    mujoco.mj_jacSite(env.model, env.data, jacp, None,
                      env.data.site(TIP_SITES["index"]).id)
    return jacp[:, list(env.index_map.dofadr)]


def step(R: dict, ctrl, frame_id, phase: str):
    """THE only physics path: budget check -> physics_step -> flushed record."""
    check_budgets(R)
    R["step_index"] += 1
    try:
        rec = vp.physics_step(R["env"], np.asarray(ctrl, dtype=float),
                              step_index=R["step_index"], frame_id=frame_id)
    except vp.PhysicsStepError as exc:
        line = dict(exc.record.to_dict(), phase=phase)
        R["teacher"].write(json.dumps(line, allow_nan=False) + "\n")
        R["teacher"].flush()
        raise
    line = dict(rec.to_dict(), phase=phase)
    R["teacher"].write(json.dumps(line, allow_nan=False) + "\n")
    R["teacher"].flush()
    R["last_record"] = line
    if phase.startswith("cleanup"):
        R["cleanup_records"].append(line)
    return rec


def send_touch(R: dict, aggregated: dict) -> None:
    """Stream actual aggregated contacts; raw payload+ack preserved BEFORE the
    ack is validated. A non-touch_ok ack or a mismatched sequence raises
    TouchRejected: the episode never continues past a rejected device frame."""
    if R.get("client") is None:
        return
    check_budgets(R)
    R["seq"] += 1
    seq = R["seq"]
    payload = {"cmd": "touch",
               **_pad.touch_v1_message(seq, float(R["env"].data.time), aggregated)}
    ack = R["client"].raw(payload)
    R["touch"].write(json.dumps({"payload": payload, "ack": ack}, allow_nan=False) + "\n")
    R["touch"].flush()
    if not isinstance(ack, dict) or ack.get("type") != "touch_ok" \
            or ack.get("sequence") != seq:
        raise TouchRejected(f"device rejected touch seq {seq}: {ack!r}")


def track(R: dict, ctrl_to, duration_s: float, phase: str, touch_every: int = 0):
    """Minimum-jerk blend R['ctrl'] -> ctrl_to over duration_s of 2 ms steps."""
    ctrl_from = np.array(R["ctrl"], dtype=float)
    ctrl_to = np.asarray(ctrl_to, dtype=float)
    n = max(1, int(round(duration_s / TIMESTEP_S)))
    rec = None
    for i in range(1, n + 1):
        R["ctrl"] = ctrl_from + (ctrl_to - ctrl_from) * min_jerk_blend(i / n)
        rec = step(R, R["ctrl"], None, phase)
        if touch_every and i % touch_every == 0:
            send_touch(R, rec.aggregate)
    return rec


def prep_phase(R: dict) -> None:
    track(R, np.zeros(vp.NU), PREP_OPEN_S, "prep_open")
    press = np.zeros(vp.NU)
    for name, value in PRESS_POSE.items():
        k = R["env"].index_map.joint_names.index(name)
        press[R["env"].index_map.act_index[k]] = value
    track(R, press, PREP_MOVE_S, "prep_move")
    track(R, press, PREP_HOLD_S, "prep_hold")
    R["z_hold"] = tip_z(R["env"])


def capture_estimate(R: dict):
    """Capture/save PNG -> decode -> extract; age measured AFTER decoding."""
    frame = R["client"].capture(save_as=f"cycle-{R['cycle']:03d}.png")
    est = extract_visual_estimate(decode_png_bytes(frame["png"]))
    age_ms = R["reconciler"].frame_age_ms(frame)
    return frame, est, age_ms


def log_capture(R: dict, frame: dict, est, age_ms: float, phase: str,
                err, prev, nxt, requested: bool) -> None:
    """Actor-side record of ONE capture (accepted, rejected or terminal). Pixel
    ref/hash + frame time + error stay here; Jacobian/contacts stay teacher-side.
    Schema note: ``action_requested`` marks only planner INTENT — this record is
    written BEFORE the first physics substep, so it never claims completion;
    actual execution is established solely by the per-substep teacher.jsonl
    StepRecords carrying the causing frame_id. Terminal/rejected frames carry
    action_requested=false with next_ctrl=null."""
    R["actor"].write(json.dumps(
        {"phase": phase, "cycle": R["cycle"],
         "png_ref": frame.get("path"),
         "png_sha256": hashlib.sha256(frame["png"]).hexdigest(),
         "frame_id": frame.get("frame_id"),
         "frame_monotonic_time_s": frame.get("monotonic_time_s"),
         "frame_age_ms": age_ms,
         "error_px": err,
         "q": [float(v) for v in R["env"].data.qpos],
         "qvel": [float(v) for v in R["env"].data.qvel],
         "prev_ctrl": None if prev is None else [float(v) for v in prev],
         "next_ctrl": None if nxt is None else [float(v) for v in nxt],
         "action_requested": bool(requested)}, allow_nan=False) + "\n")
    R["actor"].flush()


def visual_phase(R: dict) -> str:
    env = R["env"]
    for R["cycle"] in range(1, MAX_CYCLES + 1):
        check_budgets(R)
        t_cycle = time.monotonic()
        frame, est, age_ms = capture_estimate(R)
        decision = R["gate"].check(frame, est, age_ms)
        err = float(math.hypot(*est.error_px)) if est.error_px is not None else None
        if not decision.ok:
            log_capture(R, frame, est, age_ms, "gate_rejected", err, None, None, False)
            R["planner"].write(json.dumps(
                {"kind": "gate_reject", "cycle": R["cycle"],
                 "frame_id": frame.get("frame_id"), "frame_age_ms": age_ms,
                 "reasons": decision.reasons}, allow_nan=False) + "\n")
            R["planner"].flush()
            return "frame_gate_rejected: " + "; ".join(decision.reasons)
        ex, ey = est.error_px
        if R["initial_error"] is None:
            R["initial_error"] = err
        R["final_error"] = err
        R["errors"].append(err)
        if len(R["errors"]) == 1 and err < INITIAL_ERROR_MIN_PX:
            log_capture(R, frame, est, age_ms, "initial_below_min", err, None, None, False)
            return f"initial_error_below_min ({err:.2f}px < {INITIAL_ERROR_MIN_PX}px)"
        if evaluate_success(R["initial_error"], R["errors"]):
            log_capture(R, frame, est, age_ms, "success_terminal", err,
                        np.array(R["ctrl"], dtype=float), None, False)
            R["success"] = True
            return "success"
        jac = site_jacobian(env)
        z_err = R["z_hold"] - tip_z(env)
        prev = np.array(R["ctrl"], dtype=float)
        nxt, info = vp.plan_index_correction((ex, ey), z_err, jac, prev,
                                             env.index_map, PLAN_CFG)
        log_capture(R, frame, est, age_ms, "visual_step", err, prev, nxt, True)
        R["planner"].write(json.dumps(
            {"kind": "plan", "cycle": R["cycle"], "frame_id": frame["frame_id"],
             "error_px": [ex, ey], "z_error_m": float(z_err),
             "jac_3x4": [[float(v) for v in row] for row in jac],
             "task_m": info["task_m"], "dq_rad": info["dq_rad"]},
            allow_nan=False) + "\n")
        R["planner"].flush()
        R["ctrl"] = nxt
        for _ in range(VISUAL_SUBSTEPS):
            rec = step(R, nxt, frame["frame_id"], "visual")
        send_touch(R, rec.aggregate)
        dt = time.monotonic() - t_cycle
        if dt < CYCLE_MIN_WALL_S:
            time.sleep(CYCLE_MIN_WALL_S - dt)
        R["planner"].write(json.dumps(
            {"kind": "cycle_end", "cycle": R["cycle"], "frame_id": frame["frame_id"],
             "cycle_wall_s": time.monotonic() - t_cycle}, allow_nan=False) + "\n")
        R["planner"].flush()  # FULL cycle duration: physics + send + pacing
    return "max_cycles_exhausted"


def cleanup_phase(R: dict) -> bool:
    idx = R["env"].index_map
    flex0 = np.array(R["ctrl"], dtype=float)
    for name in ("if_mcp", "if_pip", "if_dip"):
        flex0[idx.act_index[idx.joint_names.index(name)]] = 0.0
    track(R, flex0, CLEAN_FLEX_S, "cleanup_flex", touch_every=TOUCH_EVERY)
    rot0 = np.array(R["ctrl"], dtype=float)
    rot0[idx.act_index[idx.joint_names.index("if_rot")]] = 0.0
    track(R, rot0, CLEAN_ROT_S, "cleanup_rot", touch_every=TOUCH_EVERY)
    track(R, rot0, CLEAN_HOLD_S, "cleanup_hold", touch_every=TOUCH_EVERY)
    return release_from_records(R["cleanup_records"])


def run(seed: int, out_dir) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()  # actual start
    R = {"t0": time.monotonic(), "out": out, "env": None, "client": None,
         "step_index": 0, "seq": 0, "cycle": 0, "ctrl": np.zeros(vp.NU),
         "z_hold": None, "initial_error": None, "final_error": None,
         "errors": [], "success": False, "cleanup_records": [],
         "last_record": None}
    with open(out / "metadata.json", "w") as fh:
        json.dump({"controller_kind": "privileged_visual_teacher",
                   "training_eligible": False, "seed": int(seed),
                   "started_utc": started_utc,
                   "budgets": {"wall_budget_s": WALL_BUDGET_S,
                               "operation_wall_s": OPERATION_WALL_S,
                               "sim_budget_s": SIM_BUDGET_S,
                               "dir_cap_bytes": DIR_CAP_BYTES,
                               "dir_headroom_bytes": DIR_HEADROOM_BYTES,
                               "max_cycles": MAX_CYCLES,
                               "teardown_reserve_s": WALL_BUDGET_S - OPERATION_WALL_S,
                               "teardown_bounds_s": {"stop_send": 1.0, "quit_wait": 1.0,
                                                     "terminate": 3.0, "kill": 5.0}},
                   "planner": {"max_joint_delta_rad": PLAN_CFG.max_joint_delta_rad,
                               "max_planar_step_m": PLAN_CFG.max_planar_step_m,
                               "max_z_step_m": PLAN_CFG.max_z_step_m,
                               "damping": PLAN_CFG.damping,
                               "px_per_m": PLAN_CFG.px_per_m},
                   "files": ["teacher.jsonl", "actor.jsonl", "planner.jsonl",
                             "touch.jsonl", "summary.json"]}, fh, indent=2)
    stop_reason, unsafe, release, cleaned = "not_started", False, False, False
    teardown = {"stop_error": None, "quit_error": None}
    try:
        try:
            R["teacher"] = open(out / "teacher.jsonl", "w")
            R["actor"] = open(out / "actor.jsonl", "w")
            R["planner"] = open(out / "planner.jsonl", "w")
            R["touch"] = open(out / "touch.jsonl", "w")
            R["env"] = vp.make_env()
            vp.reset_open(R["env"])
            prep_phase(R)
            sys.path.insert(0, str(REPO_ROOT / "tools"))
            from mac_sandbox_client import SandboxClient  # actor-only client
            R["client"] = SandboxClient(out_dir=str(out))
            # Per-instance adapter: cap every app read by the remaining
            # OPERATION budget (restored before the independent teardown).
            orig_send = R["client"]._send
            R["client_orig_send"] = orig_send

            def _send_budgeted(obj, timeout=30.0, _orig=orig_send):
                left = OPERATION_WALL_S - (time.monotonic() - R["t0"])
                return _orig(obj, timeout=max(0.001, min(timeout, left)))

            R["client"]._send = _send_budgeted
            R["client"].reset(seed=seed, task_profile="near_pointing_v1")
            send_touch(R, R["last_record"]["aggregate"])  # actual held contacts
            R["reconciler"] = ClockReconciler.sample(R["client"].capture)
            R["gate"] = FrameGate(FrameGateConfig(
                clock_uncertainty_ms=R["reconciler"].uncertainty_ms))
            stop_reason = visual_phase(R)
        except ResourceStop as exc:
            stop_reason = f"resource_stop:{exc}"
        except vp.PhysicsStepError:
            stop_reason, unsafe = "physics_unsafe", True
        except Exception as exc:  # client/build/touch errors: still finalize
            stop_reason = f"error:{type(exc).__name__}:{exc}"
        # First unsafe physics step forbids ALL further physics incl. cleanup.
        if not unsafe and R.get("env") is not None:
            try:
                release = bool(cleanup_phase(R))
                cleaned = True
            except vp.PhysicsStepError:
                stop_reason = f"{stop_reason}|cleanup_physics_unsafe"
                unsafe, release, cleaned = True, False, False
            except ResourceStop as exc:
                stop_reason = f"{stop_reason}|cleanup_resource_stop:{exc}"
                release, cleaned = False, False
            except Exception as exc:  # generic cleanup failure is reported
                stop_reason = f"{stop_reason}|cleanup_error:{type(exc).__name__}:{exc}"
                release, cleaned = False, False
    finally:
        client = R.get("client")
        child_pid, returncode, reaped = None, None, False
        if client is not None:
            child_pid = client.proc.pid
            # Independent BOUNDED teardown inside the reserved 10 s: cap every
            # teardown _send at 1 s and quit's wait at 1 s; the client's own
            # _teardown adds terminate(3 s)+kill(5 s), so the worst case stays
            # within ~10 s (see metadata budgets.teardown_bounds_s).
            if R.get("client_orig_send") is not None:
                orig_send = R["client_orig_send"]

                def _send_teardown(obj, timeout=1.0, _orig=orig_send):
                    return _orig(obj, timeout=min(float(timeout), 1.0))

                client._send = _send_teardown
            try:
                client.stop()
            except Exception as exc:
                teardown["stop_error"] = f"{type(exc).__name__}:{exc}"
            try:
                returncode = client.quit(timeout=1.0)
            except Exception as exc:
                teardown["quit_error"] = f"{type(exc).__name__}:{exc}"
                returncode = client.proc.returncode
            reaped = returncode is not None and teardown["quit_error"] is None
            teardown["send_timeout_s"] = 1.0
            teardown["quit_timeout_s"] = 1.0
            teardown["worst_case_s"] = 10.0  # 1 stop + 1 quit + 3 terminate + 5 kill
        for key in ("teacher", "actor", "planner", "touch"):
            fh = R.get(key)
            if fh is not None and not fh.closed:
                fh.close()
        summary = {
            "seed": int(seed),
            "started_utc": started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "near_pointing_success": bool(R["success"]),
            "episode_safe_complete_released": bool(cleaned and not unsafe and release),
            "release_measured": bool(release), "physics_unsafe": bool(unsafe),
            "initial_error_px": R["initial_error"], "final_error_px": R["final_error"],
            "accepted_frames": len(R["errors"]), "steps": R["step_index"],
            "wall_s": round(time.monotonic() - R["t0"], 3),
            "stop_reason": stop_reason, "bytes": dir_bytes(out),
            "controller_kind": "privileged_visual_teacher",
            "training_eligible": False,
            "clock_note": "touch simulation_time_s = physics env.data.time "
                          "(epoch: this episode's mj_resetData; relative sim clock)",
            "child_pid": child_pid, "child_returncode": returncode,
            "child_reaped": bool(reaped), "teardown": teardown,
        }
        with open(out / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, allow_nan=False)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="NEW output directory (must not exist)")
    args = ap.parse_args(argv)
    if args.seed < 0:
        ap.error("--seed must be >= 0")
    if Path(args.out).exists():
        ap.error(f"--out must be a NEW directory, already exists: {args.out}")
    summary = run(args.seed, args.out)
    print(json.dumps(summary, indent=2))
    ok = (summary["near_pointing_success"]
          and summary["episode_safe_complete_released"]
          and summary["child_reaped"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
