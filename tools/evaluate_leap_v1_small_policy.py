#!/usr/bin/env python3
"""Bounded v1 Full learned small-policy evaluation runner.

This runner executes ONE seed -> ONE fresh output directory for ONE target
class (4 or 10).  It reuses the reviewed P1 physics prep, capture/frame gate,
contact streaming, safety and cleanup/teardown paths from the privileged
teacher probe, but replaces the visual-phase planner with the frozen NumPy
small policy loaded from an NPZ checkpoint.

A private evaluation permit is required at CLI: it binds the exact checkpoint
SHA, required source-file hashes, sample-index SHA and source commit.  The
permit and checkpoint are validated BEFORE any output directory or sandbox app
is created.  The learned phase never calls force/Jacobian/planner helpers;
privileged safety supervision (raw per-step checks and the contact-loss guard)
may only stop the episode, never correct the motor action.

Prep and release are scripted and explicitly labelled as such in the metadata
and summary.  No qpos changes occur beyond the initial reset.

Usage (coordinator source/hash approval required):
    PYTHONPATH=src:. python tools/evaluate_leap_v1_small_policy.py \
        --seed 4 --class 4 \
        --checkpoint path/to/checkpoint.npz \
        --permit path/to/eval-permit.json \
        --sample-index path/to/samples.jsonl \
        --source-commit 5344a3a... \
        --out .local/v1-visual/learned-c4-SEED4
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

# Numerical-thread cap BEFORE numpy/mujoco imports.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "2"

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import mujoco  # noqa: E402
from embodied_trackpad.leap_v1 import normal_priority as npp  # noqa: E402
from embodied_trackpad.leap_v1 import pad_env as _pad  # noqa: E402
from embodied_trackpad.leap_v1 import visual_physics as vp  # noqa: E402
from embodied_trackpad.leap_v1 import visual_servo as vs  # noqa: E402
from embodied_trackpad.leap_v1.learned_evaluator import (  # noqa: E402
    REQUIRED_SOURCE_FILES,
    EvaluationContract,
    PolicyInference,
    compute_screen_dependence,
    compute_source_file_hashes,
    stop_and_reap_client,
    validate_controls_against_ranges,
    validate_evaluation_permit,
)
from embodied_trackpad.leap_v1.visual_servo import (  # noqa: E402
    ClockReconciler, FrameGate, FrameGateConfig, decode_png_bytes,
    extract_visual_estimate,
)

sys.path.insert(0, str(REPO_ROOT / "tools"))
from mac_sandbox_client import SandboxClient  # noqa: E402

# --- frozen constants (no configurable physics/safety) -----------------------
TIMESTEP_S = 0.002
WALL_BUDGET_S, SIM_BUDGET_S = 120.0, 20.0  # 10 s teardown reserve subtracted at runtime
DIR_CAP_BYTES = 60 * 1024 * 1024
DIR_HEADROOM_BYTES = 128 * 1024
PREP_OPEN_S, PREP_MOVE_S, PREP_HOLD_S = 0.6, 1.8, 0.5
PRESS_POSE = {"if_mcp": 0.6, "if_pip": 0.3, "if_dip": 0.3}
CLEAN_FLEX_S, CLEAN_ROT_S, CLEAN_HOLD_S = 0.8, 0.7, 0.7
TOUCH_EVERY = 25
SUBSTEPS_PER_FRAME = 25
MAX_ACTION_IMAGES = 120
CYCLE_MIN_WALL_S = 0.1
INITIAL_ERROR_MIN_PX, SUCCESS_PX, SUCCESS_FRAMES, SUCCESS_REDUCTION = 10.0, 3.0, 3, 0.5
RELEASE_WINDOW_S = 0.5
ALLOWED_CLASSES = frozenset({4, 10})
DATASET_SCHEMA = "v1-small-policy-smoke-v1"


def _git_head(repo_root: Path) -> str:
    """Return the current git HEAD SHA, or '' if git is unavailable."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        return out.strip()
    except Exception:
        return ""


class ResourceStop(RuntimeError):
    """Wall/sim/bytes budget exhausted."""


class TouchRejected(RuntimeError):
    """Device rejected a touch frame."""


def min_jerk_blend(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def check_budgets(R: dict, limits: dict) -> None:
    """Operation budgets with teardown reserve and directory-cap headroom.

    The wall budget honors the permit's smaller cap while reserving 10 s for
    teardown; the permit is rejected before work if it cannot accommodate that
    reserve.
    """
    if time.monotonic() - R["t0"] > R["operation_wall_s"]:
        raise ResourceStop("operation_wall_budget_exceeded")
    if float(R["env"].data.time) + TIMESTEP_S > min(SIM_BUDGET_S, limits["max_sim_s"]):
        raise ResourceStop("sim_budget_exceeded")
    if dir_bytes(R["out"]) > min(DIR_CAP_BYTES, limits["dir_cap_bytes"]) - DIR_HEADROOM_BYTES:
        raise ResourceStop("dir_cap_exceeded")


def evaluate_success(initial_error_px: float, accepted_errors: list[float]) -> bool:
    if initial_error_px < INITIAL_ERROR_MIN_PX:
        return False
    if len(accepted_errors) < SUCCESS_FRAMES:
        return False
    tail = accepted_errors[-SUCCESS_FRAMES:]
    return all(e <= SUCCESS_PX for e in tail) and tail[-1] <= SUCCESS_REDUCTION * initial_error_px


def release_from_records(records: list[dict]) -> bool:
    """True only on a measured trailing release (no pad contacts for >=0.5 s)."""
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
        if gap <= tol or gap > 1.5 * TIMESTEP_S + tol:
            return False
    t_end = times[-1]
    start = len(times) - 1
    while start > 0 and t_end - times[start - 1] <= RELEASE_WINDOW_S + tol:
        start -= 1
    if t_end - times[start] < RELEASE_WINDOW_S - tol:
        return False
    return not any(c.get("kind") in _pad.PAD_CONTACT_KINDS
                   for r in records[start:] for c in r.get("contacts", []))


def step(R: dict, ctrl, frame_id, phase: str):
    """Single validated physics step; first failure is terminal."""
    check_budgets(R, R["contract"].limits)
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


def send_touch(R: dict, aggregated: dict) -> None:
    """Stream actual aggregated contacts; validate ack before continuing."""
    if R.get("client") is None:
        return
    check_budgets(R, R["contract"].limits)
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


def capture_estimate(R: dict):
    """Capture/save PNG -> decode -> extract; age measured AFTER decoding."""
    frame = R["client"].capture(save_as=f"cycle-{R['cycle']:03d}.png")
    est = extract_visual_estimate(decode_png_bytes(frame["png"]))
    age_ms = R["reconciler"].frame_age_ms(frame)
    return frame, est, age_ms


def _actor_error_px(est) -> list[float] | None:
    if est.error_px is None:
        return None
    return [float(est.error_px[0]), float(est.error_px[1])]


def log_action_intent(
    R: dict,
    frame: dict,
    est,
    age_ms: float,
    inference_result: dict,
    counterfactual: dict,
    before_q: np.ndarray,
    before_qvel: np.ndarray,
    before_prev: np.ndarray,
    step_start: int,
) -> None:
    """Flush the full predicted/intent record BEFORE any physics step.

    The planned step range is labelled as planned; the actual executed range is
    recorded separately after the block finishes or aborts.  If the first step
    fails, this intent record still survives as evidence of what was requested.
    """
    R["actor"].write(json.dumps({
        "phase": "learned_action_intent",
        "cycle": R["cycle"],
        "png_ref": frame.get("path"),
        "png_sha256": hashlib.sha256(frame["png"]).hexdigest(),
        "frame_id": frame.get("frame_id"),
        "frame_monotonic_time_s": frame.get("monotonic_time_s"),
        "frame_age_ms": age_ms,
        "error_px": _actor_error_px(est),
        "before_q": [float(v) for v in before_q],
        "before_qvel": [float(v) for v in before_qvel],
        "before_prev_ctrl": [float(v) for v in before_prev],
        "requested_controls_rad": [[float(v) for v in row] for row in inference_result["requested_controls_rad"]],
        "predicted_deltas_rad": [[float(v) for v in row] for row in inference_result["predicted_deltas_rad"]],
        "features": [float(v) for v in inference_result["features"]],
        "counterfactual": counterfactual,
        "sensor_valid": {"fsr": False, "motor_current": False, "wrist": False},
        "fsr_N4": None,
        "motor_current_A16": None,
        "wrist": None,
        "clock_offset_s": R["reconciler"].offset_s,
        "clock_uncertainty_s": R["reconciler"].uncertainty_s,
        "cause_frame_id": frame.get("frame_id"),
        "planned_step_index_start": step_start,
        "planned_step_index_end": step_start + SUBSTEPS_PER_FRAME - 1,
        "action_requested": True,
    }, allow_nan=False) + "\n")
    R["actor"].flush()


def log_action_executed(
    R: dict,
    frame: dict,
    est,
    step_start: int,
    step_end: int,
    outcome: str,
) -> None:
    """Append the actual executed step count/end and outcome for an action block."""
    R["actor"].write(json.dumps({
        "phase": "learned_action_executed",
        "cycle": R["cycle"],
        "png_ref": frame.get("path"),
        "frame_id": frame.get("frame_id"),
        "error_px": _actor_error_px(est),
        "planned_step_index_start": step_start,
        "executed_step_count": max(0, step_end - step_start + 1),
        "executed_step_index_end": step_end,
        "outcome": outcome,
    }, allow_nan=False) + "\n")
    R["actor"].flush()


def log_capture_reject(R: dict, frame: dict, est, age_ms: float, reasons: list[str]) -> None:
    R["actor"].write(json.dumps({
        "phase": "gate_rejected",
        "cycle": R["cycle"],
        "png_ref": frame.get("path"),
        "png_sha256": hashlib.sha256(frame["png"]).hexdigest(),
        "frame_id": frame.get("frame_id"),
        "frame_monotonic_time_s": frame.get("monotonic_time_s"),
        "frame_age_ms": age_ms,
        "error_px": _actor_error_px(est),
        "reasons": reasons,
        "sensor_valid": {"fsr": False, "motor_current": False, "wrist": False},
        "fsr_N4": None,
        "motor_current_A16": None,
        "wrist": None,
        "action_requested": False,
    }, allow_nan=False) + "\n")
    R["actor"].flush()


def visual_phase_learned(R: dict, inference: PolicyInference) -> str:
    """Learned visual servo loop: one new image -> one 25-step control block."""
    env = R["env"]
    guard = npp.ContactLossGuard()
    ranges = list(R["full_map"].ranges)

    for R["cycle"] in range(1, min(MAX_ACTION_IMAGES, R["contract"].limits["max_action_images"]) + 1):
        check_budgets(R, R["contract"].limits)
        t_cycle = time.monotonic()
        frame, est, age_ms = capture_estimate(R)
        decision = R["gate"].check(frame, est, age_ms)
        err = float(math.hypot(*est.error_px)) if est.error_px is not None else None
        if not decision.ok:
            log_capture_reject(R, frame, est, age_ms, decision.reasons)
            return "frame_gate_rejected: " + "; ".join(decision.reasons)

        ex, ey = est.error_px
        if R["initial_error"] is None:
            R["initial_error"] = err
        R["final_error"] = err
        R["errors"].append(err)
        if len(R["errors"]) == 1 and err < INITIAL_ERROR_MIN_PX:
            log_capture_reject(R, frame, est, age_ms, ["initial_error_below_min"])
            return f"initial_error_below_min ({err:.2f}px < {INITIAL_ERROR_MIN_PX}px)"
        if evaluate_success(R["initial_error"], R["errors"]):
            log_capture_reject(R, frame, est, age_ms, ["success_terminal_no_action"])
            R["success"] = True
            return "success"

        # Freeze pre-action state.
        before_q = np.array(env.data.qpos, dtype=float)
        before_qvel = np.array(env.data.qvel, dtype=float)
        before_prev = np.array(R["ctrl"], dtype=float)

        actor_input = build_actor_input_for_frame(
            est.error_px, before_q, before_qvel, before_prev, frame
        )

        try:
            inference_result = inference.predict(actor_input)
            controls = inference_result["requested_controls_rad"]
            validate_controls_against_ranges(controls, ranges)
        except Exception as exc:
            log_capture_reject(R, frame, est, age_ms, [f"invalid_action:{type(exc).__name__}:{exc}"])
            return f"invalid_action:{type(exc).__name__}:{exc}"

        counterfactual = compute_screen_dependence(actor_input, inference.npz)
        step_start = R["step_index"] + 1
        log_action_intent(R, frame, est, age_ms, inference_result, counterfactual,
                          before_q, before_qvel, before_prev, step_start)
        step_end = None
        try:
            for i in range(SUBSTEPS_PER_FRAME):
                R["ctrl"] = controls[i]
                rec = step(R, R["ctrl"], frame["frame_id"], "learned")
                entry = rec.aggregate.get("index")
                lost = guard.update(TIMESTEP_S, entry is not None,
                                    None if entry is None else entry.get("normal_force_n"))
                if lost:
                    log_action_executed(R, frame, est, step_start, R["step_index"], "contact_loss")
                    return "contact_loss"
            step_end = R["step_index"]
            send_touch(R, rec.aggregate)
            log_action_executed(R, frame, est, step_start, step_end, "completed")
        except vp.PhysicsStepError:
            log_action_executed(R, frame, est, step_start, R["step_index"], "physics_unsafe")
            raise
        except ResourceStop as exc:
            log_action_executed(R, frame, est, step_start, R["step_index"], f"resource_stop:{exc}")
            raise
        except Exception as exc:
            # Preserve the actual executed step count before re-raising; do not
            # emit a misleading "completed" record when post-physics IO fails.
            log_action_executed(R, frame, est, step_start,
                                step_end if step_end is not None else R["step_index"],
                                f"error:{type(exc).__name__}:{exc}")
            raise

        dt = time.monotonic() - t_cycle
        if dt < CYCLE_MIN_WALL_S:
            time.sleep(CYCLE_MIN_WALL_S - dt)
    return "max_action_images_exhausted"


def build_actor_input_for_frame(
    error_xy_px: tuple[float, float],
    q: np.ndarray,
    qvel: np.ndarray,
    previous_ctrl: np.ndarray,
    frame: dict,
) -> dict:
    """Construct the exact 50-feature actor_input from frozen pre-action state."""
    from embodied_trackpad.leap_v1.learned_evaluator import build_actor_input
    return build_actor_input(
        error_xy_px=error_xy_px,
        q_rad16=q,
        qvel_rad_s16=qvel,
        previous_ctrl_rad16=previous_ctrl,
        png_ref=frame.get("path"),
        png_sha256=hashlib.sha256(frame["png"]).hexdigest(),
        frame_monotonic_time_s=frame.get("monotonic_time_s"),
        valid=True,
    )


def cleanup_phase(R: dict) -> bool:
    """Scripted release: flex, then rotation, then hold; return measured release."""
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


def run(seed: int, out_dir: str | Path, contract: EvaluationContract,
        inference: PolicyInference, target_class: int) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    operation_wall_s = min(WALL_BUDGET_S, contract.limits["max_wall_s"]) - 10.0
    R = {
        "t0": time.monotonic(),
        "operation_wall_s": operation_wall_s,
        "out": out,
        "env": None,
        "client": None,
        "step_index": 0,
        "seq": 0,
        "cycle": 0,
        "ctrl": np.zeros(vp.NU),
        "initial_error": None,
        "final_error": None,
        "errors": [],
        "success": False,
        "cleanup_records": [],
        "last_record": None,
        "contract": contract,
        "inference": inference,
        "target_class": target_class,
    }

    with open(out / "metadata.json", "w") as fh:
        json.dump({
            "controller_kind": "learned_small_policy",
            "training_eligible": False,
            "seed": int(seed),
            "target_class": int(target_class),
            "started_utc": started_utc,
            "prep_and_release_scripted": True,
            "checkpoint_identity": contract.checkpoint_identity,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "sample_index_sha256": contract.sample_index_sha256,
            "source_commit": contract.source_commit,
            "dataset_schema": contract.dataset_schema,
            "approved_actor_fields": list(contract.approved_actor_fields),
            "source_file_hashes": contract.source_file_hashes,
            "budgets": {
                "wall_budget_s": min(WALL_BUDGET_S, contract.limits["max_wall_s"]),
                "operation_wall_s": operation_wall_s,
                "sim_budget_s": SIM_BUDGET_S,
                "dir_cap_bytes": DIR_CAP_BYTES,
                "dir_headroom_bytes": DIR_HEADROOM_BYTES,
                "max_action_images": min(MAX_ACTION_IMAGES, contract.limits["max_action_images"]),
                "teardown_reserve_s": 10.0,
                "teardown_bounds_s": {"stop": 1.0, "quit": 1.0, "terminate": 3.0, "kill": 5.0},
            },
            "files": ["teacher.jsonl", "actor.jsonl", "planner.jsonl", "touch.jsonl", "summary.json"],
        }, fh, indent=2)

    stop_reason, unsafe, release, cleaned = "not_started", False, False, False
    try:
        try:
            R["teacher"] = open(out / "teacher.jsonl", "w")
            R["actor"] = open(out / "actor.jsonl", "w")
            R["planner"] = open(out / "planner.jsonl", "w")
            R["touch"] = open(out / "touch.jsonl", "w")
            R["env"] = vp.make_env()
            R["full_map"] = vp.build_named_mapping(R["env"].model, vp.MOTOR_JOINTS)
            vp.reset_open(R["env"])
            prep_phase(R)

            R["client"] = SandboxClient(out_dir=str(out))
            R["client"].client_orig_send = R["client"]._send

            def _send_budgeted(obj, timeout=30.0, _orig=R["client"].client_orig_send):
                left = R["operation_wall_s"] - (time.monotonic() - R["t0"])
                return _orig(obj, timeout=max(0.001, min(timeout, left)))

            R["client"]._send = _send_budgeted
            R["client"].reset(seed=seed, task_profile="near_pointing_v1")
            send_touch(R, R["last_record"]["aggregate"])
            R["reconciler"] = ClockReconciler.sample(R["client"].capture)
            R["gate"] = FrameGate(FrameGateConfig(
                clock_uncertainty_ms=R["reconciler"].uncertainty_ms))
            stop_reason = visual_phase_learned(R, inference)
        except ResourceStop as exc:
            stop_reason = f"resource_stop:{exc}"
        except vp.PhysicsStepError:
            stop_reason, unsafe = "physics_unsafe", True
        except Exception as exc:
            stop_reason = f"error:{type(exc).__name__}:{exc}"

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
            except Exception as exc:
                stop_reason = f"{stop_reason}|cleanup_error:{type(exc).__name__}:{exc}"
                release, cleaned = False, False
    finally:
        client = R.get("client")
        teardown_info = stop_and_reap_client(client, bounds={"stop": 1.0, "quit": 1.0})
        for key in ("teacher", "actor", "planner", "touch"):
            fh = R.get(key)
            if fh is not None and not fh.closed:
                fh.close()

        precision_success = bool(R["success"])
        learning_success = bool(precision_success and not unsafe and release and teardown_info["child_reaped"])
        summary = {
            "seed": int(seed),
            "target_class": int(target_class),
            "started_utc": started_utc,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "precision_success": precision_success,
            "learning_success": learning_success,
            "episode_safe_complete_released": bool(cleaned and not unsafe and release),
            "release_measured": bool(release),
            "physics_unsafe": bool(unsafe),
            "initial_error_px": R["initial_error"],
            "final_error_px": R["final_error"],
            "accepted_frames": len(R["errors"]),
            "steps": R["step_index"],
            "wall_s": round(time.monotonic() - R["t0"], 3),
            "stop_reason": stop_reason,
            "bytes": dir_bytes(out),
            "controller_kind": "learned_small_policy",
            "training_eligible": False,
            "checkpoint_identity": contract.checkpoint_identity,
            "checkpoint_sha256": contract.checkpoint_sha256,
            "source_commit": contract.source_commit,
            "clock_note": "touch simulation_time_s = physics env.data.time (relative sim epoch)",
            "child_pid": teardown_info["child_pid"],
            "child_returncode": teardown_info["child_returncode"],
            "child_reaped": bool(teardown_info["child_reaped"]),
            "teardown": teardown_info["teardown"],
        }
        with open(out / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2, allow_nan=False)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--class", dest="target_class", type=int, required=True,
                    choices=sorted(ALLOWED_CLASSES),
                    help="Target class to evaluate (only 4 or 10)")
    ap.add_argument("--checkpoint", required=True, help="Frozen NPZ checkpoint")
    ap.add_argument("--permit", required=True, help="Evaluation permit JSON")
    ap.add_argument("--sample-index", required=True,
                    help="Sample index JSONL (hashed and compared with permit)")
    ap.add_argument("--source-commit", required=False,
                    help="Current source commit; must match permit")
    ap.add_argument("--dataset-schema", required=False,
                    help="Dataset schema identifier; must match permit")
    ap.add_argument("--out", required=True, help="NEW output directory")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed != args.target_class:
        raise SystemExit(f"--seed must equal --class (frozen seeds: {sorted(ALLOWED_CLASSES)})")

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"--out must be a NEW directory, already exists: {out}")

    with open(args.permit, "r", encoding="utf-8") as fh:
        permit = json.load(fh)

    git_head = _git_head(REPO_ROOT)
    if not git_head:
        raise SystemExit("unable to determine current git HEAD for source binding")
    if args.source_commit is not None and args.source_commit != git_head:
        raise SystemExit(
            f"--source-commit {args.source_commit[:12]} does not match current HEAD {git_head[:12]}"
        )

    source_file_hashes = compute_source_file_hashes(REPO_ROOT, REQUIRED_SOURCE_FILES)
    contract = validate_evaluation_permit(
        permit,
        checkpoint_path=args.checkpoint,
        sample_index_path=args.sample_index,
        source_commit=git_head,
        dataset_schema=DATASET_SCHEMA,
        target_class=args.target_class,
        source_file_hashes=source_file_hashes,
    )

    inference = PolicyInference(args.checkpoint, contract)

    summary = run(args.seed, args.out, contract, inference, target_class=args.target_class)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("learning_success") else 1


if __name__ == "__main__":
    sys.exit(main())
