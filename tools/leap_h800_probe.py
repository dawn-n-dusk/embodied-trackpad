#!/usr/bin/env python3
"""LEAP Hand v2 Advanced bounded H800 (MuJoCo Warp) vs CPU backend validation.

Bounded task 2026-10-31: run the ACCEPTED full 17-motor / 21-joint LEAP v2
Advanced hand on one allocated NVIDIA H800 via mujoco-warp and compare it with
a CPU reference over the SAME canonical reset and the SAME 14,000-row /
7-second hand-only control schedule.  Hand-only backend validation: no pad,
no GUI, no training claim.

Canonical sources (sha256 verified at runtime, mismatch aborts):

  * model adapter  src/embodied_trackpad/leap_sim/model.py
      ef49958ce1fb38817158aeda6414c9f7fbf4a0660df38ee69379b9705afb3781
  * accepted probe tools/leap_model_probe.py (supplies ctrl_at / OPEN keyframes)
      1e75fb6c450d323ee2465846065fe8d7de6d911336dce042c93eb4d7df42f37e

The model is NEVER weakened: no geometry / inertial / effort / ctrlrange /
gravity / gain / timestep / equality / exclusion changes.  If mujoco-warp
cannot run this model as-is, that incompatibility is recorded as the result
and the run stops instead of adapting the model.

Contract:

  * ONE canonical reset: mj_resetData + open_pose_qpos + open-hold ctrl +
    mj_forward (the only live qpos write; all later motion is ctrl + step).
    GPU worlds are the broadcast of the same host reset via put_data.
    Reset audit (corrected 2026-10-31 after job 731968, whose only failure
    was comparing float64 reset bytes against float32 device arrays): the
    canonical float64 reset hash is preserved for provenance, but device
    arrays are audited in the ACTUAL backend dtype (float32 on mujoco-warp
    3.17): BEFORE any stepping, every world's qpos/qvel/ctrl must EXACTLY
    equal the broadcast of the canonical arrays explicitly converted to that
    dtype (shape + finite + exact equality; no numerical tolerance is
    substituted).  Canonical f64 hash, backend-representable expected hash,
    actual hash and per-field max conversion delta are all recorded.  f64
    precision is never claimed on the GPU.
  * ONE frozen open-loop control array: ctrl_at(m, i*timestep) for
    i in 0..13999, validated (shape / finite / ctrlrange) once, sha256
    reported; CPU and every GPU world consume exactly row i at step i.
  * Gates (both backends): finite states/control, valid raw controls,
    joint-limit excursion <= 0.01 rad, dip/pip coupling residual <= 0.01 rad,
    sampled penetration <= 1 mm, final open-pose error <= 0.05 rad,
    CPU-vs-GPU world0 final joint difference <= 0.01 rad (qvel reported too).
  * Coverage is stated exactly: CPU checks every step; GPU world0
    contacts/joints sampled every 20 steps + final; GPU all-world
    qpos/qvel/finite/coupling/limit sampled every 200 steps + final.
    Sampled gates are never asserted as unobserved every-step properties.
  * Capacity: naconmax is the AGGREGATE hard cap, njmax the PER-WORLD hard
    cap (mujoco-warp 3.17 semantics); nacon/nefc peaks are inspected at
    samples and overflow fails the stage.  nconmax is only a per-world hint.
  * GPU solver-warning counters are NOT transferred by get_data_into in
    mujoco-warp 3.17: they stay UNKNOWN, never interpreted as clean.
  * Stages run 1 -> 32 -> 128; 512 only if every prior stage passed, the
    allocated device has >= 16 GiB free and >= 5 minutes of the allocation
    remain.  Expansion stops at the first failed stage.
  * Compilation/warmup runs on an isolated throwaway batch and never
    contaminates the measured reset/run.  Timing separates compile, reset
    transfer and observation costs.  Whole-run throughput (including
    transfers, observation and Python overhead) is the PRIMARY number; the
    subtraction-derived figure is reported as observation-excluded loop
    time, never as pure GPU kernel time.
  * Evaluation instants are labeled: after mjw.step the device qpos/time
    are POST-integration, while the contact evaluation stored by that step
    is the solver-step (start-of-step, pre-integration) one and is NOT
    refreshed on the GPU.  The CPU reference refreshes contacts with
    mj_forward after stepping, so CPU/GPU contact samples belong to
    different instants by construction; both are reported with their own
    instants and no alignment is fabricated.
  * An existing --out file is rejected before any run begins; partial
    results are persisted after each completed stage so a wall-limit kill
    still preserves the evidence of completed stages.
  * Repeated identical worlds are a backend diagnostic, recorded
    (all-world final finite + max |dq|/|dv| vs world0), NOT randomized task
    generalization or a training-speedup claim.  No policy / optimizer.

Exits nonzero if any check fails; failures are preserved, never loosened.
``--tiny`` / ``--cpu-only`` are harness-mechanics smoke modes and are never
reported as GPU or full-schedule results.

Usage:
    python tools/leap_h800_probe.py --out results/leap-h800-<jobid>.json
    python tools/leap_h800_probe.py --tiny --cpu-only --out /tmp/smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as ilmd
import importlib.util
import json
import math
import os
import platform
import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Pinned accepted sources (runtime-verified; see module docstring).
EXPECTED_MODEL_PY_SHA256 = "ef49958ce1fb38817158aeda6414c9f7fbf4a0660df38ee69379b9705afb3781"
EXPECTED_CPU_PROBE_SHA256 = "1e75fb6c450d323ee2465846065fe8d7de6d911336dce042c93eb4d7df42f37e"
MODEL_PY = REPO_ROOT / "src" / "embodied_trackpad" / "leap_sim" / "model.py"
CPU_PROBE_PY = REPO_ROOT / "tools" / "leap_model_probe.py"

# Contract gates (task bounds; never loosened).
LIMIT_EXCURSION_TOL = 0.01  # rad
COUPLING_TOL = 0.01  # rad
PENETRATION_TOL = 0.001  # m (1 mm, sampled)
FINAL_OPEN_TOL = 0.05  # rad (after the >=0.5 s final open hold)
CPU_GPU_QPOS_TOL = 0.01  # rad, world0 final
MIN_REMAINING_S_FOR_512 = 300.0  # seconds of allocation left to allow stage 512
LARGE_BATCH_MIN_FREE_MIB = 16384  # allocated-device free MiB to allow stage 512

# Coverage (task bounds): world0 contacts/joints at most every 20 steps,
# all-world qpos/qvel/finite/coupling at least every 200 steps, both + final.
WORLD0_SAMPLE_EVERY = 20
ALLWORLD_SAMPLE_EVERY = 200

STAGE_ORDER = (1, 32, 128, 512)  # fixed order; 512 additionally gated

FINGERS = ("index", "middle", "ring", "pinky")

TINY_STEPS = 200


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pure helpers (stdlib only; unit-testable without mujoco/numpy)
# ---------------------------------------------------------------------------

def schedule_sha256(rows) -> str:
    """Deterministic hash of a frozen schedule array (rows x nu).

    Same byte layout as the validated h800-contact harness: magic, dims as
    <qq, then every value as a little-endian float64.
    """
    h = hashlib.sha256()
    h.update(b"schedule-v1")
    rows = list(rows)
    width = len(rows[0]) if rows else 0
    h.update(struct.pack("<qq", len(rows), width))
    for row in rows:
        for v in row:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


def reset_state_sha256(qpos, qvel, ctrl) -> str:
    """Deterministic hash of the canonical reset state (f64 LE bytes)."""
    h = hashlib.sha256()
    h.update(b"reset-v1")
    for arr in (qpos, qvel, ctrl):
        for v in arr:
            h.update(struct.pack("<d", float(v)))
    return h.hexdigest()


def _field_sha256(values, dtype) -> str:
    """sha256 of one field serialized in the GIVEN dtype (LE bytes)."""
    import numpy as np

    h = hashlib.sha256()
    h.update(np.asarray(values, dtype=np.dtype(dtype).newbyteorder("<")).tobytes())
    return h.hexdigest()


def reset_broadcast_audit(actual_fields, canonical_fields_f64) -> dict:
    """Exact broadcast audit of a backend reset against the canonical state.

    For each field, ``actual`` is the (nworld, n) array read back from the
    device BEFORE any stepping.  The expected broadcast is the canonical
    float64 array explicitly converted to the ACTUAL dtype of the device
    array (float32 on mujoco-warp 3.17).  The audit requires, per field:

      * shape: actual is 2-D with width len(canonical)
      * finite: every element finite
      * exact: EVERY world row equals the converted canonical array
        element-for-element (backend representation only -- no tolerance)

    Returns a per-field audit with the canonical f64 hash, backend dtype,
    backend-representable expected hash, world0 actual hash, max conversion
    delta and mismatch bookkeeping, plus an overall ``pass`` flag.
    """
    import numpy as np

    fields = {}
    ok_all = True
    for name, canonical in canonical_fields_f64.items():
        canonical_f64 = np.asarray(canonical, dtype=np.float64)
        actual = np.asarray(actual_fields[name])
        entry = {"canonical_f64_sha256": _field_sha256(canonical_f64, np.float64)}
        if actual.ndim != 2 or actual.shape[1] != canonical_f64.shape[0]:
            entry.update({
                "backend_dtype": str(getattr(actual, "dtype", "unknown")),
                "shape_ok": False, "finite": False, "all_worlds_exact": False,
                "n_worlds": int(actual.shape[0]) if actual.ndim >= 1 else 0,
                "mismatch_worlds": None,
                "backend_expected_sha256": None,
                "backend_actual_sha256_world0": None,
                "max_conversion_delta": None,
                "pass": False,
                "detail": "actual shape %s != (nworld, %d)"
                          % (getattr(actual, "shape", None), canonical_f64.shape[0]),
            })
            fields[name] = entry
            ok_all = False
            continue
        dtype = actual.dtype
        expected = canonical_f64.astype(dtype)
        finite = bool(np.isfinite(actual).all())
        exact_rows = [bool(np.array_equal(actual[w], expected))
                      for w in range(actual.shape[0])]
        mismatch_worlds = [w for w, ok in enumerate(exact_rows) if not ok]
        entry.update({
            "backend_dtype": str(dtype),
            "shape_ok": True,
            "finite": finite,
            "all_worlds_exact": not mismatch_worlds,
            "n_worlds": int(actual.shape[0]),
            "mismatch_worlds": mismatch_worlds[:16],
            "backend_expected_sha256": _field_sha256(expected, dtype),
            "backend_actual_sha256_world0": _field_sha256(actual[0], dtype),
            "max_conversion_delta": float(
                np.abs(canonical_f64 - expected.astype(np.float64)).max()),
        })
        entry["pass"] = bool(entry["shape_ok"] and finite and not mismatch_worlds)
        entry["detail"] = (
            "dtype=%s worlds=%d exact=%s finite=%s max|f64->%s|=%.3e"
            % (dtype, actual.shape[0], not mismatch_worlds, finite,
               dtype, entry["max_conversion_delta"]))
        if not entry["pass"]:
            ok_all = False
        fields[name] = entry
    return {"pass": ok_all, "fields": fields,
            "semantics": "exact equality against the canonical arrays explicitly "
                         "converted to the actual backend dtype; permits only the "
                         "backend representation, never a relaxed tolerance"}


def validate_schedule(rows, nu, steps, ctrlrange=None):
    """Validate a frozen schedule array.  Returns a list of problems."""
    problems = []
    if len(rows) != steps:
        problems.append("rows %d != steps %d" % (len(rows), steps))
    for i, row in enumerate(rows):
        if len(row) != nu:
            problems.append("row %d width %d != nu %d" % (i, len(row), nu))
            break
        if not all(math.isfinite(v) for v in row):
            problems.append("row %d has non-finite values" % i)
            break
        if ctrlrange is not None:
            for j, v in enumerate(row):
                lo, hi = ctrlrange[j]
                if not (lo - 1e-12 <= v <= hi + 1e-12):
                    problems.append(
                        "row %d ctrl[%d]=%.6f outside ctrlrange [%.4f, %.4f]"
                        % (i, j, v, lo, hi))
                    break
            else:
                continue
            break
    return problems


def sample_steps(steps: int, every: int) -> list[int]:
    """1-based completed-step counts at which a stream is sampled.

    Samples at multiples of ``every`` and always includes the final step;
    the interval never exceeds ``every`` (final gap may be shorter).
    """
    if steps < 1 or every < 1:
        raise ValueError("steps and every must be >= 1")
    out = [c for c in range(every, steps + 1, every)]
    if not out or out[-1] != steps:
        out.append(steps)
    return out


def coupling_residual_qpos(qpos_row, dip_adrs, pip_adrs) -> float:
    """max |qpos[dip] - qpos[pip]| over the four finger couplings."""
    err = 0.0
    for d_adr, p_adr in zip(dip_adrs, pip_adrs):
        err = max(err, abs(float(qpos_row[d_adr]) - float(qpos_row[p_adr])))
    return err


def limit_excursion_qpos(qpos_row, ranges, qposadrs) -> float:
    """max soft joint-limit excursion (rad) for actuated joints in one row."""
    exc = 0.0
    for adr, (lo, hi) in zip(qposadrs, ranges):
        q = float(qpos_row[adr])
        exc = max(exc, lo - q, q - hi, 0.0)
    return exc


def capacity_verdict(naconmax_alloc, nacon_peak_sampled, overflow_observed,
                     njmax_alloc=None, nefc_peak_sampled=None):
    """Truthful capacity verdict (mujoco-warp 3.17 units, verified source).

    nconmax is a per-world sizing hint; naconmax is the AGGREGATE hard cap
    with d.nacon (shape (1,)) the aggregate active count; njmax is the
    per-world hard cap with d.nefc (nworld,) per-world usage.  Unknown
    counters are UNKNOWN -- never a pass.  Sampled peaks are sampled-capacity
    evidence only, never a never-overflow proof.
    Returns (verdict, detail) with verdict in {"pass", "fail", "unknown"}.
    """
    if overflow_observed:
        return "fail", "overflow observed (contacts/constraints skipped)"
    if naconmax_alloc is None or (isinstance(naconmax_alloc, int) and naconmax_alloc < 0):
        return "unknown", "naconmax allocation counter unavailable"
    if nacon_peak_sampled is None or (isinstance(nacon_peak_sampled, int) and nacon_peak_sampled < 0):
        return "unknown", "active contact count (nacon) unavailable; peak unsampled"
    if nacon_peak_sampled > naconmax_alloc:
        return "fail", "nacon peak %d exceeded naconmax %d (overflow)" % (
            nacon_peak_sampled, naconmax_alloc)
    if njmax_alloc is None or (isinstance(njmax_alloc, int) and njmax_alloc < 0):
        return "unknown", "njmax allocation unknown"
    if nefc_peak_sampled is None or (isinstance(nefc_peak_sampled, int) and nefc_peak_sampled < 0):
        return "unknown", "per-world constraint usage (nefc) unavailable; peak unsampled"
    if nefc_peak_sampled > njmax_alloc:
        return "fail", "nefc peak %d exceeded njmax %d (constraint overflow)" % (
            nefc_peak_sampled, njmax_alloc)
    return ("pass",
            "sampled peaks within capacity: nacon %d<=%d (aggregate), nefc %d<=%d "
            "(per-world); sampled evidence only, not a never-overflow proof"
            % (nacon_peak_sampled, naconmax_alloc, nefc_peak_sampled, njmax_alloc))


def plan_stage_sequence(requested):
    """Validate the requested stage list (subset of STAGE_ORDER, ascending,
    must start at 1).  Returns the ordered list."""
    if not requested:
        raise ValueError("empty stage list")
    stages = []
    for nw in requested:
        if nw not in STAGE_ORDER:
            raise ValueError("invalid stage %r (allowed: %s)" % (nw, STAGE_ORDER))
        if nw in stages:
            raise ValueError("duplicate stage %d" % nw)
        stages.append(nw)
    if stages[0] != 1:
        raise ValueError("first stage must be nworld=1 (got %d)" % stages[0])
    if stages != sorted(stages):
        raise ValueError("stages must be ascending: %s" % (stages,))
    return stages


def gate_512(prev_stage_ok: bool, free_mib, remaining_s) -> tuple[bool, str]:
    """Decide whether stage 512 may run (extra gates on top of ordering)."""
    if not prev_stage_ok:
        return False, "previous stage checks failed"
    if free_mib is None:
        return False, "allocated-device free memory unknown"
    if free_mib < LARGE_BATCH_MIN_FREE_MIB:
        return False, "allocated device free MiB %s < %d" % (
            free_mib, LARGE_BATCH_MIN_FREE_MIB)
    if remaining_s is None:
        return False, "remaining allocation time unknown"
    if remaining_s < MIN_REMAINING_S_FOR_512:
        return False, "remaining allocation %.1f s < %d s" % (
            remaining_s, MIN_REMAINING_S_FOR_512)
    return True, "ok"


def allocated_device_free_mib():
    """Free MiB on the CUDA device this job was allocated.

    Resolves CUDA_VISIBLE_DEVICES (set by Slurm) to the physical index;
    without it, assumes device 0.  Returns None if the query fails.
    """
    try:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        phys = int(cvd.split(",")[0]) if cvd else 0
        out = subprocess.run(
            ["nvidia-smi", "-i", str(phys),
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        pass
    return None


def allocated_device_identity():
    """Measured identity of the allocated CUDA device via nvidia-smi."""
    info = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    try:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        phys = int(cvd.split(",")[0]) if cvd else 0
        out = subprocess.run(
            ["nvidia-smi", "-i", str(phys),
             "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
            if len(parts) == 3:
                info.update({"name": parts[0], "memory_total": parts[1],
                             "driver_version": parts[2]})
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)[:200]
    return info


# ---------------------------------------------------------------------------
# Canonical scene: accepted model + accepted schedule + canonical reset
# ---------------------------------------------------------------------------

def _import_by_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load module from %s" % path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_scene():
    """Load the accepted LEAP model and derive the frozen canonical inputs.

    Verifies the pinned source hashes BEFORE building anything.  Returns a
    dict with the model, the canonical reset callable, the frozen schedule
    rows, the frozen reset state hash, schedule hash and joint bookkeeping.
    """
    from embodied_trackpad.leap_sim import load_hand, open_pose_qpos

    model_sha = file_sha256(MODEL_PY)
    probe_sha = file_sha256(CPU_PROBE_PY)
    if model_sha != EXPECTED_MODEL_PY_SHA256:
        raise RuntimeError(
            "canonical model.py sha256 mismatch: %s != %s; refusing to run "
            "an unreviewed model" % (model_sha, EXPECTED_MODEL_PY_SHA256))
    if probe_sha != EXPECTED_CPU_PROBE_SHA256:
        raise RuntimeError(
            "accepted probe sha256 mismatch: %s != %s; refusing to recompute "
            "controls from an unreviewed source" % (probe_sha, EXPECTED_CPU_PROBE_SHA256))
    cpu_probe = _import_by_path(CPU_PROBE_PY, "leap_model_probe_accepted")

    hand = load_hand()  # verifies the pinned asset manifest itself
    m = hand.model
    nu, nq, nv = int(m.nu), int(m.nq), int(m.nv)
    dt = float(m.opt.timestep)
    t_end = float(cpu_probe.KEYFRAMES[-1][0])
    steps = int(round(t_end / dt))

    ctrlrange = [tuple(float(v) for v in row) for row in m.actuator_ctrlrange]
    rows = [tuple(float(v) for v in cpu_probe.ctrl_at(m, i * dt)) for i in range(steps)]
    problems = validate_schedule(rows, nu, steps, ctrlrange)
    sched_hash = schedule_sha256(rows)

    def canonical_reset():
        """The only live qpos/ctrl write; afterwards ctrl + physics only."""
        import mujoco

        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = open_pose_qpos(m)
        cpu_probe.apply_ctrl_checked(m, d, cpu_probe._open_vector(m))
        mujoco.mj_forward(m, d)
        return d

    reset_probe = canonical_reset()
    reset_hash = reset_state_sha256(reset_probe.qpos, reset_probe.qvel, reset_probe.ctrl)

    dip_adrs = []
    pip_adrs = []
    import mujoco as _mj
    for f in FINGERS:
        jd = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_JOINT, f"{f}_dip")
        jp = _mj.mj_name2id(m, _mj.mjtObj.mjOBJ_JOINT, f"{f}_pip")
        dip_adrs.append(int(m.jnt_qposadr[jd]))
        pip_adrs.append(int(m.jnt_qposadr[jp]))
    joint_ranges = [tuple(float(v) for v in m.jnt_range[j]) for j in range(m.njnt)]
    joint_qposadrs = [int(m.jnt_qposadr[j]) for j in range(m.njnt)]
    open_q = [float(v) for v in open_pose_qpos(m)]

    return {
        "hand": hand,
        "model": m,
        "nq": nq, "nv": nv, "nu": nu, "njnt": int(m.njnt),
        "timestep": dt,
        "t_end": t_end,
        "steps": steps,
        "rows": rows,
        "schedule_sha256": sched_hash,
        "schedule_problems": problems,
        "ctrlrange": ctrlrange,
        "canonical_reset": canonical_reset,
        "reset_state_sha256": reset_hash,
        "open_pose_qpos": open_q,
        "dip_qposadrs": dip_adrs,
        "pip_qposadrs": pip_adrs,
        "joint_ranges": joint_ranges,
        "joint_qposadrs": joint_qposadrs,
        "hashes": {
            "model_py_sha256": model_sha,
            "accepted_cpu_probe_sha256": probe_sha,
            "this_probe_sha256": file_sha256(Path(__file__).resolve()),
            "manifest_sha256": file_sha256(
                REPO_ROOT / "assets" / "manifests" / "leap-selected.json"),
            "upstream_urdf_sha256": file_sha256(
                REPO_ROOT / "assets" / "downloads" / "leap_hand_v2_adv"
                / "leap_v2_right" / "robot.urdf"),
            "derived_urdf_sha256": file_sha256(
                REPO_ROOT / "assets" / "downloads" / "leap_hand_v2_adv"
                / "leap_v2_right" / "robot_mujoco_derived.urdf"),
        },
    }


def contact_snapshot(m, d) -> dict:
    """Self-contact snapshot at the current instant.

    Contact geom IDs are read from ``contact.geom[0]`` / ``contact.geom[1]``:
    mujoco-warp 3.17 ``get_data_into`` (verified io.py source) writes ONLY the
    ``contact.geom[:]`` array field and leaves the deprecated ``geom1`` /
    ``geom2`` scalar accessors stale, so those are never used here.
    """
    pairs = {}
    min_dist = 0.0
    worst = None
    for i in range(int(d.ncon)):
        c = d.contact[i]
        b1 = m.body(int(m.geom_bodyid[int(c.geom[0])])).name or "world"
        b2 = m.body(int(m.geom_bodyid[int(c.geom[1])])).name or "world"
        key = (b1, b2)
        pairs[key] = min(pairs.get(key, 0.0), float(c.dist))
        if c.dist < min_dist:
            min_dist = float(c.dist)
            worst = [b1, b2]
    return {
        "ncon": int(d.ncon),
        "pairs": {"%s|%s" % (a, b): v for (a, b), v in pairs.items()},
        "max_penetration_m": -min_dist if min_dist < 0 else 0.0,
        "worst_pair": worst,
    }


def warning_counts(d):
    """MuJoCo warning counters; None when unreadable (UNKNOWN, never clean)."""
    try:
        import mujoco

        out = {}
        for w in range(mujoco.mjtWarning.mjNWARNING):
            n = int(d.warning[w].number)
            if n:
                out[mujoco.mjtWarning(w).name] = n
        return out
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# CPU reference (every-step coverage)
# ---------------------------------------------------------------------------

def run_cpu_reference(scene):
    """CPU backend over the identical reset + frozen 14,000-row schedule.

    Every-step coverage: finite states, joint-limit excursion, dip/pip
    coupling residual, self-contact penetration, solver warnings.  Timing is
    split into stepping and observation/refresh costs.
    """
    import mujoco
    import numpy as np

    m = scene["model"]
    rows = scene["rows"]
    steps = scene["steps"]
    d = scene["canonical_reset"]()
    reset_hash = reset_state_sha256(d.qpos, d.qvel, d.ctrl)

    max_limit = 0.0
    max_coupling = 0.0
    max_pen = 0.0
    max_ncon = 0
    first_nonfinite = None
    warn_start = [int(d.warning[w].number) for w in range(mujoco.mjtWarning.mjNWARNING)]
    warn_names = [mujoco.mjtWarning(w).name for w in range(mujoco.mjtWarning.mjNWARNING)]

    t_step = 0.0
    t_obs = 0.0
    t0 = time.perf_counter()
    for i in range(steps):
        ts = time.perf_counter()
        d.ctrl[:] = rows[i]  # frozen, already validated finite + in ctrlrange
        mujoco.mj_step(m, d)
        t_step += time.perf_counter() - ts

        ts = time.perf_counter()
        # Refresh the forward state at the observation instant (same pattern
        # as the accepted probe) so contacts match the recorded qpos.
        mujoco.mj_forward(m, d)
        if first_nonfinite is None and not (
            np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
            and np.isfinite(d.qacc).all() and np.isfinite(d.ctrl).all()
            and np.isfinite(d.actuator_force).all()
        ):
            first_nonfinite = i
        max_limit = max(max_limit, limit_excursion_qpos(
            d.qpos, scene["joint_ranges"], scene["joint_qposadrs"]))
        max_coupling = max(max_coupling, coupling_residual_qpos(
            d.qpos, scene["dip_qposadrs"], scene["pip_qposadrs"]))
        snap = contact_snapshot(m, d)
        max_ncon = max(max_ncon, snap["ncon"])
        max_pen = max(max_pen, snap["max_penetration_m"])
        t_obs += time.perf_counter() - ts
    total_s = time.perf_counter() - t0

    warn_end = [int(d.warning[w].number) for w in range(mujoco.mjtWarning.mjNWARNING)]
    warnings = {warn_names[k]: warn_end[k] - warn_start[k]
                for k in range(len(warn_names)) if warn_end[k] - warn_start[k]}

    final_qpos = [float(v) for v in d.qpos]
    final_qvel = [float(v) for v in d.qvel]
    final_open_err = max(abs(a - b) for a, b in zip(final_qpos, scene["open_pose_qpos"]))
    final_contacts = contact_snapshot(m, d)

    return {
        "steps": steps,
        "coverage": "every step (finite/limit/coupling/penetration/warnings)",
        "state_evaluation_instant": "post-integration (qpos/qvel after each mj_step)",
        "contact_evaluation_instant": "post-integration: mj_forward refresh after "
            "each mj_step, so contacts belong to the recorded qpos (unlike the GPU "
            "samples, which use the solver-step pre-integration evaluation)",
        "reset_state_sha256": reset_hash,
        "timing": {
            "total_s": total_s,
            "step_s": t_step,
            "observation_s": t_obs,
            "steps_per_s_step_only": steps / t_step if t_step > 0 else None,
            "steps_per_s_total_PRIMARY": steps / total_s if total_s > 0 else None,
        },
        "checks": {
            "states_finite": first_nonfinite is None,
            "first_nonfinite_step": first_nonfinite,
            "max_limit_excursion_rad": max_limit,
            "max_coupling_residual_rad": max_coupling,
            "max_contact_penetration_m": max_pen,
            "max_simultaneous_contacts": max_ncon,
            "solver_warnings": warnings,
            "final_open_pose_error_rad": final_open_err,
        },
        "final_qpos": final_qpos,
        "final_qvel": final_qvel,
        "final_contacts": final_contacts,
    }


# ---------------------------------------------------------------------------
# GPU staged batch (mujoco-warp on the allocated CUDA device)
# ---------------------------------------------------------------------------

def _sync(wp):
    if hasattr(wp, "synchronize_device"):
        wp.synchronize_device()
    else:
        wp.synchronize()


def remaining_allocation_s() -> float | None:
    """Seconds left in this job's allocation, from sbatch-exported bounds."""
    try:
        start = float(os.environ["LEAP_H800_JOB_START_EPOCH"])
        wall = float(os.environ.get("LEAP_H800_JOB_WALL_S", "1200"))
        return wall - (time.time() - start)
    except (KeyError, ValueError):
        return None


def run_gpu_stages(scene, cpu, requested_stages, check, persist=None):
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    wp.init()
    if not wp.is_cuda_available():
        raise RuntimeError("no CUDA device visible to warp; refusing CPU fallback")
    device = wp.get_cuda_device(0)
    m = scene["model"]
    base = scene["canonical_reset"]()
    rows = scene["rows"]
    steps = scene["steps"]
    open_q = scene["open_pose_qpos"]

    out = {
        "device_name": str(getattr(device, "name", "unknown")),
        "device_identity": allocated_device_identity(),
        "warp_version": wp.__version__,
        "mjw_version": getattr(mjw, "__version__", "unknown"),
        "gpu_solver_warnings": "unknown - warning counters are not transferred by "
                               "get_data_into in mujoco-warp 3.17 (verified source); "
                               "not interpreted as no-warning",
        "stages": [],
        "stages_skipped": [],
    }

    cpu_max_ncon = max(1, int(cpu["checks"]["max_simultaneous_contacts"]))
    with wp.ScopedDevice(device):
        put_t0 = time.perf_counter()
        mm = mjw.put_model(m)
        out["put_model_s"] = time.perf_counter() - put_t0

        prev_ok = True
        for nw in requested_stages:
            free_mib = allocated_device_free_mib()
            if nw == 1:
                go, reason = True, "ok"
            elif not prev_ok:
                go, reason = False, "previous stage checks failed"
            elif nw == 512:
                go, reason = gate_512(prev_ok, free_mib, remaining_allocation_s())
            else:
                go, reason = True, "ok"
            if not go:
                out["stages_skipped"].append(
                    {"nworld": nw, "reason": reason,
                     "free_mib_allocated_device": free_mib,
                     "remaining_allocation_s": remaining_allocation_s()})
                continue

            tag = "nworld=%d" % nw
            holder = [True]

            def scheck(name, passed, detail):
                check("%s[%s]" % (name, tag), passed, detail)
                if not passed:
                    holder[0] = False

            # Capacity sizing in verified mujoco-warp 3.17 units.
            nconmax = max(64, 4 * cpu_max_ncon)              # per-world hint
            naconmax = max(256, 4 * cpu_max_ncon * nw)       # AGGREGATE hard cap
            njmax = max(1024, 8 * (int(m.nv) + cpu_max_ncon))  # PER-WORLD hard cap

            # Isolated compile/warmup on a throwaway batch: never contaminates
            # the measured reset/run of this stage.
            warm_t0 = time.perf_counter()
            d_warm = mjw.put_data(m, base, nworld=nw, nconmax=nconmax,
                                  njmax=njmax, naconmax=naconmax)
            d_warm.ctrl.assign(np.tile(np.asarray(rows[0]), (nw, 1)))
            mjw.step(mm, d_warm)
            _sync(wp)
            compile_s = time.perf_counter() - warm_t0
            del d_warm

            # Measured reset/transfer: broadcast of the SAME canonical reset.
            tr_t0 = time.perf_counter()
            d = mjw.put_data(m, base, nworld=nw, nconmax=nconmax,
                             njmax=njmax, naconmax=naconmax)
            _sync(wp)
            reset_transfer_s = time.perf_counter() - tr_t0

            # Reset audit BEFORE any stepping, ALL worlds, exact equality in
            # the ACTUAL backend dtype (float32 on mujoco-warp 3.17).  The
            # canonical float64 hash is kept for provenance; it is NOT
            # expected to equal float32 device bytes.
            reset_audit = reset_broadcast_audit(
                {"qpos": d.qpos.numpy(), "qvel": d.qvel.numpy(),
                 "ctrl": d.ctrl.numpy()},
                {"qpos": [float(v) for v in base.qpos],
                 "qvel": [float(v) for v in base.qvel],
                 "ctrl": [float(v) for v in base.ctrl]})

            world0_samples = set(sample_steps(steps, WORLD0_SAMPLE_EVERY))
            allworld_samples = set(sample_steps(steps, ALLWORLD_SAMPLE_EVERY))

            max_limit_w0 = 0.0
            max_coupling_w0 = 0.0
            max_pen_w0 = 0.0
            max_ncon_w0 = 0
            max_limit_all = 0.0
            max_coupling_all = 0.0
            transient_nonfinite = None
            nacon_peak = -1
            nefc_peak = -1
            obs_s = 0.0
            first_step_s = None

            run_t0 = time.perf_counter()
            for i in range(steps):
                fs_t0 = time.perf_counter()
                d.ctrl.assign(np.tile(np.asarray(rows[i]), (nw, 1)))
                mjw.step(mm, d)
                if i == 0:
                    _sync(wp)
                    first_step_s = time.perf_counter() - fs_t0
                c = i + 1  # completed step count
                need_w0 = c in world0_samples
                need_all = c in allworld_samples
                if need_w0 or need_all:
                    o_t0 = time.perf_counter()
                    _sync(wp)
                    if need_all:
                        qpos = np.asarray(d.qpos.numpy())
                        qvel = np.asarray(d.qvel.numpy())
                        if transient_nonfinite is None and not (
                            np.isfinite(qpos).all() and np.isfinite(qvel).all()
                        ):
                            transient_nonfinite = i
                        for w in range(nw):
                            max_limit_all = max(max_limit_all, limit_excursion_qpos(
                                qpos[w], scene["joint_ranges"], scene["joint_qposadrs"]))
                            max_coupling_all = max(max_coupling_all, coupling_residual_qpos(
                                qpos[w], scene["dip_qposadrs"], scene["pip_qposadrs"]))
                        try:
                            nacon_peak = max(nacon_peak, int(d.nacon.numpy()[0]))
                            nefc_peak = max(nefc_peak, int(np.asarray(d.nefc.numpy()).max()))
                        except Exception:  # noqa: BLE001 - counters stay unknown (-1)
                            pass
                    if need_w0:
                        probe = mujoco.MjData(m)
                        mjw.get_data_into(probe, m, d, world_id=0)
                        if transient_nonfinite is None and not (
                            np.isfinite(probe.qpos).all() and np.isfinite(probe.qvel).all()
                            and np.isfinite(probe.ctrl).all()
                        ):
                            transient_nonfinite = i
                        max_limit_w0 = max(max_limit_w0, limit_excursion_qpos(
                            probe.qpos, scene["joint_ranges"], scene["joint_qposadrs"]))
                        max_coupling_w0 = max(max_coupling_w0, coupling_residual_qpos(
                            probe.qpos, scene["dip_qposadrs"], scene["pip_qposadrs"]))
                        snap = contact_snapshot(m, probe)
                        max_ncon_w0 = max(max_ncon_w0, snap["ncon"])
                        max_pen_w0 = max(max_pen_w0, snap["max_penetration_m"])
                        try:
                            nacon_peak = max(nacon_peak, int(d.nacon.numpy()[0]))
                            nefc_peak = max(nefc_peak, int(np.asarray(d.nefc.numpy()).max()))
                        except Exception:  # noqa: BLE001
                            pass
                    obs_s += time.perf_counter() - o_t0
            _sync(wp)
            run_total_s = time.perf_counter() - run_t0

            # Final state (all worlds) + world0 final via readback.
            qpos = np.asarray(d.qpos.numpy())
            qvel = np.asarray(d.qvel.numpy())
            try:
                nacon_peak = max(nacon_peak, int(d.nacon.numpy()[0]))
                nefc_peak = max(nefc_peak, int(np.asarray(d.nefc.numpy()).max()))
            except Exception:  # noqa: BLE001
                pass
            final0 = mujoco.MjData(m)
            mjw.get_data_into(final0, m, d, world_id=0)
            final_contacts_w0 = contact_snapshot(m, final0)
            max_pen_w0 = max(max_pen_w0, final_contacts_w0["max_penetration_m"])
            max_ncon_w0 = max(max_ncon_w0, final_contacts_w0["ncon"])
            final_limit_w0 = limit_excursion_qpos(
                final0.qpos, scene["joint_ranges"], scene["joint_qposadrs"])
            final_coupling_w0 = coupling_residual_qpos(
                final0.qpos, scene["dip_qposadrs"], scene["pip_qposadrs"])
            max_limit_w0 = max(max_limit_w0, final_limit_w0)
            max_coupling_w0 = max(max_coupling_w0, final_coupling_w0)

            final_qpos_w0 = [float(v) for v in final0.qpos]
            final_qvel_w0 = [float(v) for v in final0.qvel]
            final_open_err_w0 = max(
                abs(a - b) for a, b in zip(final_qpos_w0, open_q))
            cpu_gpu_dq = max(
                abs(a - b) for a, b in zip(final_qpos_w0, cpu["final_qpos"]))
            cpu_gpu_dv = max(
                abs(a - b) for a, b in zip(final_qvel_w0, cpu["final_qvel"]))
            cross_dq = float(np.abs(qpos - qpos[0]).max()) if nw > 1 else 0.0
            cross_dv = float(np.abs(qvel - qvel[0]).max()) if nw > 1 else 0.0
            all_finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())

            naconmax_alloc = int(getattr(d, "naconmax", -1))
            try:
                overflow = bool(int(d.nacon.numpy()[0]) > naconmax_alloc) \
                    if naconmax_alloc >= 0 else None
            except Exception:  # noqa: BLE001
                overflow = None
            verdict, verdict_detail = capacity_verdict(
                naconmax_alloc if naconmax_alloc >= 0 else None,
                nacon_peak if nacon_peak >= 0 else None,
                overflow,
                njmax_alloc=njmax,
                nefc_peak_sampled=nefc_peak if nefc_peak >= 0 else None)

            step_only_s = run_total_s - obs_s
            stage = {
                "nworld": nw,
                "free_mib_allocated_device": free_mib,
                "remaining_allocation_s_at_gate": remaining_allocation_s(),
                "capacity": {
                    "units": "nconmax per-world hint; naconmax AGGREGATE hard cap; "
                             "njmax PER-WORLD hard cap (mujoco-warp 3.17 verified)",
                    "nconmax_per_world": nconmax,
                    "naconmax_alloc": naconmax_alloc if naconmax_alloc >= 0 else None,
                    "njmax_per_world": njmax,
                    "nacon_peak_sampled": nacon_peak if nacon_peak >= 0 else None,
                    "nefc_peak_per_world_sampled": nefc_peak if nefc_peak >= 0 else None,
                    "overflow_observed": overflow,
                    "verdict": verdict,
                    "verdict_detail": verdict_detail,
                    "coverage": "nacon/nefc peaks at every observation sample "
                                "(world0 every %d, all-world every %d) + final"
                                % (WORLD0_SAMPLE_EVERY, ALLWORLD_SAMPLE_EVERY),
                },
                "timing": {
                    "compile_warmup_s_isolated": compile_s,
                    "reset_transfer_s": reset_transfer_s,
                    "run_total_s": run_total_s,
                    "observation_s_in_run": obs_s,
                    "first_measured_step_s": first_step_s,
                    "steps": steps,
                    "steps_per_s_whole_run_PRIMARY": (
                        steps / run_total_s if run_total_s > 0 else None),
                    "whole_run_note": "PRIMARY: whole measured run including ctrl "
                        "transfers, observation readbacks and Python overhead",
                    "observation_excluded_loop_s": step_only_s,
                    "steps_per_s_observation_excluded_loop": (
                        steps / step_only_s if step_only_s > 0 else None),
                    "world_steps_per_s_observation_excluded_loop": (
                        nw * steps / step_only_s if step_only_s > 0 else None),
                    "loop_note": "subtraction-derived (run_total - observation); "
                        "NOT pure GPU kernel time",
                },
                "reset_audit": reset_audit,
                "canonical_reset_state_sha256_f64_provenance": scene["reset_state_sha256"],
                "checks_data": {
                    "transient_nonfinite_step": transient_nonfinite,
                    "final_finite_all_worlds": all_finite,
                    "max_limit_excursion_rad_world0_sampled": max_limit_w0,
                    "max_coupling_residual_rad_world0_sampled": max_coupling_w0,
                    "max_limit_excursion_rad_allworld_sampled": max_limit_all,
                    "max_coupling_residual_rad_allworld_sampled": max_coupling_all,
                    "max_penetration_m_world0_sampled": max_pen_w0,
                    "max_ncon_world0_sampled": max_ncon_w0,
                    "final_open_pose_error_rad_world0": final_open_err_w0,
                    "cpu_gpu_final_max_abs_dq_world0": cpu_gpu_dq,
                    "cpu_gpu_final_max_abs_dv_world0": cpu_gpu_dv,
                    "cross_world_final_max_abs_dq": cross_dq,
                    "cross_world_final_max_abs_dv": cross_dv,
                },
                "coverage": {
                    "world0_contacts_joints": "world0 sampled every %d steps + final "
                        "(%d samples); NOT every-step, NOT all-world"
                        % (WORLD0_SAMPLE_EVERY, len(world0_samples)),
                    "allworld_state": "all-world qpos/qvel finite/limit/coupling sampled "
                        "every %d steps + final (%d samples)"
                        % (ALLWORLD_SAMPLE_EVERY, len(allworld_samples)),
                    "contacts_channel": "get_data_into readback; geom IDs from "
                        "contact.geom[0]/geom[1] (deprecated geom1/geom2 are stale "
                        "in mujoco-warp 3.17 and never used)",
                    "state_evaluation_instant": "POST-integration: device qpos/qvel "
                        "after each completed mjw.step",
                    "contact_evaluation_instant": "solver-step (start-of-step, "
                        "pre-integration) evaluation stored by that mjw.step; NOT "
                        "refreshed to the post-integration state on GPU. CPU "
                        "reference contacts were refreshed via mj_forward after "
                        "stepping, so CPU/GPU contact samples are at different "
                        "instants by construction; no alignment is fabricated",
                },
                "final_qpos_world0": final_qpos_w0,
                "final_qvel_world0": final_qvel_w0,
                "final_contacts_world0": final_contacts_w0,
            }

            scheck("gpu_reset_broadcast_exact",
                   reset_audit["pass"],
                   "pre-step exact broadcast audit in actual backend dtype: "
                   + "; ".join(
                       "%s(%s)" % (name, f["detail"]) for name, f in reset_audit["fields"].items())
                   + " | canonical f64 hash %s kept for provenance only"
                     % scene["reset_state_sha256"][:16])
            scheck("gpu_states_finite",
                   all_finite and transient_nonfinite is None,
                   "final all-world qpos/qvel finite=%s; transient sample step=%s "
                   "(sampled coverage, NOT every-step)" % (all_finite, transient_nonfinite))
            scheck("gpu_limit_excursion<=0.01",
                   max(max_limit_w0, max_limit_all) <= LIMIT_EXCURSION_TOL,
                   "world0 sampled max %.6e; all-world sampled max %.6e rad (tol %.2f)"
                   % (max_limit_w0, max_limit_all, LIMIT_EXCURSION_TOL))
            scheck("gpu_coupling<=0.01",
                   max(max_coupling_w0, max_coupling_all) <= COUPLING_TOL,
                   "world0 sampled max %.6e; all-world sampled max %.6e rad (tol %.2f)"
                   % (max_coupling_w0, max_coupling_all, COUPLING_TOL))
            scheck("gpu_penetration<=0.001",
                   max_pen_w0 <= PENETRATION_TOL,
                   "world0 sampled max penetration %.6e m (tol %.1e)"
                   % (max_pen_w0, PENETRATION_TOL))
            scheck("gpu_final_open<=0.05",
                   final_open_err_w0 <= FINAL_OPEN_TOL,
                   "world0 final open-pose error %.6e rad (tol %.2f)"
                   % (final_open_err_w0, FINAL_OPEN_TOL))
            scheck("cpu_gpu_final_qpos<=0.01",
                   cpu_gpu_dq <= CPU_GPU_QPOS_TOL,
                   "world0 vs CPU final max |dq| %.6e rad (tol %.2f); |dv| %.6e reported"
                   % (cpu_gpu_dq, CPU_GPU_QPOS_TOL, cpu_gpu_dv))
            scheck("gpu_capacity_verified", verdict == "pass",
                   "%s (unknown counters gate expansion and cannot pass)"
                   % verdict_detail)
            stage["checks_passed"] = holder[0]
            out["stages"].append(stage)
            if persist is not None:
                persist(out)  # preserve completed-stage evidence against wall kill
            prev_ok = holder[0]
            if not prev_ok:
                # Stop expansion at the first failed stage; record the stop.
                remaining = requested_stages[requested_stages.index(nw) + 1:]
                for skip_nw in remaining:
                    out["stages_skipped"].append(
                        {"nworld": skip_nw,
                         "reason": "expansion stopped: stage %d failed" % nw})
                break

        out["gating_audit"] = {
            "stages_run_in_order": [s["nworld"] for s in out["stages"]],
            "note": "stages ran 1->32->128(->512); each stage's checks were "
                    "evaluated before the next stage; 512 additionally required "
                    ">= %d MiB free on the allocated device and >= %d s of "
                    "allocation remaining" % (LARGE_BATCH_MIN_FREE_MIB,
                                              MIN_REMAINING_S_FOR_512),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get(
        "LEAP_H800_OUT", "results/leap-h800-%s.json" % (
            os.environ.get("SLURM_JOB_ID", "nominal"))))
    p.add_argument("--stages", default=os.environ.get("LEAP_H800_STAGES", "1,32,128,512"),
                   help="comma list, ascending subset of 1,32,128,512 starting at 1; "
                        "512 additionally gated on prior pass + memory + time")
    p.add_argument("--tiny", action="store_true",
                   help="200-step harness mechanics smoke; motion gates not evaluated")
    p.add_argument("--cpu-only", action="store_true",
                   help="CPU harness smoke only; NEVER a GPU result")
    args = p.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(
            "error: output file already exists, refusing to overwrite or append: "
            "%s (use a fresh path for every run)" % args.out)

    try:
        requested = plan_stage_sequence(
            [int(x) for x in args.stages.split(",") if x.strip()])
    except ValueError as exc:
        raise SystemExit("invalid --stages: %s" % exc)

    result = {
        "harness": "leap-v2-adv-h800-backend-validation",
        "schema_version": "1.1",
        "schema_note": "1.1 corrects the reset audit (exact broadcast in the actual "
                       "backend dtype instead of comparing float64 bytes against "
                       "float32 device arrays), labels state/contact evaluation "
                       "instants, makes whole-run throughput the PRIMARY timing, "
                       "declares the float32 backend control representation, and "
                       "rejects pre-existing --out files. Job 731968's raw "
                       "1.0-schema failure is preserved as history.",
        "bounded_task": "2026-10-31 LEAP v2 Advanced bounded H800 validation "
                        "(hand-only; no pad/GUI/training claim)",
        "utc_start": utcnow(),
        "source_commit": os.environ.get("LEAP_H800_SOURCE_COMMIT", "unknown"),
        "mode": "cpu_smoke" if args.cpu_only else "gpu_backend_validation",
        "tiny": bool(args.tiny),
        "host_sha256_12": sha12(socket.gethostname()),
        "user_sha256_12": sha12(os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"),
        "slurm_job_sha256_12": sha12(os.environ["SLURM_JOB_ID"]) if os.environ.get("SLURM_JOB_ID") else None,
        "slurm_allocation": {
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "gpus_requested": "1 (gres=gpu:1 via sbatch)",
            "wall_limit_s": os.environ.get("LEAP_H800_JOB_WALL_S", "1200"),
            "nodelist_sha256_12": sha12(os.environ["SLURM_NODELIST"]) if os.environ.get("SLURM_NODELIST") else None,
        },
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "packages": {},
        "checks": [],
        "errors": [],
    }
    for pkg in ("mujoco", "mujoco-warp", "warp-lang", "numpy"):
        try:
            result["packages"][pkg] = ilmd.version(pkg)
        except ilmd.PackageNotFoundError:
            result["packages"][pkg] = None

    def check(name, passed, detail):
        result["checks"].append({"name": name, "pass": bool(passed), "detail": detail})

    try:
        scene = build_scene()
    except Exception as exc:  # noqa: BLE001 - canonical-source failure aborts
        import traceback
        result["errors"].append("build_scene: %s" % exc)
        result["scene_build"] = {"failed": True, "error": str(exc)[:500],
                                 "traceback": traceback.format_exc()[-2000:]}
        result["utc_end"] = utcnow()
        result["overall"] = "fail"
        _write_result(args.out, result)
        return 1

    if args.tiny:
        scene["rows"] = scene["rows"][:TINY_STEPS]
        scene["steps"] = TINY_STEPS
        scene["schedule_sha256"] = schedule_sha256(scene["rows"])
        scene["schedule_problems"] = validate_schedule(
            scene["rows"], scene["nu"], scene["steps"], scene["ctrlrange"])

    # Backend representation of the control sequence: the frozen float64 rows
    # are the canonical artifact; on the GPU each row is converted to the
    # device ctrl dtype (float32 on mujoco-warp 3.17) by warp's assign at
    # every step.  Record the conversion delta; f64 precision is NOT claimed
    # on the GPU.
    import numpy as _np
    rows_f64 = _np.asarray(scene["rows"], dtype=_np.float64)
    rows_f32_delta = float(
        _np.abs(rows_f64 - rows_f64.astype(_np.float32).astype(_np.float64)).max())
    result["scene"] = {
        "model": "LEAP Hand v2 Advanced right hand, fixed palm, hand-only scene",
        "nq": scene["nq"], "nv": scene["nv"], "nu": scene["nu"], "njnt": scene["njnt"],
        "timestep_s": scene["timestep"],
        "steps": scene["steps"],
        "schedule": {
            "kind": "frozen_open_loop_time_indexed_from_accepted_probe_ctrl_at",
            "rows": scene["steps"],
            "nu": scene["nu"],
            "sha256": scene["schedule_sha256"],
            "sha256_dtype": "float64 (canonical; unchanged from the accepted-probe "
                            "control sequence)",
            "backend_representation": "device ctrl dtype float32 on mujoco-warp "
                "3.17: each frozen float64 row is converted by warp assign at "
                "every step; float64 precision is NOT claimed on the GPU",
            "max_f64_to_f32_conversion_delta": rows_f32_delta,
            "validation_problems": scene["schedule_problems"],
            "applied": "row i at step i on CPU and on every GPU world",
        },
        "canonical_reset_state_sha256": scene["reset_state_sha256"],
        "hashes": scene["hashes"],
        "world_variation": "repeated_identical (backend diagnostic only; NOT "
                           "randomized task generalization or training speedup)",
        "stage_plan": {"requested": requested,
                       "order": "1->32->128(->512 gated)"},
        "tolerances": {
            "limit_excursion_rad": LIMIT_EXCURSION_TOL,
            "coupling_rad": COUPLING_TOL,
            "penetration_m_sampled": PENETRATION_TOL,
            "final_open_rad": FINAL_OPEN_TOL,
            "cpu_gpu_final_qpos_rad": CPU_GPU_QPOS_TOL,
        },
        "sampling": {
            "world0_contacts_joints_every_steps": WORLD0_SAMPLE_EVERY,
            "allworld_state_every_steps": ALLWORLD_SAMPLE_EVERY,
            "plus_final": True,
        },
    }
    check("canonical_hashes_verified", True,
          "model.py %s, accepted probe %s matched pinned values before any run"
          % (scene["hashes"]["model_py_sha256"][:16],
             scene["hashes"]["accepted_cpu_probe_sha256"][:16]))
    check("schedule_valid", not scene["schedule_problems"],
          "; ".join(scene["schedule_problems"]) if scene["schedule_problems"]
          else "frozen schedule shape/finite/ctrlrange ok, sha256=%s"
               % scene["schedule_sha256"][:16])

    cpu = None
    try:
        cpu = run_cpu_reference(scene)
        result["cpu_reference"] = cpu
        cc = cpu["checks"]
        check("cpu_reset_matches_canonical",
              cpu["reset_state_sha256"] == scene["reset_state_sha256"],
              "CPU run reset hash matches frozen canonical reset")
        check("cpu_states_finite", cc["states_finite"],
              "every-step finite over %d steps" % scene["steps"])
        check("cpu_limit_excursion<=0.01",
              cc["max_limit_excursion_rad"] <= LIMIT_EXCURSION_TOL,
              "every-step max %.6e rad" % cc["max_limit_excursion_rad"])
        check("cpu_coupling<=0.01",
              cc["max_coupling_residual_rad"] <= COUPLING_TOL,
              "every-step max %.6e rad" % cc["max_coupling_residual_rad"])
        check("cpu_penetration<=0.001",
              cc["max_contact_penetration_m"] <= PENETRATION_TOL,
              "every-step max %.6e m" % cc["max_contact_penetration_m"])
        check("cpu_no_solver_warnings", not cc["solver_warnings"],
              "warnings: %s" % (cc["solver_warnings"] or "none"))
        if args.tiny:
            check("cpu_final_open_gate", True,
                  "not evaluated in --tiny mode (200 steps < final open hold)")
        else:
            check("cpu_final_open<=0.05",
                  cc["final_open_pose_error_rad"] <= FINAL_OPEN_TOL,
                  "final open-pose error %.6e rad" % cc["final_open_pose_error_rad"])
    except Exception as exc:  # noqa: BLE001
        import traceback
        result["errors"].append("cpu_reference: %s" % exc)
        result["cpu_reference"] = {"failed": True, "error": str(exc)[:500],
                                   "traceback": traceback.format_exc()[-2000:]}
        check("cpu_reference_runs", False, str(exc)[:300])

    cpu_ok = cpu is not None and all(c["pass"] for c in result["checks"])
    if args.cpu_only:
        result["gpu"] = {"skipped": True,
                         "reason": "--cpu-only smoke mode; not a GPU result"}
    elif not cpu_ok:
        result["gpu"] = {"skipped": True,
                         "reason": "CPU reference failed or unknown; GPU stages not "
                                   "launched (a failed CPU run is never an accepted "
                                   "GPU reference)"}
        check("gpu_prerequisite_cpu_passed", False, "CPU checks failed; GPU blocked")
    else:
        try:
            gpu = run_gpu_stages(
                scene, cpu, requested, check,
                persist=lambda gpu_out: _write_result(
                    args.out, {**result, "gpu": gpu_out,
                               "partial_stage_checkpoint": True}))
            result["gpu"] = gpu
            ran = [s["nworld"] for s in gpu["stages"]]
            check("gpu_stage_ran", bool(ran),
                  "stages run: %s; skipped: %s"
                  % (ran, json.dumps(gpu["stages_skipped"])))
        except Exception as exc:  # noqa: BLE001 - backend incompatibility is a result
            import traceback
            result["errors"].append("gpu_batch: %s" % exc)
            result["gpu"] = {
                "failed": True, "error": str(exc)[:500],
                "traceback": traceback.format_exc()[-2000:],
                "note": "mujoco-warp could not run the unmodified accepted model; "
                        "recorded as a backend incompatibility result - the model "
                        "was NOT weakened to force a pass",
            }
            check("gpu_backend_compatible", False, str(exc)[:300])

    result["utc_end"] = utcnow()
    result["overall"] = "pass" if (
        result["checks"] and all(c["pass"] for c in result["checks"])
        and not result["errors"]) else "fail"
    _write_result(args.out, result)
    print(json.dumps({
        "out": args.out, "overall": result["overall"],
        "checks_failed": [c["name"] for c in result["checks"] if not c["pass"]],
        "errors": result["errors"]}))
    return 0 if result["overall"] == "pass" else 1


def _write_result(out_path: str, result: dict) -> None:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    sys.exit(main())
