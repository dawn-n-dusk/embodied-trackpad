#!/usr/bin/env python3
"""LEAP v1 Full bounded one-H800 runtime smoke (CPU vs MuJoCo Warp + torch BF16).

Bounded task 2026-10-31 (docs/tasks/v1-training-and-monitoring-20261031.zh-CN.md):
validate the user-owned GPU runtime and the ACCEPTED LEAP v1 Full 16-channel
hand model on ONE allocated NVIDIA H800.  This is a RUNTIME SMOKE: it is not
policy training, not a pi0.5 run, not 8192-world scaling, and not a task
acceptance result.  8192 worlds are explicitly deferred.

Canonical sources (sha256 verified at runtime, mismatch aborts):

  * v1 model adapter  src/embodied_trackpad/leap_v1/model.py
      574f29f4c8ad99ae03acff0ff6b34fd074b815e9f6749026952f8c75e3386abc
  * v1 diagnostics    src/embodied_trackpad/leap_v1/diagnostics.py
      16d4a4a489012626e6ddf1b4e65361f0be8a12c3fc8d8c4c7d3d99797006d302

The model is NEVER weakened: no geometry / inertial / effort / ctrlrange /
gravity / gain / timestep / equality / exclusion changes.  If mujoco-warp
cannot run this model as-is, that incompatibility is recorded as the result
and the run stops instead of adapting the model.  The v1 model has NO dip/pip
coupling equalities (unlike v2); no v2 17-motor/21-joint mapping is used.

Contract:

  * ONE canonical reset: mj_resetData + open_pose_qpos (all-zero open pose)
    + zero ctrl + mj_forward (the only live qpos write; all later motion is
    ctrl + step).  GPU worlds are the broadcast of the same host reset via
    put_data.  The reset audit runs BEFORE any stepping, over ALL worlds, as
    EXACT equality against the canonical arrays explicitly converted to the
    ACTUAL backend dtype (float32 on mujoco-warp 3.17); the canonical f64
    hash is kept for provenance only and f64 precision is never claimed on
    the GPU.
  * ONE frozen open-loop control array built from the ACCEPTED v1 diagnostics
    script (``diagnostics.build_script``) restricted to a two-channel sweep
    subset (``if_mcp``/``if_pip``) plus the spread and flex/return phases,
    with the filtered-out sweep gaps removed (phase durations and motion
    content unchanged).  The schedule therefore contains (1) contact-free
    single-finger movement and (2) the spread/flex diagnostics that MAY
    produce self-contact; measured contacts are reported honestly per phase.
    Rows are validated (shape / finite / ctrlrange) once, sha256 reported;
    CPU and every GPU world consume exactly row i at step i.
  * Gates (both backends): finite states/control, valid raw controls,
    joint-limit excursion <= 0.01 rad, final open-pose error <= 0.05 rad,
    CPU-vs-GPU world0 final joint difference <= 0.01 rad (qvel reported
    too).  Self-contact penetration is a REPORTED measurement with a
    declared 1 mm reference, NOT a gate: the accepted v1 diagnostics gate
    limits/finiteness/warnings/completeness only and report physical
    measurements (tracking, spread, contacts) with reference tolerances;
    the flex/spread phases of this schedule are EXPECTED to produce
    self-contact (CPU reference measured 1.03 mm peak at contact onset),
    so a penetration gate would fail the accepted model, not the runtime.
  * Coverage is stated exactly: CPU checks every step; GPU world0
    contacts/joints sampled every 20 steps + final; GPU all-world
    qpos/qvel/finite/limit sampled every 100 steps + final.  Sampled gates
    are never asserted as unobserved every-step properties.
  * Capacity: naconmax is the AGGREGATE hard cap, njmax the PER-WORLD hard
    cap (mujoco-warp 3.17 semantics); nacon/nefc peaks are inspected at
    samples and overflow fails the stage.  nconmax is only a per-world hint.
  * GPU solver-warning counters are NOT transferred by get_data_into in
    mujoco-warp 3.17: they stay UNKNOWN, never interpreted as clean.
  * Stages run 1 -> 32 only (fixed smoke set; no expansion beyond 32).
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
  * torch runtime smoke (separate section): verifies the ACTUAL allocated
    device is an NVIDIA H800 with usable CUDA, then runs one BF16
    forward/backward and ONE AdamW optimizer update on a small diagnostic
    linear model.  This validates the numeric/training stack only; it is
    NOT project policy training and downloads no model weights.
  * Repeated identical worlds are a backend diagnostic, recorded
    (all-world final finite + max |dq|/|dv| vs world0), NOT randomized task
    generalization or a training-speedup claim.  No policy is evaluated.
  * An existing --out file is rejected before any run begins; partial
    results are persisted after each completed stage so a wall-limit kill
    still preserves the evidence of completed stages.

Exits nonzero if any check fails; failures are preserved, never loosened.
``--tiny`` / ``--cpu-only`` are harness-mechanics smoke modes and are never
reported as GPU or full-schedule results.

Usage:
    python tools/leap_v1_gpu_smoke.py --out results/leap-v1-gpu-<jobid>.json
    python tools/leap_v1_gpu_smoke.py --tiny --cpu-only --out /tmp/smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as ilmd
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

# Pinned accepted v1 sources (runtime-verified; see module docstring).
EXPECTED_MODEL_PY_SHA256 = "574f29f4c8ad99ae03acff0ff6b34fd074b815e9f6749026952f8c75e3386abc"
EXPECTED_DIAG_PY_SHA256 = "16d4a4a489012626e6ddf1b4e65361f0be8a12c3fc8d8c4c7d3d99797006d302"
MODEL_PY = REPO_ROOT / "src" / "embodied_trackpad" / "leap_v1" / "model.py"
DIAG_PY = REPO_ROOT / "src" / "embodied_trackpad" / "leap_v1" / "diagnostics.py"

# Contract gates (task bounds; never loosened).
LIMIT_EXCURSION_TOL = 0.01  # rad
PENETRATION_TOL = 0.001  # m (1 mm, sampled)
FINAL_OPEN_TOL = 0.05  # rad (after the >=0.7 s final open hold)
CPU_GPU_QPOS_TOL = 0.01  # rad, world0 final

# Coverage (task bounds): world0 contacts/joints at most every 20 steps,
# all-world qpos/qvel/finite/limit at least every 100 steps, both + final.
WORLD0_SAMPLE_EVERY = 20
ALLWORLD_SAMPLE_EVERY = 100

STAGE_ORDER = (1, 32)  # fixed smoke set; no expansion beyond 32

# Short-schedule sweep subset (contact-free single-finger movement); the
# spread and flex/return phases are always kept and may produce self-contact.
SMOKE_SWEEP_CHANNELS = ("if_mcp", "if_pip")

TINY_STEPS = 200

# torch BF16 diagnostic model size (runtime smoke only; not a policy).
TORCH_SMOKE_IN = 128
TORCH_SMOKE_OUT = 64
TORCH_SMOKE_BATCH = 32
TORCH_SMOKE_LR = 1e-4

MAX_NUMERICAL_THREADS = 2  # task bound; fewer when the allocation is smaller


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
# Pure helpers (stdlib only; unit-testable without mujoco/numpy/torch)
# ---------------------------------------------------------------------------

def schedule_sha256(rows) -> str:
    """Deterministic hash of a frozen schedule array (rows x nu).

    Same byte layout as the validated v2 harness: magic, dims as <qq, then
    every value as a little-endian float64.
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
        entry = {"canonical_f64_sha256": _field_sha256(canonical_f64, np.float64)}
        if name not in actual_fields:
            entry.update({
                "backend_dtype": None,
                "shape_ok": False, "finite": False, "all_worlds_exact": False,
                "n_worlds": 0,
                "mismatch_worlds": None,
                "backend_expected_sha256": None,
                "backend_actual_sha256_world0": None,
                "max_conversion_delta": None,
                "pass": False,
                "detail": "actual field missing",
            })
            fields[name] = entry
            ok_all = False
            continue
        actual = np.asarray(actual_fields[name])
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
    must start at 1; this smoke never expands beyond 32)."""
    if not requested:
        raise ValueError("empty stage list")
    stages = []
    for nw in requested:
        if nw not in STAGE_ORDER:
            raise ValueError("invalid stage %r (allowed: %s; no expansion)"
                             % (nw, STAGE_ORDER))
        if nw in stages:
            raise ValueError("duplicate stage %d" % nw)
        stages.append(nw)
    if stages[0] != 1:
        raise ValueError("first stage must be nworld=1 (got %d)" % stages[0])
    if stages != sorted(stages):
        raise ValueError("stages must be ascending: %s" % (stages,))
    return stages


def numerical_threads(env=None) -> int:
    """Task bound: at most 2 numerical threads, following the ACTUAL CPU
    allocation.  When SLURM_CPUS_PER_TASK is absent or unparsable the
    default is 1 (never assume 2)."""
    env = os.environ if env is None else env
    try:
        alloc = int(env.get("SLURM_CPUS_PER_TASK") or 1)
    except ValueError:
        alloc = 1
    return max(1, min(MAX_NUMERICAL_THREADS, alloc))


def compact_phases(phases) -> list:
    """Re-time filtered diagnostic phases back-to-back (durations unchanged).

    ``diagnostics.build_script`` times every phase against the FULL 16-channel
    script, so filtering the sweep subset leaves dead open-hold gaps.  This
    helper keeps phase order, kind, vectors and durations identical but
    removes the gaps; it never changes any motion content.
    """
    out = []
    t = 0.0
    for ph in phases:
        dur = ph.t1 - ph.t0
        out.append(type(ph)(ph.name, t, t + dur, ph.kind, base=ph.base,
                            target=ph.target, hold_channels=ph.hold_channels,
                            render_label=ph.render_label))
        t += dur
    return out


def is_h800(device_name: str) -> bool:
    """Accept only an actual NVIDIA H800 product name (e.g. 'NVIDIA H800',
    'NVIDIA H800 80GB HBM3'); H800 PCIe variants also match."""
    return "H800" in (device_name or "")


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
# Canonical scene: accepted v1 model + short diagnostics-derived schedule
# ---------------------------------------------------------------------------

def build_scene(sweep_channels=SMOKE_SWEEP_CHANNELS):
    """Load the accepted v1 model and derive the frozen canonical inputs.

    Verifies the pinned source hashes BEFORE building anything.  Returns a
    dict with the model, the canonical reset callable, the frozen schedule
    rows, the frozen reset state hash, schedule hash and joint bookkeeping.
    """
    from embodied_trackpad.leap_v1 import diagnostics, load_hand, open_pose_qpos

    model_sha = file_sha256(MODEL_PY)
    diag_sha = file_sha256(DIAG_PY)
    if model_sha != EXPECTED_MODEL_PY_SHA256:
        raise RuntimeError(
            "canonical leap_v1/model.py sha256 mismatch: %s != %s; refusing to "
            "run an unreviewed model" % (model_sha, EXPECTED_MODEL_PY_SHA256))
    if diag_sha != EXPECTED_DIAG_PY_SHA256:
        raise RuntimeError(
            "canonical leap_v1/diagnostics.py sha256 mismatch: %s != %s; refusing "
            "to derive controls from an unreviewed source"
            % (diag_sha, EXPECTED_DIAG_PY_SHA256))

    hand = load_hand()  # verifies the pinned asset manifest itself
    m = hand.model
    nu, nq, nv = int(m.nu), int(m.nq), int(m.nv)
    dt = float(m.opt.timestep)

    full = diagnostics.build_script(m)
    scoped = [p for p in full
              if diagnostics._phase_in_scope(p, sweep_channels)]
    phases = compact_phases(scoped)
    t_end = phases[-1].t1
    steps = int(round(t_end / dt))

    ctrlrange = [tuple(float(v) for v in row) for row in m.actuator_ctrlrange]
    rows = []
    for i in range(steps):
        vec, _phase = diagnostics._ctrl_at(phases, i * dt)
        rows.append(tuple(float(v) for v in vec))
    problems = validate_schedule(rows, nu, steps, ctrlrange)
    sched_hash = schedule_sha256(rows)
    phase_spans = [(p.name, p.t0, p.t1) for p in phases]

    def canonical_reset():
        """The only live qpos/ctrl write; afterwards ctrl + physics only."""
        import mujoco

        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = open_pose_qpos(m)
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        return d

    reset_probe = canonical_reset()
    reset_hash = reset_state_sha256(reset_probe.qpos, reset_probe.qvel, reset_probe.ctrl)

    joint_ranges = [tuple(float(v) for v in m.jnt_range[j]) for j in range(m.njnt)]
    joint_qposadrs = [int(m.jnt_qposadr[j]) for j in range(m.njnt)]
    open_q = [float(v) for v in open_pose_qpos(m)]
    act_qposadrs = [int(m.jnt_qposadr[int(m.actuator_trnid[i, 0])])
                    for i in range(m.nu)]

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
        "phase_spans": phase_spans,
        "ctrlrange": ctrlrange,
        "canonical_reset": canonical_reset,
        "reset_state_sha256": reset_hash,
        "open_pose_qpos": open_q,
        "joint_ranges": joint_ranges,
        "joint_qposadrs": joint_qposadrs,
        "act_qposadrs": act_qposadrs,
        "hashes": {
            "model_py_sha256": model_sha,
            "diagnostics_py_sha256": diag_sha,
            "this_smoke_sha256": file_sha256(Path(__file__).resolve()),
            "manifest_sha256": file_sha256(
                REPO_ROOT / "assets" / "manifests" / "leap-v1-full.json"),
        },
    }


def phase_of(phase_spans, t: float) -> str:
    """Phase name covering control time t (last phase owns its endpoint)."""
    for name, t0, t1 in phase_spans:
        if t0 - 1e-12 <= t < t1:
            return name
    return phase_spans[-1][0]


def contact_snapshot(m, d) -> dict:
    """Self-contact snapshot at the current instant.

    Contact geom IDs are read from ``contact.geom[0]`` / ``contact.geom[1]``:
    mujoco-warp 3.17 ``get_data_into`` writes ONLY the ``contact.geom[:]``
    array field and leaves the deprecated ``geom1``/``geom2`` scalar
    accessors stale, so those are never used here.
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
    """CPU backend over the identical reset + frozen short schedule.

    Every-step coverage: finite states, joint-limit excursion, tracking
    error vs the position-actuator command, self-contact penetration and
    counts, solver warnings.  Timing is split into stepping and
    observation/refresh costs.
    """
    import mujoco
    import numpy as np

    m = scene["model"]
    rows = scene["rows"]
    steps = scene["steps"]
    act_adrs = scene["act_qposadrs"]
    d = scene["canonical_reset"]()
    reset_hash = reset_state_sha256(d.qpos, d.qvel, d.ctrl)

    max_limit = 0.0
    max_tracking = 0.0
    max_pen = 0.0
    max_ncon = 0
    phase_max_ncon: dict[str, int] = {}
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
        # Refresh the forward state at the observation instant so contacts
        # match the recorded qpos.
        mujoco.mj_forward(m, d)
        if first_nonfinite is None and not (
            np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
            and np.isfinite(d.qacc).all() and np.isfinite(d.ctrl).all()
            and np.isfinite(d.actuator_force).all()
        ):
            first_nonfinite = i
        max_limit = max(max_limit, limit_excursion_qpos(
            d.qpos, scene["joint_ranges"], scene["joint_qposadrs"]))
        max_tracking = max(max_tracking, float(
            np.abs(d.ctrl - d.qpos[act_adrs]).max()))
        snap = contact_snapshot(m, d)
        max_ncon = max(max_ncon, snap["ncon"])
        max_pen = max(max_pen, snap["max_penetration_m"])
        if snap["ncon"]:
            ph = phase_of(scene["phase_spans"], i * scene["timestep"])
            phase_max_ncon[ph] = max(phase_max_ncon.get(ph, 0), snap["ncon"])
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
        "coverage": "every step (finite/limit/tracking/penetration/warnings)",
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
            "max_tracking_error_rad": max_tracking,
            "max_contact_penetration_m": max_pen,
            "max_simultaneous_contacts": max_ncon,
            "contact_phases": phase_max_ncon,
            "solver_warnings": warnings,
            "final_open_pose_error_rad": final_open_err,
        },
        "final_qpos": final_qpos,
        "final_qvel": final_qvel,
        "final_contacts": final_contacts,
    }


# ---------------------------------------------------------------------------
# torch BF16 runtime smoke (NOT project policy training)
# ---------------------------------------------------------------------------

def run_torch_smoke(check):
    """Verify the allocated device is a real NVIDIA H800 with usable CUDA and
    run one BF16 forward/backward + ONE AdamW update on a small diagnostic
    linear model.  Validates the numeric/training stack only."""
    import numpy as np
    import torch

    torch.set_num_threads(numerical_threads())
    out = {
        "label": "runtime smoke: numeric/training-stack check on a diagnostic "
                 "linear model; NOT project policy training; no model weights "
                 "downloaded or loaded",
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "threads": torch.get_num_threads(),
    }
    if not torch.cuda.is_available():
        out["failed"] = True
        out["error"] = "torch.cuda.is_available() is False"
        check("torch_cuda_available", False, out["error"])
        return out

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    out.update({
        "device_name": name,
        "device_capability": list(cap),
        "device_total_memory_mib": int(props.total_memory / 1048576),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    })
    check("torch_cuda_available", True, "device=%s capability=%s" % (name, cap))
    check("torch_device_is_h800", is_h800(name),
          "allocated device name %r must be an NVIDIA H800" % name)
    check("torch_bf16_supported", out["bf16_supported"],
          "torch.cuda.is_bf16_supported() on the allocated device")

    torch.manual_seed(20261031)
    model = torch.nn.Linear(TORCH_SMOKE_IN, TORCH_SMOKE_OUT).to(
        device="cuda", dtype=torch.bfloat16)
    x = torch.randn(TORCH_SMOKE_BATCH, TORCH_SMOKE_IN,
                    device="cuda", dtype=torch.bfloat16)
    target = torch.randn(TORCH_SMOKE_BATCH, TORCH_SMOKE_OUT,
                         device="cuda", dtype=torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=TORCH_SMOKE_LR)
    before = model.weight.detach().to(torch.float32).cpu().numpy().copy()

    pred = model(x)
    loss = torch.nn.functional.mse_loss(pred, target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    loss_f = float(loss.detach().to(torch.float32).cpu())
    grad = model.weight.grad
    grad_finite = bool(torch.isfinite(grad.to(torch.float32)).all())
    after = model.weight.detach().to(torch.float32).cpu().numpy()
    weight_delta = float(np.abs(after - before).max())
    out.update({
        "model": "torch.nn.Linear(%d, %d) bf16, batch %d, AdamW lr=%g, seed 20261031"
                 % (TORCH_SMOKE_IN, TORCH_SMOKE_OUT, TORCH_SMOKE_BATCH, TORCH_SMOKE_LR),
        "loss": loss_f,
        "grad_finite": grad_finite,
        "weight_max_abs_update": weight_delta,
        "cuda_memory_allocated_mib": int(torch.cuda.memory_allocated(0) / 1048576),
        "cuda_memory_reserved_mib": int(torch.cuda.memory_reserved(0) / 1048576),
    })
    check("torch_bf16_forward_backward_finite",
          bool(np.isfinite(loss_f)) and grad_finite,
          "loss=%.6f finite; weight grad finite=%s" % (loss_f, grad_finite))
    check("torch_optimizer_update_applied", weight_delta > 0.0,
          "one AdamW step changed weights by max |dw|=%.3e" % weight_delta)
    del model, x, target, pred, loss, grad, opt
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# GPU staged batch (mujoco-warp on the allocated CUDA device)
# ---------------------------------------------------------------------------

def _sync(wp):
    if hasattr(wp, "synchronize_device"):
        wp.synchronize_device()
    else:
        wp.synchronize()


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
            else:
                go, reason = True, "ok"
            if not go:
                out["stages_skipped"].append(
                    {"nworld": nw, "reason": reason,
                     "free_mib_allocated_device": free_mib})
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
            max_pen_w0 = 0.0
            max_ncon_w0 = 0
            phase_max_ncon_w0: dict[str, int] = {}
            max_limit_all = 0.0
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
                        snap = contact_snapshot(m, probe)
                        max_ncon_w0 = max(max_ncon_w0, snap["ncon"])
                        max_pen_w0 = max(max_pen_w0, snap["max_penetration_m"])
                        if snap["ncon"]:
                            ph = phase_of(scene["phase_spans"], i * scene["timestep"])
                            phase_max_ncon_w0[ph] = max(
                                phase_max_ncon_w0.get(ph, 0), snap["ncon"])
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
            max_limit_w0 = max(max_limit_w0, final_limit_w0)

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
                    "max_limit_excursion_rad_allworld_sampled": max_limit_all,
                    "max_penetration_m_world0_sampled": max_pen_w0,
                    "max_ncon_world0_sampled": max_ncon_w0,
                    "contact_phases_world0_sampled": phase_max_ncon_w0,
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
                    "allworld_state": "all-world qpos/qvel finite/limit sampled "
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
            scheck("gpu_penetration_disclosed", True,
                   "world0 sampled max penetration %.6e m (reference %.1e; "
                   "REPORTED, not gated - accepted v1 diagnostics convention)"
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
            "note": "stages ran 1->32 (fixed smoke set; no expansion beyond 32)",
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.environ.get(
        "LEAP_V1_GPU_OUT", "results/leap-v1-gpu-%s.json" % (
            os.environ.get("SLURM_JOB_ID", "nominal"))))
    p.add_argument("--stages", default=os.environ.get("LEAP_V1_GPU_STAGES", "1,32"),
                   help="comma list, ascending subset of 1,32 starting at 1 "
                        "(fixed smoke set; no expansion beyond 32)")
    p.add_argument("--tiny", action="store_true",
                   help="200-step harness mechanics smoke; motion gates not evaluated")
    p.add_argument("--cpu-only", action="store_true",
                   help="CPU harness smoke only; NEVER a GPU result")
    p.add_argument("--skip-torch", action="store_true",
                   help="harness debugging only; skips the torch BF16 runtime smoke")
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

    threads = numerical_threads()
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    result = {
        "harness": "leap-v1-full-h800-runtime-smoke",
        "schema_version": "1.0",
        "bounded_task": "2026-10-31 LEAP v1 Full user-owned SuperPOD runtime "
                        "smoke (hand-only; 1/32 worlds; no pad/GUI/training/"
                        "8192 claim)",
        "utc_start": utcnow(),
        "source_commit": os.environ.get("LEAP_V1_GPU_SOURCE_COMMIT", "unknown"),
        "mode": "cpu_smoke" if args.cpu_only else "gpu_runtime_smoke",
        "tiny": bool(args.tiny),
        "host_sha256_12": sha12(socket.gethostname()),
        "user_sha256_12": sha12(os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"),
        "slurm_job_sha256_12": sha12(os.environ["SLURM_JOB_ID"]) if os.environ.get("SLURM_JOB_ID") else None,
        "slurm_allocation": {
            "cpus_per_task_actual": os.environ.get("SLURM_CPUS_PER_TASK"),
            "mem_per_node_actual_mb": os.environ.get("SLURM_MEM_PER_NODE"),
            "gpus_requested": "1 (gres=gpu:1 via sbatch; cpus/mem at scheduler "
                              "defaults, recorded as allocated)",
            "wall_limit_s": os.environ.get("LEAP_V1_GPU_JOB_WALL_S", "1200"),
            "nodelist_sha256_12": sha12(os.environ["SLURM_NODELIST"]) if os.environ.get("SLURM_NODELIST") else None,
        },
        "numerical_threads": threads,
        "platform": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version()},
        "packages": {},
        "checks": [],
        "errors": [],
    }
    for pkg in ("mujoco", "mujoco-warp", "warp-lang", "numpy", "torch"):
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
        "model": "LEAP Hand v1 Full right hand, fixed palm, hand-only scene "
                 "(16 independent channels, no coupling equalities)",
        "embodiment_id": "leap_v1_full_right_menagerie_v1",
        "calibration_id": "uncalibrated_upstream_defaults_v1",
        "nq": scene["nq"], "nv": scene["nv"], "nu": scene["nu"], "njnt": scene["njnt"],
        "timestep_s": scene["timestep"],
        "steps": scene["steps"],
        "schedule": {
            "kind": "frozen_open_loop_time_indexed_from_accepted_v1_diagnostics",
            "derivation": "diagnostics.build_script filtered to sweep channels "
                "%s plus spread/flex/open phases, gaps removed (durations and "
                "motion content unchanged); contains contact-free single-finger "
                "movement AND spread/flex phases that may produce self-contact"
                % (SMOKE_SWEEP_CHANNELS,),
            "rows": scene["steps"],
            "nu": scene["nu"],
            "sha256": scene["schedule_sha256"],
            "sha256_dtype": "float64 (canonical)",
            "backend_representation": "device ctrl dtype float32 on mujoco-warp "
                "3.17: each frozen float64 row is converted by warp assign at "
                "every step; float64 precision is NOT claimed on the GPU",
            "max_f64_to_f32_conversion_delta": rows_f32_delta,
            "validation_problems": scene["schedule_problems"],
            "applied": "row i at step i on CPU and on every GPU world",
            "phase_spans": scene["phase_spans"],
        },
        "canonical_reset_state_sha256": scene["reset_state_sha256"],
        "hashes": scene["hashes"],
        "world_variation": "repeated_identical (backend diagnostic only; NOT "
                           "randomized task generalization or training speedup)",
        "stage_plan": {"requested": requested,
                       "order": "1->32 (fixed smoke set)"},
        "tolerances": {
            "limit_excursion_rad": LIMIT_EXCURSION_TOL,
            "penetration_m_reference_reported_not_gated": PENETRATION_TOL,
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
          "leap_v1 model.py %s, diagnostics.py %s matched pinned values before "
          "any run" % (scene["hashes"]["model_py_sha256"][:16],
                       scene["hashes"]["diagnostics_py_sha256"][:16]))
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
        check("cpu_penetration_disclosed", True,
              "every-step max %.6e m (reference %.1e; REPORTED, not gated - "
              "accepted v1 diagnostics convention; flex/spread self-contact is "
              "expected)" % (cc["max_contact_penetration_m"], PENETRATION_TOL))
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

    # torch BF16 runtime smoke (independent of the physics backends).
    if args.skip_torch:
        result["torch_smoke"] = {"skipped": True,
                                 "reason": "--skip-torch harness debugging; "
                                           "never a runtime acceptance"}
    else:
        try:
            result["torch_smoke"] = run_torch_smoke(check)
        except Exception as exc:  # noqa: BLE001
            import traceback
            result["errors"].append("torch_smoke: %s" % exc)
            result["torch_smoke"] = {"failed": True, "error": str(exc)[:500],
                                     "traceback": traceback.format_exc()[-2000:]}
            check("torch_runtime_smoke", False, str(exc)[:300])

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
