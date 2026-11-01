"""M2b1 capture-provenance validation for LEAP v1 pilot records.

Pure validator for per-frame capture timing provenance: no images are
decoded, no actor features are computed, and no input is mutated. This
module alone is NOT the exporter or a data-readiness claim; integration
with :mod:`pilot_dataset` and config validation are separate modules.

Contract:

* ``validate_capture_timing(record, summary, metadata, *, required=False)``
  returns a small metadata-only dict on success and raises ``ValueError``
  with a concise reason on invalid timing.
* Legacy records (no new-timing trigger anywhere) return exactly
  ``{"clock_offset_s": None, "clock_uncertainty_s": None}``; no values
  are fabricated.
* A new timing record is REQUIRED when ``required=True`` OR any trigger
  is present: a ``capture_timing`` entry in ``record``, clock fields in
  ``summary``, or ``png_backend`` in ``summary``/``metadata``. A missing
  or partial new-timing record is never silently downgraded to legacy.

Timing semantics (see :class:`visual_servo.ClockReconciler`): the app
capture clock equals ``time.monotonic() + clock_offset_s``, so a frame's
capture age at pipeline end ``t3`` is
``(t3 + clock_offset_s - frame_monotonic_time_s) * 1000`` milliseconds.
Worst-case acceptance reuses the frozen CONTACT_PROFILE limits
(``max_frame_age_ms`` = 250 ms, ``max_future_frame_ms`` = 50 ms) with the
measured clock uncertainty, never relaxed: terminal frames are validated
through the exact same path as action frames.
"""
from __future__ import annotations

import math
from typing import Any

from embodied_trackpad.leap_v1.visual_servo import CONTACT_PROFILE

__all__ = ["validate_capture_timing"]

LEGACY_TIMING_NULL = {"clock_offset_s": None, "clock_uncertainty_s": None}

_TIMING_NUMERIC_KEYS = (
    "t0_monotonic_s", "t1_monotonic_s", "t2_monotonic_s", "t3_monotonic_s",
    "capture_save_ms", "decode_ms", "extract_ms", "total_ms",
    "clock_offset_s", "clock_uncertainty_s",
)
_TIMING_REQUIRED_KEYS = _TIMING_NUMERIC_KEYS + ("png_backend", "png_backend_version")
_CLOCK_KEYS = ("clock_offset_s", "clock_uncertainty_s")
_BACKENDS = frozenset({"stdlib", "pillow"})
_DURATION_TOL_MS = 1e-6
_AGE_TOL_MS = 1e-6

# Recomputed from t0..t3 in milliseconds; keys are the declared fields.
_DERIVED_MS = (
    ("capture_save_ms", lambda t0, t1, t2, t3: (t1 - t0) * 1000.0),
    ("decode_ms", lambda t0, t1, t2, t3: (t2 - t1) * 1000.0),
    ("extract_ms", lambda t0, t1, t2, t3: (t3 - t2) * 1000.0),
    ("total_ms", lambda t0, t1, t2, t3: (t3 - t0) * 1000.0),
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require_number(mapping: dict, key: str, where: str) -> float:
    value = mapping.get(key)
    if not _is_number(value):
        raise ValueError(f"{where}.{key} must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{where}.{key} not finite: {value!r}")
    return float(value)


def _new_timing_required(record: dict, summary: dict, metadata: dict,
                         required: bool) -> bool:
    if required:
        return True
    if "capture_timing" in record:
        return True
    if any(key in summary for key in _CLOCK_KEYS):
        return True
    return "png_backend" in summary or "png_backend" in metadata


def _validate_backend(backend: Any, version: Any, where: str) -> None:
    if not isinstance(backend, str) or backend not in _BACKENDS:
        raise ValueError(f"{where}.png_backend must be 'stdlib' or 'pillow', "
                         f"got {backend!r}")
    if backend == "pillow" and (not isinstance(version, str) or not version):
        raise ValueError(
            f"{where}.png_backend_version must be a nonempty string for "
            f"pillow, got {version!r}")
    if backend == "stdlib" and version is not None and \
            (not isinstance(version, str) or not version):
        raise ValueError(
            f"{where}.png_backend_version must be null or a nonempty string "
            f"for stdlib, got {version!r}")


def validate_capture_timing(record: dict, summary: dict, metadata: dict,
                            *, required: bool = False) -> dict:
    """Validate one frame record's capture timing provenance.

    Legacy absence returns exactly ``{"clock_offset_s": None,
    "clock_uncertainty_s": None}``. When a new timing record is required
    (see module docstring), every declared value is checked: finite
    non-bool numerics, ordered stages, durations re-derived from
    t0..t3, calibration equal to the summary's real finite calibration,
    backend/version agreement across record, summary and metadata, the
    offset-adjusted frame age re-derived from ``t3``, and worst-case age
    within the frozen CONTACT_PROFILE limits. Inputs are not mutated.
    """
    if not _new_timing_required(record, summary, metadata, required):
        return dict(LEGACY_TIMING_NULL)

    timing = record.get("capture_timing")
    if not isinstance(timing, dict):
        raise ValueError(f"capture_timing must be an object, got {timing!r}")
    missing = [key for key in _TIMING_REQUIRED_KEYS if key not in timing]
    if missing:
        raise ValueError(f"capture_timing missing keys: {', '.join(missing)}")

    for key in _TIMING_NUMERIC_KEYS:
        _require_number(timing, key, "capture_timing")
    t0 = timing["t0_monotonic_s"]
    t1 = timing["t1_monotonic_s"]
    t2 = timing["t2_monotonic_s"]
    t3 = timing["t3_monotonic_s"]
    if not (t0 <= t1 <= t2 <= t3):
        raise ValueError(
            f"capture stages out of order: {t0!r} <= {t1!r} <= {t2!r} "
            f"<= {t3!r} violated")

    for key, derive in _DERIVED_MS:
        declared = timing[key]
        if declared < 0.0:
            raise ValueError(f"capture_timing.{key} negative: {declared!r}")
        expected = derive(t0, t1, t2, t3)
        if abs(declared - expected) > _DURATION_TOL_MS:
            raise ValueError(
                f"capture_timing.{key}={declared!r} does not match derived "
                f"{expected!r} ms from t0..t3")

    offset = timing["clock_offset_s"]
    uncertainty = timing["clock_uncertainty_s"]
    if uncertainty < 0.0:
        raise ValueError(f"clock uncertainty negative: {uncertainty!r}")

    for key in _CLOCK_KEYS:
        _require_number(summary, key, "summary")
    if timing["clock_offset_s"] != summary["clock_offset_s"]:
        raise ValueError(
            f"capture_timing.clock_offset_s={timing['clock_offset_s']!r} does "
            f"not equal summary clock_offset_s={summary['clock_offset_s']!r}")
    if timing["clock_uncertainty_s"] != summary["clock_uncertainty_s"]:
        raise ValueError(
            f"capture_timing.clock_uncertainty_s="
            f"{timing['clock_uncertainty_s']!r} does not equal summary "
            f"clock_uncertainty_s={summary['clock_uncertainty_s']!r}")

    backend = timing["png_backend"]
    version = timing["png_backend_version"]
    _validate_backend(backend, version, "capture_timing")
    for side, name in ((summary, "summary"), (metadata, "metadata")):
        for key in ("png_backend", "png_backend_version"):
            if key not in side:
                raise ValueError(f"{name} missing required {key}")
        if side["png_backend"] != backend:
            raise ValueError(
                f"{name}.png_backend={side['png_backend']!r} does not match "
                f"capture_timing.png_backend={backend!r}")
        if side["png_backend_version"] != version:
            raise ValueError(
                f"{name}.png_backend_version={side['png_backend_version']!r} "
                f"does not match capture_timing.png_backend_version={version!r}")

    frame_time = _require_number(record, "frame_monotonic_time_s", "record")
    declared_age = _require_number(record, "frame_age_ms", "record")
    age_ms = (t3 + offset - frame_time) * 1000.0
    if abs(declared_age - age_ms) > _AGE_TOL_MS:
        raise ValueError(
            f"record.frame_age_ms={declared_age!r} does not match derived "
            f"{age_ms!r} ms from t3 + clock_offset_s - frame_monotonic_time_s")

    worst_age_ms = age_ms + uncertainty * 1000.0
    worst_future_ms = age_ms - uncertainty * 1000.0
    max_age = CONTACT_PROFILE["max_frame_age_ms"]
    max_future = CONTACT_PROFILE["max_future_frame_ms"]
    if worst_age_ms > max_age:
        raise ValueError(
            f"worst-case frame age {worst_age_ms:.6f} ms exceeds "
            f"{max_age} ms limit")
    if worst_future_ms < -max_future:
        raise ValueError(
            f"worst-case future frame {-worst_future_ms:.6f} ms exceeds "
            f"{max_future} ms limit")

    return {
        "clock_offset_s": offset,
        "clock_uncertainty_s": uncertainty,
        "png_backend": backend,
        "png_backend_version": version,
        "capture_timing": dict(timing),
        "frame_age_ms": age_ms,
        "worst_frame_age_ms": worst_age_ms,
    }
