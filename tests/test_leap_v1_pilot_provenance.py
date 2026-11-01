"""M2b1 tests for pilot_provenance.validate_capture_timing.

Synthetic fixtures only; no real episodes, no image decode, no actor
features. Validates the pure capture-timing provenance contract:
legacy exact-null path, full new-timing validation shared by action and
terminal records, and no input mutation.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from embodied_trackpad.leap_v1.pilot_provenance import (  # noqa: E402
    validate_capture_timing,
)
from embodied_trackpad.leap_v1.visual_servo import CONTACT_PROFILE  # noqa: E402

OFFSET = 0.25
UNC = 0.001
T0, T1, T2, T3 = 100.0, 100.004, 100.006, 100.010
AGE_MS = 20.0  # frame_monotonic_time_s = t3 + offset - age/1000
LEGACY_NULL = {"clock_offset_s": None, "clock_uncertainty_s": None}


def _timing(**over):
    base = {
        "t0_monotonic_s": T0, "t1_monotonic_s": T1,
        "t2_monotonic_s": T2, "t3_monotonic_s": T3,
        "capture_save_ms": (T1 - T0) * 1000.0,
        "decode_ms": (T2 - T1) * 1000.0,
        "extract_ms": (T3 - T2) * 1000.0,
        "total_ms": (T3 - T0) * 1000.0,
        "clock_offset_s": OFFSET, "clock_uncertainty_s": UNC,
        "png_backend": "stdlib", "png_backend_version": None,
    }
    base.update(over)
    return base


def _sides(backend="stdlib", version=None):
    side = {"png_backend": backend, "png_backend_version": version}
    return ({"clock_offset_s": OFFSET, "clock_uncertainty_s": UNC, **side},
            dict(side))


def _record(timing=None, age_ms=AGE_MS, **over):
    rec = {
        "frame_id": 2,
        "frame_monotonic_time_s": T3 + OFFSET - age_ms / 1000.0,
        "frame_age_ms": age_ms,
        "capture_timing": _timing() if timing is None else timing,
    }
    rec.update(over)
    return rec


def test_valid_action_and_terminal_records():
    summary, metadata = _sides()
    for phase in ("action", "terminal"):
        rec = _record(phase=phase)
        out = validate_capture_timing(rec, summary, metadata)
        assert out["clock_offset_s"] == OFFSET
        assert out["clock_uncertainty_s"] == UNC
        assert out["png_backend"] == "stdlib"
        assert out["png_backend_version"] is None
        assert out["frame_age_ms"] == pytest.approx(AGE_MS, abs=1e-9)
        assert out["worst_frame_age_ms"] == pytest.approx(
            AGE_MS + UNC * 1000.0, abs=1e-9)
        assert out["capture_timing"] == rec["capture_timing"]
        assert out["capture_timing"] is not rec["capture_timing"]


def test_legacy_absence_returns_exact_null_dict():
    assert validate_capture_timing({"frame_id": 1}, {}, {}) == LEGACY_NULL
    assert list(validate_capture_timing({}, {}, {}).keys()) == [
        "clock_offset_s", "clock_uncertainty_s"]


def test_terminal_phase_does_not_bypass_helper():
    summary, metadata = _sides()
    rec = _record(phase="terminal", age_ms=400.0)
    with pytest.raises(ValueError, match="worst-case frame age"):
        validate_capture_timing(rec, summary, metadata)


@pytest.mark.parametrize("key,value", [
    ("clock_offset_s", True), ("clock_uncertainty_s", False),
    ("t0_monotonic_s", "100.0"), ("total_ms", float("nan")),
    ("decode_ms", float("inf")), ("capture_save_ms", None),
])
def test_nonfinite_or_nonnumeric_timing_fields_rejected(key, value):
    timing = _timing(**{key: value})
    summary, metadata = _sides()
    with pytest.raises(ValueError):
        validate_capture_timing(_record(timing=timing), summary, metadata)


@pytest.mark.parametrize("key,value", [
    ("frame_monotonic_time_s", float("nan")), ("frame_age_ms", True),
    ("frame_age_ms", float("inf")),
])
def test_record_frame_fields_validated(key, value):
    summary, metadata = _sides()
    with pytest.raises(ValueError):
        validate_capture_timing(_record(**{key: value}), summary, metadata)


def test_stage_order_and_duration_recomputation():
    summary, metadata = _sides()
    with pytest.raises(ValueError, match="out of order"):
        validate_capture_timing(
            _record(timing=_timing(t2_monotonic_s=T0 - 1.0)),
            summary, metadata)
    with pytest.raises(ValueError, match="capture_save_ms"):
        validate_capture_timing(
            _record(timing=_timing(capture_save_ms=5.0)), summary, metadata)
    with pytest.raises(ValueError, match="total_ms"):
        validate_capture_timing(
            _record(timing=_timing(total_ms=-1.0)), summary, metadata)


def test_frame_age_recomputed_with_plus_offset():
    summary, metadata = _sides()
    # Wrong-sign age (t3 - offset - fmt) must not validate.
    rec = _record()
    rec["frame_age_ms"] = (T3 - OFFSET - rec["frame_monotonic_time_s"]) * 1000.0
    with pytest.raises(ValueError, match="frame_age_ms"):
        validate_capture_timing(rec, summary, metadata)
    # Exact derived age passes (already covered by valid test; nonzero
    # offset there proves the PLUS-offset formula).
    rec["frame_age_ms"] = (T3 + OFFSET - rec["frame_monotonic_time_s"]) * 1000.0
    assert validate_capture_timing(rec, summary, metadata)["frame_age_ms"] == \
        pytest.approx(rec["frame_age_ms"], abs=1e-9)


def test_uncertainty_pushing_age_beyond_250_rejected():
    summary, metadata = _sides()
    summary = {**summary, "clock_uncertainty_s": 0.02}
    rec = _record(timing=_timing(clock_uncertainty_s=0.02),
                  age_ms=CONTACT_PROFILE["max_frame_age_ms"] - 5.0)
    with pytest.raises(ValueError, match="250"):
        validate_capture_timing(rec, summary, metadata)


def test_future_frame_beyond_50_rejected():
    summary, metadata = _sides()
    summary = {**summary, "clock_uncertainty_s": 0.02}
    rec = _record(timing=_timing(clock_uncertainty_s=0.02),
                  age_ms=-(CONTACT_PROFILE["max_future_frame_ms"] - 5.0))
    with pytest.raises(ValueError, match="future"):
        validate_capture_timing(rec, summary, metadata)


def test_missing_or_partial_new_timing_never_downgraded():
    summary, metadata = _sides()
    # Trigger via summary backend but no capture_timing object.
    with pytest.raises(ValueError, match="capture_timing"):
        validate_capture_timing({"frame_id": 1}, summary, metadata)
    # Partial: one required key missing.
    timing = _timing()
    del timing["decode_ms"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_capture_timing(_record(timing=timing), summary, metadata)
    # required=True forces new timing even with no other trigger.
    with pytest.raises(ValueError):
        validate_capture_timing({}, {}, {}, required=True)


def test_backend_and_version_bindings_enforced():
    summary, metadata = _sides()
    good = _record()
    # Bad backend types/values -> ValueError, not TypeError.
    for bad in ("bogus", ["stdlib"], 3):
        with pytest.raises(ValueError):
            validate_capture_timing(
                _record(timing=_timing(png_backend=bad)), summary, metadata)
    # Pillow needs a nonempty version.
    with pytest.raises(ValueError, match="pillow"):
        validate_capture_timing(
            _record(timing=_timing(png_backend="pillow",
                                   png_backend_version="")),
            summary, metadata)
    # Missing binding on either side is rejected.
    summ_no_backend = {k: v for k, v in summary.items() if k != "png_backend"}
    with pytest.raises(ValueError, match="summary"):
        validate_capture_timing(good, summ_no_backend, metadata)
    meta_no_version = {k: v for k, v in metadata.items()
                       if k != "png_backend_version"}
    with pytest.raises(ValueError, match="metadata"):
        validate_capture_timing(good, summary, meta_no_version)
    # Conflicting bindings are rejected.
    with pytest.raises(ValueError, match="does not match"):
        validate_capture_timing(
            good, summary, {**metadata, "png_backend": "pillow"})
    with pytest.raises(ValueError, match="does not match"):
        validate_capture_timing(
            good, {**summary, "png_backend_version": "3.12.0"}, metadata)


def test_stdlib_python_version_string_supported():
    version = "3.12.7"
    summary, metadata = _sides(version=version)
    out = validate_capture_timing(
        _record(timing=_timing(png_backend_version=version)),
        summary, metadata)
    assert out["png_backend_version"] == version


def test_pillow_backend_with_version_supported():
    summary, metadata = _sides(backend="pillow", version="10.3.0")
    out = validate_capture_timing(
        _record(timing=_timing(png_backend="pillow",
                               png_backend_version="10.3.0")),
        summary, metadata)
    assert out["png_backend"] == "pillow" and out["png_backend_version"] == "10.3.0"


def test_summary_clock_calibration_must_match():
    summary, metadata = _sides()
    with pytest.raises(ValueError, match="clock_offset_s"):
        validate_capture_timing(
            good := _record(), {**summary, "clock_offset_s": OFFSET + 0.5},
            metadata)
    with pytest.raises(ValueError, match="clock_uncertainty_s"):
        validate_capture_timing(
            good, {**summary, "clock_uncertainty_s": UNC + 0.5}, metadata)
    # Non-finite summary calibration is rejected, not propagated.
    with pytest.raises(ValueError):
        validate_capture_timing(
            good, {**summary, "clock_offset_s": float("nan")}, metadata)


def test_inputs_are_not_mutated():
    record, summary, metadata = _record(), *_sides()
    snapshot = copy.deepcopy((record, summary, metadata))
    validate_capture_timing(record, summary, metadata)
    validate_capture_timing({"frame_id": 1}, {}, {})
    assert (record, summary, metadata) == snapshot
