"""Boundary tests for tools/leap_h800_probe.py (LEAP v2 H800 validation).

Pure-helper tests run with the stdlib only (they never touch MuJoCo, numpy or
the asset cache).  Scene/schedule tests importorskip numpy/MuJoCo and require
the pinned asset cache; they exercise the REAL accepted model and probe,
nothing is mocked.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "tools" / "leap_h800_probe.py"


def _load_probe():
    spec = importlib.util.spec_from_file_location("leap_h800_probe", PROBE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load_probe()


# ---------------------------------------------------------------------------
# Frozen schedule hashing / validation
# ---------------------------------------------------------------------------

def test_schedule_sha256_deterministic_and_layout():
    rows = [(0.05, 1.4, -0.3), (0.1, 0.2, 0.3)]
    h1 = probe.schedule_sha256(rows)
    h2 = probe.schedule_sha256([tuple(r) for r in rows])
    assert h1 == h2
    assert len(h1) == 64
    # Changing any value changes the hash.
    changed = [rows[0], (0.1, 0.2, 0.30000000000000004 + 1e-12)]
    assert probe.schedule_sha256(changed) != h1


def test_schedule_sha256_width_change_changes_hash():
    a = probe.schedule_sha256([(1.0, 2.0)])
    b = probe.schedule_sha256([(1.0, 2.0, 3.0)])
    assert a != b


def test_validate_schedule_accepts_valid():
    rows = [(0.0, 0.5), (0.25, 0.5)]
    problems = probe.validate_schedule(rows, 2, 2, [(0.0, 0.5), (0.0, 1.0)])
    assert problems == []


def test_validate_schedule_rejects_bad_rows():
    assert probe.validate_schedule([(0.0,)], 2, 1)  # width mismatch
    assert probe.validate_schedule([(0.0, float("nan"))], 2, 1)  # non-finite
    assert probe.validate_schedule([(0.0, 2.0)], 2, 1, [(0.0, 1.0)] * 2)  # range
    assert probe.validate_schedule([(0.0, 0.0)], 2, 2)  # row count


def test_reset_state_sha256_deterministic_and_sensitive():
    h1 = probe.reset_state_sha256([0.0, 0.05], [0.0, 0.0], [0.05, 0.05])
    assert h1 == probe.reset_state_sha256([0.0, 0.05], [0.0, 0.0], [0.05, 0.05])
    assert h1 != probe.reset_state_sha256([0.0, 0.051], [0.0, 0.0], [0.05, 0.05])
    assert h1 != probe.reset_state_sha256([0.0, 0.05], [0.0, 1e-9], [0.05, 0.05])


# ---------------------------------------------------------------------------
# Sampling coverage math
# ---------------------------------------------------------------------------

def test_sample_steps_interval_and_final_inclusion():
    s = probe.sample_steps(14000, 20)
    assert s[0] == 20 and s[-1] == 14000
    assert all(b - a == 20 for a, b in zip(s, s[1:]))
    assert len(s) == 700
    a = probe.sample_steps(14000, 200)
    assert a[0] == 200 and a[-1] == 14000 and len(a) == 70
    assert set(a) <= set(s)  # all-world samples are a subset of world0 samples


def test_sample_steps_forces_final_when_not_multiple():
    s = probe.sample_steps(205, 20)
    assert s[-1] == 205 and s[-2] == 200
    assert all((b - a) <= 20 for a, b in zip(s, s[1:]))


def test_sample_steps_rejects_bad_args():
    with pytest.raises(ValueError):
        probe.sample_steps(0, 20)
    with pytest.raises(ValueError):
        probe.sample_steps(10, 0)


# ---------------------------------------------------------------------------
# Joint bookkeeping helpers (pure)
# ---------------------------------------------------------------------------

def test_coupling_residual_qpos():
    qpos = [0.0, 0.5, 0.5, 0.3, 0.35]
    assert probe.coupling_residual_qpos(qpos, [1, 3], [2, 4]) == pytest.approx(0.05)
    assert probe.coupling_residual_qpos(qpos, [1], [2]) == 0.0


def test_limit_excursion_qpos():
    qpos = [0.1, 1.2, -0.4]
    ranges = [(0.0, 1.0), (-1.0, 1.0), (-0.3, 0.3)]
    assert probe.limit_excursion_qpos(qpos, ranges, [0, 1, 2]) == pytest.approx(0.2)
    assert probe.limit_excursion_qpos([0.5, 0.0, 0.0], ranges, [0, 1, 2]) == 0.0


# ---------------------------------------------------------------------------
# Capacity verdict (naconmax aggregate, njmax per-world)
# ---------------------------------------------------------------------------

def test_capacity_verdict_pass_fail_unknown():
    v, _ = probe.capacity_verdict(100, 50, False, njmax_alloc=10, nefc_peak_sampled=5)
    assert v == "pass"
    v, d = probe.capacity_verdict(100, 101, False, njmax_alloc=10, nefc_peak_sampled=5)
    assert v == "fail" and "nacon" in d
    v, d = probe.capacity_verdict(100, 50, False, njmax_alloc=10, nefc_peak_sampled=11)
    assert v == "fail" and "nefc" in d
    v, _ = probe.capacity_verdict(100, 50, True, njmax_alloc=10, nefc_peak_sampled=5)
    assert v == "fail"
    assert probe.capacity_verdict(None, 50, False, njmax_alloc=10, nefc_peak_sampled=5)[0] == "unknown"
    assert probe.capacity_verdict(100, None, False, njmax_alloc=10, nefc_peak_sampled=5)[0] == "unknown"
    assert probe.capacity_verdict(100, 50, False, njmax_alloc=None, nefc_peak_sampled=5)[0] == "unknown"
    assert probe.capacity_verdict(100, 50, False, njmax_alloc=10, nefc_peak_sampled=None)[0] == "unknown"


# ---------------------------------------------------------------------------
# Stage planning and 512 gating
# ---------------------------------------------------------------------------

def test_plan_stage_sequence():
    assert probe.plan_stage_sequence([1, 32, 128, 512]) == [1, 32, 128, 512]
    assert probe.plan_stage_sequence([1, 32]) == [1, 32]
    with pytest.raises(ValueError):
        probe.plan_stage_sequence([32, 1])
    with pytest.raises(ValueError):
        probe.plan_stage_sequence([1, 128, 32])
    with pytest.raises(ValueError):
        probe.plan_stage_sequence([1, 7])
    with pytest.raises(ValueError):
        probe.plan_stage_sequence([])


def test_gate_512():
    ok, _ = probe.gate_512(True, 40000, 600.0)
    assert ok
    assert not probe.gate_512(False, 40000, 600.0)[0]
    assert not probe.gate_512(True, None, 600.0)[0]
    assert not probe.gate_512(True, 1024, 600.0)[0]
    assert not probe.gate_512(True, 40000, None)[0]
    assert not probe.gate_512(True, 40000, 299.9)[0]
    assert probe.gate_512(True, 40000, 300.0)[0]


def test_main_rejects_existing_out():
    import os
    import sys
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    old_argv = sys.argv
    sys.argv = ["leap_h800_probe.py", "--out", path, "--tiny", "--cpu-only"]
    try:
        with pytest.raises(SystemExit):
            probe.main()
    finally:
        sys.argv = old_argv
        os.unlink(path)


# ---------------------------------------------------------------------------
# Real scene: canonical schedule and reset from the ACCEPTED sources
# ---------------------------------------------------------------------------

np = pytest.importorskip("numpy", reason="numpy not installed")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad import leap_sim  # noqa: E402

ASSETS_PRESENT = leap_sim.DOWNLOAD_ROOT.is_dir()


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned asset cache not downloaded")
def test_scene_schedule_matches_accepted_probe():
    scene = probe.build_scene()
    assert scene["steps"] == 14000
    assert scene["nu"] == 17 and scene["nq"] == 21 and scene["njnt"] == 21
    assert scene["schedule_problems"] == []
    # Row 0 is exactly the open-pose hold vector; every row stays in ctrlrange.
    m = scene["model"]
    assert len(scene["rows"]) == 14000
    for row in (scene["rows"][0], scene["rows"][4000], scene["rows"][-1]):
        for j, v in enumerate(row):
            lo, hi = m.actuator_ctrlrange[j]
            assert lo - 1e-12 <= v <= hi + 1e-12
    # The frozen rows are exactly the accepted probe's ctrl_at samples.
    cpu_probe = probe._import_by_path(probe.CPU_PROBE_PY, "leap_model_probe_for_test")
    for i in (0, 1, 999, 4000, 6999, 10399, 13999):
        expected = cpu_probe.ctrl_at(m, i * scene["timestep"])
        assert np.allclose(scene["rows"][i], expected, rtol=0, atol=0)


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned asset cache not downloaded")
def test_scene_canonical_reset_matches_open_pose():
    from embodied_trackpad.leap_sim import open_pose_qpos

    scene = probe.build_scene()
    d = scene["canonical_reset"]()
    assert np.allclose(np.asarray(d.qpos), open_pose_qpos(scene["model"]), atol=0)
    assert np.allclose(np.asarray(d.qvel), 0.0, atol=0)
    assert np.allclose(np.asarray(d.ctrl), np.asarray(scene["rows"][0]), atol=0)
    h1 = probe.reset_state_sha256(d.qpos, d.qvel, d.ctrl)
    assert h1 == scene["reset_state_sha256"]


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned asset cache not downloaded")
def test_scene_pinned_hashes_enforced():
    scene = probe.build_scene()
    assert scene["hashes"]["model_py_sha256"] == probe.EXPECTED_MODEL_PY_SHA256
    assert scene["hashes"]["accepted_cpu_probe_sha256"] == probe.EXPECTED_CPU_PROBE_SHA256
    # Tampering with the expected hash must abort the scene build.
    orig = probe.EXPECTED_MODEL_PY_SHA256
    probe.EXPECTED_MODEL_PY_SHA256 = "0" * 64
    try:
        with pytest.raises(RuntimeError):
            probe.build_scene()
    finally:
        probe.EXPECTED_MODEL_PY_SHA256 = orig


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned asset cache not downloaded")
def test_coupling_addresses_cover_four_fingers():
    scene = probe.build_scene()
    assert len(scene["dip_qposadrs"]) == 4
    assert len(scene["pip_qposadrs"]) == 4
    names = [scene["model"].joint(j).name for j in range(scene["model"].njnt)]
    for f in ("index", "middle", "ring", "pinky"):
        assert f"{f}_dip" in names and f"{f}_pip" in names


# ---------------------------------------------------------------------------
# Reset broadcast audit (backend-dtype exact, no tolerance substitution)
# ---------------------------------------------------------------------------

def _audit_inputs(dtype, nworld=3):
    canonical = {
        "qpos": [0.0, 0.05, 1.4, -0.3],
        "qvel": [0.0, 0.0, 0.0, 0.0],
        "ctrl": [0.05, 0.05, 1.2, -0.45],
    }
    expected = {k: np.asarray(v, dtype=np.float64).astype(dtype)
                for k, v in canonical.items()}
    actual = {k: np.tile(expected[k], (nworld, 1)) for k in canonical}
    return canonical, actual


def test_reset_broadcast_audit_exact_pass_float32():
    canonical, actual = _audit_inputs(np.float32)
    audit = probe.reset_broadcast_audit(actual, canonical)
    assert audit["pass"]
    f = audit["fields"]["qpos"]
    assert f["backend_dtype"] == "float32"
    assert f["shape_ok"] and f["finite"] and f["all_worlds_exact"]
    assert f["backend_expected_sha256"] == f["backend_actual_sha256_world0"]
    # The canonical f64 hash is preserved for provenance and is DISTINCT from
    # the float32 backend hash: 0.05 / 1.4 / -0.3 are not float32-exact.
    assert f["canonical_f64_sha256"] != f["backend_expected_sha256"]
    assert f["max_conversion_delta"] > 0.0


def test_reset_broadcast_audit_exact_pass_float64():
    canonical, actual = _audit_inputs(np.float64)
    audit = probe.reset_broadcast_audit(actual, canonical)
    assert audit["pass"]
    f = audit["fields"]["ctrl"]
    assert f["backend_dtype"] == "float64"
    assert f["max_conversion_delta"] == 0.0
    assert f["canonical_f64_sha256"] == f["backend_expected_sha256"]


def test_reset_broadcast_audit_one_ulp_is_not_a_tolerance():
    # A single float32 ulp deviation must FAIL: only the backend
    # representation is permitted, never a relaxed numerical tolerance.
    canonical, actual = _audit_inputs(np.float32)
    actual["qpos"][0, 1] = np.nextafter(np.float32(0.05), np.float32(1.0))
    audit = probe.reset_broadcast_audit(actual, canonical)
    assert not audit["pass"]
    assert audit["fields"]["qpos"]["mismatch_worlds"] == [0]


def test_reset_broadcast_audit_perturbations_fail():
    canonical, actual = _audit_inputs(np.float32)

    def perturbed(field, world, index, value):
        bad = {k: v.copy() for k, v in actual.items()}
        bad[field][world, index] = value
        return bad

    # one state (qpos) value in one world
    a = probe.reset_broadcast_audit(perturbed("qpos", 1, 2, np.float32(1.41)), canonical)
    assert not a["pass"] and a["fields"]["qpos"]["mismatch_worlds"] == [1]
    # one control value
    a = probe.reset_broadcast_audit(perturbed("ctrl", 0, 0, np.float32(0.051)), canonical)
    assert not a["pass"] and a["fields"]["ctrl"]["mismatch_worlds"] == [0]
    # one field entirely wrong (qvel non-zero everywhere)
    bad = {k: v.copy() for k, v in actual.items()}
    bad["qvel"][:] = np.float32(1e-6)
    a = probe.reset_broadcast_audit(bad, canonical)
    assert not a["pass"] and a["fields"]["qvel"]["mismatch_worlds"] == [0, 1, 2]
    # one world row only
    a = probe.reset_broadcast_audit(perturbed("qpos", 2, 3, np.float32(-0.31)), canonical)
    assert not a["pass"] and a["fields"]["qpos"]["mismatch_worlds"] == [2]
    # non-finite
    bad = {k: v.copy() for k, v in actual.items()}
    bad["qpos"][0, 0] = np.float32("nan")
    a = probe.reset_broadcast_audit(bad, canonical)
    assert not a["pass"] and not a["fields"]["qpos"]["finite"]
    # wrong shape
    a = probe.reset_broadcast_audit(
        {k: v[:, :2] for k, v in actual.items()}, canonical)
    assert not a["pass"] and not a["fields"]["qpos"]["shape_ok"]
