"""Boundary tests for tools/leap_v1_gpu_smoke.py (LEAP v1 H800 runtime smoke).

Pure-helper tests run with the stdlib only (they never touch MuJoCo, numpy,
torch or the asset cache).  Scene/schedule tests importorskip numpy/MuJoCo and
require the pinned v1 asset cache; they exercise the REAL accepted v1 model
and diagnostics, nothing is mocked.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "tools" / "leap_v1_gpu_smoke.py"


def _load_smoke():
    spec = importlib.util.spec_from_file_location("leap_v1_gpu_smoke", SMOKE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke = _load_smoke()


# ---------------------------------------------------------------------------
# Frozen schedule hashing / validation
# ---------------------------------------------------------------------------

def test_schedule_sha256_deterministic_and_layout():
    rows = [(0.05, 1.4, -0.3), (0.1, 0.2, 0.3)]
    h1 = smoke.schedule_sha256(rows)
    h2 = smoke.schedule_sha256([tuple(r) for r in rows])
    assert h1 == h2
    assert len(h1) == 64
    changed = [rows[0], (0.1, 0.2, 0.30000000000000004 + 1e-12)]
    assert smoke.schedule_sha256(changed) != h1


def test_schedule_sha256_width_change_changes_hash():
    a = smoke.schedule_sha256([(1.0, 2.0)])
    b = smoke.schedule_sha256([(1.0, 2.0, 3.0)])
    assert a != b


def test_validate_schedule_accepts_valid():
    rows = [(0.0, 0.5), (0.25, 0.5)]
    problems = smoke.validate_schedule(rows, 2, 2, [(0.0, 0.5), (0.0, 1.0)])
    assert problems == []


def test_validate_schedule_rejects_bad_rows():
    assert smoke.validate_schedule([(0.0,)], 2, 1)  # width mismatch
    assert smoke.validate_schedule([(0.0, float("nan"))], 2, 1)  # non-finite
    assert smoke.validate_schedule([(0.0, 2.0)], 2, 1, [(0.0, 1.0)] * 2)  # range


def test_reset_state_sha256_deterministic_and_sensitive():
    h1 = smoke.reset_state_sha256([0.0, 1.0], [0.0], [0.0, 0.0])
    h2 = smoke.reset_state_sha256([0.0, 1.0], [0.0], [0.0, 0.0])
    assert h1 == h2
    assert smoke.reset_state_sha256([0.0, 1.0], [0.0], [0.0, 1e-15]) != h1


def test_sample_steps_interval_and_final_inclusion():
    assert smoke.sample_steps(1000, 200) == [200, 400, 600, 800, 1000]
    assert smoke.sample_steps(1050, 200)[-1] == 1050
    assert smoke.sample_steps(50, 200) == [50]


def test_sample_steps_rejects_bad_args():
    with pytest.raises(ValueError):
        smoke.sample_steps(0, 10)
    with pytest.raises(ValueError):
        smoke.sample_steps(10, 0)


def test_limit_excursion_qpos():
    ranges = [(-1.0, 1.0), (0.0, 0.5)]
    adrs = [0, 3]
    assert smoke.limit_excursion_qpos([0.2, 0, 0, 0.4], ranges, adrs) == 0.0
    exc = smoke.limit_excursion_qpos([1.2, 0, 0, -0.1], ranges, adrs)
    assert abs(exc - 0.2) < 1e-12


def test_capacity_verdict_pass_fail_unknown():
    v, _ = smoke.capacity_verdict(256, 100, False, njmax_alloc=1024,
                                  nefc_peak_sampled=512)
    assert v == "pass"
    v, _ = smoke.capacity_verdict(256, 300, False, njmax_alloc=1024,
                                  nefc_peak_sampled=512)
    assert v == "fail"
    v, _ = smoke.capacity_verdict(256, 100, True, njmax_alloc=1024,
                                  nefc_peak_sampled=512)
    assert v == "fail"
    v, _ = smoke.capacity_verdict(None, 100, False, njmax_alloc=1024,
                                  nefc_peak_sampled=512)
    assert v == "unknown"
    v, _ = smoke.capacity_verdict(256, -1, False, njmax_alloc=1024,
                                  nefc_peak_sampled=512)
    assert v == "unknown"
    v, _ = smoke.capacity_verdict(256, 100, False, njmax_alloc=1024,
                                  nefc_peak_sampled=2048)
    assert v == "fail"
    v, _ = smoke.capacity_verdict(256, 100, False, njmax_alloc=None,
                                  nefc_peak_sampled=512)
    assert v == "unknown"


def test_plan_stage_sequence_smoke_set_only():
    assert smoke.plan_stage_sequence([1, 32]) == [1, 32]
    assert smoke.plan_stage_sequence([1]) == [1]
    with pytest.raises(ValueError):
        smoke.plan_stage_sequence([32, 1])  # must be ascending
    with pytest.raises(ValueError):
        smoke.plan_stage_sequence([32])  # must start at 1
    with pytest.raises(ValueError):
        smoke.plan_stage_sequence([1, 128])  # beyond the fixed smoke set
    with pytest.raises(ValueError):
        smoke.plan_stage_sequence([1, 8192])  # explicitly deferred scale
    with pytest.raises(ValueError):
        smoke.plan_stage_sequence([1, 1])  # duplicates
    with pytest.raises(ValueError):
        smoke.plan_stage_sequence([])


def test_numerical_threads_bound():
    assert smoke.numerical_threads({}) == 1  # absent allocation -> default 1
    assert smoke.numerical_threads({"SLURM_CPUS_PER_TASK": "8"}) == 2
    assert smoke.numerical_threads({"SLURM_CPUS_PER_TASK": "2"}) == 2
    assert smoke.numerical_threads({"SLURM_CPUS_PER_TASK": "1"}) == 1
    assert smoke.numerical_threads({"SLURM_CPUS_PER_TASK": "junk"}) == 1
    assert smoke.numerical_threads({"SLURM_CPUS_PER_TASK": "0"}) == 1


def test_is_h800_accepts_only_real_h800_names():
    assert smoke.is_h800("NVIDIA H800 80GB HBM3")
    assert smoke.is_h800("NVIDIA H800 PCIe")
    assert not smoke.is_h800("NVIDIA H100 80GB HBM3")
    assert not smoke.is_h800("NVIDIA A800-SXM4-80GB")
    assert not smoke.is_h800("")
    assert not smoke.is_h800(None)


def test_compact_phases_preserves_durations_and_order():
    class P:
        def __init__(self, name, t0, t1, kind, base=None, target=None,
                     hold_channels=(), render_label=None):
            self.name, self.t0, self.t1, self.kind = name, t0, t1, kind
            self.base, self.target = base, target
            self.hold_channels, self.render_label = hold_channels, render_label

    phases = [P("a", 0.0, 0.2, "hold"), P("b", 1.4, 2.5, "ramp"),
              P("c", 2.5, 3.2, "hold")]
    out = smoke.compact_phases(phases)
    assert [(p.name, round(p.t0, 6), round(p.t1, 6)) for p in out] == [
        ("a", 0.0, 0.2), ("b", 0.2, 1.3), ("c", 1.3, 2.0)]
    for src, dst in zip(phases, out):
        assert abs((src.t1 - src.t0) - (dst.t1 - dst.t0)) < 1e-12
        assert dst.kind == src.kind


def test_phase_of_boundaries():
    spans = [("a", 0.0, 0.2), ("b", 0.2, 1.3), ("c", 1.3, 2.0)]
    assert smoke.phase_of(spans, 0.0) == "a"
    assert smoke.phase_of(spans, 0.2) == "b"
    assert smoke.phase_of(spans, 1.299) == "b"
    assert smoke.phase_of(spans, 1.3) == "c"
    assert smoke.phase_of(spans, 99.0) == "c"  # past end -> last phase


def test_main_rejects_existing_out():
    import os
    import sys
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    old_argv = sys.argv
    sys.argv = ["leap_v1_gpu_smoke.py", "--out", path, "--tiny", "--cpu-only"]
    try:
        with pytest.raises(SystemExit):
            smoke.main()
    finally:
        sys.argv = old_argv
        os.unlink(path)


# ---------------------------------------------------------------------------
# Real scene: canonical schedule and reset from the ACCEPTED v1 sources
# ---------------------------------------------------------------------------

np = pytest.importorskip("numpy", reason="numpy not installed")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad import leap_v1  # noqa: E402

ASSETS_PRESENT = leap_v1.DOWNLOAD_ROOT.is_dir()


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned v1 asset cache not downloaded")
def test_scene_structure_is_exact_v1_16_channel():
    scene = smoke.build_scene()
    # Exact 16-channel v1 Full: 16 actuators / 16 joints, no equalities, no
    # v2 17-motor/21-joint mapping.
    assert scene["nu"] == 16 and scene["njnt"] == 16
    assert scene["nq"] == 16 and scene["nv"] == 16
    assert int(scene["model"].neq) == 0
    assert scene["schedule_problems"] == []


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned v1 asset cache not downloaded")
def test_scene_schedule_is_short_and_contains_spread_and_flex():
    scene = smoke.build_scene()
    names = [n for n, _t0, _t1 in scene["phase_spans"]]
    # Two-channel sweep subset only: the other 14 sweeps are absent.
    assert "sweep_if_mcp_hold" in names and "sweep_if_pip_hold" in names
    assert not any(n.startswith("sweep_mf_") for n in names)
    # Contact-free movement + spread/flex diagnostics are both present.
    assert "spread_hold" in names
    assert "flex_hold" in names
    assert names[0] == "open_settle" and names[-1] == "open_final"
    # Compact re-timing: no gaps, contiguous coverage of [0, t_end].
    spans = scene["phase_spans"]
    assert spans[0][1] == 0.0
    for (_n0, _t0, t1), (_n1, t0_next, _t1n) in zip(spans, spans[1:]):
        assert abs(t1 - t0_next) < 1e-12
    t_end = spans[-1][2]
    # Short schedule: the full 16-sweep script is ~22.4 s; the compacted
    # two-sweep smoke must be under 8 s (exactly 6.6 s of diagnostics time).
    assert t_end < 8.0
    assert scene["steps"] == int(round(t_end / scene["timestep"]))
    # Row 0 is exactly the open-pose hold vector; every row stays in range.
    assert all(v == 0.0 for v in scene["rows"][0])
    for row in (scene["rows"][500], scene["rows"][-1]):
        assert len(row) == scene["nu"]
        for v, (lo, hi) in zip(row, scene["ctrlrange"]):
            assert lo - 1e-12 <= v <= hi + 1e-12


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned v1 asset cache not downloaded")
def test_scene_flex_phase_reaches_flexion_target():
    """The kept flex phase must command the accepted diagnostics targets."""
    from embodied_trackpad.leap_v1.diagnostics import FLEX_TARGET

    scene = smoke.build_scene()
    t0, t1 = next((t0, t1) for n, t0, t1 in scene["phase_spans"]
                  if n == "flex_hold")
    dt = scene["timestep"]
    row = scene["rows"][int(round(((t0 + t1) / 2) / dt))]
    m = scene["model"]
    names = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    for jname, target in FLEX_TARGET.items():
        assert abs(row[names.index(jname)] - target) < 1e-9


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned v1 asset cache not downloaded")
def test_scene_canonical_reset_matches_open_pose():
    scene = smoke.build_scene()
    d = scene["canonical_reset"]()
    assert np.allclose(d.qpos, scene["open_pose_qpos"])
    assert np.all(d.qvel == 0.0)
    assert np.all(d.ctrl == 0.0)
    assert smoke.reset_state_sha256(d.qpos, d.qvel, d.ctrl) == \
        scene["reset_state_sha256"]


@pytest.mark.skipif(not ASSETS_PRESENT, reason="pinned v1 asset cache not downloaded")
def test_scene_pinned_hashes_enforced(monkeypatch):
    monkeypatch.setattr(smoke, "EXPECTED_MODEL_PY_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="model.py sha256 mismatch"):
        smoke.build_scene()


def test_reset_broadcast_audit_exact_pass_float32():
    canonical = {"qpos": [0.0, 0.1, -0.25], "qvel": [0.0, 0.0, 0.0],
                 "ctrl": [0.0, 0.5, 0.5]}
    expected = np.asarray([np.float32(v) for v in canonical["qpos"]])
    actual = np.tile(expected, (4, 1))
    audit = smoke.reset_broadcast_audit(
        {"qpos": actual,
         "qvel": np.zeros((4, 3), dtype=np.float32),
         "ctrl": np.tile(np.asarray([np.float32(v) for v in canonical["ctrl"]]),
                         (4, 1))},
        canonical)
    assert audit["pass"], audit


def test_reset_broadcast_audit_one_ulp_is_not_a_tolerance():
    canonical = {"qpos": [0.1]}
    actual = np.asarray([[np.nextafter(np.float32(0.1), np.float32(1.0))]],
                        dtype=np.float32)
    audit = smoke.reset_broadcast_audit({"qpos": actual}, canonical)
    assert not audit["pass"]
    assert not audit["fields"]["qpos"]["all_worlds_exact"]


def test_reset_broadcast_audit_perturbations_fail():
    canonical = {"qpos": [0.0, 0.1], "qvel": [0.0, 0.0]}
    base = np.tile(np.asarray([0.0, 0.1], dtype=np.float32), (3, 1))
    bad = base.copy()
    bad[2, 1] = np.float32(0.10000001)
    audit = smoke.reset_broadcast_audit(
        {"qpos": bad, "qvel": np.zeros((3, 2), dtype=np.float32)}, canonical)
    assert not audit["pass"]
    assert audit["fields"]["qpos"]["mismatch_worlds"] == [2]
    # Shape mismatch fails.
    audit = smoke.reset_broadcast_audit(
        {"qpos": np.zeros((3, 3), dtype=np.float32)}, canonical)
    assert not audit["pass"]
    assert not audit["fields"]["qpos"]["shape_ok"]
    # Non-finite fails.
    bad = base.copy()
    bad[0, 0] = np.float32("nan")
    audit = smoke.reset_broadcast_audit({"qpos": bad}, canonical)
    assert not audit["pass"]
    assert not audit["fields"]["qpos"]["finite"]
