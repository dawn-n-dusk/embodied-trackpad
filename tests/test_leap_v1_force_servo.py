"""Pure-math tests for LEAP v1 normal-force admittance increment."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad.leap_v1.force_servo import (  # noqa: E402
    NormalForceConfig,
    normal_force_admittance,
)
from embodied_trackpad.leap_v1.visual_physics import NU, NamedJointMap  # noqa: E402

NU = int(NU)

# A simple, non-degenerate normal row aligned mostly with the first index joint.
N_UNIT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
# A recorded normal row (cycle 20 seed2 probe) for nontrivial scaling checks.
N20 = np.array([-0.08065036440031177, 0.00023046520152102817,
                -0.03697772552088247, -0.013269084200966147], dtype=float)

# Default mapping, identical to normal_priority tests.
MAP = NamedJointMap(
    ("if_mcp", "if_rot", "if_pip", "if_dip"),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    ((-0.314, 2.23), (-1.047, 1.047), (-0.506, 1.885), (-0.366, 2.042)),
)

# Mapping for the coordinator's reproduced floating-boundary case.
MAP_UNIT_RANGE = NamedJointMap(
    ("if_mcp", "if_rot", "if_pip", "if_dip"),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
)


def _call(current, anchor, n, force=0.8, dt=0.002, mapping=MAP,
          cfg=None):
    cfg = cfg or NormalForceConfig()
    return normal_force_admittance(
        np.asarray(current, dtype=float),
        np.asarray(anchor, dtype=float),
        np.asarray(n, dtype=float),
        force, dt, mapping, cfg,
    )


def test_force_signs_and_dt_proportionality():
    cfg = NormalForceConfig()
    ctrl = np.zeros(NU)
    anchor = np.zeros(NU)
    # Higher than target -> UPWARD (+z) correction.
    out, info = _call(ctrl, anchor, N_UNIT, force=1.0, dt=0.002)
    assert info["force_error_n"] == pytest.approx(0.2)
    assert info["requested_normal_speed_m_s"] == pytest.approx(0.0008)
    assert info["requested_dz_m"] == pytest.approx(1.6e-6)
    assert info["predicted_normal_dz_m"] == pytest.approx(info["requested_dz_m"])
    assert info["common_scale"] == pytest.approx(1.0)
    assert info["applied_dq"][0] > 0.0
    # Lower than target -> DOWNWARD (-z) correction.
    out, info = _call(ctrl, anchor, N_UNIT, force=0.0, dt=0.002)
    assert info["force_error_n"] == pytest.approx(-0.8)
    assert info["requested_normal_speed_m_s"] == pytest.approx(-0.002)  # clipped
    assert info["requested_dz_m"] == pytest.approx(-4.0e-6)
    assert info["predicted_normal_dz_m"] == pytest.approx(info["requested_dz_m"])
    assert info["applied_dq"][0] < 0.0
    # Exact target -> zero action.
    out, info = _call(ctrl, anchor, N_UNIT, force=0.8, dt=0.002)
    assert info["force_error_n"] == pytest.approx(0.0)
    assert info["requested_dz_m"] == pytest.approx(0.0)
    assert np.allclose(info["applied_dq"], 0.0)
    assert np.allclose(out, ctrl)
    assert info["bound_limited"] == []
    # dt proportionality (same force, double dt -> double requested dz).
    _, info1 = _call(ctrl, anchor, N_UNIT, force=1.0, dt=0.002)
    _, info2 = _call(ctrl, anchor, N_UNIT, force=1.0, dt=0.004)
    assert info2["requested_dz_m"] == pytest.approx(2.0 * info1["requested_dz_m"])
    assert np.allclose(np.array(info2["applied_dq"]), 2.0 * np.array(info1["applied_dq"]))


@pytest.mark.parametrize("kind", ["substep", "range", "cycle_envelope"])
def test_common_bounds_preserve_scaled_normal_constraint(kind):
    # Use force=2.0 / dt=0.01 -> requested dz = 2e-5 m, dq = 2e-5 rad on N_UNIT,
    # large enough to bind each 1e-5 headroom below the bound.
    if kind == "substep":
        ctrl = np.zeros(NU)
        anchor = np.zeros(NU)
        cfg = NormalForceConfig(max_joint_step_rad=1e-5)
        out, info = _call(ctrl, anchor, N_UNIT, force=2.0, dt=0.01, cfg=cfg)
        assert max(abs(v) for v in info["applied_dq"]) == pytest.approx(1e-5)
    elif kind == "range":
        cfg = NormalForceConfig()
        hi = MAP.ranges[0][1]
        anchor = np.zeros(NU)
        anchor[0] = hi - 1e-5
        ctrl = anchor.copy()
        out, info = _call(ctrl, anchor, N_UNIT, force=2.0, dt=0.01, cfg=cfg)
        assert out[0] == pytest.approx(hi, abs=1e-12)
    else:  # cycle_envelope
        cfg = NormalForceConfig(max_cycle_delta_rad=0.005)
        anchor = np.zeros(NU)
        ctrl = np.zeros(NU)
        ctrl[0] = 0.005 - 1e-5  # 1e-5 headroom before |ctrl-anchor| = 0.005.
        out, info = _call(ctrl, anchor, N_UNIT, force=2.0, dt=0.01, cfg=cfg)
        assert out[0] == pytest.approx(0.005, abs=1e-12)
    s = info["common_scale"]
    assert 0.0 < s < 1.0
    applied = np.array(info["applied_dq"])
    n = np.asarray(N_UNIT, dtype=float)
    assert n @ applied == pytest.approx(s * info["requested_dz_m"], abs=1e-15)
    assert info["predicted_normal_dz_m"] == pytest.approx(n @ applied)
    assert info["bound_limited"]


def test_reproduced_float_boundary_and_repeat_stability():
    # Coordinator reproduction: 0.605 - 0.6 rounds to 5.0000000000000044e-3,
    # must NOT fail a 0.005 envelope.
    ctrl = np.zeros(NU)
    ctrl[0] = 0.605
    anchor = np.zeros(NU)
    anchor[0] = 0.6
    out, info = _call(ctrl, anchor, N_UNIT, force=0.8, dt=0.002,
                      mapping=MAP_UNIT_RANGE)
    assert np.allclose(out, ctrl)
    assert info["common_scale"] == pytest.approx(1.0)
    # Repeated outward calls at the boundary: scale 0, no drift outside envelope.
    for _ in range(5):
        out, info = _call(ctrl, anchor, N_UNIT, force=1.0, dt=0.002,
                          mapping=MAP_UNIT_RANGE)
        assert info["common_scale"] == pytest.approx(0.0)
        assert np.allclose(out, ctrl)
        # residual must stay within tolerance, not creep outward.
        assert out[0] <= 0.6 + 0.005 + 1e-12
        ctrl = out.copy()
    # A true meaningful violation (1e-9 rad) is rejected.
    bad = np.zeros(NU)
    bad[0] = 0.605000001
    with pytest.raises(ValueError, match="max_cycle_delta_rad"):
        _call(bad, anchor, N_UNIT, force=0.8, dt=0.002, mapping=MAP_UNIT_RANGE)


def test_scattered_mapping_and_nonindex_preservation():
    ctrl = np.arange(NU, dtype=float) * 0.01
    keep = ctrl.copy()
    anchor = ctrl.copy()
    scattered = NamedJointMap(MAP.joint_names, MAP.qposadr, MAP.dofadr,
                              (5, 2, 9, 13), MAP.ranges)
    out1, info1 = _call(ctrl, anchor, N20, force=1.0, dt=0.002,
                        mapping=scattered)
    # reversed order with reversed normal row gives the same physical result
    rev = NamedJointMap(MAP.joint_names[::-1], MAP.qposadr[::-1],
                        MAP.dofadr[::-1], (13, 9, 2, 5), MAP.ranges[::-1])
    out2, info2 = _call(ctrl, anchor, N20[::-1], force=1.0, dt=0.002,
                        mapping=rev)
    assert info2["applied_dq"] == pytest.approx(info1["applied_dq"][::-1])
    for name in MAP.joint_names:
        a = scattered.act_index[scattered.joint_names.index(name)]
        b = rev.act_index[rev.joint_names.index(name)]
        assert out1[a] == pytest.approx(out2[b])
    # non-index channels are bitwise unchanged
    for i in range(NU):
        if i not in scattered.act_index:
            assert out1[i] == keep[i]


@pytest.mark.parametrize("kwargs,match", [
    ({"current": np.full(NU, np.nan)}, "current_ctrl"),
    ({"anchor": np.full(NU, np.nan)}, "cycle_anchor"),
    ({"n": np.zeros(4)}, "degenerate"),
    ({"force": -0.1}, "measured_force_n"),
    ({"force": float("nan")}, "measured_force_n"),
    ({"force": True}, "measured_force_n"),
    ({"dt": 0.0}, "dt_s"),
    ({"dt": 0.011}, "dt_s"),
    ({"dt": float("inf")}, "dt_s"),
    ({"dt": True}, "dt_s"),
    ({"n": np.array([1.0, 0.0])}, "normal_row"),
    ({"n": np.full(4, np.nan)}, "normal_row"),
])
def test_invalid_inputs_rejected(kwargs, match):
    ctrl = np.zeros(NU)
    kw = {"current": ctrl, "anchor": ctrl, "n": N_UNIT, "force": 0.8,
          "dt": 0.002, "mapping": MAP_UNIT_RANGE}
    kw.update(kwargs)
    with pytest.raises(ValueError, match=match):
        _call(**kw)


def test_config_and_current_envelope_validation():
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            NormalForceConfig(target_force_n=bad)
    ctrl = np.zeros(NU)
    ctrl[0] = 3.0  # outside declared [-0.314, 2.23]
    with pytest.raises(ValueError, match="declared range"):
        _call(ctrl, np.zeros(NU), N_UNIT, force=0.8, dt=0.002)
    ctrl = np.zeros(NU)
    anchor = np.zeros(NU)
    anchor[0] = -0.006
    ctrl[0] = 0.0  # |0 - (-0.006)| = 0.006 > 0.005
    with pytest.raises(ValueError, match="max_cycle_delta_rad"):
        _call(ctrl, anchor, N_UNIT, force=0.8, dt=0.002)


def test_zero_and_limited_correction_info_semantics():
    out, info = _call(np.zeros(NU), np.zeros(NU), N_UNIT, force=0.8, dt=0.002)
    assert info["teacher_only"] is True
    assert info["bound_limited"] == []
    assert info["common_scale"] == pytest.approx(1.0)
    assert info["predicted_normal_dz_m"] == pytest.approx(0.0)
    # limited speed: error large enough to hit max_normal_speed_m_s
    out, info = _call(np.zeros(NU), np.zeros(NU), N_UNIT, force=2.0, dt=0.002)
    assert info["requested_normal_speed_m_s"] == pytest.approx(0.002)
    assert info["requested_dz_m"] == pytest.approx(0.002 * 0.002)
