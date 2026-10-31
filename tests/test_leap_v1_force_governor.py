"""Pure-math tests for LEAP v1 force-governed tangential composer."""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad.leap_v1.force_governor import (  # noqa: E402
    ForceGovernorConfig,
    compose_governed_substep,
)
from embodied_trackpad.leap_v1.visual_physics import NU, NamedJointMap  # noqa: E402

NU = int(NU)

MAP = NamedJointMap(
    ("if_mcp", "if_rot", "if_pip", "if_dip"),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    (0, 1, 2, 3),
    ((-0.314, 2.23), (-1.047, 1.047), (-0.506, 1.885), (-0.366, 2.042)),
)


def _gov(current, normal, nominal, anchor, force=0.8, cfg=None, mapping=MAP):
    cfg = cfg or ForceGovernorConfig()
    return compose_governed_substep(
        np.asarray(current, dtype=float),
        np.asarray(normal, dtype=float),
        np.asarray(nominal, dtype=float),
        np.asarray(anchor, dtype=float),
        force, mapping, cfg,
    )


def test_high_force_inhibits_tangent_preserving_normal():
    current = np.zeros(NU)
    normal = current.copy()
    normal[0] += 0.0001  # accepted normal increment
    nominal = np.zeros(NU)
    nominal[0] = 0.01  # mcp frame delta
    cfg = ForceGovernorConfig(tangential_stop_force_n=1.1, target_force_n=0.8)
    out, info = _gov(current, normal, nominal, current, force=1.2, cfg=cfg)
    assert info["gate"] == 0.0
    assert np.allclose(out, normal)
    assert info["applied_tangent"] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_low_force_divides_nominal_by_substeps():
    current = np.zeros(NU)
    normal = current.copy()
    nominal = np.zeros(NU)
    nominal[0] = 0.025  # 25 * 0.001
    cfg = ForceGovernorConfig(substeps_per_frame=25)
    out, info = _gov(current, normal, nominal, current, force=0.5)
    assert info["gate"] == pytest.approx(1.0)
    assert info["requested_tangent"][0] == pytest.approx(0.001)
    # Requested tangent exceeds per-substep cap, so it is scaled to 0.0002.
    assert info["common_scale"] == pytest.approx(0.2)
    assert out[0] == pytest.approx(0.0002)
    assert info["applied_tangent"][0] == pytest.approx(0.0002)


def test_smooth_accumulation_over_substeps_reaches_endpoint():
    current = np.zeros(NU)
    nominal = np.zeros(NU)
    nominal[0] = 0.005  # exactly 25 * cap, so cap-limited accumulation reaches it
    cfg = ForceGovernorConfig(substeps_per_frame=25)
    total = 0.0
    for _ in range(25):
        normal = current.copy()  # no normal correction in this pure-tangent test
        out, info = _gov(current, normal, nominal, np.zeros(NU), force=0.5, cfg=cfg)
        total += info["applied_tangent"][0]
        current = out
    assert total == pytest.approx(0.005)


def test_shared_scale_preserves_direction_and_normal_priority_at_cap():
    current = np.zeros(NU)
    normal = current.copy()
    normal[0] = 0.00015  # uses 0.15 mrad of the 0.2 mrad cap
    nominal = np.zeros(NU)
    nominal[0] = 0.005  # would request 0.005/25=0.0002 mrad on mcp alone -> exceed cap
    cfg = ForceGovernorConfig(substeps_per_frame=25,
                              per_substep_total_joint_delta_cap_rad=0.0002)
    out, info = _gov(current, normal, nominal, np.zeros(NU), force=0.5, cfg=cfg)
    # normal unchanged, tangent scaled to keep total within cap.
    assert out[0] - current[0] <= 0.0002 + 1e-15
    assert info["common_scale"] < 1.0
    assert info["common_scale"] > 0.0
    assert info["applied_tangent"][0] == pytest.approx(0.0002 - 0.00015)
    assert info["normal_delta"][0] == pytest.approx(0.00015)
    assert info["bound_limited"]


def test_cycle_envelope_exhausted_alpha_zero_keeps_normal():
    current = np.zeros(NU)
    current[0] = 0.005  # exactly at envelope hi boundary
    normal = current.copy()
    nominal = np.zeros(NU)
    nominal[0] = 0.005  # any positive tangent would exceed envelope
    cfg = ForceGovernorConfig(substeps_per_frame=25, max_cycle_delta_rad=0.005)
    out, info = _gov(current, normal, nominal, np.zeros(NU), force=0.5, cfg=cfg)
    assert info["common_scale"] == 0.0
    assert np.allclose(out, normal)
    assert "cycle_envelope" in info["bound_limited"][0]


def test_non_index_channels_unchanged():
    current = np.arange(NU, dtype=float) * 0.01
    normal = current.copy()
    normal[0] += 0.0001
    nominal = np.zeros(NU)
    nominal[0] = 0.025
    out, _ = _gov(current, normal, nominal, current, force=0.5)
    for i in range(NU):
        if i not in MAP.act_index:
            assert out[i] == current[i]


def test_invalid_inputs_rejected():
    current = np.zeros(NU)
    normal = current.copy()
    normal[0] += 0.0001
    nominal = np.zeros(NU)
    with pytest.raises(ValueError):
        _gov(np.full(NU, np.nan), normal, nominal, current, force=0.5)
    with pytest.raises(ValueError):
        _gov(current, np.full(NU, np.nan), nominal, current, force=0.5)
    with pytest.raises(ValueError):
        _gov(current, normal, nominal, current, force=-0.1)
    with pytest.raises(ValueError):
        _gov(current, normal, nominal, current, force=float("nan"))
    with pytest.raises(ValueError):
        _gov(current, normal, nominal, current, force=True)
    with pytest.raises(ValueError):
        _gov(current, normal, nominal, current, force=None)


def test_normal_ctrl_bounds_validated_with_tolerance():
    current = np.zeros(NU)
    normal = current.copy()
    normal[0] = 0.0002 + 1e-13  # within 1e-12 tolerance of cap
    nominal = np.zeros(NU)
    _gov(current, normal, nominal, current, force=0.5)
    normal[0] = 0.0002 + 1e-9  # meaningful violation
    with pytest.raises(ValueError):
        _gov(current, normal, nominal, current, force=0.5)


def test_config_validation():
    for bad in (0, -1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            ForceGovernorConfig(target_force_n=bad)
    with pytest.raises(ValueError):
        ForceGovernorConfig(tangential_stop_force_n=0.7)  # <= target
    # substeps_per_frame must be a strict positive integer.
    with pytest.raises(ValueError):
        ForceGovernorConfig(substeps_per_frame=1.5)
    with pytest.raises(ValueError):
        ForceGovernorConfig(substeps_per_frame=0)
    with pytest.raises(ValueError):
        ForceGovernorConfig(substeps_per_frame=True)


def test_mapping_validation():
    ctrl = np.zeros(NU)
    bad = NamedJointMap(MAP.joint_names[:3], (0, 1, 2), (0, 1, 2), (0, 1, 2),
                        MAP.ranges[:3])
    with pytest.raises(ValueError):
        _gov(ctrl, ctrl, np.zeros(NU), ctrl, force=0.5, mapping=bad)
    dup = NamedJointMap(MAP.joint_names, MAP.qposadr, MAP.dofadr,
                        (0, 1, 2, 2), MAP.ranges)
    with pytest.raises(ValueError):
        _gov(ctrl, ctrl, np.zeros(NU), ctrl, force=0.5, mapping=dup)
