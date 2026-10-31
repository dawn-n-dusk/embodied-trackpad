"""Pure-math tests for the normal-priority planner candidate.

Fixtures are the ACTUAL recorded cycle-1/cycle-20 Jacobians and tasks from the
private seed2 probe (copied numerically; no private paths). These tests prove
the mathematical constraint only; they assert NOTHING about contact success.
No solver stepping, no assets, no app.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad.leap_v1.normal_priority import (  # noqa: E402
    ContactLossGuard,
    plan_normal_priority,
)
from embodied_trackpad.leap_v1.visual_physics import (  # noqa: E402
    NU,
    NamedJointMap,
    PlannerConfig,
)

# Recorded cycle 1 (seed2 probe): J at the open/approach pose.
J1 = np.array([
    [-0.08862352161866313, -0.0003481623685944457, -0.05873215186048128, -0.031233653935638372],
    [-5.317425626543551e-07, -0.07714197781506119, -1.2455647247196076e-06, -8.539048956151235e-07],
    [-0.07960853380227861, 0.00022924448356084065, -0.035713879424659076, -0.012324937342469588],
])
# Recorded cycle 20: the step where the old DLS was requested downward z but
# predicted upward z (+0.000120532 m for task [.0005, -.00001698, -.0002]).
J20 = np.array([
    [-0.08828257644093039, -0.0003444501931893114, -0.058067735260634024, -0.030844393688657726],
    [-5.296969103407153e-07, -0.07803766899473195, -2.0048989444168305e-06, -1.3575474996445507e-06],
    [-0.08065036440031177, 0.00023046520152102817, -0.03697772552088247, -0.013269084200966147],
])
ERR20 = (18.77334803293502, -0.0679260649762341)
Z20 = -0.0003409451777388328  # clipped to -0.0002 by the candidate cfg

MAP = NamedJointMap(
    ("if_mcp", "if_rot", "if_pip", "if_dip"), (0, 1, 2, 3), (0, 1, 2, 3),
    (0, 1, 2, 3), ((-0.314, 2.23), (-1.047, 1.047), (-0.506, 1.885), (-0.366, 2.042)),
)
CFG = PlannerConfig(max_planar_step_m=0.0005, max_z_step_m=0.0002,
                    max_joint_delta_rad=0.005, damping=0.05)


def test_cycle20_exact_normal_constraint_negative_never_positive():
    out, info = plan_normal_priority(ERR20, Z20, J20, np.zeros(NU), MAP, CFG)
    assert info["common_scale"] == pytest.approx(1.0)
    assert info["requested_task"] == pytest.approx([0.0005, -1.6981516244058524e-05, -0.0002])
    # exact first-order normal constraint: predicted z == requested z
    assert info["predicted_task"][2] == pytest.approx(-0.0002, abs=1e-12)
    assert info["predicted_task"][2] < 0.0  # negative request never predicts +
    # contrast with the recorded old-DLS failure (+0.000120532 m predicted z)
    n = J20[2, :]
    assert abs(n @ np.array(info["dq"]) - (-0.0002)) < 1e-12
    assert info["teacher_only"] and info["mapping"] == list(MAP.joint_names)


def test_zero_z_request_predicts_zero_z():
    _, info = plan_normal_priority((20.0, 0.0), 0.0, J1, np.zeros(NU), MAP, CFG)
    assert info["requested_task"][2] == 0.0
    assert info["predicted_task"][2] == pytest.approx(0.0, abs=1e-10)
    # old scalar-damped DLS predicted +0.000196281 m z for the same task


def test_common_scaling_preserves_normal_not_per_component_clip():
    _, probe = plan_normal_priority(ERR20, Z20, J20, np.zeros(NU), MAP, CFG)
    dq_u = np.array(probe["dq_unscaled"])
    k = int(np.argmax(np.abs(dq_u)))  # bind the fastest-moving channel
    lo, hi = MAP.ranges[k]
    ctrl = np.zeros(NU)
    ctrl[k] = (hi - 0.0001) if dq_u[k] > 0 else (lo + 0.0001)  # 0.1 mrad headroom
    out, info = plan_normal_priority(ERR20, Z20, J20, ctrl, MAP, CFG)
    s = info["common_scale"]
    assert 0.0 < s < 1.0  # range-limited, one common factor
    assert s == pytest.approx(0.0001 / abs(dq_u[k]))  # binding channel
    # scaled normal task equals scale * z_task exactly
    assert info["predicted_task"][2] == pytest.approx(s * -0.0002, abs=1e-12)
    assert out[k] == pytest.approx(hi if dq_u[k] > 0 else lo)  # stops AT the limit
    # per-component clipping would break the normal constraint: demonstrate
    per_component = np.array([
        np.clip(dq_u[j], MAP.ranges[j][0] - ctrl[j], MAP.ranges[j][1] - ctrl[j])
        for j in range(4)
    ])
    viol = abs(J20[2, :] @ per_component - s * -0.0002)
    assert viol > 1e-6  # clipping breaks it; common scaling does not


def test_at_range_limit_scale_zero_reported_not_extrapolated():
    _, probe = plan_normal_priority(ERR20, Z20, J20, np.zeros(NU), MAP, CFG)
    dq_u = np.array(probe["dq_unscaled"])
    k = int(np.argmax(np.abs(dq_u)))
    lo, hi = MAP.ranges[k]
    ctrl = np.zeros(NU)
    ctrl[k] = hi if dq_u[k] > 0 else lo  # exactly at the blocking limit
    out, info = plan_normal_priority(ERR20, Z20, J20, ctrl, MAP, CFG)
    assert info["common_scale"] == 0.0
    assert np.allclose(out, ctrl)
    assert info["predicted_task"] == pytest.approx([0.0, 0.0, 0.0])


def test_joint_delta_cap_scales_whole_vector():
    huge = PlannerConfig(max_planar_step_m=0.05, max_z_step_m=0.05,
                         max_joint_delta_rad=0.005, damping=0.05)
    out, info = plan_normal_priority((4000.0, 4000.0), -0.05, J20, np.zeros(NU), MAP, huge)
    s = info["common_scale"]
    assert 0.0 < s < 1.0
    assert max(abs(v) for v in info["dq"]) == pytest.approx(0.005)
    assert info["predicted_task"][2] == pytest.approx(s * info["requested_task"][2], abs=1e-12)


def test_reordered_mapping_with_reordered_columns_same_correction():
    ctrl = np.full(NU, 0.1)
    scattered = NamedJointMap(MAP.joint_names, MAP.qposadr, MAP.dofadr,
                              (5, 2, 9, 13), MAP.ranges)
    out1, info1 = plan_normal_priority(ERR20, Z20, J20, ctrl, scattered, CFG)
    rev = NamedJointMap(MAP.joint_names[::-1], MAP.qposadr[::-1], MAP.dofadr[::-1],
                        (13, 9, 2, 5), MAP.ranges[::-1])
    out2, info2 = plan_normal_priority(ERR20, Z20, J20[:, ::-1], ctrl, rev, CFG)
    assert info2["dq"] == pytest.approx(info1["dq"][::-1])
    for name in MAP.joint_names:
        a = scattered.act_index[scattered.joint_names.index(name)]
        b = rev.act_index[rev.joint_names.index(name)]
        assert out1[a] == pytest.approx(out2[b])


def test_non_index_channels_and_ordering_preserved():
    ctrl = np.arange(NU, dtype=float) * 0.01  # distinct sentinels
    keep = ctrl.copy()
    out, _ = plan_normal_priority(ERR20, Z20, J20, ctrl, MAP, CFG)
    for i in range(NU):
        if i not in MAP.act_index:
            assert out[i] == keep[i]


def test_guards_reject_degenerate_nonfinite_bad_shapes():
    ctrl = np.zeros(NU)
    with pytest.raises(ValueError, match="degenerate"):
        plan_normal_priority(ERR20, Z20, np.zeros((3, 4)), ctrl, MAP, CFG)
    with pytest.raises(ValueError, match="degenerate"):
        plan_normal_priority(ERR20, Z20, J20 * 1e-15, ctrl, MAP, CFG)
    with pytest.raises(ValueError):
        plan_normal_priority((float("nan"), 0.0), 0.0, J20, ctrl, MAP, CFG)
    with pytest.raises(ValueError):
        plan_normal_priority(ERR20, float("inf"), J20, ctrl, MAP, CFG)
    with pytest.raises(ValueError):
        plan_normal_priority(ERR20, Z20, np.full((3, 4), np.nan), ctrl, MAP, CFG)
    with pytest.raises(ValueError):
        plan_normal_priority(ERR20, Z20, np.zeros((4, 4)), ctrl, MAP, CFG)
    bad = ctrl.copy(); bad[3] = np.nan
    with pytest.raises(ValueError):
        plan_normal_priority(ERR20, Z20, J20, bad, MAP, CFG)
    with pytest.raises(ValueError):
        plan_normal_priority(ERR20, Z20, J20, np.zeros(NU - 1), MAP, CFG)
    badmap = NamedJointMap(MAP.joint_names[:3], (0, 1, 2), (0, 1, 2), (0, 1, 2),
                           MAP.ranges[:3])
    with pytest.raises(ValueError):
        plan_normal_priority(ERR20, Z20, J20, ctrl, badmap, CFG)


def test_out_of_range_current_ctrl_rejected_even_with_zero_task():
    # coordinator reproduction: current_ctrl[act0]=2 with range [-1,1] and a
    # zero task was previously accepted unchanged; it must now raise.
    ctrl = np.zeros(NU)
    ctrl[0] = 3.0  # outside if_mcp's declared [-0.314, 2.23]
    with pytest.raises(ValueError, match="outside its declared range"):
        plan_normal_priority((0.0, 0.0), 0.0, J20, ctrl, MAP, CFG)
    ctrl = np.zeros(NU)
    ctrl[3] = MAP.ranges[3][1]  # exactly at the limit: valid boundary control
    out, info = plan_normal_priority((0.0, 0.0), 0.0, J20, ctrl, MAP, CFG)
    assert np.allclose(out, ctrl) and info["common_scale"] == 1.0


def test_mapping_identity_validated():
    ctrl = np.zeros(NU)
    short_q = NamedJointMap(MAP.joint_names, (0, 1, 2), MAP.dofadr,
                            MAP.act_index, MAP.ranges)
    with pytest.raises(ValueError, match="length 4"):
        plan_normal_priority(ERR20, Z20, J20, ctrl, short_q, CFG)
    bool_act = NamedJointMap(MAP.joint_names, MAP.qposadr, MAP.dofadr,
                             (0, 1, 2, True), MAP.ranges)
    with pytest.raises(ValueError, match="invalid actuator channel"):
        plan_normal_priority(ERR20, Z20, J20, ctrl, bool_act, CFG)
    dup_act = NamedJointMap(MAP.joint_names, MAP.qposadr, MAP.dofadr,
                            (0, 1, 2, 2), MAP.ranges)
    with pytest.raises(ValueError, match="duplicated"):
        plan_normal_priority(ERR20, Z20, J20, ctrl, dup_act, CFG)
    dup_name = NamedJointMap(("if_mcp", "if_mcp", "if_pip", "if_dip"),
                             MAP.qposadr, MAP.dofadr, MAP.act_index, MAP.ranges)
    with pytest.raises(ValueError, match="permutation"):
        plan_normal_priority(ERR20, Z20, J20, ctrl, dup_name, CFG)
    alien = NamedJointMap(("mf_mcp", "if_rot", "if_pip", "if_dip"),
                          MAP.qposadr, MAP.dofadr, MAP.act_index, MAP.ranges)
    with pytest.raises(ValueError, match="permutation"):
        plan_normal_priority(ERR20, Z20, J20, ctrl, alien, CFG)


def test_contact_loss_guard():
    g = ContactLossGuard()
    for _ in range(49):  # 0.098 s below threshold: not yet terminated
        assert g.update(0.002, True, 0.01) is False
    assert g.update(0.002, True, 0.01) is True  # 0.100 s reached
    assert g.terminated and g.update(0.002, True, 1.0) is True  # stays stopped
    g = ContactLossGuard()
    for _ in range(40):
        g.update(0.002, False, None)  # missing aggregate accumulates
    g.update(0.002, True, 0.10)  # valid contact resets
    assert g.lost_s == 0.0 and not g.terminated
    g = ContactLossGuard()
    assert g.update(0.002, True, float("nan")) is True  # nonfinite: immediate
    with pytest.raises(ValueError):
        ContactLossGuard().update(-0.1, True, 1.0)


def test_contact_loss_guard_config_validation():
    # coordinator reproduction: max_loss_s=NaN was accepted and never terminated
    with pytest.raises(ValueError):
        ContactLossGuard(max_loss_s=float("nan"))
    for kw in ({"force_min_n": 0.0}, {"force_min_n": -1.0}, {"force_min_n": float("inf")},
               {"max_loss_s": -0.1}, {"lost_s": -0.5}, {"lost_s": float("nan")},
               {"terminated": "yes"}):
        with pytest.raises(ValueError):
            ContactLossGuard(**kw)
    g = ContactLossGuard()  # default .05 N / .1 s boundary unchanged
    assert g.update(0.1, False, None) is True  # exactly max_loss_s terminates
