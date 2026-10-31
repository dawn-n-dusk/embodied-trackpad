"""Tests for the v1 visual physics core.

No physics stepping here: the step function is exercised with small fake
model/data records and scripted stub callables; loaded-model checks are
static (compile + field reads only) and skip when the pinned v1 assets are
not downloaded. These tests claim NO closed-loop performance.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad import leap_v1  # noqa: E402
from embodied_trackpad.leap_v1 import pad_env as _pad  # noqa: E402
from embodied_trackpad.leap_v1 import visual_physics as vp  # noqa: E402
from embodied_trackpad.leap_v1.model import MOTOR_JOINTS, open_pose_qpos  # noqa: E402
from embodied_trackpad.leap_v1.visual_servo import CONTACT_PROFILE  # noqa: E402

ASSETS_PRESENT = leap_v1.DOWNLOAD_ROOT.is_dir()
requires_assets = pytest.mark.skipif(
    not ASSETS_PRESENT, reason="pinned v1 assets not downloaded"
)
P1 = CONTACT_PROFILE["scene_contact"]
NWARN = 8


# --- fake records/stubs for the step function --------------------------------

def fake_model():
    return SimpleNamespace(
        nu=vp.NU,
        actuator_ctrlrange=np.tile(np.array([[-2.5, 2.5]]), (vp.NU, 1)),
        actuator=lambda i: SimpleNamespace(name=f"motor_{i}"),
        njnt=vp.NU,
        jnt_range=np.tile(np.array([[-1.0, 1.0]]), (vp.NU, 1)),
        jnt_qposadr=np.arange(vp.NU),
    )


def fake_data():
    return SimpleNamespace(
        ctrl=np.zeros(vp.NU),
        qpos=np.zeros(vp.NU),
        qvel=np.zeros(vp.NU),
        time=0.0,
        warning=[SimpleNamespace(number=0) for _ in range(NWARN)],
    )


def fake_env():
    scene = SimpleNamespace(model=fake_model(), pad_geom_id=0)
    return vp.VisualPhysicsEnv(
        scene=scene,
        index_map=vp.NamedJointMap(vp.INDEX_JOINTS, (0, 1, 2, 3), (0, 1, 2, 3),
                                   (0, 1, 2, 3), ((-1.0, 1.0),) * 4),
        data=fake_data(),
    )


def raw_contact(kind="tip_pad_top", force=0.5, dist=0.0):
    return _pad.RawContact(
        geom1=1, geom2=0, body1="if_ds", body2="world",
        pos_world=[0.0, 0.0, 0.1], frame=[1.0] + [0.0] * 8,
        force6=[force] + [0.0] * 5, normal_force_n=force, dist_m=dist,
        pad_local=[0.01, 0.02, 0.0], top_face_cos=1.0, kind=kind,
    )


def run_steps(env, contacts_per_step, n_steps, **kw):
    """Drive physics_step with scripted contacts; returns (records, error, calls)."""
    calls = []

    def step_fn(m, d):
        calls.append(len(calls))
        d.time = 0.002 * (len(calls))

    def contact_reader(scene, d):
        return contacts_per_step[len(calls) - 1]

    records = []
    err = None
    try:
        for k in range(n_steps):
            records.append(
                vp.physics_step(
                    env, np.zeros(vp.NU), step_index=k, frame_id=f"frame_{k}",
                    step_fn=step_fn, forward_fn=lambda m, d: None,
                    contact_reader=contact_reader,
                    aggregator=lambda scene, raw: ({}, [c for c in raw
                                                        if c.kind != "tip_pad_top"]),
                    **kw,
                )
            )
    except vp.PhysicsStepError as exc:
        err = exc
    return records, err, calls


def test_step_returns_complete_record_with_empty_contacts():
    env = fake_env()
    records, err, calls = run_steps(env, [[]], 1)
    assert err is None and len(records) == 1
    rec = records[0]
    assert rec.ok and rec.contacts == [] and rec.aggregate == {}
    assert rec.step_index == 0 and rec.frame_id == "frame_0"
    assert rec.sim_time_s == pytest.approx(0.002)
    assert len(rec.ctrl) == vp.NU and len(rec.qpos) == vp.NU and len(rec.qvel) == vp.NU
    assert rec.warning_delta == [0] * NWARN
    assert all(rec.checks.values())
    rec2 = rec.to_dict()
    assert rec2["ok"] and rec2["contacts"] == []


def test_failing_intermediate_substep_cannot_be_hidden():
    env = fake_env()
    bad = raw_contact(force=6.0)  # exceeds the 5 N all-pad gate
    records, err, calls = run_steps(env, [[], [bad], []], 3)
    assert err is not None, "middle substep failure must surface"
    assert err.record.step_index == 1  # FIRST failure, not a later sample
    assert err.record.contacts[0]["normal_force_n"] == pytest.approx(6.0)
    assert err.record.checks["pad_force<=5N"] is False
    assert any("pad_force_all" in r for r in err.record.failure_reasons)
    assert len(records) == 1 and records[0].ok  # only the good first substep returned
    assert len(calls) == 2  # the third substep was NEVER executed


def test_invalid_ctrl_fails_before_any_stepping():
    env = fake_env()
    called = []

    def step_fn(m, d):
        called.append(1)

    with pytest.raises(vp.PhysicsStepError) as ei:
        vp.physics_step(env, np.full(vp.NU, np.nan), step_index=0,
                        step_fn=step_fn, forward_fn=lambda m, d: None,
                        contact_reader=lambda s, d: [], aggregator=lambda s, r: ({}, []))
    assert ei.value.record.checks == {"ctrl_valid": False}
    assert "ctrl_invalid" in ei.value.record.failure_reasons[0]
    assert called == []
    with pytest.raises(vp.PhysicsStepError):
        vp.physics_step(env, np.zeros(vp.NU - 1), step_index=0,
                        step_fn=step_fn, forward_fn=lambda m, d: None,
                        contact_reader=lambda s, d: [], aggregator=lambda s, r: ({}, []))
    assert called == []


def test_malformed_ctrl_shape_retains_serialized_evidence():
    env = fake_env()
    called = []
    with pytest.raises(vp.PhysicsStepError) as ei:
        vp.physics_step(env, np.zeros((2, 8)), step_index=0,
                        step_fn=lambda m, d: called.append(1),
                        forward_fn=lambda m, d: None,
                        contact_reader=lambda s, d: [], aggregator=lambda s, r: ({}, []))
    rec = ei.value.record  # PhysicsStepError evidence, NOT a raw TypeError
    assert rec.ctrl_shape == [2, 8]
    assert len(rec.ctrl) == 16  # requested values retained (flattened)
    assert rec.checks == {"ctrl_valid": False}
    assert called == []
    json.dumps(rec.to_dict(), allow_nan=False)


def test_non_finite_prestate_fails_before_stepping():
    env = fake_env()
    env.data.qpos[5] = np.nan  # invalid prestate must not be stepped over
    called = []
    with pytest.raises(vp.PhysicsStepError) as ei:
        vp.physics_step(env, np.zeros(vp.NU), step_index=0,
                        step_fn=lambda m, d: called.append(1),
                        forward_fn=lambda m, d: None,
                        contact_reader=lambda s, d: [], aggregator=lambda s, r: ({}, []))
    assert ei.value.record.checks == {"pre_step_state_finite": False}
    assert called == []  # solver never ran on the invalid prestate
    assert ei.value.record.qpos[5] == "nan"  # documented string representation
    json.dumps(ei.value.record.to_dict(), allow_nan=False)


def test_non_finite_sim_time_fails_post_step_with_safe_evidence():
    env = fake_env()

    def step_fn(m, d):
        d.time = float("nan")  # solver-produced non-finite time must surface

    with pytest.raises(vp.PhysicsStepError) as ei:
        vp.physics_step(env, np.zeros(vp.NU), step_index=0, step_fn=step_fn,
                        forward_fn=lambda m, d: None,
                        contact_reader=lambda s, d: [], aggregator=lambda s, r: ({}, []))
    rec = ei.value.record
    assert rec.checks["finite_state"] is False
    assert "non_finite_state" in rec.failure_reasons
    assert rec.sim_time_s == "nan"  # documented string representation
    json.dumps(rec.to_dict(), allow_nan=False)


def test_nan_raw_contact_rejected_before_aggregation():
    env = fake_env()
    spy = []
    nan_contact = raw_contact(force=float("nan"), dist=float("nan"))

    def spy_aggregator(scene, raw):
        spy.append(1)
        return ({}, [])

    with pytest.raises(vp.PhysicsStepError) as ei:
        vp.physics_step(env, np.zeros(vp.NU), step_index=0,
                        step_fn=lambda m, d: None, forward_fn=lambda m, d: None,
                        contact_reader=lambda s, d: [nan_contact],
                        aggregator=spy_aggregator)
    rec = ei.value.record
    assert spy == []  # aggregator NEVER called on invalid raw contacts
    assert rec.aggregate == {}
    assert rec.checks["raw_contacts_valid"] is False
    assert "pad_force<=5N" not in rec.checks  # no summaries from invalid raw
    assert rec.contacts[0]["normal_force_n"] == "nan"  # evidence preserved
    json.dumps(rec.to_dict(), allow_nan=False)


def test_unintended_contact_penetration_excursion_warnings_each_fail():
    env = fake_env()
    _, err, _ = run_steps(env, [[raw_contact(kind="hand_self")]], 1)
    assert err is not None and err.record.checks["no_unintended_contact"] is False

    env = fake_env()
    _, err, _ = run_steps(env, [[raw_contact(dist=-0.002)]], 1)
    assert err is not None and err.record.checks["penetration<=tol"] is False

    env = fake_env()
    env.data.qpos[3] = 1.05  # beyond the fake +/-1.0 range

    def step_fn(m, d):
        d.warning[0].number += 1  # a new solver warning on the same substep

    with pytest.raises(vp.PhysicsStepError) as ei:
        vp.physics_step(env, np.zeros(vp.NU), step_index=0, step_fn=step_fn,
                        forward_fn=lambda m, d: None,
                        contact_reader=lambda s, d: [], aggregator=lambda s, r: ({}, []))
    assert ei.value.record.checks["joint_excursion<=tol"] is False
    assert ei.value.record.checks["no_new_warnings"] is False
    assert len(ei.value.record.failure_reasons) == 2


# --- pure planner tests --------------------------------------------------------

JAC = np.array([
    [0.05, 0.02, 0.03, 0.01],
    [0.00, 0.04, 0.01, 0.02],
    [0.04, 0.00, 0.05, 0.03],
])
MAP = vp.NamedJointMap(
    vp.INDEX_JOINTS, (0, 1, 2, 3), (0, 1, 2, 3), (0, 1, 2, 3),
    ((-0.314, 2.23), (-1.047, 1.047), (-0.506, 1.885), (-0.366, 2.042)),
)


def test_pixel_error_direction_determines_correction_direction():
    ctrl0 = np.full(vp.NU, 0.3)
    out_pos, info_pos = vp.plan_index_correction((40.0, -20.0), 0.0, JAC, ctrl0, MAP)
    out_neg, info_neg = vp.plan_index_correction((-40.0, 20.0), 0.0, JAC, ctrl0, MAP)
    assert info_pos["task_m"][0] == pytest.approx(40.0 / 4000.0)
    assert info_pos["task_m"][1] == pytest.approx(-20.0 / 4000.0)
    for a, b in zip(info_pos["dq_rad"], info_neg["dq_rad"]):
        assert a == pytest.approx(-b)  # opposite pixel error -> opposite joints
    for a, b, c in zip(out_pos, out_neg, ctrl0):
        assert (a - c) == pytest.approx(-(b - c))


def test_mapping_scatters_by_channel_and_reorders():
    ctrl0 = np.zeros(vp.NU)
    scattered = vp.NamedJointMap(MAP.joint_names, MAP.qposadr, MAP.dofadr,
                                 (5, 2, 9, 13), MAP.ranges)
    out, info = vp.plan_index_correction((40.0, 40.0), 0.001, JAC, ctrl0, scattered)
    changed = {i for i in range(vp.NU) if out[i] != ctrl0[i]}
    assert changed <= {5, 2, 9, 13}
    for k, act in enumerate(scattered.act_index):
        assert out[act] == pytest.approx(info["dq_rad"][k])
    # Column-order contract: jac column k belongs to joint_names[k]. Reversing
    # the mapping TOGETHER with the Jacobian columns must give the identical
    # named-joint physical correction (dq permutes with the columns).
    rev = vp.NamedJointMap(MAP.joint_names[::-1], MAP.qposadr[::-1],
                           MAP.dofadr[::-1], (13, 9, 2, 5), MAP.ranges[::-1])
    out2, info2 = vp.plan_index_correction((40.0, 40.0), 0.001, JAC[:, ::-1], ctrl0, rev)
    assert info2["dq_rad"] == pytest.approx(info["dq_rad"][::-1])
    for name in MAP.joint_names:
        act_a = scattered.act_index[scattered.joint_names.index(name)]
        act_b = rev.act_index[rev.joint_names.index(name)]
        assert out[act_a] == pytest.approx(out2[act_b])  # same joint, same correction


def test_commands_bounded_and_ranges_respected():
    ctrl0 = np.zeros(vp.NU)
    out, info = vp.plan_index_correction((1e6, -1e6), 1.0, JAC * 1000.0, ctrl0, MAP)
    assert abs(info["task_m"][0]) == pytest.approx(0.010)  # planar clip
    assert info["task_m"][2] == pytest.approx(0.005)  # z-hold clip
    assert all(abs(dq) <= 0.10 + 1e-12 for dq in info["dq_rad"])  # joint delta cap
    near_hi = np.array([MAP.ranges[k][0] if i != k else MAP.ranges[k][1] - 0.02
                        for k in range(4) for i in [k]][:4] + [0.0] * 12)
    near_hi = np.zeros(vp.NU)
    for k, (lo, hi) in enumerate(MAP.ranges):
        near_hi[k] = hi - 0.02
    out2, _ = vp.plan_index_correction((1e6, 1e6), 1.0, JAC * 1000.0, near_hi, MAP)
    for k, (lo, hi) in enumerate(MAP.ranges):
        assert lo <= out2[k] <= hi
    assert np.all(np.isfinite(out))


def test_planner_rejects_non_finite_and_bad_shapes():
    ctrl0 = np.zeros(vp.NU)
    with pytest.raises(ValueError):
        vp.plan_index_correction((float("nan"), 0.0), 0.0, JAC, ctrl0, MAP)
    with pytest.raises(ValueError):
        vp.plan_index_correction((0.0, 0.0), 0.0, np.full((3, 4), np.nan), ctrl0, MAP)
    bad_ctrl = ctrl0.copy()
    bad_ctrl[7] = np.inf
    with pytest.raises(ValueError):
        vp.plan_index_correction((1.0, 1.0), 0.0, JAC, bad_ctrl, MAP)
    with pytest.raises(ValueError):
        vp.damped_least_squares(np.zeros((4, 4)), np.zeros(3), 0.05)


def test_invalid_planner_config_rejected_before_acting():
    for kw in ({"px_per_m": 0.0}, {"px_per_m": float("nan")},
               {"damping": -0.1}, {"damping": True},
               {"max_planar_step_m": float("inf")}, {"max_joint_delta_rad": 0.0},
               {"max_z_step_m": "fast"}):
        with pytest.raises(ValueError):
            vp.PlannerConfig(**kw)
    default = vp.PlannerConfig()
    assert default.px_per_m == 4000.0  # fixed device gain preserved
    out, info = vp.plan_index_correction((40.0, 0.0), 0.0, JAC,
                                         np.zeros(vp.NU), MAP, default)
    assert np.all(np.isfinite(out))
    assert all(abs(dq) <= default.max_joint_delta_rad for dq in info["dq_rad"])


# --- static loaded-model checks (skip without pinned assets) -------------------

@requires_assets
def test_loaded_scene_carries_p1_profile_and_preserved_hand():
    env = vp.make_env()
    observed = vp.verify_scene_profile(env)
    assert observed["timestep_s"] == pytest.approx(P1["timestep_s"])
    assert observed["pad_priority"] == 1
    assert observed["pad_solref"] == pytest.approx(P1["solref"])
    assert observed["pad_solimp"] == pytest.approx(P1["solimp"])
    assert observed["pad_friction"] == pytest.approx(P1["friction"])
    assert observed["nu"] == 16 and observed["motor_order_preserved"]
    m = env.model
    assert m.neq == 0  # no coupling reintroduced
    expected = [MOTOR_JOINTS.index(j) for j in vp.INDEX_JOINTS]
    assert list(env.index_map.act_index) == expected
    for k, jname in enumerate(vp.INDEX_JOINTS):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert env.index_map.qposadr[k] == int(m.jnt_qposadr[jid])
        assert env.index_map.dofadr[k] == int(m.jnt_dofadr[jid])


@requires_assets
def test_verify_detects_profile_tamper():
    env = vp.make_env()
    env.model.geom_friction[env.scene.pad_geom_id] = [9.0, 9.0, 9.0]
    with pytest.raises(ValueError, match="pad_friction"):
        vp.verify_scene_profile(env)
    env = vp.make_env()
    env.model.opt.timestep = 0.004
    with pytest.raises(ValueError, match="timestep_s"):
        vp.verify_scene_profile(env)


@requires_assets
def test_named_mapping_follows_joint_order_not_positions():
    env = vp.make_env()
    rev = vp.build_named_mapping(env.model, list(reversed(vp.INDEX_JOINTS)))
    assert rev.joint_names == tuple(reversed(vp.INDEX_JOINTS))
    assert list(rev.act_index) == [MOTOR_JOINTS.index(j) for j in reversed(vp.INDEX_JOINTS)]
    with pytest.raises(ValueError):
        vp.build_named_mapping(env.model, ["no_such_joint"])


@requires_assets
def test_reset_open_is_contact_free_open_pose():
    env = vp.make_env()
    vp.reset_open(env)
    assert np.allclose(env.data.qpos, open_pose_qpos(env.model))
    assert env.warn_counts is not None
    # STATIC only: injected no-op step (no solver stepping in this task);
    # real forward/contact reads on the reset state.
    rec = vp.physics_step(env, vp.open_release_ctrl(env.model), step_index=0,
                          step_fn=lambda m, d: None, forward_fn=mujoco.mj_forward)
    assert rec.ok and rec.contacts == []
