"""Focused structural tests for the tap/slide 17-DoF scene (ADR 0010).

Runtime tests need mujoco and the pinned LEAP v1 assets in the ignored
download cache; they skip cleanly when either is absent. They verify scene
STRUCTURE only (counts, orders, P1 contact profile, wrist parameters, cameras,
named indexing) — never contact, gestures or training behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from embodied_trackpad.leap_v1.assets import LeapAssetError

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")

from embodied_trackpad.leap_v1.model import MOTOR_JOINTS, TIP_SITES  # noqa: E402
from embodied_trackpad.leap_v1.visual_servo import CONTACT_PROFILE  # noqa: E402
from embodied_trackpad.tap_slide import scene as ts  # noqa: E402


def _load():
    try:
        return ts.load_scene()
    except (LeapAssetError, FileNotFoundError) as exc:  # assets absent
        pytest.skip(f"pinned LEAP v1 assets unavailable: {exc}")


@pytest.fixture(scope="module")
def scene():
    return _load()


def test_model_counts_and_orders(scene):
    m = scene.model
    assert (m.njnt, m.nu, m.nq, m.nv) == (17, 17, 17, 17)
    assert scene.joint_names == ts.JOINT_NAMES17
    assert scene.joint_names[0] == ts.WRIST_JOINT
    assert scene.joint_names[1:] == list(MOTOR_JOINTS)
    assert scene.actuator_names == ts.ACTUATOR_NAMES17
    assert scene.actuator_names[:16] == [f"{j}_act" for j in MOTOR_JOINTS]
    assert scene.actuator_names[16] == ts.WRIST_ACTUATOR


def test_new_17_channel_identities_and_hand_provenance(scene):
    integrity = ts.scene_integrity(scene)
    assert integrity["embodiment_id"] == "leap_v1_full_right_wrist_pitch_menagerie_v1"
    assert integrity["calibration_id"] == "uncalibrated_conservative_wrist_defaults_v1"
    # Old upstream IDs retained as hand-only provenance, never relabelled.
    assert integrity["hand_source_embodiment_id"] != integrity["embodiment_id"]
    assert integrity["hand_source_calibration_id"] != integrity["calibration_id"]


def test_upstream_options_and_exclusions_preserved(scene):
    m = scene.model
    assert m.opt.timestep == pytest.approx(0.002)
    assert int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert int(m.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    assert m.opt.impratio == pytest.approx(100.0)
    assert m.neq == 0
    assert m.nexclude == 18


def test_pad_contact_profile_matches_p1_selection(scene):
    m = scene.model
    p1 = CONTACT_PROFILE["scene_contact"]
    assert p1["pad_geom"] == ts.PAD_GEOM_NAME
    assert list(m.geom_solref[scene.pad_geom_id]) == list(p1["solref"]) == [0.02, 1.0]
    assert list(m.geom_solimp[scene.pad_geom_id]) == list(p1["solimp"])
    assert list(m.geom_friction[scene.pad_geom_id]) == list(p1["friction"])
    # Explicitly NOT the old stiff default profile.
    assert list(m.geom_solref[scene.pad_geom_id]) != [0.0001, 1.0]


def test_wrist_hinge_and_actuator_parameters(scene):
    m = scene.model
    w = scene.config.wrist
    jid = scene.wrist_joint_id
    aid = scene.wrist_actuator_id
    assert [float(v) for v in m.jnt_axis[jid]] == [float(v) for v in w.axis]
    assert [float(v) for v in m.jnt_range[jid]] == pytest.approx(list(w.range_rad))
    assert m.dof_damping[int(m.jnt_dofadr[jid])] == pytest.approx(w.damping)
    assert m.dof_frictionloss[int(m.jnt_dofadr[jid])] == pytest.approx(w.frictionloss)
    assert m.dof_armature[int(m.jnt_dofadr[jid])] == pytest.approx(w.armature)
    assert [float(v) for v in m.actuator_ctrlrange[aid]] == pytest.approx(list(w.range_rad))
    assert m.actuator_gainprm[aid, 0] == pytest.approx(w.kp)
    assert -m.actuator_biasprm[aid, 2] == pytest.approx(w.kv)
    assert [float(v) for v in m.actuator_forcerange[aid]] == pytest.approx(list(w.forcerange_nm))


def test_cameras(scene):
    m = scene.model
    assert int(m.cam_bodyid[scene.wrist_cam_id]) == int(
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    )
    assert int(m.cam_bodyid[scene.overview_cam_id]) == 0  # worldbody


def test_named_indexing_and_open_pose(scene):
    assert scene.joint_index(ts.WRIST_JOINT) == scene.wrist_joint_id
    assert scene.actuator_index(ts.WRIST_ACTUATOR) == scene.wrist_actuator_id
    assert scene.joint_index("if_mcp") == 1
    with pytest.raises(KeyError):
        scene.joint_index("does_not_exist")
    q = scene.open_pose_qpos()
    assert q.shape == (17,)
    assert float(q[scene.wrist_joint_id]) == 0.0


def test_fingertip_sites_on_tip_bodies(scene):
    m = scene.model
    for finger, sid in scene.fingertip_site_ids.items():
        assert m.site(sid).name == TIP_SITES[finger]


def test_reset_reproduces_reference_tip_positions(scene):
    """At wrist=0 open pose, tips match the accepted fixed-base v1 scene."""
    d = mujoco.MjData(scene.model)
    mujoco.mj_resetData(scene.model, d)
    d.qpos[:] = scene.open_pose_qpos()
    mujoco.mj_forward(scene.model, d)
    tips = scene.fingertip_positions(d)
    assert tips["index"] == pytest.approx([0.1158, 0.0081, 0.2042], abs=2e-4)
    assert tips["middle"] == pytest.approx([0.1157, -0.0373, 0.2042], abs=2e-4)


def test_source_xml_tracks_download_root():
    root = Path("/nonexistent/download/root")
    with pytest.raises((LeapAssetError, FileNotFoundError)):
        ts.load_scene(ts.SceneConfig(download_root=root, manifest_path=root / "missing.json"))


def test_working_pose_has_landscape_pad_and_rising_palm(scene):
    import numpy as np
    working, info = ts.load_working_scene()
    m = working.model
    d = mujoco.MjData(m)
    for name, value in info['pose'].items():
        d.qpos[m.jnt_qposadr[working.joint_index(name)]] = value
    mujoco.mj_forward(m, d)
    assert d.geom_xmat[working.pad_geom_id].reshape(3, 3)[:, 0] == pytest.approx([0, 1, 0])
    assert d.body('palm').xmat.reshape(3, 3)[2, 0] > 0  # +X rises toward fingers
    assert .14 < d.body('palm').xpos[2] < .16
    assert info['measured_working']['knuckle_xpos_m'][2] > d.body('wrist_link').xpos[2]
    tips = working.fingertip_positions(d)
    assert (tips['index'][:2] + tips['middle'][:2]) / 2 == pytest.approx([0, 0], abs=1e-7)
    for finger in ('index', 'middle'):
        distance = min(mujoco.mj_geomDistance(m, d, g, working.pad_geom_id, 1, None)
                       for g in working.tip_geom_ids[finger])
        measured = ts._lowest_collision_z(working, d, finger) - .1
        assert measured == pytest.approx(distance, abs=2e-6)
        assert .0059 < distance < .0061
    for finger in ('ring', 'thumb'):
        assert ts._lowest_collision_z(working, d, finger) > .11
    visual = m.geom('visual_table').id
    assert m.geom_contype[visual] == m.geom_conaffinity[visual] == 0


def test_working_mount_preserves_base_configuration(scene):
    import numpy as np
    from dataclasses import replace
    base = ts.SceneConfig(pad_center=(.01, -.02, .12), pad_half_extents=(.07, .05, .006),
                         pad_yaw_deg=45, overview_cam_target=(.01, .02, .15),
                         wrist=replace(ts.WristConfig(), kp=9, mass_kg=.18,
                                       wrist_quat=(.995004165, 0, 0, .099833417)))
    working, info = ts.load_working_scene(base=base)
    assert working.config.pad_center == base.pad_center
    assert working.config.pad_half_extents == base.pad_half_extents
    assert working.config.pad_yaw_deg == 45
    assert working.config.overview_cam_target == base.overview_cam_target
    assert working.config.wrist.kp == 9
    assert working.config.wrist.mass_kg == .18
    rot = np.array(info['measured_working']['palm_xmat']).reshape(3, 3)
    eye = np.array(info['measured_working']['palm_xpos_m']) + rot @ working.config.wrist_cam_local_pos
    assert eye == pytest.approx(info['wrist_cam']['world_eye_at_reset'], abs=2e-7)
