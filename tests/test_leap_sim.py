"""Real-MuJoCo boundary tests for the LEAP v2 Advanced import and probe contract.

These tests exercise the actual pinned assets and solver: nothing is mocked.
They skip only when MuJoCo or the asset cache is genuinely unavailable; a
PRESENT but corrupt cache raises LeapAssetError from load_hand/verify_manifest
and fails the test.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad import leap_sim  # noqa: E402
from embodied_trackpad.leap_sim import open_pose_qpos  # noqa: E402
from embodied_trackpad.leap_sim.model import MOTOR_JOINTS, ROOT_BODY  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# Skip only when the download root itself is absent; an existing but
# incomplete/corrupt cache (including a missing robot.urdf) must fail inside
# load_hand/verify_manifest, never silently skip.
ASSETS_PRESENT = leap_sim.DOWNLOAD_ROOT.is_dir()


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "leap_model_probe", REPO_ROOT / "tools" / "leap_model_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


@pytest.fixture(scope="module")
def hand():
    if not ASSETS_PRESENT:
        pytest.skip("LEAP asset cache not available")
    return leap_sim.load_hand()  # corrupt present assets raise -> test failure


def test_motor_joint_equality_mapping_and_mass_labels(hand):
    m = hand.model
    assert m.nu == 17
    assert m.njnt == 21
    assert m.neq == 4
    actuated = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    assert actuated == list(MOTOR_JOINTS)
    for i in range(m.neq):
        j1 = m.joint(m.eq_obj1id[i]).name
        j2 = m.joint(m.eq_obj2id[i]).name
        assert j1.endswith("_dip") and j2.endswith("_pip")
        assert j1[: -len("_dip")] == j2[: -len("_pip")]
    inv = leap_sim.model_inventory(hand)
    mass = inv["mass_kg"]
    assert mass["total_cad"] == pytest.approx(1.505, abs=1e-9)
    assert mass["fixed_root"] == pytest.approx(0.140, abs=1e-9)
    assert mass["moving_links"] == pytest.approx(1.365, abs=1e-9)


def test_reset_and_short_finite_physics(hand):
    m = hand.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = open_pose_qpos(m)
    probe.apply_ctrl_checked(m, d, probe._open_vector(m))
    mujoco.mj_forward(m, d)
    for _ in range(100):
        probe.apply_ctrl_checked(m, d, probe._open_vector(m))
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        assert np.all(np.isfinite(d.qpos))
        assert np.all(np.isfinite(d.qvel))
        assert np.all(np.isfinite(d.qacc))
        assert np.all(np.isfinite(d.ctrl))
        assert np.all(np.isfinite(d.actuator_force))


def test_invalid_ctrl_rejected_and_ctrl_left_unchanged(hand):
    m = hand.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = open_pose_qpos(m)
    base = probe._open_vector(m)
    probe.apply_ctrl_checked(m, d, base)
    before = d.ctrl.copy()

    with pytest.raises(ValueError):
        probe.apply_ctrl_checked(m, d, np.zeros(m.nu - 1))  # wrong shape
    np.testing.assert_array_equal(d.ctrl, before)

    bad = base.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError):
        probe.apply_ctrl_checked(m, d, bad)  # non-finite
    np.testing.assert_array_equal(d.ctrl, before)

    bad = base.copy()
    bad[3] = float(m.actuator_ctrlrange[3, 1]) + 0.5
    with pytest.raises(ValueError):
        probe.apply_ctrl_checked(m, d, bad)  # out of range
    np.testing.assert_array_equal(d.ctrl, before)


def test_world_mounted_geom_contacts_palm_with_root_exclusions_active(hand):
    m0 = hand.model
    d0 = mujoco.MjData(m0)
    mujoco.mj_resetData(m0, d0)
    d0.qpos[:] = open_pose_qpos(m0)
    mujoco.mj_forward(m0, d0)
    palm_pos = d0.body("4_finger_frame_mirror").xpos.copy()

    def add_world_pad(spec):
        spec.worldbody.add_geom(
            name="test_world_pad",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0.0, 0.0],
            pos=palm_pos.tolist(),
        )

    hand2 = leap_sim.load_hand(spec_hook=add_world_pad)
    m = hand2.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = open_pose_qpos(m)
    mujoco.mj_forward(m, d)

    pad_palm_contacts = []
    excluded_pair_contacts = 0
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = m.geom(int(c.geom1)).name
        g2 = m.geom(int(c.geom2)).name
        b1 = m.body(int(m.geom_bodyid[int(c.geom1)])).name
        b2 = m.body(int(m.geom_bodyid[int(c.geom2)])).name
        if "test_world_pad" in (g1, g2) and "4_finger_frame_mirror" in (b1, b2):
            pad_palm_contacts.append((g1, g2))
        if {b1, b2} == {ROOT_BODY, "4_finger_frame_mirror"}:
            excluded_pair_contacts += 1
    assert pad_palm_contacts, (
        "world-mounted geom must contact the palm body 4_finger_frame_mirror "
        "(contacts with the excluded static root do not count)"
    )
    assert excluded_pair_contacts == 0, "root exclusion must remain active"


def test_existing_output_dir_rejected(tmp_path):
    out = tmp_path / "already-there"
    out.mkdir()
    with pytest.raises(FileExistsError):
        probe.run_probe(out, tiny=True)
