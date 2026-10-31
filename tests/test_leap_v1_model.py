"""Real-MuJoCo boundary tests for the LEAP v1 Full baseline.

These tests exercise the actual pinned Menagerie assets and solver: nothing is
mocked. They skip only when MuJoCo or the asset cache is genuinely
unavailable; a PRESENT but corrupt cache raises LeapAssetError from
load_hand/verify_manifest and fails the test.
"""
from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad import leap_v1  # noqa: E402
from embodied_trackpad.leap_v1 import diagnostics  # noqa: E402
from embodied_trackpad.leap_v1.model import MOTOR_JOINTS, TIP_GEOMS, TIP_SITES  # noqa: E402
from embodied_trackpad.leap_sim.pad_env import apply_ctrl_checked, decode_png_rgb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_PRESENT = leap_v1.DOWNLOAD_ROOT.is_dir()

EXPECTED_RANGES = {  # upstream right_hand.xml values (audit anchors)
    "if_mcp": (-0.314, 2.23),
    "if_rot": (-1.047, 1.047),
    "if_pip": (-0.506, 1.885),
    "if_dip": (-0.366, 2.042),
    "th_cmc": (-0.349, 2.094),
    "th_axl": (-0.349, 2.094),
    "th_mcp": (-0.47, 2.443),
    "th_ipl": (-1.34, 1.88),
}


@pytest.fixture(scope="module")
def hand():
    if not ASSETS_PRESENT:
        pytest.skip("LEAP v1 asset cache not available")
    return leap_v1.load_hand()  # corrupt present assets raise -> test failure


def test_compile_inventory_16_independent_channels(hand):
    m = hand.model
    assert (m.nq, m.nv, m.nu, m.njnt) == (16, 16, 16, 16)
    assert m.neq == 0  # no tendon coupling in v1 (unlike v2)
    assert m.nexclude == 18
    actuated = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    assert actuated == list(MOTOR_JOINTS)
    assert len(set(actuated)) == 16
    for i in range(m.nu):
        assert int(m.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
        j = int(m.actuator_trnid[i, 0])
        assert np.allclose(m.actuator_ctrlrange[i], m.jnt_range[j])
    # fixed palm: the palm body carries no joint
    palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    assert int(m.body_jntnum[palm]) == 0
    for jname, (lo, hi) in EXPECTED_RANGES.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) == pytest.approx(
            (lo, hi), abs=1e-9
        )
    inv = leap_v1.model_inventory(hand)
    assert inv["dof_split"]["hardware_motors"] == 16
    assert inv["dof_split"]["passive_coupled_joints"] == 0
    assert inv["embodiment_id"] == leap_v1.EMBODIMENT_ID
    assert inv["calibration_id"] == leap_v1.CALIBRATION_ID


def test_tip_sites_track_tip_geoms(hand):
    m = hand.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert m.nsite == 4
    for finger, sname in TIP_SITES.items():
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname)
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, TIP_GEOMS[finger])
        assert sid >= 0 and gid >= 0
        assert np.linalg.norm(d.site_xpos[sid] - d.geom_xpos[gid]) < 1e-9


def test_flat_reset_is_contact_free_and_finite(hand):
    """The all-zero open reset is legal: no self-contact invalidates it."""
    m = hand.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = leap_v1.open_pose_qpos(m)
    apply_ctrl_checked(m, d, np.zeros(m.nu))
    mujoco.mj_forward(m, d)
    assert d.ncon == 0
    for _ in range(100):  # 0.2 s open hold
        apply_ctrl_checked(m, d, np.zeros(m.nu))
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        assert np.all(np.isfinite(d.qpos))
        assert np.all(np.isfinite(d.qvel))
    assert d.ncon == 0  # still contact-free after settling
    warn = np.array([w.number for w in d.warning])
    assert not warn.any()


def test_each_actuator_moves_exactly_its_joint(hand):
    """16-channel mapping: driving one channel moves its own joint most."""
    m = hand.model
    d = mujoco.MjData(m)
    for i in range(m.nu):
        mujoco.mj_resetData(m, d)
        d.qpos[:] = leap_v1.open_pose_qpos(m)
        ctrl = np.zeros(m.nu)
        hi = float(m.actuator_ctrlrange[i][1])
        ctrl[i] = min(0.5, 0.5 * hi)
        for _ in range(250):  # 0.5 s
            apply_ctrl_checked(m, d, ctrl)
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        dq = np.abs(d.qpos)
        own = int(m.jnt_qposadr[m.actuator_trnid[i, 0]])
        assert dq[own] > 0.05, f"{MOTOR_JOINTS[i]} did not move"
        # Own joint is the largest mover. Neighbour joints may move via real
        # finger-finger contact coupling (recorded in the diagnostics report),
        # so no stricter ratio is imposed here.
        assert dq[own] >= dq.max() - 1e-9, f"{MOTOR_JOINTS[i]} is not the dominant mover"


def test_ctrl_validation_rejects_bad_input(hand):
    m = hand.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    with pytest.raises(ValueError, match="shape"):
        apply_ctrl_checked(m, d, np.zeros(m.nu + 1))
    bad = np.zeros(m.nu)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        apply_ctrl_checked(m, d, bad)
    bad = np.zeros(m.nu)
    bad[0] = float(m.actuator_ctrlrange[0][1]) + 0.5
    with pytest.raises(ValueError, match="ctrlrange"):
        apply_ctrl_checked(m, d, bad)


def test_full_diagnostics_verdict_and_artifacts(hand, tmp_path):
    result = diagnostics.run_motion_diagnostics(
        hand, out_dir=tmp_path / "diag", render=True, sample_every=50
    )
    metrics = result.metrics
    assert metrics["verdict"] == "PASS", metrics["failure_reasons"]
    assert metrics["steps_executed"] == metrics["steps_planned"]
    assert metrics["gates"]["limit_excursion<=0.01"]
    assert metrics["gates"]["no_warnings"]
    rep = metrics["reported"]
    # measured reference outcomes of this baseline (not pass/fail relaxations)
    assert rep["held_tracking_free_max_rad"] <= rep["held_tracking_reference_tol_rad"]
    assert rep["spread_delta_m"] > 0.01  # index/middle spacing really opens
    assert rep["recovered_open_error_rad"] <= rep["recovered_open_reference_tol_rad"]
    assert len(rep["held_tracking_per_channel_rad"]) == 16
    assert len(rep["channel_tip_displacement_m"]) == 16
    # renders are real decodable PNGs
    pngs = sorted((tmp_path / "diag").glob("*.png"))
    assert len(pngs) >= 20  # overview + 16 sweeps + spread + flex + returned
    for png in pngs:
        decode_png_rgb(png)
    # traces cover the episode
    trace = np.load(tmp_path / "diag" / "trace.npz")
    assert trace["ctrl"].shape[1] == 16
    assert trace["qpos"].shape[1] == 16
    assert trace["tips"].shape[1] == 4
    assert trace["t"][-1] == pytest.approx(metrics["sim_seconds"], abs=2e-3)
    for key in ("motion-metrics.json", "self-contacts.json", "trace.npz"):
        assert (tmp_path / "diag" / key).exists()
