"""Focused tests for the LEAP v2 static-pad contact environment.

Covers the boundary contract the coordinator reviews: scene integrity (the
accepted hand is preserved and the pad is static), contact classification and
per-finger aggregation math, the pre-aggregation finite-value guard (NaN can
never vanish through the positive-force filter), touch.v1 message shape,
PNG encode/decode roundtrip, and a short SMOKE episode that must stay
contact-free at the fixed mount. Skips only when MuJoCo or the asset cache is
genuinely unavailable; a present but corrupt cache fails inside load_hand.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from embodied_trackpad import leap_sim  # noqa: E402
from embodied_trackpad.leap_sim import open_pose_qpos  # noqa: E402
from embodied_trackpad.leap_sim import pad_env  # noqa: E402
from embodied_trackpad.leap_sim.model import MOTOR_JOINTS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_PRESENT = leap_sim.DOWNLOAD_ROOT.is_dir()


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "leap_pad_probe", REPO_ROOT / "tools" / "leap_pad_probe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


@pytest.fixture(scope="module")
def scene():
    if not ASSETS_PRESENT:
        pytest.skip("LEAP asset cache not available")
    return pad_env.load_pad_scene()


def _raw(kind="tip_pad_top", force=1.0, geom1=5, geom2=0, pad_local=(0.01, 0.02, 0.005),
         top_cos=1.0, dist=-0.0005):
    return pad_env.RawContact(
        geom1=geom1,
        geom2=geom2,
        body1="dip_tip_soft_upward_cut" if geom1 == 5 else "world",
        body2="world" if geom2 == 0 else "dip_tip_soft_upward_cut",
        pos_world=[0.01, -0.03, -0.1715],
        frame=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        force6=[force, 0.0, 0.0, 0.0, 0.0, 0.0],
        normal_force_n=force,
        dist_m=dist,
        pad_local=list(pad_local) if pad_local is not None else None,
        top_face_cos=top_cos,
        kind=kind,
    )


# --- scene integrity ---------------------------------------------------------

def test_scene_preserves_accepted_hand(scene):
    info = pad_env.scene_integrity(scene)
    assert info["njnt"] == 21
    assert info["nu"] == 17
    assert info["neq"] == 4
    assert info["motor_order_preserved"]
    assert info["motor_joint_order"] == MOTOR_JOINTS
    m = scene.model
    hand_only = leap_sim.load_hand()
    assert info["nexclude"] == hand_only.model.nexclude == 8


def test_pad_is_static_world_geom(scene):
    info = pad_env.scene_integrity(scene)
    assert info["pad_body_is_world"]
    assert scene.pad_body_id == 0  # worldbody: static by construction
    assert info["pad_contype"] == 1 and info["pad_conaffinity"] == 1
    half = scene.pad_half_extents
    assert half == (0.080, 0.055, 0.005)  # 160 x 110 mm face


def test_tip_mapping_covers_five_distal_bodies(scene):
    assert set(scene.tip_geom_ids) == {"index", "middle", "ring", "pinky", "thumb"}
    m = scene.model
    expected_bodies = {
        "index": "dip_tip_soft_upward_cut",
        "middle": "dip_tip_soft_upward_cut_2",
        "ring": "dip_tip_soft_upward_cut_3",
        "pinky": "dip_tip_soft_upward_cut_4",
        "thumb": "thumb_tip_hard",
    }
    for finger, body_name in expected_bodies.items():
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for g in scene.tip_geom_ids[finger]:
            assert int(m.geom_bodyid[g]) == bid
        site_body = int(m.site_bodyid[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, scene.tip_site[finger])])
        assert site_body == bid


def test_reset_open_pose_is_contact_free(scene):
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = open_pose_qpos(m)
    d.ctrl[:] = pad_env._open_ctrl(m)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0  # fixed mount clears every body at reset
    for _ in range(200):  # and stays clear over a 0.1 s servo hold
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    assert d.ncon == 0


# --- classification / aggregation / finite guard ------------------------------

def test_validate_raw_contacts_rejects_nonfinite(scene):
    tip_geom = scene.tip_geom_ids["index"][0]
    good = _raw(geom1=tip_geom)
    pad_env.validate_raw_contacts([good])  # must not raise
    for bad_kwargs in (
        {"force": np.nan},
        {"force": np.inf},
        {"dist": np.nan},
        {"pad_local": (np.nan, 0.0, 0.0)},
        {"top_cos": np.nan},
    ):
        with pytest.raises(ValueError):
            pad_env.validate_raw_contacts([_raw(geom1=tip_geom, **bad_kwargs)])
    bad_frame = _raw(geom1=tip_geom)
    bad_frame.frame[3] = np.inf
    with pytest.raises(ValueError):
        pad_env.validate_raw_contacts([bad_frame])


def test_nan_force_cannot_vanish_in_aggregation(scene):
    # validate-then-aggregate is the required order; a NaN must be rejected
    # up front, because `NaN > 0` is False and would otherwise be filtered.
    tip_geom = scene.tip_geom_ids["index"][0]
    nan_contact = _raw(force=np.nan, geom1=tip_geom)
    with pytest.raises(ValueError):
        pad_env.validate_raw_contacts([nan_contact])
    zero = _raw(force=0.0, geom1=tip_geom)
    pad_env.validate_raw_contacts([zero])
    aggregated, _ = pad_env.aggregate_fingers(scene, [zero])
    assert aggregated == {}  # zero force is not contact, but NaN never gets here


def test_aggregation_force_weighted_single_entry(scene):
    tip_geom = scene.tip_geom_ids["index"][0]
    c1 = _raw(force=1.0, geom1=tip_geom, pad_local=(0.01, 0.02, 0.005))
    c2 = _raw(force=3.0, geom1=tip_geom, pad_local=(0.03, 0.0, 0.005))
    pad_env.validate_raw_contacts([c1, c2])
    aggregated, unintended = pad_env.aggregate_fingers(scene, [c1, c2])
    assert unintended == []
    assert list(aggregated) == ["index"]  # one entry, never duplicated
    a = aggregated["index"]
    assert a["normal_force_n"] == pytest.approx(4.0)
    assert a["x_m"] == pytest.approx((1.0 * 0.01 + 3.0 * 0.03) / 4.0)
    assert a["y_m"] == pytest.approx((1.0 * 0.02 + 3.0 * 0.0) / 4.0)
    assert a["contact_points"] == 2


def test_classification_edge_and_nontip_are_unintended(scene):
    tip_geom = scene.tip_geom_ids["index"][0]
    edge = _raw(kind="tip_pad_edge", geom1=tip_geom, top_cos=0.5)
    nontip = _raw(kind="nontip_pad", geom1=3)  # finger_v2_with_motor
    selfc = _raw(kind="hand_self", geom1=3, geom2=4, pad_local=None, top_cos=None)
    aggregated, unintended = pad_env.aggregate_fingers(scene, [edge, nontip, selfc])
    assert aggregated == {}
    assert [c.kind for c in unintended] == ["tip_pad_edge", "nontip_pad", "hand_self"]


def test_touch_v1_message_shape_and_order():
    aggregated = {
        "index": {"x_m": 0.01, "y_m": -0.02, "normal_force_n": 1.5},
        "thumb": {"x_m": -0.03, "y_m": 0.01, "normal_force_n": 0.4},
    }
    msg = pad_env.touch_v1_message(7, 1.25, aggregated)
    assert msg["schema_version"] == "touch.v1"
    assert msg["sequence"] == 7
    assert msg["simulation_time_s"] == 1.25
    ids = [c["finger_id"] for c in msg["contacts"]]
    assert ids == ["index", "thumb"]  # deterministic FINGERS order, unique
    c0 = msg["contacts"][0]
    assert set(c0) == {"finger_id", "x_m", "y_m", "normal_force_n"}
    assert abs(c0["x_m"]) <= pad_env.TOUCH_HALF_X
    assert abs(c0["y_m"]) <= pad_env.TOUCH_HALF_Y
    empty = pad_env.touch_v1_message(0, 0.0, {})
    assert empty["contacts"] == []  # clear frames exist and carry no contacts


# --- PNG roundtrip -------------------------------------------------------------

def test_png_roundtrip_decodes_actual_image(tmp_path):
    h, w = 8, 12
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.arange(w, dtype=np.uint8)[None, :] * 10
    img[:, :, 1] = np.arange(h, dtype=np.uint8)[:, None] * 20
    img[:, :, 2] = 128
    path = tmp_path / "t.png"
    pad_env.write_png(path, img)
    decoded = pad_env.decode_png_rgb(path)
    assert np.array_equal(decoded, img)


def test_png_decoder_rejects_broken_writer_output(tmp_path):
    # regression for the tiny-001 failure: 44-byte files (IDAT without data)
    import struct
    import zlib

    def bad_chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    broken = (
        b"\x89PNG\r\n\x1a\n"
        + bad_chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
        + bad_chunk(b"IDAT", zlib.compress(b""))
        + bad_chunk(b"IEND", b"")
    )
    path = tmp_path / "broken.png"
    path.write_bytes(broken)
    with pytest.raises((ValueError, Exception)):
        pad_env.decode_png_rgb(path)


# --- short smoke episode -------------------------------------------------------

def test_smoke_episode_is_contact_free_and_jsonable(scene, tmp_path):
    cfg = pad_env.PadEpisodeConfig(
        candidate_id="pytest-smoke",
        clear1_end=0.3, press_ramp_end=0.35, press_hold_end=0.4,
        slide_end=0.45, lift_ramp_end=0.5, t_end=0.7,
        press_mcpf=0.05, press_dip=0.05, slide_mcps=0.0,
        smoke=True,
    )
    res = pad_env.run_pad_episode(scene, cfg, out_dir=tmp_path / "smoke", render=True)
    m = res.metrics
    assert m["verdict"] == "SMOKE"  # smoke runs can never claim acceptance
    assert m["episode_complete"]
    c = m["checks"]
    assert c["unintended_contact_steps"] == 0
    assert c["unintended_contact_events"] == 0
    assert c["max_pad_normal_force_all_n"] == 0.0
    assert c["initial_clearance"]["ok"] and c["final_clearance"]["ok"]
    assert c["initial_clearance"]["coverage_complete"]
    assert c["warnings"] == {}
    ts = c["touch_stream"]
    assert ts["monotonic"] and ts["bounded_interval_ok"]
    assert ts["max_frame_interval_s"] <= pad_env.TOUCH_MAX_FRAME_INTERVAL_S + 1e-9
    assert ts["empty_frames"] == ts["frames"]  # open hold: all frames empty
    # artefacts: metrics JSON roundtrip, per-step series, decodable renders
    import json

    reloaded = json.loads((tmp_path / "smoke" / "metrics.json").read_text())
    assert reloaded["verdict"] == "SMOKE"
    trace = np.load(tmp_path / "smoke" / "trace.npz")
    n_steps = m["steps_planned"]
    assert len(trace["series_t"]) == n_steps  # every-step gate evidence
    assert trace["qpos"].shape[0] >= 2  # sampled trace incl. endpoints
    for png in (tmp_path / "smoke").glob("*.png"):
        pad_env.decode_png_rgb(png)  # raises on broken/blank renders


def test_probe_candidate_grid_is_within_bounds():
    grid = probe.candidate_grid()
    assert 1 <= len(grid) <= probe.MAX_CANDIDATES
    ids = [c.candidate_id for c in grid]
    assert len(ids) == len(set(ids))
    for cfg in grid:
        assert not cfg.smoke
        # pad stays at the reviewed fixed mount unless explicitly varied
        assert len(cfg.pad_center) == 3
