"""Focused unit tests for the continuous live viewer (no server, no scene).

Covers lifetime normalization (unlimited default), camera command
validation/clamping, and camera-state helpers of tools/tap_slide_live_view.
"""

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest
mujoco = pytest.importorskip("mujoco")
import numpy as np

TOOL = Path(__file__).resolve().parents[1] / "tools" / "tap_slide_live_view.py"
spec = importlib.util.spec_from_file_location("tap_slide_live_view", TOOL)
lv = importlib.util.module_from_spec(spec)
sys.modules["tap_slide_live_view"] = lv  # argparse/docstring import safety
spec.loader.exec_module(lv)

CURRENT = {"azimuth": 135.0, "elevation": -30.0, "distance": 0.45,
           "lookat": [0.02, 0.0, 0.13]}


class TestLifetime(unittest.TestCase):
    def test_unlimited_cases(self):
        for v in (None, 0, 0.0, -5, -0.001, float("nan"), float("inf"),
                  float("-inf"), "junk", []):
            self.assertEqual(lv.normalize_lifetime(v), 0.0, repr(v))

    def test_finite_positive_kept(self):
        self.assertEqual(lv.normalize_lifetime(1200), 1200.0)
        self.assertAlmostEqual(lv.normalize_lifetime("60.5"), 60.5)


class TestCameraCommand(unittest.TestCase):
    def test_absolute_and_delta(self):
        out = lv.apply_camera_command(CURRENT, {"d_azimuth": 10.0})
        self.assertAlmostEqual(out["azimuth"], 145.0)
        out = lv.apply_camera_command(CURRENT, {"distance": 1.0})
        self.assertAlmostEqual(out["distance"], 1.0)
        out = lv.apply_camera_command(CURRENT, {"distance": 1.0, "d_distance": 0.2})
        self.assertAlmostEqual(out["distance"], 1.2)

    def test_clamps(self):
        out = lv.apply_camera_command(CURRENT, {"distance": 99.0})
        self.assertEqual(out["distance"], lv.CAM_DIST_MAX_M)
        out = lv.apply_camera_command(CURRENT, {"distance": 0.001})
        self.assertEqual(out["distance"], lv.CAM_DIST_MIN_M)
        out = lv.apply_camera_command(CURRENT, {"elevation": 500.0})
        self.assertEqual(out["elevation"], lv.CAM_ELEV_MAX_DEG)
        out = lv.apply_camera_command(CURRENT, {"elevation": -500.0})
        self.assertEqual(out["elevation"], lv.CAM_ELEV_MIN_DEG)
        out = lv.apply_camera_command(CURRENT, {"d_azimuth": 400.0})
        self.assertTrue(0.0 <= out["azimuth"] < 360.0)

    def test_rejects_bad_input(self):
        for cmd in ({}, {"distance": float("nan")}, {"d_azimuth": float("inf")},
                    "not-a-dict", {"distance": "abc"}, {"preset": "nope"}):
            with self.assertRaises((ValueError, TypeError), msg=repr(cmd)):
                lv.apply_camera_command(CURRENT, cmd,
                                        presets=lv.CAMERA_PRESETS)

    def test_preset_replaces_state_and_rejects_absolute_mix(self):
        presets = {"top": {"azimuth": 0.0, "elevation": -89.0, "distance": 0.5,
                           "lookat": [0.0, 0.0, 0.1]}}
        out = lv.apply_camera_command(CURRENT, {"preset": "top"}, presets=presets)
        self.assertEqual((out["azimuth"], out["elevation"], out["distance"]),
                         (0.0, -89.0, 0.5))
        self.assertEqual(out["lookat"], [0.0, 0.0, 0.1])
        with self.assertRaises(ValueError):
            lv.apply_camera_command(CURRENT, {"preset": "top", "distance": 1.0},
                                    presets=presets)

    def test_lookat_preserved_and_finite(self):
        out = lv.apply_camera_command(CURRENT, {"d_distance": -0.4})
        self.assertEqual(out["lookat"], CURRENT["lookat"])
        for v in out.values():
            if isinstance(v, list):
                self.assertTrue(all(math.isfinite(x) for x in v))
            else:
                self.assertTrue(math.isfinite(v))


class TestInitialCamera(unittest.TestCase):
    def test_coordinator_verified_convention(self):
        # Ground truth from independent mjv_updateScene review:
        # az=0,el=0 -> eye=lookat+(-1,0,0)*d ; az=90,el=0 -> (0,-1,0)*d ;
        # az=0,el=-45 -> (-.707,0,+.707)*d.
        cases = [
            ((-0.30, 0.0, 0.0), 0.0, 0.0),
            ((0.0, -0.30, 0.0), 90.0, 0.0),
            ((-0.21213, 0.0, +0.21213), 0.0, -45.0),
        ]
        for off, want_az, want_el in cases:
            off = np.asarray(off, dtype=float)
            d = float(np.linalg.norm(off))
            cfg = SimpleNamespace(
                overview_cam_pos=(0.02 + off[0], 0.0 + off[1], 0.13 + off[2]),
                overview_cam_target=(0.02, 0.0, 0.13),
            )
            cam = lv.initial_overview_camera(cfg)
            self.assertAlmostEqual(cam["azimuth"], want_az, places=4,
                                   msg=f"off={off}")
            self.assertAlmostEqual(cam["elevation"], want_el, places=4,
                                   msg=f"off={off}")
            self.assertAlmostEqual(cam["distance"], d, places=6)

    def test_scene_defaults_point_at_config_eye(self):
        cfg = SimpleNamespace(overview_cam_pos=(0.32, -0.28, 0.28),
                              overview_cam_target=(0.02, 0.0, 0.13))
        cam = lv.initial_overview_camera(cfg)
        # Coordinator-pinned expected values for the real overview camera.
        self.assertAlmostEqual(cam["azimuth"], 136.9746, places=2)
        self.assertAlmostEqual(cam["elevation"], -20.0786, places=2)
        self.assertEqual(cam["lookat"], [0.02, 0.0, 0.13])

    def test_freecam_eye_matches_config_pos(self):
        """mjv_updateScene freecam eye must land on config.overview_cam_pos."""
        target = [0.02, 0.0, 0.13]
        off = [0.30, -0.28, 0.15]  # eye = target + off
        xml = f"""
        <mujoco><worldbody>
          <body pos="{' '.join(map(str, target))}">
            <camera name="ov" pos="{' '.join(map(str, off))}"
                    xyaxes="1 0 0 0 1 0" fovy="45"/>
            <geom type="sphere" size="0.01"/>
          </body>
        </worldbody></mujoco>
        """
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cfg = SimpleNamespace(
            overview_cam_pos=[target[i] + off[i] for i in range(3)],
            overview_cam_target=target,
        )
        cam = lv.initial_overview_camera(cfg)
        fc = mujoco.MjvCamera()
        fc.type = mujoco.mjtCamera.mjCAMERA_FREE
        fc.azimuth, fc.elevation, fc.distance = (
            cam["azimuth"], cam["elevation"], cam["distance"])
        fc.lookat[:] = cam["lookat"]
        scn = mujoco.MjvScene(m, maxgeom=64)
        opt = mujoco.MjvOption()
        mujoco.mjv_updateScene(m, d, opt, None, fc,
                               mujoco.mjtCatBit.mjCAT_ALL, scn)
        eye = 0.5 * (np.asarray(scn.camera[0].pos)
                     + np.asarray(scn.camera[1].pos))
        want = np.asarray(cfg.overview_cam_pos, dtype=float)
        self.assertLess(float(np.linalg.norm(eye - want)), 1e-4)


if __name__ == "__main__":
    unittest.main()


def test_target_validation_and_pending_delta_accumulation():
    import threading
    sim = lv.LiveSim.__new__(lv.LiveSim)
    sim.lock = threading.Lock()
    sim.camera_state = dict(CURRENT)
    sim.camera_revision = 0
    sim.apply_camera_command({'d_azimuth': 10})
    second = sim.apply_camera_command({'d_azimuth': 10})
    assert second['azimuth'] == 155
    assert second['revision'] == 2
    for target in ([0, 0], [0, float('nan'), 0], [0, 2, 0]):
        with pytest.raises(ValueError):
            lv.apply_camera_command(CURRENT, {'lookat': target})
    good = lv.apply_camera_command(CURRENT, {'lookat': [-.07, .02, .1]})
    assert good['lookat'] == [-.07, .02, .1]


def test_old_shutdown_cannot_replace_current_runtime(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(lv, 'LIVE_DIR', tmp_path)
    monkeypatch.setattr(lv, 'RUNTIME_PATH', tmp_path / 'runtime.json')
    lv.write_runtime({'pid': 102, 'status': 'running'})
    lv.write_runtime({'pid': 101, 'status': 'stopped'})
    assert json.loads((tmp_path / 'runtime.json').read_text())['pid'] == 102
    assert json.loads((tmp_path / 'runtime-101.json').read_text())['status'] == 'stopped'
