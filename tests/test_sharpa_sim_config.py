"""Config contract tests for the Sharpa trackpad scene (pure Python)."""
import dataclasses
import math

import pytest

from embodied_trackpad.sharpa_sim.config import (
    ACTION_SCHEMA, ARM_PANDA_7DOF, HAND_22_CHANNELS, PANDA_LIMITS,
    SharpaSceneConfig,
)


def test_action_schema_dim_and_order():
    assert ACTION_SCHEMA["dim"] == 29
    assert len(ACTION_SCHEMA["arm"]["joints"]) == 7
    assert ACTION_SCHEMA["arm"]["joints"][0] == "panda_joint1"
    assert ACTION_SCHEMA["arm"]["joints"][-1] == "panda_joint7"
    assert ACTION_SCHEMA["arm"]["units"] == "rad"
    assert "TEMPORARY_SIM_ARM" in ARM_PANDA_7DOF
    assert HAND_22_CHANNELS.startswith("sharpawave_right")


def test_panda_limits_are_finite_and_ordered():
    assert len(PANDA_LIMITS) == 7
    for lo, hi in PANDA_LIMITS:
        assert math.isfinite(lo) and math.isfinite(hi)
        assert lo < hi
        assert hi - lo > 0.1


def test_default_config_validates():
    cfg = SharpaSceneConfig()
    cfg.validate()


def test_config_roundtrip():
    cfg = SharpaSceneConfig(seed=3, base_xy=(0.5, 0.1), pad_yaw=0.2,
                            pad_tilt_deg=5.0)
    d = cfg.to_dict()
    cfg2 = SharpaSceneConfig.from_dict(d)
    assert cfg2.seed == 3
    assert abs(cfg2.base_xy[0] - 0.5) < 1e-9
    assert abs(cfg2.pad_yaw - 0.2) < 1e-9
    assert len(cfg2.cameras) == len(cfg.cameras)


def test_config_rejects_bad_values():
    cfg = SharpaSceneConfig(pad_tilt_deg=80.0)
    with pytest.raises(ValueError):
        cfg.validate()
    cfg = SharpaSceneConfig(physics_dt=0.0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_pad_size_is_magic_trackpad_landscape():
    cfg = SharpaSceneConfig()
    w, h, t = cfg.pad_size
    assert abs(w - 0.160) < 1e-9
    assert abs(h - 0.1149) < 1e-9
    assert w > h  # landscape
    assert "Magic Trackpad 2" in cfg.pad_source


def test_camera_roles():
    cfg = SharpaSceneConfig()
    roles = {c.name: c.role for c in cfg.cameras}
    assert roles["overview"] == "overview"
    assert roles["screen_rgb"] == "actor"
    assert roles["wrist_rgb"] == "actor"
