"""Unit tests for the relative-trackpad feedback expert.

These tests use fixture-only mock environments and never launch Isaac Sim.
They exercise the expert's contract (finite normalized actions), relative
pointer semantics, phase transitions, joint-limit clipping, recovery, and the
``TeacherEnabledEnv`` wrapper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from embodied_trackpad.sharpa_learning.env import ACTION_DIM, ARM_DELTA_SCALE, HAND_DELTA_SCALE
from embodied_trackpad.sharpa_learning.feedback_expert import (
    FeedbackExpert, TeacherEnabledEnv, _is_flexion_joint,
)


# ---------------------------------------------------------------------------
# Minimal fake environment
# ---------------------------------------------------------------------------
class _FakeTip:
    def __init__(self, pos=(0.0, 0.0, 0.0)):
        self._pos = np.asarray(pos, dtype=float)

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=float)

    def get_world_poses(self):
        # Mimics isaacsim.core.experimental.prims.RigidPrim.get_world_poses()
        return (self._pos.reshape(1, 3), np.array([[1.0, 0.0, 0.0, 0.0]]))


@dataclass
class _FakeSceneConfig:
    base_xy: tuple[float, float] = (0.42, 0.30)
    base_yaw: float = math.radians(180.0)
    pad_xy: tuple[float, float] = (0.0, 0.0)
    pad_yaw: float = 0.0
    pad_size: tuple[float, float, float] = (0.160, 0.1149, 0.0109)
    table_top_z: float = 0.72
    mount_pos0: tuple[float, ...] = (0.0, 0.0, 0.107)
    mount_rot0_wxyz: tuple[float, ...] = (0.70710677, 0.0, 0.70710677, 0.0)
    mount_rot1_wxyz: tuple[float, ...] = (
        0.6532815, -0.2705981, 0.6532814, 0.27059814)


class _FakeTracker:
    def __init__(self, gain=3200.0):
        self.gain = float(gain)


class _FakeAdapter:
    def __init__(self, gain=3200.0):
        self.tracker = _FakeTracker(gain)


class _FakeEnv:
    """Pure-Python stand-in for SharpaEnv."""

    SUBSTEPS = 8

    def __init__(self):
        self.action_names = (
            [f"panda_joint{i}" for i in range(1, 8)]
            + [
                "right_index_finger_MCP", "right_index_finger_PIP",
                "right_index_finger_DIP", "right_index_finger_abduction",
                "right_middle_finger_MCP", "right_middle_finger_PIP",
                "right_middle_finger_DIP", "right_middle_finger_abduction",
                "right_ring_finger_MCP", "right_ring_finger_PIP",
                "right_ring_finger_DIP", "right_ring_finger_abduction",
                "right_pinky_finger_MCP", "right_pinky_finger_PIP",
                "right_pinky_finger_DIP", "right_pinky_finger_abduction",
                "right_thumb_CMC", "right_thumb_MCP", "right_thumb_IP",
                "right_thumb_abduction", "right_hand_flange_roll",
                "right_hand_flange_pitch",
            ]
        )
        assert len(self.action_names) == ACTION_DIM
        lo = np.full(ACTION_DIM, -0.6, dtype=float)
        hi = np.full(ACTION_DIM, 0.6, dtype=float)
        from embodied_trackpad.sharpa_sim.config import PANDA_LIMITS
        for i in range(7):
            lo[i], hi[i] = PANDA_LIMITS[i]
        self.joint_limits = (lo, hi)
        is_arm = np.array([n.startswith("panda_joint") for n in self.action_names])
        self.delta_scale = np.where(is_arm, ARM_DELTA_SCALE, HAND_DELTA_SCALE).astype(np.float32)
        self.scene_config = _FakeSceneConfig()
        self.home_q7 = np.array(
            [0.089940, 0.608528, -0.120211, -2.272798,
             1.527936, 1.695347, 1.050544], dtype=float)
        self.dt = 1.0 / 120.0
        self.adapter = _FakeAdapter(gain=3200.0)
        self.gui = type("G", (), {"width": 640, "height": 480})()
        self.tips = {"index": _FakeTip(pos=(0.0, 0.0, 0.735))}
        self._cursor = (320.0, 240.0)
        self._target = (448.0, 160.0)
        self._cursor_over_target = False
        self._click_on_target = 0

    def _privileged(self):
        return {
            "oracle_cursor_px": self._cursor,
            "oracle_target_px": self._target,
            "oracle_cursor_over_target": self._cursor_over_target,
            "oracle_click_on_target": self._click_on_target,
        }

    def _pad_local_xy(self, p_world):
        cfg = self.scene_config
        dx = float(p_world[0]) - cfg.pad_xy[0]
        dy = float(p_world[1]) - cfg.pad_xy[1]
        c, s = math.cos(-cfg.pad_yaw), math.sin(-cfg.pad_yaw)
        return c * dx - s * dy, s * dx + c * dy

    def set_cursor(self, x, y, over_target=False):
        self._cursor = (float(x), float(y))
        self._cursor_over_target = bool(over_target)

    def set_target(self, x, y):
        self._target = (float(x), float(y))

    def set_click_on_target(self, v):
        self._click_on_target = int(v)


@pytest.fixture
def env():
    return _FakeEnv()


@pytest.fixture
def expert(env):
    return FeedbackExpert(env)


def _obs(q=None, index_force=0.0, index_valid=True):
    q = np.zeros(ACTION_DIM, dtype=float) if q is None else np.asarray(q, dtype=float)
    qd = np.zeros(ACTION_DIM, dtype=float)
    touch = np.zeros(5, dtype=np.float32)
    touch[0] = float(index_force)
    valid = np.zeros(5, dtype=np.uint8)
    valid[0] = 1 if index_valid else 0
    return {
        "proprio": np.concatenate([q, qd]).astype(np.float32),
        "touch": touch,
        "touch_valid": valid,
    }


# ---------------------------------------------------------------------------
# Contract / sanity
# ---------------------------------------------------------------------------
def test_action_shape_and_range(expert):
    expert.reset()
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    obs = _obs()
    for t in range(5):
        a = expert.action(obs, t)
        assert a.shape == (ACTION_DIM,)
        assert a.dtype == np.float32
        assert np.all(np.isfinite(a))
        assert np.all(a >= -1.0) and np.all(a <= 1.0)


def test_reset_clears_phase_machine(expert):
    expert.reset()
    assert expert._phase == "clear"
    expert._enter_phase("slide")
    expert.reset()
    assert expert._phase == "clear"
    assert expert._recontact_attempts == 0
    assert expert._reanchor_attempts == 0


def test_is_flexion_excludes_abduction_lateral_joints():
    assert _is_flexion_joint("right_index_finger_MCP")
    assert _is_flexion_joint("right_index_finger_PIP")
    assert not _is_flexion_joint("right_index_finger_abduction")
    assert not _is_flexion_joint("right_index_finger_MCP_AA")
    assert not _is_flexion_joint("right_thumb_CMC_AA")


# ---------------------------------------------------------------------------
# Relative vs absolute cursor semantics
# ---------------------------------------------------------------------------
def test_slide_uses_relative_cursor_delta_not_absolute_target(env, expert):
    """The expert must move the actual fingertip by (cursor_error / gain) on the
    pad, with pad y sign inverted, then solve differentially from current q7."""
    expert.reset()
    expert._enter_phase("slide")
    env.set_cursor(320, 240)
    env.set_target(420, 140)  # target is right and above cursor
    env.tips["index"].set_pos((0.0, 0.0, 0.735))
    captured = []

    def fake_arm_move(qm, measured_tip, world_delta, max_step):
        captured.append(np.asarray(world_delta, dtype=float).copy())
        return expert.env.home_q7.copy()

    expert._arm_move = fake_arm_move
    obs = _obs(index_force=0.5)
    expert.action(obs, t=10)

    assert len(captured) == 1
    err_x, err_y = 420 - 320, 140 - 240  # +100, -100
    gain = 3200.0
    dx_pad = float(np.clip(err_x / gain, -0.003, 0.003))
    dy_pad = float(np.clip(-err_y / gain, -0.003, 0.003))
    # pad yaw is 0, so world delta equals pad-local delta
    expected = np.array([dx_pad, dy_pad, 0.0])
    np.testing.assert_allclose(captured[0], expected, atol=1e-9)


def test_slide_does_not_teleport_to_target_coordinates(env, expert):
    """Absolute mapping would place the fingertip at the target's world
    coordinates; the relative expert must not do that."""
    expert.reset()
    expert._enter_phase("slide")
    env.set_cursor(0, 0)
    env.set_target(640, 480)
    env.tips["index"].set_pos((0.0, 0.0, 0.735))
    captured = []

    def fake_arm_move(qm, measured_tip, world_delta, max_step):
        captured.append(np.asarray(world_delta, dtype=float).copy())
        return expert.env.home_q7.copy()

    expert._arm_move = fake_arm_move
    obs = _obs(index_force=0.5)
    expert.action(obs, t=10)

    assert len(captured) == 1
    assert abs(captured[0][0]) <= FeedbackExpert.MAX_SLIDE_DELTA_M + 1e-9
    assert abs(captured[0][1]) <= FeedbackExpert.MAX_SLIDE_DELTA_M + 1e-9
    assert captured[0][2] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Phase transitions on measured contact / release
# ---------------------------------------------------------------------------
def test_approach_transitions_to_slide_on_index_contact(env, expert):
    expert.reset()
    expert._enter_phase("approach")
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    obs = _obs(index_force=FeedbackExpert.CONTACT_ON_N + 0.05)
    expert.action(obs, t=0)
    assert expert._phase == "slide"
    assert expert._contact_step == 0


def test_slide_transitions_to_tap_when_cursor_on_target(env, expert):
    expert.reset()
    expert._enter_phase("slide")
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    env.set_cursor(448.0, 160.0, over_target=True)
    env.set_target(448.0, 160.0)
    obs = _obs(index_force=0.5)
    expert.action(obs, t=0)
    assert expert._phase == "tap_lift"
    assert expert._tap_subphase == "lift"


def test_tap_subphase_transitions_on_force(env, expert):
    expert.reset()
    expert._enter_phase("tap_lift")
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    # lift clear
    obs = _obs(index_force=0.0)
    expert.action(obs, t=0)
    assert expert._tap_subphase == "descend"
    # press
    obs = _obs(index_force=FeedbackExpert.CONTACT_ON_N + 0.1)
    expert.action(obs, t=1)
    assert expert._tap_subphase == "release"
    # release clear with click -> done
    env.set_click_on_target(1)
    obs = _obs(index_force=0.0)
    expert.action(obs, t=2)
    assert expert._phase == "done"


def test_tap_retries_when_click_missing(env, expert):
    expert.reset()
    expert._enter_phase("tap_lift")
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    env.set_click_on_target(0)
    # lift clear
    expert.action(_obs(index_force=0.0), t=0)
    # press
    expert.action(_obs(index_force=FeedbackExpert.CONTACT_ON_N + 0.1), t=1)
    # release clear but no click -> retry tap_lift
    expert.action(_obs(index_force=0.0), t=2)
    assert expert._phase == "tap_lift"
    assert expert._tap_attempts == 1


# ---------------------------------------------------------------------------
# Joint-limit / action-legality clipping
# ---------------------------------------------------------------------------
def test_actions_respect_joint_limits(env, expert):
    expert.reset()
    expert._enter_phase("approach")
    q_meas = np.zeros(ACTION_DIM, dtype=float)
    lo = q_meas - 0.02
    hi = q_meas + 0.02
    env.joint_limits = (lo, hi)

    def bad_arm_move(qm, mt, wd, ms):
        return np.ones(7, dtype=float) * 10.0

    expert._arm_move = bad_arm_move
    obs = _obs(q=q_meas)
    a = expert.action(obs, t=0)
    q_ref = q_meas + a * env.delta_scale
    assert np.all(q_ref >= lo + 0.005 - 1e-6)
    assert np.all(q_ref <= hi - 0.005 + 1e-6)


def test_actions_respect_normalized_bounds(expert):
    expert.reset()
    expert._enter_phase("approach")
    expert._arm_move = lambda qm, mt, wd, ms: np.ones(7, dtype=float) * 1e6
    obs = _obs()
    a = expert.action(obs, t=0)
    assert np.all(a >= -1.0) and np.all(a <= 1.0)


# ---------------------------------------------------------------------------
# Recovery and timeout/failure
# ---------------------------------------------------------------------------
def test_slide_recovers_lost_contact(expert):
    expert.reset()
    expert._enter_phase("slide")
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    obs = _obs(index_force=0.0)
    a = expert.action(obs, t=0)
    assert expert._phase == "recontact"
    assert np.all(np.isfinite(a))


def test_recontact_presses_downward(env, expert):
    expert.reset()
    expert._enter_phase("recontact")
    captured = []

    def fake_arm_move(qm, measured_tip, world_delta, max_step):
        captured.append(np.asarray(world_delta, dtype=float).copy())
        return expert.env.home_q7.copy()

    expert._arm_move = fake_arm_move
    env.tips["index"].set_pos((0.0, 0.0, 0.735))
    obs = _obs(index_force=0.0)
    expert.action(obs, t=0)
    assert len(captured) == 1
    assert captured[0][2] < -0.001


def test_pad_edge_enters_reanchor_not_tap(env, expert):
    expert.reset()
    expert._enter_phase("slide")
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    # Place fingertip near the right edge with a cursor error that would push
    # it further out.
    half_w = env.scene_config.pad_size[0] / 2
    env.tips["index"].set_pos((half_w - 0.001, 0.0, 0.735))
    env.set_cursor(320, 240)
    env.set_target(640, 240)  # large +x error
    obs = _obs(index_force=0.5)
    expert.action(obs, t=0)
    assert expert._phase == "reanchor"


def test_excessive_recontact_enters_fail(expert):
    expert.reset()
    expert._enter_phase("slide")
    expert._recontact_attempts = FeedbackExpert.MAX_RECONTACT_ATTEMPTS
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    obs = _obs(index_force=0.0)
    expert.action(obs, t=0)
    assert expert._phase == "fail"


def test_approach_timeout_enters_fail(expert):
    expert.reset()
    expert._enter_phase("approach")
    expert._arm_move = lambda qm, mt, wd, ms: expert.env.home_q7.copy()
    obs = _obs(index_force=0.0)
    expert._step = FeedbackExpert.APPROACH_TIMEOUT_STEPS + 1
    expert._phase_enter_step = 0
    a = expert.action(obs, t=expert._step)
    assert expert._phase == "fail"
    assert np.all(a == 0.0)


# ---------------------------------------------------------------------------
# TeacherEnabledEnv wrapper
# ---------------------------------------------------------------------------
class _MiniEnv:
    """Tiny env with the subset TeacherEnabledEnv delegates to."""

    dt = 1.0 / 120.0
    SUBSTEPS = 8

    def __init__(self):
        self.scene_config = _FakeSceneConfig()
        self.home_q7 = np.zeros(7, dtype=float)
        self.action_names = [f"panda_joint{i}" for i in range(1, 8)] + ["j" + str(i) for i in range(22)]
        lo = np.full(ACTION_DIM, -1.0, dtype=float)
        hi = np.full(ACTION_DIM, 1.0, dtype=float)
        self.joint_limits = (lo, hi)
        self.delta_scale = np.full(ACTION_DIM, 0.05, dtype=np.float32)
        self.adapter = _FakeAdapter()
        self.gui = type("G", (), {"width": 640, "height": 480})()
        self.tips = {"index": _FakeTip()}
        self._cursor = (320.0, 240.0)
        self._target = (448.0, 160.0)
        self._cursor_over_target = False
        self._click_on_target = 0
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False

    def reset(self, seed=0, options=None):
        self.reset_calls += 1
        obs = _obs()
        return obs, {"provenance": {}}

    def step(self, action):
        self.step_calls += 1
        return _obs(), 0.0, False, False, {}

    def _privileged(self):
        return {
            "oracle_cursor_px": self._cursor,
            "oracle_target_px": self._target,
            "oracle_cursor_over_target": self._cursor_over_target,
            "oracle_click_on_target": self._click_on_target,
        }

    def _pad_local_xy(self, p_world):
        return float(p_world[0]), float(p_world[1])

    def close(self):
        self.closed = True


def test_teacher_enabled_env_calls_teacher_for_expert_action():
    inner = _MiniEnv()
    wrapped = TeacherEnabledEnv(inner)
    wrapped.reset()
    wrapped.teacher._arm_move = lambda qm, mt, wd, ms: inner.home_q7.copy()
    a = wrapped.expert_action()
    assert a.shape == (ACTION_DIM,)
    assert np.all(np.isfinite(a))
    assert a.dtype == np.float32
    wrapped.step(np.zeros(ACTION_DIM, dtype=np.float32))
    a2 = wrapped.expert_action()
    assert np.all(np.isfinite(a2))
    wrapped.close()
    assert inner.closed


def test_teacher_enabled_env_reset_resets_teacher():
    inner = _MiniEnv()
    wrapped = TeacherEnabledEnv(inner)
    wrapped.teacher._enter_phase("slide")
    wrapped.reset()
    assert wrapped.teacher._phase == "clear"


def test_teacher_enabled_env_factory_accepts_config():
    """The wrapper can be used as a BC factory taking a config dict."""
    # We cannot construct a real SharpaEnv here, so just validate the
    # signature path: passing config=None without env must raise.
    with pytest.raises(ValueError):
        TeacherEnabledEnv()


# ---------------------------------------------------------------------------
# Coordinate frame correctness (catches the world/base double-transform bug)
# ---------------------------------------------------------------------------
def test_world_to_base_and_back_are_inverses(env, expert):
    p_world = np.array([0.05, -0.12, 0.81])
    p_base = expert._world_to_base(p_world)
    p_back = expert._base_to_world(p_base)
    np.testing.assert_allclose(p_back, p_world, atol=1e-9)


def test_arm_move_converts_base_tip_to_world_before_solving_ik(env, expert):
    """The old bug added a world delta to a base-frame point and then
    transformed the sum with _world_to_base again.  The fixed code converts
    the zero-model tip to world first, adds the world delta, and passes that
    world target to _solve_ik (which itself converts world->base)."""
    expert.reset()
    q7 = np.array(env.home_q7, dtype=float)
    q_meas = np.zeros(ACTION_DIM, dtype=float)
    q_meas[:7] = q7

    zero_tip_base = np.array([0.050, -0.020, 0.110])
    zero_tip_world = np.array([0.120, 0.230, 0.830])
    expert._fk_zero_hand_tip = lambda q: zero_tip_base.copy()
    expert._base_to_world = lambda p: zero_tip_world.copy()
    captured = []
    expert._solve_ik = lambda target, seeds=(): captured.append(np.asarray(target, dtype=float).copy()) or q7.copy()

    delta = np.array([0.003, -0.002, 0.001])
    expert._arm_move(q_meas, zero_tip_world, delta, 1.0)

    assert len(captured) == 1
    np.testing.assert_allclose(captured[0], zero_tip_world + delta, atol=1e-9)


def test_arm_move_zero_delta_returns_seed_q(env, expert):
    """FK/IK round-trip with mocked identity IK: zero delta keeps q7."""
    expert.reset()
    q7 = np.array(env.home_q7, dtype=float)
    q_meas = np.zeros(ACTION_DIM, dtype=float)
    q_meas[:7] = q7
    expert._fk_zero_hand_tip = lambda q: np.array([0.0, 0.0, 0.8])
    expert._base_to_world = lambda p: np.array([0.42, 0.30, 0.80 + 0.72])
    expert._solve_ik = lambda target, seeds=(): seeds[0] if seeds else q7.copy()
    q_arm = expert._arm_move(q_meas, np.zeros(3), np.array([0.0, 0.0, 0.0]), 1.0)
    np.testing.assert_allclose(q_arm, q7, atol=1e-9)


def test_approach_contact_before_ik_failure_transitions_to_slide(env, expert):
    """Actual contact feedback must win over an IK failure early return."""
    expert.reset()
    expert._enter_phase("approach")
    expert._arm_move = lambda qm, mt, wd, ms: None  # IK fails
    obs = _obs(index_force=FeedbackExpert.CONTACT_ON_N + 0.05)
    a = expert.action(obs, t=0)
    assert expert._phase == "slide"
    assert expert._contact_step == 0
    assert np.all(a == 0.0)
