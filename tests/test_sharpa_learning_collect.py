"""Focused collector contract tests (no Isaac Sim)."""
from __future__ import annotations

import numpy as np
import pytest

from embodied_trackpad.sharpa_learning.collect import collect_episode
from embodied_trackpad.sharpa_learning.data import EpisodeWriter


class _MockEnv:
    """Deterministic mock with exactly T transitions."""

    ACTION_DIM = 29
    SUBSTEPS = 8
    dt = 1.0 / 120.0

    def __init__(self, terminate_on_step: int | None = None,
                 click_on_target: bool = False):
        self._step = 0
        self.terminate_on = terminate_on_step
        self.click_on_target = click_on_target
        self.delta_scale = np.ones(self.ACTION_DIM, dtype=np.float32) * 0.05

    def reset(self, seed=0, options=None):
        self._step = 0
        obs = {
            "screen_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
            "proprio": np.zeros(58, dtype=np.float32),
            "touch": np.zeros(5, dtype=np.float32),
            "touch_valid": np.zeros(5, dtype=np.uint8),
        }
        info = {
            "provenance": {
                "action_names": [f"j{i}" for i in range(self.ACTION_DIM)],
                "delta_scale": [0.05] * self.ACTION_DIM,
                "contact_backend": "mock",
            },
            "oracle_click_on_target": 0,
            "oracle_click_registered": 0,
        }
        return obs, info

    def step(self, action):
        self._step += 1
        obs = {
            "screen_rgb": np.zeros((128, 128, 3), dtype=np.uint8) + self._step,
            "wrist_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
            "proprio": np.zeros(58, dtype=np.float32),
            "touch": np.zeros(5, dtype=np.float32),
            "touch_valid": np.zeros(5, dtype=np.uint8),
        }
        terminated = (self.terminate_on is not None
                      and self._step == self.terminate_on)
        truncated = False
        reward = 1.0 if (terminated and self.click_on_target) else -0.1
        info = {
            "executed_q_ref": np.zeros(self.ACTION_DIM, dtype=np.float32),
            "oracle_click_on_target": 1 if (terminated and self.click_on_target) else 0,
            "oracle_click_registered": 1 if (terminated and self.click_on_target) else 0,
            "sim_time": self._step * self.SUBSTEPS * self.dt,
        }
        return obs, reward, terminated, truncated, info

    def expert_action(self):
        return np.zeros(self.ACTION_DIM, dtype=np.float32)

    def close(self):
        pass


class _CountingExpert:
    def __init__(self):
        self.calls = 0

    def action(self, obs, t):
        self.calls += 1
        return np.zeros(29, dtype=np.float32)

    def reset(self):
        self.calls = 0


def test_collector_writes_t_plus_one_observations_and_times(tmp_path):
    env = _MockEnv(terminate_on_step=2, click_on_target=True)
    expert = _CountingExpert()
    out = tmp_path / "ep"
    manifest = collect_episode(env, expert, str(out), "ep0", max_decisions=10)
    assert manifest["outcome"] == "success"
    assert manifest["training_eligible_for_bc"] is True

    with np.load(out / "trajectory.npz") as z:
        T = int(manifest["length_T"])
        assert T == 2
        assert z["screen_rgb"].shape[0] == T + 1
        assert z["sim_time"].shape[0] == T + 1
        assert z["action"].shape[0] == T
        assert z["reward"].shape[0] == T
        assert z["teacher_valid"].shape[0] == T
        # observations are distinct post-step frames
        assert not np.array_equal(z["screen_rgb"][0], z["screen_rgb"][1])
        # sim_time starts at 0 and advances by substeps*dt per transition
        assert z["sim_time"][0] == pytest.approx(0.0)
        assert z["sim_time"][1] == pytest.approx(env.SUBSTEPS * env.dt)


def test_failed_termination_not_labelled_success(tmp_path):
    env = _MockEnv(terminate_on_step=1, click_on_target=False)
    expert = _CountingExpert()
    out = tmp_path / "ep"
    manifest = collect_episode(env, expert, str(out), "ep1", max_decisions=10)
    assert manifest["outcome"] != "success"
    assert manifest["training_eligible_for_bc"] is False


def test_truncation_flag_recorded_on_max_decisions(tmp_path):
    env = _MockEnv(terminate_on_step=None, click_on_target=False)
    expert = _CountingExpert()
    out = tmp_path / "ep"
    manifest = collect_episode(env, expert, str(out), "ep2", max_decisions=3)
    assert manifest["outcome"] == "truncated"
    with np.load(out / "trajectory.npz") as z:
        assert bool(z["truncated"][-1]) is True
        assert z["truncated"].shape[0] == manifest["length_T"]


def test_writer_refuses_overwrite(tmp_path):
    out = tmp_path / "ep"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        EpisodeWriter(out, "ep", {})
