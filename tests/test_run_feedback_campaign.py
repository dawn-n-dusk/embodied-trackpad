"""Focused tests for run_feedback_campaign timestamp + truncation honesty."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.sharpa_learning.data import EpisodeWriter
from embodied_trackpad.sharpa_learning.env import ACTION_DIM
from embodied_trackpad.sharpa_learning import run_feedback_campaign as rfc
from embodied_trackpad.sharpa_learning.run_feedback_campaign import run_feedback_episode


@pytest.fixture(autouse=True)
def stub_video_encoding(monkeypatch):
    """Video encoding needs PIL/ffmpeg which are not in the local test env."""
    monkeypatch.setattr(rfc, "_encode_video_from_npz",
                        lambda npz_path, out_path, key, fps=15:
                        {"returncode": 0, "frames": 4, "key": key})


class _MockContactBody:
    def __init__(self):
        self.max_force = 0.0
        self.frames_valid = 1


class _MockEnv:
    """Minimal mock implementing the SharpaEnv surface used by the campaign."""

    def __init__(self, step_returns=None, reset_sim_time=0.052):
        self.provenance = {
            "action_names": [f"j{i}" for i in range(ACTION_DIM)],
            "delta_scale": [0.025] * 7 + [0.05] * (ACTION_DIM - 7),
            "contact_backend": "PhysX mock",
        }
        self._step_returns = step_returns or []
        self._step_index = 0
        self._reset_sim_time = reset_sim_time
        self.rep = type("Rep", (), {"orchestrator": type("Orchestrator", (), {"step": lambda *a, **k: None})()})()
        self.products = {}

    def reset(self, seed=0, options=None):
        obs = {
            "screen_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
            "wrist_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
            "proprio": np.zeros(58, dtype=np.float32),
            "touch": np.zeros(5, dtype=np.float32),
            "touch_valid": np.zeros(5, dtype=np.uint8),
        }
        info = {
            "sim_time": self._reset_sim_time,
            "oracle_target_px": (111.0, 222.0),
            "oracle_cursor_px": (320.0, 240.0),
            "contacts": {},
        }
        self._step_index = 0
        return obs, info

    def step(self, action):
        if self._step_index >= len(self._step_returns):
            ret = (False, False)
        else:
            ret = self._step_returns[self._step_index]
        self._step_index += 1
        terminated, truncated = ret
        obs = {
            "screen_rgb": np.zeros((128, 128, 3), dtype=np.uint8) + self._step_index,
            "wrist_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
            "proprio": np.zeros(58, dtype=np.float32),
            "touch": np.zeros(5, dtype=np.float32),
            "touch_valid": np.zeros(5, dtype=np.uint8),
        }
        info = {
            "sim_time": self._reset_sim_time + self._step_index * 0.066,
            "executed_q_ref": np.zeros(ACTION_DIM, dtype=np.float32),
            "oracle_target_px": (111.0, 222.0),
            "oracle_click_on_target": 1 if terminated else 0,
            "oracle_click_registered": 1 if terminated else 0,
            "contacts": {},
        }
        return obs, 0.0, terminated, truncated, info

    def close(self):
        pass


class _FailExpert:
    """Expert that declares fail immediately."""

    def __init__(self, env=None):
        self._phase = "approach"
        self._tap_subphase = None
        self.env = env or type("E", (), {"_privileged": lambda self: {
            "oracle_cursor_px": (0.0, 0.0),
            "oracle_target_px": (0.0, 0.0),
            "oracle_cursor_over_target": False,
            "oracle_click_on_target": 0,
        }})()

    def reset(self):
        self._phase = "approach"

    def action(self, obs, t):
        self._phase = "fail"
        return np.zeros(ACTION_DIM, dtype=np.float32)


def test_reset_sim_time_used_for_first_observation(tmp_path):
    env = _MockEnv(reset_sim_time=0.0517)
    expert = _FailExpert()
    ep_dir = tmp_path / "ep"
    manifest = run_feedback_episode(env, expert, ep_dir, "ep0", seed=0,
                                    max_decisions=400)
    with np.load(ep_dir / "trajectory.npz") as z:
        assert z["sim_time"][0] == pytest.approx(0.0517, abs=1e-6)


def test_expert_fail_records_one_real_truncated_step(tmp_path):
    env = _MockEnv(reset_sim_time=0.0517)
    expert = _FailExpert()
    ep_dir = tmp_path / "ep"
    manifest = run_feedback_episode(env, expert, ep_dir, "ep0", seed=0,
                                    max_decisions=400)
    assert manifest["outcome"] == "truncated"
    assert manifest["failure_reason"] == "expert_phase_fail"
    assert manifest["training_eligible_for_bc"] is False
    # The fail branch must perform exactly one real env.step, not invent one.
    assert env._step_index == 1
    with np.load(ep_dir / "trajectory.npz") as z:
        T = int(z["action"].shape[0])
        assert T == 1
        assert bool(z["truncated"][-1]) is True
        assert bool(z["terminated"][-1]) is False
        # observations are T+1 and timestamps must strictly increase
        assert z["sim_time"].shape[0] == T + 1
        assert z["sim_time"][1] > z["sim_time"][0]


def test_max_decision_cap_marks_last_step_truncated(tmp_path):
    env = _MockEnv(step_returns=[(False, False)] * 10, reset_sim_time=0.05)
    expert = type("HoldExpert", (), {
        "reset": lambda self: None,
        "action": lambda self, obs, t: np.zeros(ACTION_DIM, dtype=np.float32),
        "_phase": "slide",
        "_tap_subphase": None,
        "env": type("E", (), {"_privileged": lambda self: {
            "oracle_cursor_px": (0.0, 0.0),
            "oracle_target_px": (0.0, 0.0),
            "oracle_cursor_over_target": False,
            "oracle_click_on_target": 0,
        }})(),
    })()
    ep_dir = tmp_path / "ep"
    manifest = run_feedback_episode(env, expert, ep_dir, "ep0", seed=0,
                                    max_decisions=3)
    assert manifest["outcome"] == "truncated"
    assert env._step_index == 3
    with np.load(ep_dir / "trajectory.npz") as z:
        assert z["action"].shape[0] == 3
        assert bool(z["truncated"][-1]) is True
        assert bool(z["terminated"][-1]) is False
        # No invented observations and timestamps strictly increase.
        assert z["sim_time"].shape[0] == 4
        assert np.all(np.diff(z["sim_time"]) > 0)


class _ImmediateSuccessExpert:
    """Mock expert that completes on the first env step."""

    def __init__(self, env=None):
        self.env = self
        self._phase = "slide"
        self._tap_subphase = None

    def reset(self):
        pass

    def _privileged(self):
        return {
            "oracle_cursor_px": (0.0, 0.0),
            "oracle_target_px": (0.0, 0.0),
            "oracle_cursor_over_target": False,
            "oracle_click_on_target": 0,
        }

    def action(self, obs, t):
        return np.zeros(ACTION_DIM, dtype=np.float32)


def test_campaign_ids_unique_across_same_second_calls(tmp_path, monkeypatch):
    """Distinct run_campaign invocations in the same second must never reuse IDs."""
    fixed_time = "20261118T051219Z"

    def fake_strftime(fmt, gmtime):
        return fixed_time

    monkeypatch.setattr(time, "strftime", fake_strftime)
    monkeypatch.setattr(rfc, "SharpaEnv", lambda config: _MockEnv(
        step_returns=[(True, False)], reset_sim_time=0.05))
    monkeypatch.setattr(rfc, "FeedbackExpert", _ImmediateSuccessExpert)

    out1 = tmp_path / "run1"
    data1 = tmp_path / "data1"
    out2 = tmp_path / "run2"
    data2 = tmp_path / "data2"

    res1 = rfc.run_campaign("dummy", str(out1), str(data1),
                            seed=0, n_episodes=2, max_decisions=5)
    res2 = rfc.run_campaign("dummy", str(out2), str(data2),
                            seed=0, n_episodes=2, max_decisions=5)

    ids1 = [e["episode_id"] for e in res1["episodes"]]
    ids2 = [e["episode_id"] for e in res2["episodes"]]

    assert len(ids1) == 2 and len(set(ids1)) == 2
    assert len(ids2) == 2 and len(set(ids2)) == 2
    assert set(ids1).isdisjoint(ids2), f"colliding IDs: {set(ids1) & set(ids2)}"

    # Each dataset index and manifest agrees on its own episode IDs.
    for data_dir, out_dir, ids in [(data1, out1, ids1), (data2, out2, ids2)]:
        index = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
        index_ids = [e["episode_id"] for e in index["episodes"]]
        assert index_ids == ids
        for eid in ids:
            manifest = json.loads((out_dir / eid / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["episode_id"] == eid
