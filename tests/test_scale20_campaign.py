"""Focused tests for the 20x data+BC campaign additions.

Lossless NPZ compression option (data.py), atomic checkpoint serialization and
the fixed-seed raw-trace evaluator path (bc_ppo.py). Software fixtures only.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_trackpad.sharpa_learning.bc_ppo import (
    ACTION_DIM,
    IMAGE_SIZE,
    PROPRIO_DIM,
    TOUCH_DIM,
    TOUCH_VALID_DIM,
    HistoryActorCritic,
    ModelConfig,
    evaluate_policy,
    load_checkpoint,
    save_checkpoint,
)
from embodied_trackpad.sharpa_learning.data import EpisodeWriter, SCHEMA


def _obs(T, rng):
    return {
        "screen_rgb": rng.integers(0, 256, size=(T + 1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "wrist_rgb": rng.integers(0, 256, size=(T + 1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8),
        "proprio": rng.normal(size=(T + 1, PROPRIO_DIM)).astype(np.float32),
        "touch": rng.normal(size=(T + 1, TOUCH_DIM)).astype(np.float32),
        "touch_valid": rng.integers(0, 2, size=(T + 1, TOUCH_VALID_DIM), dtype=np.uint8),
    }


def _write_episode(tmp_path: Path, name: str, compress: bool) -> Path:
    rng = np.random.default_rng(7)
    ep_dir = tmp_path / name
    T = 6
    w = EpisodeWriter(
        ep_dir, name,
        {"action_names": [f"a{i}" for i in range(ACTION_DIM)],
         "delta_scale": [0.01] * ACTION_DIM,
         "joint_limits": [[-1.0] * ACTION_DIM, [1.0] * ACTION_DIM],
         "contact_backend": "physx measured", "stage": "test"},
        compress=compress)
    obs = _obs(T, rng)
    for t in range(T + 1):
        w.add_observation({k: obs[k][t] for k in obs}, sim_time=0.1 * t)
    for t in range(T):
        info = {"executed_q_ref": np.zeros(ACTION_DIM, dtype=np.float32)}
        w.add_step(np.zeros(ACTION_DIM, dtype=np.float32), info, 0.0,
                   t == T - 1, False)
    w.finalize({"split": "train", "source_kind": "feedback_expert_v1",
                "training_eligible_for_bc": True,
                "measured_contact_backend": "physx measured",
                "config": {"seed": 1}, "action_names": w.provenance["action_names"],
                "delta_scale": w.provenance["delta_scale"],
                "joint_limits": w.provenance["joint_limits"]})
    return ep_dir


def test_npz_compression_roundtrip_lossless(tmp_path):
    ep_plain = _write_episode(tmp_path, "ep_plain", compress=False)
    ep_comp = _write_episode(tmp_path, "ep_comp", compress=True)
    with np.load(ep_plain / "trajectory.npz") as a, \
         np.load(ep_comp / "trajectory.npz") as b:
        assert a.files == b.files
        for k in a.files:
            assert a[k].dtype == b[k].dtype
            assert a[k].shape == b[k].shape
            assert np.array_equal(a[k], b[k]), f"array {k} changed by compression"


def test_compression_default_preserves_old_api(tmp_path):
    # Default path must remain a plain (uncompressed) np.savez archive.
    ep = _write_episode(tmp_path, "ep_default", compress=False)
    with open(ep / "trajectory.npz", "rb") as fh:
        assert fh.read(2) == b"PK"
    w = EpisodeWriter(tmp_path / "ep_api", "ep_api", {})
    assert w.compress is False


def test_save_checkpoint_atomic_and_loadable(tmp_path):
    cfg = ModelConfig(image_hw=(128, 128), conv_channels=(4, 8),
                      conv_feature_dim=16, vector_hidden=16, fused_hidden=32,
                      gru_hidden=32, gru_layers=1)
    model = HistoryActorCritic(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ckpt = tmp_path / "sub" / "bc_step10.pt"
    save_checkpoint(ckpt, model, opt, cfg, "bc", 10, {"seed": 42})
    assert ckpt.exists()
    assert not (tmp_path / "sub" / "bc_step10.pt.tmp").exists()
    model2, payload = load_checkpoint(ckpt, "cpu")
    assert payload["global_step"] == 10
    for k, v in model.state_dict().items():
        assert torch.equal(v, model2.state_dict()[k])


class _TraceEnv:
    """Deterministic stub: success at step 2, oracle fields in info."""

    def __init__(self, report_probe=None):
        self._t = 0
        self.report_probe = report_probe
        self.closed = False

    def reset(self, seed=None, options=None):
        self._t = 0
        self.seed = seed
        rng = np.random.default_rng(0)
        return ({k: v[0] for k, v in _obs(4, rng).items()}, {})

    def step(self, action):
        self._t += 1
        rng = np.random.default_rng(self._t)
        obs = {k: v[0] for k, v in _obs(4, rng).items()}
        terminated = self._t >= 2
        info = {
            "sim_time": float(self._t),
            "oracle_cursor_px": (300.0 + self._t, 200.0),
            "oracle_target_px": (310.0, 210.0),
            "oracle_cursor_over_target": terminated,
            "oracle_click_registered": 1 if terminated else 0,
            "oracle_click_on_target": 1 if terminated else 0,
            "contacts": {"index": [{"force_n": 0.5}]},
            "success": terminated,
            "events": ["tap"],
        }
        return obs, 0.0, terminated, False, info

    def close(self):
        self.closed = True
        if self.report_probe is not None:
            # The report must exist BEFORE the env is closed.
            assert self.report_probe.exists(), "report written after env close"


def _small_model_ckpt(tmp_path):
    cfg = ModelConfig(image_hw=(128, 128), conv_channels=(4, 8),
                      conv_feature_dim=16, vector_hidden=16, fused_hidden=32,
                      gru_hidden=32, gru_layers=1)
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "bc_step5.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 5, {"seed": 42})
    return ckpt


def test_evaluate_policy_fixed_seeds_trace_and_no_overwrite(tmp_path):
    ckpt = _small_model_ckpt(tmp_path)
    out_dir = tmp_path / "eval_out"
    out_dir.mkdir()
    env_holder = {}

    def factory(**kwargs):
        env = _TraceEnv(report_probe=None)
        env_holder["env"] = env
        return env

    seeds = [90000000, 90000001, 90000002]
    # Probe report-before-close on the second call (output file path pattern).
    report = evaluate_policy(
        factory, ckpt, episodes=3, seed=0, seeds=seeds,
        max_steps=5, output=out_dir, trace_dir=tmp_path / "traces",
        trace_episodes=2)
    assert report["seeds"] == seeds
    assert report["episodes_attempted"] == 3
    assert report["successes"] == 3
    tdir = tmp_path / "traces"
    traces = sorted(tdir.glob("episode_*"))
    assert len(traces) == 2
    with np.load(traces[0] / "raw_trace.npz") as z:
        assert {"screen_rgb", "wrist_rgb", "proprio", "touch", "touch_valid"} <= set(z.files)
        assert z["screen_rgb"].shape[0] == 3  # 2 steps -> T+1 = 3 frames
    meta = json.loads((traces[0] / "trace_meta.json").read_text())
    assert meta["seed"] == 90000000
    assert meta["privileged_evaluator_trace"] is True
    assert meta["steps"][-1]["click_on_target"] == 1
    assert meta["steps"][-1]["contacts"]["index"][0]["force_n"] == 0.5
    assert env_holder["env"].closed is True

    # Re-run: existing raw files must not be overwritten.
    before = (traces[0] / "raw_trace.npz").read_bytes()
    report2 = evaluate_policy(
        factory, ckpt, episodes=3, seed=0, seeds=seeds,
        max_steps=5, output=out_dir, trace_dir=tmp_path / "traces",
        trace_episodes=2)
    assert report2["traces"][0]["skipped_existing"] is True
    assert (traces[0] / "raw_trace.npz").read_bytes() == before


def test_evaluate_policy_copies_obs_buffers_and_records_terminal_flags(tmp_path):
    ckpt = _small_model_ckpt(tmp_path)

    class ReuseEnv:
        """Backend that reuses one buffer per observation key in-place."""

        def __init__(self):
            self._t = 0
            rng = np.random.default_rng(0)
            self._obs = {k: v[0] for k, v in _obs(8, rng).items()}
            self.closed = False

        def reset(self, seed=None, options=None):
            self._t = 0
            return self._obs, {}

        def step(self, action):
            self._t += 1
            for k in self._obs:
                self._obs[k] = self._obs[k]  # same object, mutated below
            self._obs["proprio"][:] = float(self._t)
            terminated = self._t >= 2
            info = {"sim_time": float(self._t), "oracle_cursor_px": (1.0, 2.0),
                    "oracle_target_px": (3.0, 4.0), "success": terminated,
                    "executed_q_ref": np.full(ACTION_DIM, self._t, dtype=np.float32)}
            return self._obs, 0.0, terminated, False, info

        def close(self):
            self.closed = True

    def factory(**kwargs):
        return ReuseEnv()

    report = evaluate_policy(factory, ckpt, episodes=1, seed=0, seeds=[123],
                             max_steps=6, trace_dir=tmp_path / "traces2",
                             trace_episodes=1)
    ep_dir = tmp_path / "traces2" / "episode_000_seed123"
    with np.load(ep_dir / "raw_trace.npz") as z:
        # Buffer reuse would collapse every post-reset frame to the final value.
        assert [float(v) for v in z["proprio"][1:, 0]] == [1.0, 2.0]
    meta = json.loads((ep_dir / "trace_meta.json").read_text())
    assert meta["steps"][-1]["terminated"] is True
    assert meta["steps"][-1]["executed_q_ref"][0] == 2.0
    assert list(meta["steps"][-1]["cursor_px"]) == [1.0, 2.0]
    assert report["traces"][0]["video_screen"].endswith("screen_rgb.mp4")
    assert report["traces"][0]["video_wrist"].endswith("wrist_rgb.mp4")


def test_evaluate_policy_time_budget_stops_at_boundary(tmp_path):
    ckpt = _small_model_ckpt(tmp_path)

    class SlowEnv(_TraceEnv):
        def step(self, action):
            import time as _t
            _t.sleep(0.05)
            return super().step(action)

    def factory(**kwargs):
        return SlowEnv()

    report = evaluate_policy(factory, ckpt, episodes=10, seed=0, max_steps=5,
                             time_budget_s=0.12)
    assert report["stop_reason"] == "time_budget"
    assert 0 < report["episodes_attempted"] < 10


def test_run_campaign_forwards_compress_to_writer(monkeypatch, tmp_path):
    """Public run_campaign must thread compress= into run_feedback_episode."""
    import embodied_trackpad.sharpa_learning.run_feedback_campaign as rfc

    captured: list[dict] = []

    class _FakeEnv:
        provenance = {"action_names": [f"a{i}" for i in range(ACTION_DIM)],
                      "delta_scale": [0.01] * ACTION_DIM}

        def __init__(self, cfg):
            pass

        def close(self):
            pass

    def _fake_episode(env, expert, episode_dir, episode_id, seed,
                      max_decisions, source_manifest_path=None, split="train",
                      compress=False):
        captured.append({"compress": compress, "seed": seed, "split": split})
        return {"split": split, "source_kind": "feedback_expert_v1",
                "training_eligible_for_bc": True,
                "action_names": self_action_names(), "delta_scale": [0.01] * ACTION_DIM}

    def self_action_names():
        return [f"a{i}" for i in range(ACTION_DIM)]

    monkeypatch.setattr(rfc, "SharpaEnv", _FakeEnv)
    monkeypatch.setattr(rfc, "FeedbackExpert", lambda env: object())
    monkeypatch.setattr(rfc, "run_feedback_episode", _fake_episode)

    rfc.run_campaign(stage="stage", out_dir=str(tmp_path / "out"),
                     data_root=str(tmp_path / "data"), gpu_index=0, seed=100000,
                     n_episodes=2, max_decisions=10, compress=True)
    assert [c["compress"] for c in captured] == [True, True]
    captured.clear()
    rfc.run_campaign(stage="stage", out_dir=str(tmp_path / "out2"),
                     data_root=str(tmp_path / "data2"), gpu_index=0, seed=100000,
                     n_episodes=1, max_decisions=10)
    assert [c["compress"] for c in captured] == [False]


def test_evaluate_policy_laststep_success_is_not_truncated(tmp_path):
    """Env terminates exactly on the final allowed step: the evaluator cap
    must NOT be set (terminated=True, truncated=False, no failure reason)."""
    ckpt = _small_model_ckpt(tmp_path)

    class LastStepSuccessEnv:
        def __init__(self):
            self._t = 0
            self.closed = False

        def reset(self, seed=None, options=None):
            self._t = 0
            rng = np.random.default_rng(0)
            return {k: v[0] for k, v in _obs(4, rng).items()}, {}

        def step(self, action):
            self._t += 1
            rng = np.random.default_rng(self._t)
            obs = {k: v[0] for k, v in _obs(4, rng).items()}
            terminated = self._t >= 3  # exactly max_steps -> success at cap
            info = {"sim_time": float(self._t),
                    "oracle_cursor_px": (1.0, 2.0),
                    "oracle_target_px": (3.0, 4.0),
                    "success": terminated}
            return obs, 0.0, terminated, False, info

        def close(self):
            self.closed = True

    def factory(**kwargs):
        return LastStepSuccessEnv()

    report = evaluate_policy(factory, ckpt, episodes=1, seed=0, seeds=[7],
                             max_steps=3, trace_dir=tmp_path / "traces3",
                             trace_episodes=1)
    row = report["rows"][0]
    assert row["terminated"] is True
    assert row["truncated"] is False
    assert row["failure_reason"] is None
    assert row["success"] is True
    ep_dir = tmp_path / "traces3" / "episode_000_seed7"
    meta = json.loads((ep_dir / "trace_meta.json").read_text())
    last = meta["steps"][-1]
    assert last["terminated"] is True
    assert last["truncated"] is False
    assert last["evaluator_step_cap"] is False
    assert last["failure_reason"] is None


def test_evaluate_policy_step_cap_sets_eval_step_limit(tmp_path):
    """Env never terminates: last record is truncated with the cap flag and
    failure_reason=eval_step_limit."""
    ckpt = _small_model_ckpt(tmp_path)

    class NeverDoneEnv:
        def __init__(self):
            self._t = 0
            self.closed = False

        def reset(self, seed=None, options=None):
            self._t = 0
            rng = np.random.default_rng(0)
            return {k: v[0] for k, v in _obs(8, rng).items()}, {}

        def step(self, action):
            self._t += 1
            rng = np.random.default_rng(self._t)
            obs = {k: v[0] for k, v in _obs(8, rng).items()}
            info = {"sim_time": float(self._t),
                    "oracle_cursor_px": (1.0, 2.0),
                    "oracle_target_px": (3.0, 4.0),
                    "success": False}
            return obs, 0.0, False, False, info

        def close(self):
            self.closed = True

    def factory(**kwargs):
        return NeverDoneEnv()

    report = evaluate_policy(factory, ckpt, episodes=1, seed=0, seeds=[8],
                             max_steps=4, trace_dir=tmp_path / "traces4",
                             trace_episodes=1)
    row = report["rows"][0]
    assert row["terminated"] is False
    assert row["truncated"] is True
    assert row["failure_reason"] == "eval_step_limit"
    ep_dir = tmp_path / "traces4" / "episode_000_seed8"
    meta = json.loads((ep_dir / "trace_meta.json").read_text())
    last = meta["steps"][-1]
    assert last["truncated"] is True
    assert last["evaluator_step_cap"] is True
    assert last["failure_reason"] == "eval_step_limit"


def test_episode_writer_refuses_npz_overwrite(tmp_path):
    ep_dir = _write_episode(tmp_path, "ep_once", compress=False)
    # Simulate a crashed first attempt that left the NPZ but no manifest:
    # the writer must still refuse to clobber the raw trajectory.
    (ep_dir / "manifest.json").unlink()
    rng = np.random.default_rng(7)
    w = EpisodeWriter(
        ep_dir, "ep_once",
        {"action_names": [f"a{i}" for i in range(ACTION_DIM)],
         "delta_scale": [0.01] * ACTION_DIM})
    obs = _obs(6, rng)
    for t in range(7):
        w.add_observation({k: obs[k][t] for k in obs}, sim_time=0.1 * t)
    for t in range(6):
        w.add_step(np.zeros(ACTION_DIM, dtype=np.float32),
                   {"executed_q_ref": np.zeros(ACTION_DIM, dtype=np.float32)},
                   0.0, t == 5, False)
    with pytest.raises(FileExistsError):
        w.finalize({"split": "train"})


def test_run_campaign_explicit_seed_list(monkeypatch, tmp_path):
    """Explicit seeds list must reach run_feedback_episode verbatim (the
    launch receipt then matches the actual reset seeds)."""
    import embodied_trackpad.sharpa_learning.run_feedback_campaign as rfc

    captured: list[int] = []

    class _FakeEnv:
        provenance = {"action_names": [f"a{i}" for i in range(ACTION_DIM)],
                      "delta_scale": [0.01] * ACTION_DIM}

        def __init__(self, cfg):
            pass

        def close(self):
            pass

    def _fake_episode(env, expert, episode_dir, episode_id, seed,
                      max_decisions, source_manifest_path=None, split="train",
                      compress=False):
        captured.append(int(seed))
        return {"split": split, "source_kind": "feedback_expert_v1",
                "training_eligible_for_bc": True,
                "action_names": [f"a{i}" for i in range(ACTION_DIM)],
                "delta_scale": [0.01] * ACTION_DIM}

    monkeypatch.setattr(rfc, "SharpaEnv", _FakeEnv)
    monkeypatch.setattr(rfc, "FeedbackExpert", lambda env: object())
    monkeypatch.setattr(rfc, "run_feedback_episode", _fake_episode)

    seeds = [100000, 100005, 100009, 100017]
    rfc.run_campaign(stage="stage", out_dir=str(tmp_path / "out3"),
                     data_root=str(tmp_path / "data3"), gpu_index=0,
                     seed=1, n_episodes=1, max_decisions=10, seeds=seeds)
    assert captured == seeds


def test_run_campaign_cli_seeds_forwarding(monkeypatch, tmp_path, capsys):
    """The public CLI must parse --seeds and forward the exact list."""
    import embodied_trackpad.sharpa_learning.run_feedback_campaign as rfc

    captured: list[list[int]] = []

    def _fake_run_campaign(**kwargs):
        captured.append(kwargs["seeds"])
        return {"episodes": [], "index": str(tmp_path / "dataset.json")}

    monkeypatch.setattr(rfc, "run_campaign", _fake_run_campaign)
    rc = rfc.main(["--stage", "s", "--out-dir", str(tmp_path / "o"),
                   "--data-root", str(tmp_path / "d"),
                   "--seeds", "100000,100001,100002"])
    assert rc == 0
    assert captured == [[100000, 100001, 100002]]
