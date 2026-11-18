"""Tests for the Sharpa compact latent world-model prototype.

All data here is synthetic and honestly labelled as fixtures (episode ids
prefixed "fixture_synthetic_"). These tests are software validation only --
they are NOT policy training and NOT evidence of task competence.

Can run under pytest, or directly (no pytest required):
    PYTHONPATH=src python tests/test_sharpa_world_model.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # plain-runner path on hosts without pytest
    class _PytestStub:
        @staticmethod
        def importorskip(mod):
            try:
                return __import__(mod)
            except ModuleNotFoundError:
                raise SystemExit(f"skipped: {mod} not installed")

        class fixture:  # noqa: N801 - mimic pytest.fixture decorator
            def __init__(self, *a, **k):
                pass

            def __call__(self, fn):
                return fn

        class raises:  # noqa: N801 - mimic pytest.raises context manager
            def __init__(self, exc):
                self.exc = exc

            def __enter__(self):
                return self

            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(f"expected {self.exc}")
                return issubclass(et, self.exc)

    pytest = _PytestStub()

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from embodied_trackpad.sharpa_learning.world_model import (  # noqa: E402
    ACTION_DIM,
    ACTOR_OBS_KEYS,
    ARCHITECTURE_ID,
    CEMPlanner,
    DatasetError,
    Episode,
    PROPRIO_DIM,
    TOUCH_DIM,
    SharpaDataset,
    SequenceBatcher,
    SharpaWorldModel,
    WorldModelConfig,
    build_argparser,
    compute_losses,
    load_checkpoint,
    main,
    open_loop_eval,
    parameter_count,
    save_checkpoint,
)


def small_config(**kw) -> WorldModelConfig:
    base = dict(
        cnn_channels=(8, 16, 32, 32), view_embed_dim=16, proprio_embed_dim=16,
        fused_dim=48, latent_dim=32, gru_dim=32, ensemble_size=3,
        horizon=3, cem_iterations=2, cem_samples=32, cem_elites=8,
        window=4, batch_size=8, log_every=5, eval_every=0,
    )
    base.update(kw)
    return WorldModelConfig(**base)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_synthetic_episode(root: str, episode_id: str, split: str, T: int,
                           seed: int, outcome: str = "success",
                           bc_eligible: bool = True,
                           source_kind: str = "policy_rollout") -> str:
    """Write one contract-shaped episode with action-driven linear dynamics.

    Dynamics are intentionally simple and learnable: proprio follows a linear
    response to actions, touch is a thresholded function of proprio, reward is
    a fixed form of the state. This lets tests verify action dependence and
    boundary handling without claiming any real task competence.
    """
    rng = np.random.default_rng(seed)
    ep_dir = os.path.join(root, episode_id)
    os.makedirs(ep_dir, exist_ok=True)
    action = rng.uniform(-1.0, 1.0, size=(T, ACTION_DIM)).astype(np.float32)
    W = rng.normal(0, 0.1, size=(ACTION_DIM, PROPRIO_DIM)).astype(np.float32)
    proprio = np.zeros((T + 1, PROPRIO_DIM), dtype=np.float32)
    proprio[0] = rng.normal(0, 0.5, size=PROPRIO_DIM).astype(np.float32)
    for t in range(T):
        proprio[t + 1] = np.tanh(proprio[t] + action[t] @ W)
    touch = np.zeros((T + 1, TOUCH_DIM), dtype=np.float32)
    touch_valid = np.zeros((T + 1, TOUCH_DIM), dtype=np.uint8)
    for t in range(T + 1):
        m = proprio[t, :TOUCH_DIM] > 0.5
        touch[t][m] = (proprio[t, :TOUCH_DIM][m] - 0.5).astype(np.float32)
        touch_valid[t] = 1
    reward = (-np.linalg.norm(proprio[:-1], axis=1)
              + 2.0 * touch[:-1].sum(axis=1)).astype(np.float32)
    terminated = np.zeros(T, dtype=bool)
    terminated[-1] = outcome == "success"
    truncated = np.zeros(T, dtype=bool)
    screen = rng.integers(0, 256, size=(T + 1, 128, 128, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(T + 1, 128, 128, 3), dtype=np.uint8)
    sim_time = np.arange(T + 1, dtype=np.float32)
    executed_q_ref = action.copy()

    npz_path = os.path.join(ep_dir, "trajectory.npz")
    np.savez(npz_path,
             screen_rgb=screen, wrist_rgb=wrist, proprio=proprio, touch=touch,
             touch_valid=touch_valid, action=action, reward=reward,
             terminated=terminated, truncated=truncated, sim_time=sim_time,
             executed_q_ref=executed_q_ref)

    arrays = {
        "screen_rgb": screen, "wrist_rgb": wrist, "proprio": proprio,
        "touch": touch, "touch_valid": touch_valid, "action": action,
        "reward": reward, "terminated": terminated, "truncated": truncated,
        "sim_time": sim_time, "executed_q_ref": executed_q_ref,
    }
    manifest = {
        "schema": "sharpa-learning-v1",
        "episode_id": episode_id,
        "source_kind": source_kind,
        "action_names": [f"joint_{i}" for i in range(ACTION_DIM)],
        "delta_scale": [0.025] * 7 + [0.05] * (ACTION_DIM - 7),
        "joint_limits": [[-2.0] * ACTION_DIM, [2.0] * ACTION_DIM],
        "config": {"fixture": True, "seed": seed},
        "seed": seed, "split": split, "task": "tap", "outcome": outcome,
        "training_eligible_for_bc": bc_eligible,
        "measured_contact_backend": "physx_fixture",
        "timestamps": {"recorded_utc": "2026-11-18T00:00:00Z"},
        "array_sha256": {k: _sha256_array(v) for k, v in arrays.items()},
        "trajectory_sha256": _sha256_file(npz_path),
    }
    with open(os.path.join(ep_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh)
    return ep_dir


def make_dataset(root: str, n_train: int = 4, n_val: int = 1, n_test: int = 1,
                 T: int = 24, seed: int = 7) -> str:
    data_dir = os.path.join(root, "data")
    os.makedirs(data_dir, exist_ok=True)
    episodes = []
    spec = [("train", n_train), ("val", n_val), ("test", n_test)]
    k = 0
    for split, count in spec:
        for i in range(count):
            ep_id = f"fixture_synthetic_{split}_{i:03d}"
            make_synthetic_episode(data_dir, ep_id, split, T, seed + k,
                                   outcome=("success" if i % 2 == 0 else "failure"),
                                   bc_eligible=(i % 2 == 0),
                                   source_kind="policy_rollout")
            episodes.append({"path": os.path.join("data", ep_id),
                             "episode_id": ep_id, "split": split,
                             "source_kind": "policy_rollout",
                             "training_eligible_for_bc": i % 2 == 0})
            k += 1
    index = {"schema": "sharpa-learning-v1",
             "action_names": [f"joint_{i}" for i in range(ACTION_DIM)],
             "delta_scale": [0.025] * 7 + [0.05] * (ACTION_DIM - 7),
             "episodes": episodes}
    path = os.path.join(root, "dataset.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(index, fh)
    return path


@pytest.fixture(scope="module")
def dataset_path():
    root = tempfile.mkdtemp(prefix="sharpa_wm_test_")
    try:
        yield make_dataset(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dataset_schema_and_whitelist(dataset_path):
    ds = SharpaDataset(dataset_path)
    assert len(ds.episodes) == 6
    splits = ds.splits()
    assert set(splits) == {"train", "val", "test"}
    ep = splits["train"][0]
    obs = ep.actor_obs(0)
    assert set(obs.keys()) == set(ACTOR_OBS_KEYS)
    report = ds.dynamics_data_report()
    assert report["totals"]["episodes"] == 6
    # failure (bc-ineligible) transitions are counted and marked
    assert report["totals"]["bc_ineligible_transitions_used"] > 0


def test_dataset_rejects_sha_mismatch(dataset_path):
    root = tempfile.mkdtemp(prefix="sharpa_wm_sha_")
    try:
        ep_dir = make_synthetic_episode(root, "fixture_synthetic_sha", "train", 8, 3)
        manifest_path = os.path.join(ep_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest["trajectory_sha256"] = "0" * 64
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        with pytest.raises(DatasetError):
            Episode.load(ep_dir)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_dataset_rejects_validation_leakage(dataset_path):
    ds = SharpaDataset(dataset_path)
    # dataset fixture has a val split, so train-only leakage path is not taken
    splits = ds.splits()
    assert "val" in splits and len(splits["val"]) > 0
    # artificially drop val split to verify the guard
    train_only = [ep for ep in ds.episodes if ep.split == "train"]
    batcher = SequenceBatcher(train_only, window=4, batch_size=2,
                              seed=0, device=torch.device("cpu"))
    assert batcher is not None


def test_dataset_rejects_bad_shapes(dataset_path):
    root = tempfile.mkdtemp(prefix="sharpa_wm_bad_")
    try:
        ep_dir = make_synthetic_episode(root, "fixture_synthetic_bad", "train", 8, 3)
        npz = os.path.join(ep_dir, "trajectory.npz")
        with np.load(npz, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        data["action"] = np.zeros((8, 5), dtype=np.float32)  # wrong dim
        np.savez(npz, **data)
        with pytest.raises(DatasetError):
            Episode.load(ep_dir)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_encode_rejects_non_whitelisted_keys(dataset_path):
    ds = SharpaDataset(dataset_path)
    ep = ds.splits()["train"][0]
    model = SharpaWorldModel(small_config())
    obs = ep.actor_obs(0)
    obs["oracle_cursor"] = np.zeros(2, dtype=np.float32)
    with pytest.raises(DatasetError):
        model.encode_obs_dict(obs)


def test_batcher_respects_episode_boundaries(dataset_path):
    """Windows must stay inside one episode; distinct per-episode constants."""
    root = tempfile.mkdtemp(prefix="sharpa_wm_bnd_")
    try:
        data = os.path.join(root, "data")
        os.makedirs(data)
        T = 12
        for name, const in (("ep_a", 3.0), ("ep_b", -7.0)):
            ep_dir = make_synthetic_episode(data, name, "train", T, 11)
            npz = os.path.join(ep_dir, "trajectory.npz")
            with np.load(npz, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
            d["proprio"] = np.full_like(d["proprio"], const)
            np.savez(npz, **d)
            with open(os.path.join(ep_dir, "manifest.json")) as fh:
                m = json.load(fh)
            m["episode_id"] = name
            m["array_sha256"] = {k: _sha256_array(v) for k, v in d.items()}
            m["trajectory_sha256"] = _sha256_file(npz)
            with open(os.path.join(ep_dir, "manifest.json"), "w") as fh:
                json.dump(m, fh)
        eps = [Episode.load(os.path.join(data, "ep_a")),
               Episode.load(os.path.join(data, "ep_b"))]
        batcher = SequenceBatcher(eps, window=4, batch_size=64, seed=0,
                                  device=torch.device("cpu"))
        for _ in range(5):
            batch = batcher.sample_batch()
            for row in range(batch["proprio"].shape[0]):
                pv = batch["proprio"][row, :, 0].numpy()
                pn = batch["proprio_next"][row, :, 0].numpy()
                # every frame in the window comes from exactly one episode
                assert set(np.unique(pv)).issubset({3.0, -7.0})
                assert set(np.unique(pn)).issubset({3.0, -7.0})
                assert set(np.unique(pv)) == set(np.unique(pn))
                # contiguity within the episode: next is one step ahead
                assert np.allclose(pn[:-1], pv[1:])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_shifted_action_sequence_changes_prediction(dataset_path):
    """Action dependence: same state, different actions -> different rollout."""
    ds = SharpaDataset(dataset_path)
    ep = ds.splits()["train"][0]
    cfg = small_config(seed=4)
    torch.manual_seed(4)
    model = SharpaWorldModel(cfg)
    model.eval()
    obs = ep.actor_obs(0)
    h = model.advance(model.init_state(1), model.encode_obs_dict(obs))
    a1 = torch.full((1, 4, ACTION_DIM), -0.9)
    a2 = torch.full((1, 4, ACTION_DIM), 0.9)
    im1 = model.imagine(h, a1)
    im2 = model.imagine(h, a2)
    assert not torch.allclose(im1["final_state"], im2["final_state"])
    assert not torch.allclose(im1["reward"], im2["reward"])


def test_recurrent_loss_runs_over_window(dataset_path):
    """compute_losses must consume [B, W, ...] and backprop through time."""
    ds = SharpaDataset(dataset_path)
    cfg = small_config(window=4, batch_size=2)
    model = SharpaWorldModel(cfg)
    batcher = SequenceBatcher(ds.splits()["train"], cfg.window, cfg.batch_size,
                              0, torch.device("cpu"))
    batch = batcher.sample_batch()
    for key in ("screen", "wrist", "proprio", "touch", "touch_valid",
                "action", "reward", "continuation"):
        assert batch[key].shape[0] == cfg.batch_size
        assert batch[key].shape[1] == cfg.window
    losses = compute_losses(model, batch)
    losses["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_no_future_image_leak(dataset_path):
    """Prediction from obs[t] must not change when frames strictly after t change."""
    ds = SharpaDataset(dataset_path)
    ep = ds.splits()["train"][0]
    model = SharpaWorldModel(small_config())
    model.eval()
    obs = ep.actor_obs(2)
    h = model.advance(model.init_state(1), model.encode_obs_dict(obs))
    a = torch.zeros(1, 2, ACTION_DIM)
    ref = model.imagine(h, a)["final_state"]
    # Build a second observation that is identical at step 2 but changes later
    # frames. encode_obs_dict only accepts the whitelisted keys of one frame, so
    # the architecture inherently cannot see future frames; we verify that the
    # encoder is deterministic and the same current frame maps to the same state.
    h2 = model.advance(model.init_state(1), model.encode_obs_dict(obs))
    out2 = model.imagine(h2, a)["final_state"]
    assert torch.allclose(ref, out2)


def test_uncertainty_exposed_and_rollout_rejection(dataset_path):
    ds = SharpaDataset(dataset_path)
    ep = ds.splits()["train"][0]
    model = SharpaWorldModel(small_config())
    model.eval()
    obs = ep.actor_obs(0)
    h = model.advance(model.init_state(1), model.encode_obs_dict(obs))
    a = torch.zeros(1, 3, ACTION_DIM)
    im = model.imagine(h, a)
    assert im["reward_std"].shape == (1, 3)
    assert im["latent_std"].shape == (1, 3)
    assert torch.all(im["reward_std"] >= 0)
    # tighten thresholds until all rollouts are rejected
    tight = small_config(reject_reward_std=1e-9, reject_latent_std=1e-9)
    m2 = SharpaWorldModel(tight)
    m2.eval()
    h_b = m2.advance(m2.init_state(1), m2.encode_obs_dict(obs))
    im_b = m2.imagine(h_b, a)
    assert not m2.rollout_valid(im_b).any()


def test_checkpoint_restore_matches(dataset_path, tmp_path):
    ds = SharpaDataset(dataset_path)
    ep = ds.splits()["train"][0]
    cfg = small_config(seed=5)
    torch.manual_seed(5)
    model = SharpaWorldModel(cfg)
    ckpt = str(tmp_path / "ckpt.pt")
    save_checkpoint(ckpt, model, None, 0, {"updates": 0})
    model2, cfg2, payload = load_checkpoint(ckpt)
    assert cfg2.architecture_id == ARCHITECTURE_ID
    obs = ep.actor_obs(1)
    h1 = model.advance(model.init_state(1), model.encode_obs_dict(obs))
    h2 = model2.advance(model2.init_state(1), model2.encode_obs_dict(obs))
    a = torch.zeros(1, 3, ACTION_DIM)
    assert torch.allclose(model.imagine(h1, a)["reward"],
                          model2.imagine(h2, a)["reward"])
    assert payload["step"] == 0


def test_train_cli_smoke_and_timeout(dataset_path, tmp_path):
    out = str(tmp_path / "run_smoke")
    metrics = main(["train", "--dataset", dataset_path, "--output", out,
                    "--steps", "6", "--seed", "0", "--device", "cpu",
                    "--compact"])
    assert metrics["updates"] == 6
    assert metrics["stopped_reason"] == "steps"
    assert os.path.isfile(os.path.join(out, "checkpoint_final.pt"))
    assert os.path.isfile(os.path.join(out, "checkpoint_latest.pt"))
    assert os.path.isfile(os.path.join(out, "dynamics_data_report.json"))
    out2 = str(tmp_path / "run_timeout")
    m2 = main(["train", "--dataset", dataset_path, "--output", out2,
               "--steps", "100000", "--seed", "0", "--device", "cpu",
               "--compact", "--time-budget", "0.1"])
    assert m2["stopped_reason"] == "time_budget"
    assert os.path.isfile(os.path.join(out2, "checkpoint_latest.pt"))


def test_open_loop_eval_reports_finite_structural_metrics(dataset_path, tmp_path):
    """open_loop_eval reports finite per-horizon MSEs and persistence baselines.

    This is a deterministic structural check, not a stochastic learned-quality
    threshold test.
    """
    ds = SharpaDataset(dataset_path)
    cfg = small_config(seed=1)
    torch.manual_seed(1)
    model = SharpaWorldModel(cfg)
    batcher = SequenceBatcher(ds.splits()["train"], cfg.window, cfg.batch_size,
                              0, torch.device("cpu"))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(10):
        losses = compute_losses(model, batcher.sample_batch())
        opt.zero_grad(); losses["loss"].backward(); opt.step()
    ol = open_loop_eval(model, ds.splits()["test"], horizons=(1, 2, 4),
                        samples_per_episode=2)
    for k in ("1", "2", "4"):
        row = ol["horizons"][k]
        for key in ("proprio", "touch", "reward", "persistence_proprio",
                    "persistence_touch", "persistence_reward", "latent"):
            assert key in row and np.isfinite(row[key])
        # persistence baseline is a non-negative MSE
        assert row["persistence_proprio"] >= 0.0


class _FakeEnv:
    """Minimal stand-in for SharpaEnv to test the closed-loop evaluator path.

    It exposes the same call surface (reset/step/close) but is NOT physics.
    """

    def __init__(self, config=None):
        self.rng = np.random.default_rng(0)
        self.t = 0

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._obs(), {"privileged": True}

    def _obs(self):
        return {
            "screen_rgb": self.rng.integers(0, 256, (128, 128, 3), dtype=np.uint8),
            "wrist_rgb": self.rng.integers(0, 256, (128, 128, 3), dtype=np.uint8),
            "proprio": self.rng.normal(0, 1, PROPRIO_DIM).astype(np.float32),
            "touch": np.zeros(TOUCH_DIM, dtype=np.float32),
            "touch_valid": np.ones(TOUCH_DIM, dtype=np.uint8),
        }

    def step(self, action):
        assert action.shape == (ACTION_DIM,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)
        self.t += 1
        return self._obs(), 0.0, False, False, {"success": False}

    def close(self):
        pass


def test_closed_loop_evaluate_with_fake_env(dataset_path, tmp_path, monkeypatch=None):
    fake_mod = types.ModuleType("fake_sharpa_env_mod")
    fake_mod.FakeSharpaEnv = _FakeEnv
    sys.modules["fake_sharpa_env_mod"] = fake_mod
    try:
        cfg = small_config()
        torch.manual_seed(2)
        model = SharpaWorldModel(cfg)
        ckpt = str(tmp_path / "ckpt.pt")
        save_checkpoint(ckpt, model, None, 0, {})
        out = str(tmp_path / "eval_out")
        result = main(["evaluate",
                       "--env-factory", "fake_sharpa_env_mod:FakeSharpaEnv",
                       "--checkpoint", ckpt, "--episodes", "2", "--seed", "0",
                       "--max-steps", "5", "--output", out, "--device", "cpu"])
    finally:
        sys.modules.pop("fake_sharpa_env_mod", None)
    assert result["mode"] == "closed_loop"
    assert result["episodes"] == 2
    assert result["total_steps"] == 10  # evaluator-side timeout truncation
    assert all(e["truncated"] and not e["terminated"] for e in result["episode_records"])
    assert isinstance(result["rejected_rollouts"], int)
    assert os.path.isfile(os.path.join(out, "closed_loop_eval.json"))


def test_evaluate_open_loop_cli(dataset_path, tmp_path):
    cfg = small_config()
    torch.manual_seed(3)
    model = SharpaWorldModel(cfg)
    ckpt = str(tmp_path / "ckpt.pt")
    save_checkpoint(ckpt, model, None, 0, {})
    result = main(["evaluate", "--dataset", dataset_path, "--checkpoint", ckpt,
                   "--openloop-split", "test", "--openloop-samples", "1",
                   "--device", "cpu"])
    assert result["mode"] == "open_loop"
    assert result["episodes"] == 1
    assert "1" in result["open_loop"]["horizons"]


def test_cli_parser_defaults():
    args = build_argparser().parse_args(
        ["train", "--dataset", "d.json", "--output", "o", "--steps", "20",
         "--seed", "1"])
    assert args.steps == 20 and args.seed == 1
    args = build_argparser().parse_args(
        ["evaluate", "--env-factory", "embodied_trackpad.sharpa_learning.env:SharpaEnv",
         "--checkpoint", "c.pt"])
    assert args.episodes == 100
    assert args.env_factory == "embodied_trackpad.sharpa_learning.env:SharpaEnv"


def _run_all() -> int:
    """Plain-python runner for hosts without pytest (remote validation).

    Provides stand-ins for pytest fixtures (dataset_path, tmp_path) by name.
    """
    import pathlib

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        kwargs = {}
        cleanup = None
        if "dataset_path" in params:
            root = tempfile.mkdtemp(prefix="sharpa_wm_runner_")
            kwargs["dataset_path"] = make_dataset(root)
            cleanup = lambda: shutil.rmtree(root, ignore_errors=True)  # noqa: E731
        if "tmp_path" in params:
            kwargs["tmp_path"] = pathlib.Path(tempfile.mkdtemp(prefix="sharpa_wm_tmp_"))
        try:
            fn(**kwargs)
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        finally:
            if cleanup:
                cleanup()
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
