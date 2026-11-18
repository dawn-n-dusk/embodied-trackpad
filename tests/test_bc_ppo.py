"""Focused tests for sharpa_learning/bc_ppo.py.

Software-only fixtures; no real data index or physical runtime.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from types import SimpleNamespace

from embodied_trackpad.sharpa_learning.bc_ppo import (
    ACTION_DIM,
    PROPRIO_DIM,
    TOUCH_DIM,
    TOUCH_VALID_DIM,
    IMAGE_SIZE,
    ContractError,
    Episode,
    HistoryActorCritic,
    ModelConfig,
    PPOConfig,
    bc_loss,
    build_arg_parser,
    collect_ppo_rollout,
    compute_gae,
    gather_corrections,
    load_checkpoint,
    load_dataset,
    pad_episodes,
    ppo_update,
    save_checkpoint,
    train_bc,
    _obs_to_tensors,
    _teacher_label_valid,
)

torch.set_num_threads(2)


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def small_cfg() -> ModelConfig:
    # Keep image_hw at the contract 128 so episode arrays and env obs line up.
    return ModelConfig(
        image_hw=(128, 128),
        conv_channels=(4, 8),
        conv_feature_dim=16,
        vector_hidden=16,
        fused_hidden=32,
        gru_hidden=32,
        gru_layers=1,
    )


def make_arrays(T: int, seed: int = 0, eligible: bool = True, image_size: int = IMAGE_SIZE,
                source_kind: str = "sim_geometric_expert") -> dict:
    rng = np.random.default_rng(seed)
    arrays = {
        "screen_rgb": rng.integers(0, 256, size=(T + 1, image_size, image_size, 3), dtype=np.uint8),
        "wrist_rgb": rng.integers(0, 256, size=(T + 1, image_size, image_size, 3), dtype=np.uint8),
        "proprio": rng.normal(size=(T + 1, PROPRIO_DIM)).astype(np.float32),
        "touch": rng.normal(size=(T + 1, TOUCH_DIM)).astype(np.float32),
        "touch_valid": np.ones((T + 1, TOUCH_VALID_DIM), dtype=np.uint8),
        "action": rng.uniform(-1, 1, size=(T, ACTION_DIM)).astype(np.float32),
        "reward": rng.normal(size=(T,)).astype(np.float32),
        "terminated": np.zeros((T,), dtype=np.uint8),
        "truncated": np.zeros((T,), dtype=np.uint8),
        "sim_time": np.arange(T + 1, dtype=np.float64) * 0.05 + 1.0,
        "executed_q_ref": rng.normal(size=(T, ACTION_DIM)).astype(np.float32),
    }
    if not eligible:
        arrays["touch_valid"] = np.zeros((T + 1, TOUCH_VALID_DIM), dtype=np.uint8)
    if source_kind == "student_rollin_correction":
        arrays["teacher_action"] = rng.uniform(-1, 1, size=(T, ACTION_DIM)).astype(np.float32)
        arrays["teacher_valid"] = np.ones((T,), dtype=np.uint8)
        if not eligible:
            arrays["teacher_valid"] = np.zeros((T,), dtype=np.uint8)
    return arrays


def write_episode(ep_dir: Path, arrays: dict, seed: int = 0, split: str = "train",
                  eligible: bool = True, manifest_overrides: dict | None = None,
                  source_kind: str = "sim_geometric_expert") -> Path:
    ep_dir = Path(ep_dir)
    ep_dir.mkdir(parents=True, exist_ok=True)
    np.savez(ep_dir / "trajectory.npz", **arrays)
    traj_sha = _sha256_file(ep_dir / "trajectory.npz")
    array_sha = {k: _sha256_array(v) for k, v in arrays.items()}
    manifest = {
        "schema": "sharpa-learning-v1",
        "episode_id": ep_dir.name,
        "source_kind": source_kind,
        "action_names": [f"a{i}" for i in range(ACTION_DIM)],
        "delta_scale": [0.1] * ACTION_DIM,
        "joint_limits": [[-1.0] * ACTION_DIM, [1.0] * ACTION_DIM],
        "config": {"seed": seed},
        "seed": seed,
        "split": split,
        "task": "tap",
        "outcome": "success",
        "training_eligible_for_bc": eligible,
        "measured_contact_backend": "physx",
        "array_sha256": array_sha,
        "trajectory_sha256": traj_sha,
    }
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (ep_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ep_dir


def write_dataset(root: Path, episode_dirs: list[Path]) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    episodes = []
    for ep_dir in episode_dirs:
        manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
        episodes.append({
            "path": str(ep_dir.relative_to(root)),
            "episode_id": ep_dir.name,
            "split": manifest["split"],
            "source_kind": manifest["source_kind"],
            "training_eligible_for_bc": manifest["training_eligible_for_bc"],
        })
    doc = {
        "schema": "sharpa-learning-v1",
        "action_names": [f"a{i}" for i in range(ACTION_DIM)],
        "delta_scale": [0.1] * ACTION_DIM,
        "episodes": episodes,
    }
    (root / "dataset.json").write_text(json.dumps(doc), encoding="utf-8")
    return root / "dataset.json"


def make_env_obs(cfg: ModelConfig, rng: np.random.Generator) -> dict:
    h, w = cfg.image_hw
    return {
        "screen_rgb": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "wrist_rgb": rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        "proprio": rng.normal(size=(PROPRIO_DIM,)).astype(np.float32),
        "touch": rng.normal(size=(TOUCH_DIM,)).astype(np.float32),
        "touch_valid": rng.integers(0, 2, size=(TOUCH_VALID_DIM,), dtype=np.uint8),
    }


# ---------------------------------------------------------------------------
# Actor shape
# ---------------------------------------------------------------------------

def test_act_returns_batched_action_vector():
    cfg = small_cfg()
    h, w = cfg.image_hw
    model = HistoryActorCritic(cfg)
    model.eval()
    obs = {
        "screen_rgb": torch.randint(0, 256, (1, 1, h, w, 3), dtype=torch.uint8),
        "wrist_rgb": torch.randint(0, 256, (1, 1, h, w, 3), dtype=torch.uint8),
        "proprio": torch.zeros(1, 1, PROPRIO_DIM),
        "touch": torch.zeros(1, 1, TOUCH_DIM),
        "touch_valid": torch.ones(1, 1, TOUCH_VALID_DIM, dtype=torch.uint8),
    }
    action, raw, lp, value, hidden = model.act(obs, None, deterministic=True)
    assert action.shape == (1, ACTION_DIM)
    assert raw.shape == (1, ACTION_DIM)
    assert action[0].shape == (ACTION_DIM,)
    assert raw[0].shape == (ACTION_DIM,)
    assert lp.dim() == 0
    assert value.dim() == 0


# ---------------------------------------------------------------------------
# PPO rollout and unchanged-policy ratio
# ---------------------------------------------------------------------------

class FakeEnv:
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, max_steps: int = 5):
        self.cfg = cfg
        self.rng = rng
        self.max_steps = max_steps
        self._t = 0

    def reset(self, seed=None, options=None):
        self._t = 0
        return make_env_obs(self.cfg, self.rng), {}

    def step(self, action):
        self._t += 1
        obs = make_env_obs(self.cfg, self.rng)
        reward = 0.0
        terminated = self._t >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass


def test_collect_ppo_rollout_unchanged_policy_ratio_is_one():
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    env = FakeEnv(cfg, np.random.default_rng(0), max_steps=5)
    rollout = collect_ppo_rollout(
        model, env, cfg, torch.device("cpu"), max_steps=10, seed=0, gamma=0.99, gae_lambda=0.95
    )
    assert rollout["raw_action"].shape == (5, ACTION_DIM)

    # Build a single-batch padded dict for ppo_update.
    batch = {
        "screen_rgb": rollout["screen_rgb"].unsqueeze(0),
        "wrist_rgb": rollout["wrist_rgb"].unsqueeze(0),
        "proprio": rollout["proprio"].unsqueeze(0),
        "touch": rollout["touch"].unsqueeze(0),
        "touch_valid": rollout["touch_valid"].unsqueeze(0),
        "raw_action": rollout["raw_action"].unsqueeze(0),
        "old_logp": rollout["old_logp"].unsqueeze(0),
        "value": rollout["value"].unsqueeze(0),
        "adv": rollout["adv"].unsqueeze(0),
        "returns": rollout["returns"].unsqueeze(0),
        "mask": rollout["mask"].unsqueeze(0),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = ppo_update(model, optimizer, batch, None, PPOConfig())
    assert metrics["ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["clip_fraction"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# GAE terminal vs timeout
# ---------------------------------------------------------------------------

def test_gae_terminal_cuts_bootstrap_timeout_uses_it():
    rewards = [1.0, 1.0]
    values = [0.0, 0.0, 10.0]
    gamma, lam = 0.99, 0.95

    adv_term, ret_term = compute_gae(
        rewards, values, [False, True], [False, False], gamma, lam
    )
    assert ret_term[1] == pytest.approx(1.0, abs=1e-9)

    adv_time, ret_time = compute_gae(
        rewards, values, [False, False], [False, True], gamma, lam
    )
    assert ret_time[1] > 5.0  # bootstrap value 10 contributes


# ---------------------------------------------------------------------------
# NPZ / manifest rejection
# ---------------------------------------------------------------------------

def test_reject_missing_executed_q_ref():
    arrays = make_arrays(T=3)
    del arrays["executed_q_ref"]
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "ep", arrays)
        with pytest.raises(ContractError, match="missing npz keys"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


def test_reject_bad_sim_time_shape():
    arrays = make_arrays(T=3)
    arrays["sim_time"] = np.arange(3, dtype=np.float32)  # should be T+1
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "ep", arrays)
        with pytest.raises(ContractError, match="sim_time shape"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


def test_reject_non_increasing_sim_time():
    arrays = make_arrays(T=3)
    arrays["sim_time"] = np.array([1.0, 0.5, 2.0, 3.0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "ep", arrays)
        with pytest.raises(ContractError, match="strictly increasing"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


def test_reject_bad_trajectory_sha256():
    arrays = make_arrays(T=3)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "ep", arrays,
                              manifest_overrides={"trajectory_sha256": "0" * 64})
        with pytest.raises(ContractError, match="trajectory_sha256 mismatch"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


def test_reject_non_bool_eligibility():
    arrays = make_arrays(T=3)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "ep", arrays,
                              manifest_overrides={"training_eligible_for_bc": "true"})
        with pytest.raises(ContractError, match="explicit bool"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


def test_reject_invalid_touch_valid_dtype():
    arrays = make_arrays(T=3)
    arrays["touch_valid"] = arrays["touch_valid"].astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "ep", arrays)
        with pytest.raises(ContractError, match="touch_valid must be uint8"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


def test_reject_all_invalid_touch_when_eligible():
    arrays = make_arrays(T=3, eligible=False)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "ep", arrays, eligible=True)
        with pytest.raises(ContractError, match="all touch_valid are zero"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


# ---------------------------------------------------------------------------
# BC train + checkpoint + resume determinism
# ---------------------------------------------------------------------------

def test_bc_four_steps_matches_two_plus_resume(tmp_path):
    cfg = small_cfg()
    ds_root = tmp_path / "ds"
    ep_dir = write_episode(ds_root / "ep", make_arrays(T=4, seed=1))
    idx_path = write_dataset(ds_root, [ep_dir])
    dataset = load_dataset(idx_path)

    out4 = train_bc(
        dataset, tmp_path / "out4", steps=4, seed=7, cfg=cfg,
        batch_episodes=1, lr=1e-3, checkpoint_every=100,
    )
    out2 = train_bc(
        dataset, tmp_path / "out2", steps=2, seed=7, cfg=cfg,
        batch_episodes=1, lr=1e-3, checkpoint_every=100,
    )
    out2r = train_bc(
        dataset, tmp_path / "out2r", steps=2, seed=7, cfg=cfg,
        init_from=out2, batch_episodes=1, lr=1e-3, checkpoint_every=100,
    )

    m4, p4 = load_checkpoint(out4, restore_rng=False)
    mr, pr = load_checkpoint(out2r, restore_rng=False)

    for key in m4.state_dict():
        assert torch.allclose(m4.state_dict()[key], mr.state_dict()[key]), key

    # Optimizer state tensors.
    s4, sr = p4["optimizer_state"]["state"], pr["optimizer_state"]["state"]
    assert set(s4.keys()) == set(sr.keys())
    for k in s4:
        for kk in s4[k]:
            if torch.is_tensor(s4[k][kk]):
                assert torch.allclose(s4[k][kk], sr[k][kk]), (k, kk)

    # RNG + sampler state.
    assert p4["rng_state"]["python"] == pr["rng_state"]["python"]
    assert np.array_equal(p4["rng_state"]["numpy"][1], pr["rng_state"]["numpy"][1])
    assert torch.equal(p4["rng_state"]["torch_cpu"], pr["rng_state"]["torch_cpu"])
    assert p4["sampler_state"] == pr["sampler_state"]


# ---------------------------------------------------------------------------
# Student-state corrections with measured timestamps
# ---------------------------------------------------------------------------

class FakeCorrectionEnv(FakeEnv):
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, max_steps: int = 5):
        super().__init__(cfg, rng, max_steps=max_steps)
        self.action_names = [f"a{i}" for i in range(ACTION_DIM)]
        self.delta_scale = [0.1] * ACTION_DIM
        self.joint_limits = [[-1.0] * ACTION_DIM, [1.0] * ACTION_DIM]
        self.config = {"mock": True}
        self.provenance = {"contact_backend": "physx"}
        self._q_ref = np.zeros(ACTION_DIM, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self._t = 0
        self._q_ref = np.zeros(ACTION_DIM, dtype=np.float32)
        return make_env_obs(self.cfg, self.rng), {"sim_time": 1.0}

    def expert_action(self):
        return self.rng.uniform(-1, 1, size=ACTION_DIM).astype(np.float64)

    def step(self, action):
        self._t += 1
        obs = make_env_obs(self.cfg, self.rng)
        reward = 0.0
        terminated = self._t >= self.max_steps
        truncated = False
        self._q_ref += 0.01
        info = {
            "sim_time": 1.0 + self._t * 0.05,
            "executed_q_ref": self._q_ref.copy(),
        }
        return obs, reward, terminated, truncated, info


def test_gather_corrections_uses_measured_sim_time(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        return FakeCorrectionEnv(cfg, np.random.default_rng(0), max_steps=5)

    corr_idx = gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=5)
    bundle = load_dataset(corr_idx)
    assert len(bundle.episodes) == 1
    ep = bundle.episodes[0]
    sim_time = ep.arrays["sim_time"]
    assert sim_time.shape[0] == ep.T + 1
    assert np.all(np.isfinite(sim_time))
    assert np.all(np.diff(sim_time) > 0)
    assert sim_time[0] == pytest.approx(1.0)
    assert "executed_q_ref" in ep.arrays
    assert ep.arrays["executed_q_ref"].shape == (ep.T, ACTION_DIM)


# ---------------------------------------------------------------------------
# Oracle observation rejection
# ---------------------------------------------------------------------------

def test_obs_to_tensors_rejects_oracle_keys():
    cfg = small_cfg()
    rng = np.random.default_rng(0)
    obs = make_env_obs(cfg, rng)
    obs["oracle_cursor_xy"] = np.zeros(2, dtype=np.float32)
    with pytest.raises(ContractError):
        _obs_to_tensors(obs, cfg, torch.device("cpu"))


def test_obs_to_tensors_rejects_missing_and_extra_keys():
    cfg = small_cfg()
    rng = np.random.default_rng(0)
    obs = make_env_obs(cfg, rng)
    del obs["touch"]
    with pytest.raises(ContractError):
        _obs_to_tensors(obs, cfg, torch.device("cpu"))

    obs = make_env_obs(cfg, rng)
    obs["extra"] = np.zeros(1)
    with pytest.raises(ContractError):
        _obs_to_tensors(obs, cfg, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Correction-learning integrity
# ---------------------------------------------------------------------------

def test_correction_bc_target_uses_teacher_action():
    """For student_rollin_correction the BC label is teacher_action, not action."""
    cfg = small_cfg()
    T = 3
    arrays = make_arrays(T, source_kind="student_rollin_correction")
    arrays["action"][:] = 0.7
    arrays["teacher_action"][:] = -0.7
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "cor", arrays, source_kind="student_rollin_correction")
        ep = Episode.load(ep_dir)
        tensors = ep.bc_batch_tensors(cfg, torch.device("cpu"))
        assert torch.allclose(tensors["action"], torch.full((T, ACTION_DIM), -0.7))
        batch = pad_episodes([ep], cfg, torch.device("cpu"))
        assert torch.allclose(batch["action"], torch.full((1, T, ACTION_DIM), -0.7))
        assert torch.allclose(batch["mask"], torch.ones(1, T))


def test_correction_invalid_teacher_step_masked():
    """teacher_valid=0 steps must be masked out in the BC batch."""
    cfg = small_cfg()
    T = 4
    arrays = make_arrays(T, source_kind="student_rollin_correction")
    arrays["teacher_valid"] = np.array([1, 0, 0, 1], dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "cor", arrays, source_kind="student_rollin_correction")
        ep = Episode.load(ep_dir)
        batch = pad_episodes([ep], cfg, torch.device("cpu"))
        expected = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        assert torch.allclose(batch["mask"], expected)


def test_train_bc_rejects_ordinary_expert_as_corrections(tmp_path):
    """Passing a non-correction dataset as --corrections must fail."""
    cfg = small_cfg()
    ds_root = tmp_path / "ds"
    ep_dir = write_episode(ds_root / "ep", make_arrays(T=4, seed=1))
    idx_path = write_dataset(ds_root, [ep_dir])
    dataset = load_dataset(idx_path)

    corr_root = tmp_path / "cor"
    corr_ep = write_episode(corr_root / "ord", make_arrays(T=4, seed=2))
    corr_idx = write_dataset(corr_root, [corr_ep])

    with pytest.raises(ContractError, match="corrections must be student_rollin_correction"):
        train_bc(dataset, tmp_path / "out", steps=1, seed=7, cfg=cfg,
                 corrections=[corr_idx], batch_episodes=1, lr=1e-3)


def test_correction_episode_requires_teacher_arrays():
    arrays = make_arrays(T=3, source_kind="student_rollin_correction")
    del arrays["teacher_action"]
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "cor", arrays, source_kind="student_rollin_correction")
        with pytest.raises(ContractError, match="correction episode missing teacher_action"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


def test_correction_eligible_requires_one_valid():
    arrays = make_arrays(T=3, source_kind="student_rollin_correction")
    arrays["teacher_valid"] = np.zeros((3,), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as td:
        ep_dir = write_episode(Path(td) / "cor", arrays, source_kind="student_rollin_correction", eligible=True)
        with pytest.raises(ContractError, match="no teacher_valid==1"):
            load_dataset(write_dataset(Path(td), [ep_dir]))


class FakeEligibilityEnv(FakeCorrectionEnv):
    """Alternates valid/invalid labels across rollouts via teacher._phase."""

    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, max_steps: int = 3):
        super().__init__(cfg, rng, max_steps=max_steps)
        self._rollout_count = -1
        self.teacher = type("Teacher", (), {"_phase": "ok"})()

    def reset(self, seed=None, options=None):
        self._rollout_count += 1
        self.teacher._phase = "ok" if self._rollout_count % 2 == 0 else "fail"
        return super().reset(seed=seed, options=options)


def test_gather_corrections_per_episode_eligibility_preserved(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        return FakeEligibilityEnv(cfg, np.random.default_rng(0), max_steps=3)

    corr_idx = gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=4, seed=1, max_steps=3)
    bundle = load_dataset(corr_idx)
    assert len(bundle.episodes) == 4
    # Rollouts alternate eligible/ineligible, so two True and two False.
    eligibilities = [ep.training_eligible_for_bc for ep in bundle.episodes]
    assert eligibilities.count(True) == 2
    assert eligibilities.count(False) == 2


class FakeSystemExitEnv(FakeCorrectionEnv):
    def close(self):
        raise SystemExit("simulator fast shutdown")


def test_gather_corrections_writes_index_before_systemexit_close(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        return FakeSystemExitEnv(cfg, np.random.default_rng(0), max_steps=3)

    with pytest.raises(SystemExit):
        gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=3)

    index_path = tmp_path / "corr" / "corrections.json"
    assert index_path.is_file()
    bundle = load_dataset(index_path)
    assert len(bundle.episodes) == 1


class FakePhaseFailEnv(FakeCorrectionEnv):
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, max_steps: int = 3):
        super().__init__(cfg, rng, max_steps=max_steps)
        self.teacher = type("Teacher", (), {"_phase": "fail"})()

    def expert_action(self):
        # Legal zero label that must still be rejected because teacher is in fail phase.
        return np.zeros(ACTION_DIM, dtype=np.float64)


def test_gather_corrections_teacher_phase_fail_invalidates_label(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        return FakePhaseFailEnv(cfg, np.random.default_rng(0), max_steps=3)

    corr_idx = gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=3)
    bundle = load_dataset(corr_idx)
    ep = bundle.episodes[0]
    assert ep.arrays["teacher_valid"].sum() == 0
    assert not ep.training_eligible_for_bc


class FakeExplicitInvalidEnv(FakeCorrectionEnv):
    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, max_steps: int = 3):
        super().__init__(cfg, rng, max_steps=max_steps)
        self.expert_label_valid = False

    def expert_action(self):
        return np.zeros(ACTION_DIM, dtype=np.float64)


def test_gather_corrections_expert_label_valid_false_invalidates(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        return FakeExplicitInvalidEnv(cfg, np.random.default_rng(0), max_steps=3)

    corr_idx = gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=3)
    bundle = load_dataset(corr_idx)
    ep = bundle.episodes[0]
    assert ep.arrays["teacher_valid"].sum() == 0
    assert not ep.training_eligible_for_bc


def test_gather_corrections_refuses_overwrite(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    # Pre-create the episode directory and trajectory.npz that gather_corrections would write.
    existing = tmp_path / "corr" / "episodes" / "corrections-0000"
    existing.mkdir(parents=True)
    np.savez(existing / "trajectory.npz", dummy=np.zeros(1))

    def factory():
        return FakeCorrectionEnv(cfg, np.random.default_rng(0), max_steps=3)

    with pytest.raises(ContractError, match="refusing to overwrite"):
        gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=3)


def test_gather_corrections_sim_time_float64(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        return FakeCorrectionEnv(cfg, np.random.default_rng(0), max_steps=3)

    corr_idx = gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=3)
    bundle = load_dataset(corr_idx)
    assert bundle.episodes[0].arrays["sim_time"].dtype == np.float64


def test_feedback_expert_v1_source_kind_accepted(tmp_path):
    arrays = make_arrays(T=3, source_kind="feedback_expert_v1")
    ep_dir = write_episode(tmp_path / "fb", arrays, source_kind="feedback_expert_v1")
    bundle = load_dataset(write_dataset(tmp_path, [ep_dir]))
    assert bundle.episodes[0].source_kind == "feedback_expert_v1"


def test_train_bc_correction_counts_valid_steps_only(tmp_path):
    """correction_steps must count teacher_valid==1 steps, not all frames."""
    cfg = small_cfg()
    ds_root = tmp_path / "ds"
    ep_dir = write_episode(ds_root / "ep", make_arrays(T=4, seed=1))
    idx_path = write_dataset(ds_root, [ep_dir])
    dataset = load_dataset(idx_path)

    corr_arrays = make_arrays(T=5, source_kind="student_rollin_correction")
    corr_arrays["teacher_valid"] = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
    corr_ep = write_episode(tmp_path / "cor" / "c0", corr_arrays, source_kind="student_rollin_correction")
    corr_idx = write_dataset(tmp_path / "cor", [corr_ep])

    out = tmp_path / "out"
    train_bc(dataset, out, steps=1, seed=7, cfg=cfg, corrections=[corr_idx],
             batch_episodes=1, lr=1e-3, checkpoint_every=100)

    log_lines = (out / "train_log.jsonl").read_text(encoding="utf-8").strip().split("\n")
    begin = json.loads(log_lines[0])
    assert begin["event"] == "train_begin"
    assert begin["data_counts"]["correction_steps"] == 3


def test_train_bc_rejects_duplicate_correction_id(tmp_path):
    cfg = small_cfg()
    ds_root = tmp_path / "ds"
    base_ep = write_episode(ds_root / "ep", make_arrays(T=4, seed=1))
    idx_path = write_dataset(ds_root, [base_ep])
    dataset = load_dataset(idx_path)

    corr_arrays = make_arrays(T=4, source_kind="student_rollin_correction")
    corr_ep = write_episode(tmp_path / "cor" / "ep", corr_arrays, source_kind="student_rollin_correction")
    corr_idx = write_dataset(tmp_path / "cor", [corr_ep])

    with pytest.raises(ContractError, match="duplicates an already-used episode"):
        train_bc(dataset, tmp_path / "out", steps=1, seed=7, cfg=cfg,
                 corrections=[corr_idx], batch_episodes=1, lr=1e-3)


# ---------------------------------------------------------------------------
# Coordinator second-review cases
# ---------------------------------------------------------------------------

def test_teacher_label_valid_calls_callable_false():
    """Callable expert_label_valid returning False must invalidate the label."""
    env = SimpleNamespace(expert_label_valid=lambda: False)
    expert = np.zeros(ACTION_DIM, dtype=np.float64)
    assert not _teacher_label_valid(env, None, expert)


def test_gather_corrections_factory_exception_preserved(tmp_path):
    """A factory RuntimeError must propagate unchanged, not become UnboundLocalError."""
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        raise RuntimeError("factory_sentinel")

    with pytest.raises(RuntimeError, match="factory_sentinel"):
        gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=3)


class FakeToggleValidEnv(FakeCorrectionEnv):
    """Pre-step flag is False; post-step info flag is True (must stay invalid)."""

    def __init__(self, cfg: ModelConfig, rng: np.random.Generator, max_steps: int = 3):
        super().__init__(cfg, rng, max_steps=max_steps)
        self.expert_label_valid = False

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["expert_label_valid"] = True
        return obs, reward, terminated, truncated, info


def test_gather_corrections_pre_invalid_and_post_valid_stays_invalid(tmp_path):
    cfg = small_cfg()
    model = HistoryActorCritic(cfg)
    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, None, cfg, "bc", 0, {})

    def factory():
        return FakeToggleValidEnv(cfg, np.random.default_rng(0), max_steps=3)

    corr_idx = gather_corrections(factory, ckpt, tmp_path / "corr", rollouts=1, seed=1, max_steps=3)
    bundle = load_dataset(corr_idx)
    ep = bundle.episodes[0]
    assert ep.arrays["teacher_valid"].sum() == 0


def test_retrain_bc_parser_accepts_batch_episodes():
    parser = build_arg_parser()
    args = parser.parse_args([
        "retrain-bc",
        "--dataset", "ds.json",
        "--corrections", "c1.json",
        "--checkpoint", "ckpt.pt",
        "--output", "out",
        "--steps", "10",
        "--seed", "0",
        "--batch-episodes", "16",
    ])
    assert args.batch_episodes == 16

    parser2 = build_arg_parser()
    args2 = parser2.parse_args([
        "retrain-bc",
        "--dataset", "ds.json",
        "--corrections", "c1.json",
        "--checkpoint", "ckpt.pt",
        "--output", "out",
        "--steps", "10",
        "--seed", "0",
    ])
    assert args2.batch_episodes == 8
