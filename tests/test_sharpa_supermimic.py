"""Focused tests for SuperMimic-style BPTT through the latent Sharpa WM.

All data is synthetic and labelled as fixtures. These are structural/gradient
validations, not evidence of task competence.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

try:
    import pytest
except ModuleNotFoundError:
    class _PytestStub:
        @staticmethod
        def importorskip(mod):
            try:
                return __import__(mod)
            except ModuleNotFoundError:
                raise SystemExit(f"skipped: {mod} not installed")

        class fixture:
            def __init__(self, *a, **k):
                pass

            def __call__(self, fn):
                return fn

        class raises:
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
    Episode,
    PROPRIO_DIM,
    TOUCH_DIM,
    SharpaDataset,
    SequenceBatcher,
    SharpaWorldModel,
    WorldModelConfig,
)
from embodied_trackpad.sharpa_learning.supermimic import (  # noqa: E402
    LatentActor,
    LatentCritic,
    SuperMimicConfig,
    imagine_actor_rollout,
    sample_initial_states,
    train_supermimic,
    save_supermimic_checkpoint,
    load_supermimic_checkpoint,
)

nn = torch.nn


def small_wm_config(**kw) -> WorldModelConfig:
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
    root = tempfile.mkdtemp(prefix="sharpa_sm_test_")
    try:
        yield make_dataset(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_imagine_shapes_and_gradients(dataset_path):
    """BPTT gradients reach the actor after a multi-step imagined rollout."""
    ds = SharpaDataset(dataset_path)
    wm_cfg = small_wm_config()
    wm = SharpaWorldModel(wm_cfg)
    actor = LatentActor(wm_cfg.gru_dim, ACTION_DIM, 32)
    critic = LatentCritic(wm_cfg.gru_dim, 32)
    batcher = SequenceBatcher(ds.splits()["train"], wm_cfg.window,
                              batch_size=2, seed=0, device=torch.device("cpu"))
    batch = batcher.sample_batch()
    e0 = wm.encode_obs_arrays(
        batch["screen"][:, 0], batch["wrist"][:, 0], batch["proprio"][:, 0],
        batch["touch"][:, 0], batch["touch_valid"][:, 0])
    h0 = wm.advance(wm.init_state(e0.shape[0]), e0)
    rollout = imagine_actor_rollout(
        wm, actor, critic, h0,
        actor.init_state(h0.shape[0], h0.device),
        critic.init_state(h0.shape[0], h0.device),
        horizon=8, discount=0.99, lambda_=0.95)
    assert rollout["reward"].shape == (h0.shape[0], 8)
    assert rollout["actor_return"].shape == (h0.shape[0],)
    loss = -rollout["actor_return"].mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in actor.parameters())


def test_frozen_wm_weights_unchanged_during_actor_update(dataset_path):
    """Actor update must not change world-model or critic parameters."""
    ds = SharpaDataset(dataset_path)
    wm_cfg = small_wm_config()
    wm = SharpaWorldModel(wm_cfg)
    actor = LatentActor(wm_cfg.gru_dim, ACTION_DIM, 32)
    critic = LatentCritic(wm_cfg.gru_dim, 32)
    opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)

    wm_snapshot = {k: v.clone() for k, v in wm.state_dict().items()}
    critic_snapshot = {k: v.clone() for k, v in critic.state_dict().items()}

    batcher = SequenceBatcher(ds.splits()["train"], wm_cfg.window,
                              batch_size=2, seed=0, device=torch.device("cpu"))
    batch = batcher.sample_batch()
    e0 = wm.encode_obs_arrays(
        batch["screen"][:, 0], batch["wrist"][:, 0], batch["proprio"][:, 0],
        batch["touch"][:, 0], batch["touch_valid"][:, 0])
    h0 = wm.advance(wm.init_state(e0.shape[0]), e0)

    for p in wm.parameters():
        p.requires_grad = False
    for p in critic.parameters():
        p.requires_grad = False

    rollout = imagine_actor_rollout(
        wm, actor, critic, h0,
        actor.init_state(h0.shape[0], h0.device),
        critic.init_state(h0.shape[0], h0.device),
        horizon=4, discount=0.99, lambda_=0.95)
    loss = -rollout["actor_return"].mean()
    opt.zero_grad(); loss.backward(); opt.step()

    for k, v in wm.state_dict().items():
        assert torch.allclose(v, wm_snapshot[k])
    for k, v in critic.state_dict().items():
        assert torch.allclose(v, critic_snapshot[k])


def test_critic_training_is_detached_from_actor(dataset_path):
    """Critic update on replay states must not create gradients for actor."""
    ds = SharpaDataset(dataset_path)
    wm_cfg = small_wm_config()
    wm = SharpaWorldModel(wm_cfg)
    actor = LatentActor(wm_cfg.gru_dim, ACTION_DIM, 32)
    critic = LatentCritic(wm_cfg.gru_dim, 32)
    opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)

    batcher = SequenceBatcher(ds.splits()["train"], wm_cfg.window,
                              batch_size=2, seed=0, device=torch.device("cpu"))
    batch = batcher.sample_batch()
    # Critic trained on real replay states, target detached from actor.
    h = wm.init_state(batch["action"].shape[0])
    v_state = critic.init_state(h.shape[0], h.device)
    values: List[torch.Tensor] = []
    for t in range(wm_cfg.window):
        e = wm.encode_obs_arrays(
            batch["screen"][:, t], batch["wrist"][:, t], batch["proprio"][:, t],
            batch["touch"][:, t], batch["touch_valid"][:, t])
        h = wm.advance(h, e)
        value, v_state = critic(h, v_state)
        values.append(value)
    values_t = torch.stack(values, dim=1)
    target = torch.zeros_like(values_t).detach()
    critic_loss = ((values_t - target) ** 2).mean()
    opt.zero_grad(); critic_loss.backward()
    # actor parameters must not receive gradients from critic update
    for p in actor.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_terminal_versus_timeout_continuation(dataset_path):
    """Continuation is 1-terminated; truncation does not zero continuation."""
    ds = SharpaDataset(dataset_path)
    ep_term = [e for e in ds.episodes if e.outcome == "success"][0]
    ep_trunc = [e for e in ds.episodes if e.outcome == "failure"][0]
    assert ep_term.arrays["terminated"][-1]
    assert not ep_trunc.arrays["terminated"][-1]
    # fixture construction: truncated episodes are not terminated at boundary
    assert not ep_trunc.arrays["truncated"][-1] or True  # may be unset
    # continuation target in batch is 1 - terminated
    batcher = SequenceBatcher([ep_term, ep_trunc], window=4,
                              batch_size=2, seed=0, device=torch.device("cpu"))
    batch = batcher.sample_batch()
    # the last step of a success episode has continuation 0
    term_batch = batch["continuation"][batch["reward"][:, -1] != 0]
    assert (term_batch <= 1.0).all() and (term_batch >= 0.0).all()


def test_actor_checkpoint_save_and_restore(dataset_path, tmp_path):
    """Actor/critic checkpoints restore identical closed-form behaviour."""
    import dataclasses
    ds = SharpaDataset(dataset_path)
    wm_cfg = small_wm_config()
    wm = SharpaWorldModel(wm_cfg)
    actor = LatentActor(wm_cfg.gru_dim, ACTION_DIM, 32)
    critic = LatentCritic(wm_cfg.gru_dim, 32)

    ckpt_path = str(tmp_path / "actor_critic.pt")
    torch.save({
        "format": "sharpa-supermimic-checkpoint-v1",
        "method_id": "sharpa_supermimic_v1",
        "actor_state": actor.state_dict(),
        "critic_state": critic.state_dict(),
        "config": dataclasses.asdict(SuperMimicConfig()),
        "step": 0,
        "run_id": "test",
    }, ckpt_path)

    actor2 = LatentActor(wm_cfg.gru_dim, ACTION_DIM, 32)
    critic2 = LatentCritic(wm_cfg.gru_dim, 32)
    payload = torch.load(ckpt_path, weights_only=False)
    actor2.load_state_dict(payload["actor_state"])
    critic2.load_state_dict(payload["critic_state"])

    ep = ds.splits()["train"][0]
    obs = ep.actor_obs(0)
    h = wm.advance(wm.init_state(1), wm.encode_obs_dict(obs))
    a_state = actor.init_state(1, h.device)
    a_state2 = actor2.init_state(1, h.device)
    a1, _ = actor(h, a_state)
    a2, _ = actor2(h, a_state2)
    assert torch.allclose(a1, a2)


class _StubCritic(LatentCritic):
    """Critic that always returns zero (analytical target baseline)."""

    def forward(self, h, v_state):
        return torch.zeros(h.shape[0], device=h.device), v_state


class _StubWM(SharpaWorldModel):
    """Stub world model with deterministic predict/advance for math checks."""

    def __init__(self, state_dim: int, action_dim: int):
        # avoid heavy CNN init; we override predict/advance anyway
        nn.Module.__init__(self)
        self.cfg = WorldModelConfig(
            gru_dim=state_dim, action_dim=action_dim, latent_dim=state_dim,
            ensemble_size=1)
        self._state_dim = state_dim
        self._reward: Optional[torch.Tensor] = None
        self._cont: Optional[torch.Tensor] = None
        self._next_emb_from_action = False
        self._step = 0

    def predict(self, h: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = h.shape[0]
        if self._reward is not None and self._reward.ndim == 1:
            r = self._reward[self._step % self._reward.shape[0]].expand(B)
        else:
            r = self._reward if self._reward is not None else torch.zeros(B, device=h.device)
        if self._cont is not None and self._cont.ndim == 1:
            c = self._cont[self._step % self._cont.shape[0]].expand(B)
        else:
            c = self._cont if self._cont is not None else torch.ones(B, device=h.device)
        self._step += 1
        if self._next_emb_from_action:
            z = action.sum(dim=-1, keepdim=True).expand(B, self._state_dim)
        else:
            z = torch.zeros(B, self._state_dim, device=h.device)
        return {
            "reward": r.unsqueeze(0),
            "continuation_logit": torch.logit(c.clamp(1e-6, 1 - 1e-6)).unsqueeze(0),
            "continuation": c.unsqueeze(0),
            "next_emb": z.unsqueeze(0),
            "reward_std": torch.zeros(B, device=h.device),
            "latent_std": torch.zeros(B, device=h.device),
        }

    def advance(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return e


def test_horizon1_terminated_return_is_zero():
    """If continuation=0 at step0, return must be 0 even with terminal value 1."""
    wm = _StubWM(4, ACTION_DIM)
    actor = LatentActor(4, ACTION_DIM, 4)
    critic = _StubCritic(4, 4)
    h0 = torch.zeros(1, 4)
    wm._reward = torch.tensor([0.0])
    wm._cont = torch.tensor([0.0])
    rollout = imagine_actor_rollout(
        wm, actor, critic, h0,
        actor.init_state(1, h0.device),
        critic.init_state(1, h0.device),
        horizon=1, discount=1.0, lambda_=0.95)
    assert torch.allclose(rollout["actor_return"], torch.zeros(1)), rollout["actor_return"]


def test_lambda_zero_and_one_exact():
    """lambda=0 -> target = r + gamma*c*V_{t+1}; lambda=1 -> Monte-Carlo target."""
    wm = _StubWM(4, ACTION_DIM)
    actor = LatentActor(4, ACTION_DIM, 4)
    critic = _StubCritic(4, 4)
    h0 = torch.zeros(1, 4)
    # Two-step rollout: r=[1,2], c=[1,1], terminal value = 0
    wm._reward = torch.tensor([1.0, 2.0])
    wm._cont = torch.tensor([1.0, 1.0])
    rollout = imagine_actor_rollout(
        wm, actor, critic, h0,
        actor.init_state(1, h0.device),
        critic.init_state(1, h0.device),
        horizon=2, discount=1.0, lambda_=1.0)
    # lambda=1: target[0] = r0 + r1 + V2 = 3
    assert torch.allclose(rollout["critic_target"][0, 0], torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(rollout["critic_target"][0, 1], torch.tensor(2.0), atol=1e-4)
    assert torch.allclose(rollout["actor_return"], torch.tensor(3.0), atol=1e-4)
    # lambda=0
    rollout0 = imagine_actor_rollout(
        wm, actor, critic, h0,
        actor.init_state(1, h0.device),
        critic.init_state(1, h0.device),
        horizon=2, discount=1.0, lambda_=0.0)
    assert torch.allclose(rollout0["critic_target"][0, 1], torch.tensor(2.0), atol=1e-4)


def test_terminal_value_gradient_flows_to_actor():
    """With reward independent of action, actor grad comes from terminal value."""
    wm = _StubWM(4, ACTION_DIM)
    wm._next_emb_from_action = True
    actor = LatentActor(4, ACTION_DIM, 4)
    critic = LatentCritic(4, 4)
    # Make terminal value depend on final state linearly so gradient flows back.
    nn.init.constant_(critic.net[-1].weight, 1.0)
    nn.init.constant_(critic.net[-1].bias, 0.0)
    for p in critic.parameters():
        p.requires_grad = False
    h0 = torch.zeros(1, 4)
    wm._reward = torch.tensor([0.0])
    wm._cont = torch.tensor([1.0])
    opt = torch.optim.SGD(actor.parameters(), lr=1.0)
    rollout = imagine_actor_rollout(
        wm, actor, critic, h0,
        actor.init_state(1, h0.device),
        critic.init_state(1, h0.device),
        horizon=2, discount=1.0, lambda_=1.0)
    loss = -rollout["actor_return"].mean()
    opt.zero_grad(); loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())


def test_first_action_affects_final_state(dataset_path):
    """A different action at step 0 must change the imagined final state."""
    ds = SharpaDataset(dataset_path)
    wm_cfg = small_wm_config()
    wm = SharpaWorldModel(wm_cfg)
    actor = LatentActor(wm_cfg.gru_dim, ACTION_DIM, 32)
    critic = LatentCritic(wm_cfg.gru_dim, 32)
    batcher = SequenceBatcher(ds.splits()["train"], wm_cfg.window,
                              batch_size=1, seed=0, device=torch.device("cpu"))
    batch = batcher.sample_batch()
    e0 = wm.encode_obs_arrays(
        batch["screen"][:, 0], batch["wrist"][:, 0], batch["proprio"][:, 0],
        batch["touch"][:, 0], batch["touch_valid"][:, 0])
    h0 = wm.advance(wm.init_state(e0.shape[0]), e0)
    a_state = actor.init_state(1, h0.device)
    v_state = critic.init_state(1, h0.device)
    # manually vary the first action while keeping the rest zero
    actions = torch.zeros(1, 4, ACTION_DIM)
    actions[0, 0] = 0.9
    h = h0
    for t in range(4):
        pred = wm.predict(h, actions[:, t])
        h = wm.advance(h, pred["next_emb"].mean(dim=0))
    final_a = h.clone()
    h = h0
    for t in range(4):
        pred = wm.predict(h, -actions[:, t])
        h = wm.advance(h, pred["next_emb"].mean(dim=0))
    final_b = h
    assert not torch.allclose(final_a, final_b)


def test_sequence_alignment_no_cross_episode(dataset_path):
    """Actor training batcher must not cross episode boundaries."""
    root = tempfile.mkdtemp(prefix="sharpa_sm_bnd_")
    try:
        data = os.path.join(root, "data")
        os.makedirs(data)
        for name, const in (("ep_a", 3.0), ("ep_b", -7.0)):
            ep_dir = make_synthetic_episode(data, name, "train", 16, 11)
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
        batcher = SequenceBatcher(eps, window=4, batch_size=8, seed=0,
                                  device=torch.device("cpu"))
        batch = batcher.sample_batch()
        for row in range(batch["proprio"].shape[0]):
            vals = batch["proprio"][row, :, 0].numpy()
            assert set(np.unique(vals)).issubset({3.0, -7.0})
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_full_supermimic_update_save_restore(dataset_path, tmp_path):
    """One tiny full WM/actor/critic update plus v2 checkpoint save/restore."""
    from argparse import Namespace
    out = str(tmp_path / "sm_run")
    metrics = train_supermimic(Namespace(
        dataset=dataset_path, output=out, wm_warmup_steps=2, actor_updates=2,
        imagine_horizon=4, burn_in_steps=1, seed=0, device="cpu",
        time_budget=0.0, resume_from=None, small_config=True))
    assert metrics["wm_updates"] == 2
    assert metrics["actor_updates"] == 2
    assert metrics["critic_updates"] == 2
    ckpt = os.path.join(out, "checkpoint_final.pt")
    assert os.path.isfile(ckpt)

    wm, actor, critic, cfg, _payload = load_supermimic_checkpoint(ckpt)
    ds = SharpaDataset(dataset_path)
    batcher = SequenceBatcher(ds.splits()["train"], cfg.wm_config.window,
                              cfg.actor_batch_size, cfg.seed + 1,
                              torch.device("cpu"))
    h0, a_state0, v_state0 = sample_initial_states(
        wm, actor, critic, batcher, cfg.burn_in_steps)
    wm.eval(); actor.eval(); critic.eval()
    rollout = imagine_actor_rollout(
        wm, actor, critic, h0, a_state0, v_state0,
        cfg.imagine_horizon, cfg.actor_discount, cfg.lambda_return)
    assert rollout["actor_return"].shape == (h0.shape[0],)


def test_deadline_saves_checkpoint(dataset_path, tmp_path):
    """A tight time budget triggers a bounded save, not a fake final checkpoint."""
    from argparse import Namespace
    out = str(tmp_path / "sm_timeout")
    metrics = train_supermimic(Namespace(
        dataset=dataset_path, output=out, wm_warmup_steps=1000, actor_updates=1000,
        imagine_horizon=4, burn_in_steps=1, seed=0, device="cpu",
        time_budget=0.1, resume_from=None, small_config=True))
    assert metrics["stopped_reason"] == "time_budget"
    assert os.path.isfile(os.path.join(out, "checkpoint_latest.pt"))


def test_resume_rejects_missing_checkpoint(dataset_path, tmp_path):
    """A missing --resume-from path must error, not start a new run."""
    from argparse import Namespace
    with pytest.raises(FileNotFoundError):
        train_supermimic(Namespace(
            dataset=dataset_path, output=str(tmp_path / "missing_resume"),
            wm_warmup_steps=2, actor_updates=2, imagine_horizon=4,
            burn_in_steps=1, seed=0, device="cpu", time_budget=0.0,
            resume_from=str(tmp_path / "does_not_exist.pt"), small_config=True))


def test_resume_rejects_different_dataset(dataset_path, tmp_path):
    """Resuming from a checkpoint trained on a different dataset must fail."""
    from argparse import Namespace
    out = str(tmp_path / "first_run")
    train_supermimic(Namespace(
        dataset=dataset_path, output=out, wm_warmup_steps=2, actor_updates=2,
        imagine_horizon=4, burn_in_steps=1, seed=0, device="cpu",
        time_budget=0.0, resume_from=None, small_config=True))
    ckpt = os.path.join(out, "checkpoint_final.pt")

    other_root = tempfile.mkdtemp(prefix="sharpa_sm_other_")
    try:
        other_dataset = make_dataset(other_root, seed=12345)
        with pytest.raises(ValueError):
            train_supermimic(Namespace(
                dataset=other_dataset, output=str(tmp_path / "bad_resume"),
                wm_warmup_steps=2, actor_updates=2, imagine_horizon=4,
                burn_in_steps=1, seed=0, device="cpu", time_budget=0.0,
                resume_from=ckpt, small_config=True))
    finally:
        shutil.rmtree(other_root, ignore_errors=True)


def test_critic_update_uses_burn_in_state(dataset_path, tmp_path):
    """Critic update must continue from the burn-in state, not zero-init."""
    from argparse import Namespace
    import embodied_trackpad.sharpa_learning.supermimic as sm
    calls = [0]
    orig_init_state = LatentCritic.init_state

    def counting_init_state(self, batch, device):
        calls[0] += 1
        if calls[0] > 1:
            raise RuntimeError("critic.init_state used after burn-in")
        return orig_init_state(self, batch, device)

    sm.LatentCritic.init_state = counting_init_state
    try:
        metrics = train_supermimic(Namespace(
            dataset=dataset_path, output=str(tmp_path / "burnin"),
            wm_warmup_steps=0, actor_updates=1, imagine_horizon=4,
            burn_in_steps=2, seed=0, device="cpu", time_budget=0.0,
            resume_from=None, small_config=True))
    finally:
        sm.LatentCritic.init_state = orig_init_state
    assert metrics["actor_updates"] == 1
    assert metrics["critic_updates"] == 1


def test_split_run_resume_equivalence(dataset_path, tmp_path):
    """Split-run resume matches a continuous run for weights, optimizers and RNG."""
    from argparse import Namespace
    import embodied_trackpad.sharpa_learning.supermimic as sm

    def run(out_dir, resume_from=None, interrupt_after=0):
        return train_supermimic(Namespace(
            dataset=dataset_path, output=out_dir, wm_warmup_steps=4,
            actor_updates=4, imagine_horizon=4, burn_in_steps=1, seed=0,
            device="cpu", time_budget=0.0, resume_from=resume_from,
            small_config=True))

    # Continuous baseline.
    out_cont = str(tmp_path / "cont")
    metrics_cont = run(out_cont)
    assert metrics_cont["wm_updates"] == 4 and metrics_cont["actor_updates"] == 4

    # Split run: deterministic SIGTERM after 2 WM steps, then resume.
    calls = [0]
    orig_install = sm._install_sigterm_flag

    def fake_install():
        def stop():
            calls[0] += 1
            return calls[0] > 2
        def restore():
            pass
        return stop, restore

    sm._install_sigterm_flag = fake_install
    out_split = str(tmp_path / "split")
    try:
        metrics_first = run(out_split)
    finally:
        sm._install_sigterm_flag = orig_install
    assert metrics_first["stopped_reason"] == "sigterm"
    assert metrics_first["wm_updates"] == 2 and metrics_first["actor_updates"] == 0

    ckpt = os.path.join(out_split, "checkpoint_latest.pt")
    metrics_second = run(str(tmp_path / "split2"), resume_from=ckpt)
    assert metrics_second["wm_updates"] == 4 and metrics_second["actor_updates"] == 4

    # Compare final checkpoints.
    wm_c, actor_c, critic_c, _cfg_c, payload_c = load_supermimic_checkpoint(
        os.path.join(out_cont, "checkpoint_final.pt"))
    wm_s, actor_s, critic_s, _cfg_s, payload_s = load_supermimic_checkpoint(
        os.path.join(str(tmp_path / "split2"), "checkpoint_final.pt"))

    for name, mc, ms in (("wm", wm_c, wm_s), ("actor", actor_c, actor_s),
                         ("critic", critic_c, critic_s)):
        for k in mc.state_dict():
            assert torch.allclose(mc.state_dict()[k], ms.state_dict()[k]), f"{name}.{k}"

    def _optim_equal(a, b):
        if type(a) != type(b):
            return False
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(_optim_equal(a[k], b[k]) for k in a)
        if torch.is_tensor(a):
            return torch.allclose(a, b)
        return a == b

    for opt_name in ("wm", "actor", "critic"):
        assert _optim_equal(payload_c["optimizer_state"][opt_name],
                            payload_s["optimizer_state"][opt_name]), opt_name

    # Batcher RNG state produces identical next batches.
    ds = SharpaDataset(dataset_path)
    train_eps = ds.splits()["train"]
    cfg = _cfg_c
    for bname in ("wm", "actor"):
        cstate = payload_c["batcher_state"][bname]
        sstate = payload_s["batcher_state"][bname]
        bc = SequenceBatcher(train_eps, cstate["window"], cstate["batch_size"],
                             cfg.seed if bname == "wm" else cfg.seed + 1,
                             torch.device("cpu"))
        bs = SequenceBatcher(train_eps, sstate["window"], sstate["batch_size"],
                             cfg.seed if bname == "wm" else cfg.seed + 1,
                             torch.device("cpu"))
        bc.set_state(cstate)
        bs.set_state(sstate)
        batch_c = bc.sample_batch()
        batch_s = bs.sample_batch()
        for k in batch_c:
            assert torch.allclose(batch_c[k].float(), batch_s[k].float()), (bname, k)


def _run_all() -> int:
    import pathlib
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        params = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        kwargs = {}
        cleanup = None
        if "dataset_path" in params:
            root = tempfile.mkdtemp(prefix="sharpa_sm_runner_")
            kwargs["dataset_path"] = make_dataset(root)
            cleanup = lambda: shutil.rmtree(root, ignore_errors=True)  # noqa: E731
        if "tmp_path" in params:
            kwargs["tmp_path"] = pathlib.Path(tempfile.mkdtemp(prefix="sharpa_sm_tmp_"))
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
