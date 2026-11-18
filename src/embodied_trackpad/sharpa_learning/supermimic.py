"""SuperMimic-style BPTT through a differentiable latent world model for Sharpa.

Visual adaptation (not an exact donor reproduction) of SuperMimic:
  https://github.com/HybridRobotics/supermimic  (official, Apache-2.0)

This module keeps the two-RGB + proprio/touch observation encoder from
``world_model.py`` and adds:
- deterministic recurrent latent actor,
- recurrent value critic,
- BPTT actor training through the frozen-but-differentiable world model,
- detached lambda-return critic update,
- autonomous actor evaluation (no oracle actor fallback).

The world model is reused directly from ``world_model.py``. CEM-MPC remains
optional and is not required for this learner.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "embodied_trackpad.sharpa_learning.supermimic requires PyTorch."
    ) from exc

from embodied_trackpad.sharpa_learning.world_model import (
    ACTION_DIM,
    ACTOR_OBS_KEYS,
    ARCHITECTURE_ID,
    DatasetError,
    Episode,
    PROPRIO_DIM,
    SequenceBatcher,
    SharpaDataset,
    SharpaWorldModel,
    WorldModelConfig,
    compute_losses,
    load_checkpoint,
    parameter_count,
    resolve_device,
    save_checkpoint,
    _load_env_factory,
)
from embodied_trackpad.sharpa_learning.bc_ppo import (
    capture_rng_state,
    restore_rng_state,
)

METHOD_ID = "sharpa_supermimic_v1"
RUN_SCHEMA = "sharpa-supermimic-run-v1"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SuperMimicConfig:
    """Hyper-parameters for SuperMimic-style latent BPTT."""

    method_id: str = METHOD_ID

    # world model (pretrained or trained together from scratch)
    wm_config: WorldModelConfig = dataclasses.field(default_factory=WorldModelConfig)

    # actor / critic architecture
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256

    # imagination
    imagine_horizon: int = 16  # configurable 8/32
    actor_discount: float = 0.99
    lambda_return: float = 0.95

    # training
    wm_warmup_steps: int = 2000
    actor_updates: int = 5000
    actor_batch_size: int = 32
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    actor_grad_clip: float = 20.0
    critic_grad_clip: float = 20.0
    actor_log_every: int = 200
    actor_update_wm_every: int = 0  # 0 = freeze WM after warmup
    burn_in_steps: int = 2  # real-history steps to seed actor/critic state

    seed: int = 0
    device: str = "auto"


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class LatentActor(nn.Module):
    """Deterministic recurrent actor mapping WM latent state -> action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(state_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def forward(self, h: torch.Tensor, a_state: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_state = self.gru(h, a_state)
        return self.net(a_state), a_state


class LatentCritic(nn.Module):
    """Recurrent value critic over WM latent states."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRUCell(state_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def init_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_dim, device=device)

    def forward(self, h: torch.Tensor, v_state: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_state = self.gru(h, v_state)
        return self.net(v_state).squeeze(-1), v_state


# --------------------------------------------------------------------------- #
# Imagination
# --------------------------------------------------------------------------- #
def imagine_actor_rollout(
    wm: SharpaWorldModel,
    actor: LatentActor,
    critic: LatentCritic,
    h0: torch.Tensor,
    a_state0: torch.Tensor,
    v_state0: torch.Tensor,
    horizon: int,
    discount: float,
    lambda_: float,
) -> Dict[str, torch.Tensor]:
    """Unroll actor + WM for ``horizon`` steps starting from latent state h0.

    Returns dictionaries of per-step rewards, continuations, values, and the
    lambda-return target for the critic. The actor recurrent state is NOT
    detached so BPTT gradients reach the earliest action. The critic target is
    detached before being returned.
    """
    B = h0.shape[0]
    dev = h0.device
    h = h0
    a_state = a_state0
    v_state = v_state0
    rewards: List[torch.Tensor] = []
    continuations: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    states: List[torch.Tensor] = []
    for t in range(horizon):
        states.append(h)
        value, v_state = critic(h, v_state)
        values.append(value)
        action, a_state = actor(h, a_state)
        pred = wm.predict(h, action)
        rewards.append(pred["reward"].mean(dim=0))
        cont = pred["continuation"].mean(dim=0)
        continuations.append(cont)
        h = wm.advance(h, pred["next_emb"].mean(dim=0))
    # terminal bootstrap
    v_term, _ = critic(h, v_state)
    values.append(v_term)

    rewards_t = torch.stack(rewards, dim=1)          # [B, H]
    cont_t = torch.stack(continuations, dim=1)       # [B, H]
    values_t = torch.stack(values, dim=1)            # [B, H+1]

    # discounted return with soft survival (multiply ALL continuation probs)
    disc = torch.tensor([discount ** t for t in range(horizon)], device=dev)
    cum_cont = torch.cumprod(
        torch.cat([torch.ones(B, 1, device=dev), cont_t], dim=1), dim=1)
    returns = (rewards_t * cum_cont[:, :-1] * disc).sum(dim=1)
    # terminal bootstrap keeps gradient through final state (critic params frozen)
    returns = returns + (discount ** horizon) * cum_cont[:, -1] * v_term

    # lambda-return target (detached from actor) computed recursively:
    # G_H = V_H; G_t = r_t + gamma*c_t*((1-lambda)*V_{t+1} + lambda*G_{t+1})
    g = values_t[:, -1].detach()
    lambda_target: List[torch.Tensor] = []
    for t in reversed(range(horizon)):
        td = rewards_t[:, t] + discount * cont_t[:, t] * g
        lambda_target.append(td)
        g = (1 - lambda_) * values_t[:, t + 1].detach() + lambda_ * td
    lambda_target = torch.stack(list(reversed(lambda_target)), dim=1)  # [B, H]

    return {
        "states": torch.stack(states, dim=1),      # [B, H, state_dim]
        "reward": rewards_t,
        "continuation": cont_t,
        "values": values_t[:, :-1],
        "actor_return": returns,                   # scalar per sample
        "critic_target": lambda_target.detach(),
    }


# --------------------------------------------------------------------------- #
# Dataset helpers
# --------------------------------------------------------------------------- #
def sample_initial_states(
    wm: SharpaWorldModel,
    actor: LatentActor,
    critic: LatentCritic,
    batcher: SequenceBatcher,
    burn_in: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Burn in recurrent states from real history before imagination.

    Returns (h, a_state, v_state) after ``burn_in`` real steps, so the actor
    and critic recurrent states carry touch-gesture history instead of zero.
    """
    batch = batcher.sample_batch()
    B = batch["action"].shape[0]
    device = next(wm.parameters()).device
    h = wm.init_state(B).to(device)
    a_state = actor.init_state(B, device)
    v_state = critic.init_state(B, device)
    burn_in = max(0, min(burn_in, batch["action"].shape[1] - 1))
    for t in range(burn_in):
        e = wm.encode_obs_arrays(
            batch["screen"][:, t], batch["wrist"][:, t],
            batch["proprio"][:, t], batch["touch"][:, t],
            batch["touch_valid"][:, t])
        h = wm.advance(h, e)
        _, v_state = critic(h, v_state)
        _, a_state = actor(h, a_state)
    return h, a_state, v_state


# --------------------------------------------------------------------------- #
# Checkpointing (full resume: WM/actor/critic + optimizers + RNG + batcher)
# --------------------------------------------------------------------------- #
SUPERMIMIC_CHECKPOINT_FORMAT = "sharpa-supermimic-checkpoint-v2"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _dataset_fingerprint(dataset: SharpaDataset) -> str:
    """SHA-256 over the dataset index and every referenced trajectory."""
    h = hashlib.sha256()
    with open(dataset.index_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    for ep in sorted(dataset.episodes, key=lambda e: e.path):
        h.update(str(ep.manifest.get("trajectory_sha256", "")).encode("utf-8"))
    return h.hexdigest()


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save_supermimic_checkpoint(
    path: str,
    wm: SharpaWorldModel,
    actor: LatentActor,
    critic: LatentCritic,
    wm_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    cfg: SuperMimicConfig,
    metrics: Dict[str, Any],
    wm_batcher: SequenceBatcher,
    actor_batcher: SequenceBatcher,
    run_id: str,
    provenance: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "format": SUPERMIMIC_CHECKPOINT_FORMAT,
        "method_id": METHOD_ID,
        "run_id": run_id,
        "config": dataclasses.asdict(cfg),
        "model_state": {
            "wm": {k: v.detach().cpu() for k, v in wm.state_dict().items()},
            "actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "critic": {k: v.detach().cpu() for k, v in critic.state_dict().items()},
        },
        "optimizer_state": {
            "wm": wm_optimizer.state_dict(),
            "actor": actor_optimizer.state_dict(),
            "critic": critic_optimizer.state_dict(),
        },
        "rng_state": capture_rng_state(),
        "batcher_state": {
            "wm": wm_batcher.get_state(),
            "actor": actor_batcher.get_state(),
        },
        "metrics": metrics,
        "provenance": provenance or {},
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(payload, path)


def load_supermimic_checkpoint(
    path: str,
    map_location: Optional[str] = None,
) -> Tuple[SharpaWorldModel, LatentActor, LatentCritic, SuperMimicConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    if payload.get("format") != SUPERMIMIC_CHECKPOINT_FORMAT:
        raise ValueError(f"not a {SUPERMIMIC_CHECKPOINT_FORMAT}: {path}")
    cfg_dict = dict(payload["config"])
    cfg_dict["wm_config"] = WorldModelConfig(**cfg_dict["wm_config"])
    cfg = SuperMimicConfig(**cfg_dict)
    wm = SharpaWorldModel(cfg.wm_config)
    wm.load_state_dict(payload["model_state"]["wm"])
    actor = LatentActor(cfg.wm_config.gru_dim, ACTION_DIM, cfg.actor_hidden_dim)
    actor.load_state_dict(payload["model_state"]["actor"])
    critic = LatentCritic(cfg.wm_config.gru_dim, cfg.critic_hidden_dim)
    critic.load_state_dict(payload["model_state"]["critic"])
    return wm, actor, critic, cfg, payload


def _install_sigterm_flag() -> Tuple[Any, Any]:
    """Return (stop_requested, restore) handling SIGTERM during training."""
    import signal
    state = {"flag": False}
    old_handler = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):  # pragma: no cover - exercised via integration
        state["flag"] = True

    signal.signal(signal.SIGTERM, _handler)

    def stop_requested() -> bool:
        return state["flag"]

    def restore() -> None:
        signal.signal(signal.SIGTERM, old_handler)

    return stop_requested, restore


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_supermimic(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    device = resolve_device(args.device)
    cfg = SuperMimicConfig(
        seed=args.seed,
        device=str(device),
        imagine_horizon=args.imagine_horizon,
        actor_updates=args.actor_updates,
        wm_warmup_steps=args.wm_warmup_steps,
        burn_in_steps=args.burn_in_steps,
    )
    if args.small_config:
        cfg = dataclasses.replace(
            cfg,
            wm_config=dataclasses.replace(
                cfg.wm_config,
                cnn_channels=(8, 16, 32, 32),
                view_embed_dim=16,
                proprio_embed_dim=16,
                fused_dim=48,
                latent_dim=32,
                gru_dim=32,
                ensemble_size=3,
                window=4,
                batch_size=4,
            ),
            actor_hidden_dim=32,
            critic_hidden_dim=32,
            actor_batch_size=4,
            imagine_horizon=min(cfg.imagine_horizon, 4),
        )

    dataset = SharpaDataset(args.dataset)
    dataset_sha256 = _dataset_fingerprint(dataset)
    report = dataset.dynamics_data_report()
    splits = dataset.splits()
    train_eps = splits.get("train", [])
    val_eps = splits.get("val") or splits.get("validation") or []
    if not train_eps:
        raise DatasetError("no train split episodes")
    if not val_eps:
        raise DatasetError("no validation split episodes")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "dynamics_data_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    log_path = os.path.join(args.output, "train_log.jsonl")
    t_deadline = (started + args.time_budget
                  if args.time_budget and args.time_budget > 0 else None)
    stop_requested, restore_sigterm = _install_sigterm_flag()

    resume_payload: Optional[Dict[str, Any]] = None
    resume_provenance: Dict[str, Any] = {}
    if args.resume_from:
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"--resume-from not found: {args.resume_from}")
        payload = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        if payload.get("format") != SUPERMIMIC_CHECKPOINT_FORMAT:
            raise ValueError(f"not a {SUPERMIMIC_CHECKPOINT_FORMAT}: {args.resume_from}")
        resume_provenance = payload.get("provenance") or {}
        if resume_provenance.get("dataset_sha256") != dataset_sha256:
            raise ValueError(
                f"resume dataset mismatch: checkpoint {resume_provenance.get('dataset_sha256', '?')[:16]}... "
                f"vs current {dataset_sha256[:16]}..."
            )
        cfg_dict = dict(payload["config"])
        cfg_dict["wm_config"] = WorldModelConfig(**cfg_dict["wm_config"])
        cfg = SuperMimicConfig(**cfg_dict)
        resume_payload = payload

    run_id = resume_provenance.get("run_id") or str(uuid.uuid4())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    wm_cfg = cfg.wm_config
    wm = SharpaWorldModel(wm_cfg).to(device)
    actor = LatentActor(wm_cfg.gru_dim, ACTION_DIM, cfg.actor_hidden_dim).to(device)
    critic = LatentCritic(wm_cfg.gru_dim, cfg.critic_hidden_dim).to(device)

    wm_optimizer = torch.optim.AdamW(wm.parameters(), lr=wm_cfg.learning_rate)
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=cfg.actor_learning_rate)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=cfg.critic_learning_rate)

    wm_batcher = SequenceBatcher(train_eps, wm_cfg.window, wm_cfg.batch_size,
                                 cfg.seed, device)
    actor_batcher = SequenceBatcher(train_eps, wm_cfg.window, cfg.actor_batch_size,
                                    cfg.seed + 1, device)

    start_wm_step = 0
    start_actor_step = 0
    if resume_payload:
        wm.load_state_dict(resume_payload["model_state"]["wm"])
        actor.load_state_dict(resume_payload["model_state"]["actor"])
        critic.load_state_dict(resume_payload["model_state"]["critic"])
        wm_optimizer.load_state_dict(resume_payload["optimizer_state"]["wm"])
        actor_optimizer.load_state_dict(resume_payload["optimizer_state"]["actor"])
        critic_optimizer.load_state_dict(resume_payload["optimizer_state"]["critic"])
        _optimizer_to_device(wm_optimizer, device)
        _optimizer_to_device(actor_optimizer, device)
        _optimizer_to_device(critic_optimizer, device)
        wm_batcher.set_state(resume_payload["batcher_state"]["wm"])
        actor_batcher.set_state(resume_payload["batcher_state"]["actor"])
        metrics = dict(resume_payload.get("metrics", {}))
        start_wm_step = int(metrics.get("wm_updates", 0))
        start_actor_step = int(metrics.get("actor_updates", 0))
        rng_state = resume_payload.get("rng_state")
        if rng_state is None:
            raise ValueError("checkpoint missing rng_state; cannot resume reproducibly")
        restore_rng_state(rng_state)
    else:
        metrics: Dict[str, Any] = {
            "updates": 0,
            "wm_updates": 0,
            "actor_updates": 0,
            "critic_updates": 0,
            "run_id": run_id,
            "method_id": METHOD_ID,
            "parameters": {
                "wm": parameter_count(wm),
                "actor": parameter_count(actor),
                "critic": parameter_count(critic),
            },
            "device": str(device),
            "seed": cfg.seed,
            "stopped_reason": None,
        }

    provenance: Dict[str, Any] = {
        "dataset_path": args.dataset,
        "dataset_sha256": dataset_sha256,
        "run_id": run_id,
    }

    def _save(step_label: str) -> None:
        save_supermimic_checkpoint(
            os.path.join(args.output, f"checkpoint_{step_label}.pt"),
            wm, actor, critic, wm_optimizer, actor_optimizer, critic_optimizer,
            cfg, metrics, wm_batcher, actor_batcher, run_id, provenance)

    stop_reason = "steps"
    try:
        # 1. Offline world-model warmup (autoregressive latent loss).
        for step in range(start_wm_step + 1, cfg.wm_warmup_steps + 1):
            if t_deadline is not None and time.time() > t_deadline:
                stop_reason = "time_budget"
                break
            if stop_requested():
                stop_reason = "sigterm"
                break
            batch = wm_batcher.sample_batch()
            losses = compute_losses(wm, batch)
            wm_optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), wm_cfg.grad_clip)
            wm_optimizer.step()
            metrics["wm_updates"] = step
            metrics["updates"] = step
            if step == 1 or step % cfg.actor_log_every == 0:
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(json.dumps({
                        "phase": "wm_warmup",
                        "step": step,
                        "losses": {k: round(v.item(), 6) for k, v in losses.items()},
                        "elapsed_s": round(time.time() - started, 1)}) + "\n")

        if stop_reason == "steps":
            # 2. Actor + critic BPTT training.
            for step in range(start_actor_step + 1, cfg.actor_updates + 1):
                if t_deadline is not None and time.time() > t_deadline:
                    stop_reason = "time_budget"
                    break
                if stop_requested():
                    stop_reason = "sigterm"
                    break

                # Optionally train WM alongside actor.
                if cfg.actor_update_wm_every > 0 and step % cfg.actor_update_wm_every == 0:
                    batch = wm_batcher.sample_batch()
                    losses = compute_losses(wm, batch)
                    wm_optimizer.zero_grad(set_to_none=True)
                    losses["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(wm.parameters(), wm_cfg.grad_clip)
                    wm_optimizer.step()
                    metrics["wm_updates"] += 1

                # Actor update: freeze WM/critic weights, keep WM graph differentiable.
                for p in wm.parameters():
                    p.requires_grad = False
                for p in critic.parameters():
                    p.requires_grad = False
                for p in actor.parameters():
                    p.requires_grad = True

                h0, a_state0, v_state0 = sample_initial_states(
                    wm, actor, critic, actor_batcher, cfg.burn_in_steps)
                rollout = imagine_actor_rollout(
                    wm, actor, critic, h0, a_state0, v_state0,
                    cfg.imagine_horizon, cfg.actor_discount, cfg.lambda_return)
                actor_loss = -rollout["actor_return"].mean()
                actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.actor_grad_clip)
                actor_optimizer.step()
                metrics["actor_updates"] = step

                # Critic update: detached imagined states + burn-in-matched history.
                for p in wm.parameters():
                    p.requires_grad = True
                for p in critic.parameters():
                    p.requires_grad = True
                for p in actor.parameters():
                    p.requires_grad = False

                detached_states = rollout["states"].detach()
                v_state = v_state0.detach()
                critic_values: List[torch.Tensor] = []
                for t in range(cfg.imagine_horizon):
                    value, v_state = critic(detached_states[:, t], v_state)
                    critic_values.append(value)
                values_t = torch.stack(critic_values, dim=1)
                critic_loss = F.mse_loss(values_t, rollout["critic_target"])
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.critic_grad_clip)
                critic_optimizer.step()
                metrics["critic_updates"] = step

                # Restore gradients for next phase.
                for p in actor.parameters():
                    p.requires_grad = True

                metrics["updates"] = cfg.wm_warmup_steps + step
                if step == 1 or step % cfg.actor_log_every == 0:
                    with open(log_path, "a", encoding="utf-8") as log:
                        log.write(json.dumps({
                            "phase": "actor_critic",
                            "step": step,
                            "actor_loss": round(actor_loss.item(), 6),
                            "critic_loss": round(critic_loss.item(), 6),
                            "mean_return": round(rollout["actor_return"].mean().item(), 6),
                            "elapsed_s": round(time.time() - started, 1)}) + "\n")

                if step == cfg.actor_updates or step % max(1, cfg.actor_log_every * 5) == 0:
                    _save("latest")
    finally:
        _save("latest")
        restore_sigterm()

    metrics["stopped_reason"] = stop_reason
    metrics["wallclock_s"] = round(time.time() - started, 1)
    _save("final")
    with open(os.path.join(args.output, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": RUN_SCHEMA, "metrics": metrics,
                   "config": dataclasses.asdict(cfg),
                   "provenance": provenance}, fh, indent=2)
    return metrics


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_supermimic(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_device(args.device)
    wm, actor, critic, cfg, payload = load_supermimic_checkpoint(
        args.checkpoint, map_location=str(device))
    wm.to(device).eval()
    actor.to(device).eval()
    critic.to(device).eval()
    wm_cfg = cfg.wm_config

    if args.dataset:
        # open-loop policy rollouts against a dataset split (no env)
        dataset = SharpaDataset(args.dataset)
        eps = dataset.splits().get(args.openloop_split, [])
        if not eps:
            raise DatasetError(f"no episodes in split {args.openloop_split}")
        dev = next(wm.parameters()).device
        acc: Dict[str, List[float]] = {"return": [], "steps": [], "success": []}
        for ep in eps:
            if ep.T < 2:
                continue
            obs0 = ep.actor_obs(0)
            h = wm.advance(wm.init_state(1), wm.encode_obs_dict(obs0))
            a_state = actor.init_state(1, dev)
            ep_return = 0.0
            alive = 1.0
            for t in range(min(ep.T, args.max_steps)):
                action, a_state = actor(h, a_state)
                pred = wm.predict(h, action)
                reward = pred["reward"].mean(dim=0).item()
                cont = pred["continuation"].mean(dim=0).item()
                ep_return += alive * reward
                alive *= cont
                h = wm.advance(h, pred["next_emb"].mean(dim=0))
            acc["return"].append(ep_return)
            acc["steps"].append(min(ep.T, args.max_steps))
        result = {
            "schema": RUN_SCHEMA,
            "mode": "open_loop",
            "split": args.openloop_split,
            "episodes": len(acc["return"]),
            "mean_return": float(np.mean(acc["return"])) if acc["return"] else 0.0,
            "checkpoint": args.checkpoint,
        }
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            with open(os.path.join(args.output, "open_loop_eval.json"), "w") as fh:
                json.dump(result, fh, indent=2)
        return result

    if not args.env_factory:
        raise ValueError("evaluate requires --dataset or --env-factory")
    env_factory = _load_env_factory(args.env_factory)
    env_config = None
    if args.env_config and os.path.isfile(args.env_config):
        with open(args.env_config, "r", encoding="utf-8") as fh:
            env_config = json.load(fh)
    env = env_factory(env_config)
    episodes_out: List[Dict[str, Any]] = []
    total_steps = 0
    try:
        for ep_i in range(args.episodes):
            options: Dict[str, Any] = {"split": "heldout"}
            if args.task:
                options["task_id"] = args.task
            obs, info = env.reset(seed=args.seed * 100003 + ep_i, options=options)
            h = wm.init_state(1).to(device)
            a_state = actor.init_state(1, device)
            steps = 0
            ep_return = 0.0
            terminated = truncated = False
            done = False
            while not done:
                e = wm.encode_obs_dict(obs)
                h = wm.advance(h, e)
                action_t, a_state = actor(h, a_state)
                action = action_t[0].detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_steps += 1
                ep_return += float(reward)
                terminated = bool(terminated)
                truncated = bool(truncated)
                if steps >= args.max_steps and not (terminated or truncated):
                    truncated = True
                done = terminated or truncated
            success = None
            if isinstance(info, dict):
                for key in (args.success_key, "success", "task_success"):
                    if key in info:
                        success = bool(info[key])
                        break
            episodes_out.append({
                "episode_index": ep_i, "steps": steps, "return": ep_return,
                "terminated": terminated, "truncated": truncated,
                "success": success})
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    n_succ = sum(1 for e in episodes_out if e["success"] is True)
    result = {
        "schema": RUN_SCHEMA,
        "mode": "closed_loop_actor",
        "env_factory": args.env_factory,
        "checkpoint": args.checkpoint,
        "episodes": len(episodes_out),
        "total_steps": total_steps,
        "success_rate": n_succ / max(1, len(episodes_out)),
        "mean_return": (float(np.mean([e["return"] for e in episodes_out]))
                        if episodes_out else 0.0),
        "episode_records": episodes_out,
    }
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "closed_loop_eval.json"), "w") as fh:
            json.dump(result, fh, indent=2)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sharpa-supermimic",
        description="SuperMimic-style BPTT through a latent Sharpa world model.")
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="fit world model then train actor/critic")
    tr.add_argument("--dataset", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--wm-warmup-steps", type=int, default=2000)
    tr.add_argument("--actor-updates", type=int, default=5000)
    tr.add_argument("--imagine-horizon", type=int, default=16)
    tr.add_argument("--burn-in-steps", type=int, default=2)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--time-budget", type=float, default=0.0,
                    help="wall-clock seconds; <=0 disables. Checkpoint+exit on timeout.")
    tr.add_argument("--resume-from", default=None,
                    help="path to a v2 supermimic checkpoint to resume")
    tr.add_argument("--small-config", action="store_true",
                    help="tiny model + short horizon for software testing only")
    tr.set_defaults(func=train_supermimic)

    ev = sub.add_parser("evaluate", help="autonomous actor evaluation")
    ev.add_argument("--checkpoint", required=True,
                    help="unified sharpa-supermimic-checkpoint-v2 path")
    ev.add_argument("--env-factory", default=None)
    ev.add_argument("--dataset", default=None)
    ev.add_argument("--openloop-split", default="test")
    ev.add_argument("--episodes", type=int, default=100)
    ev.add_argument("--max-steps", type=int, default=600)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--task", default=None)
    ev.add_argument("--env-config", default=None)
    ev.add_argument("--success-key", default="success")
    ev.add_argument("--output", default=None)
    ev.add_argument("--device", default="auto")
    ev.set_defaults(func=evaluate_supermimic)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_argparser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    json.dump(main(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
