"""Compact recurrent latent world model + ensemble/CEM-MPC prototype for Sharpa learning.

This is an explicitly named prototype: "proto_sharpa_wm_v1". It is NOT the
official TD-MPC2 implementation (https://github.com/nicklashansen/tdmpc2) and
NOT DreamerV3 (https://github.com/danijar/dreamerv3). The design follows the
same family of ideas -- action-conditioned latent dynamics, learned
reward/continuation, visual encoder, receding-horizon action selection via
stochastic trajectory optimization -- but is a compact PyTorch re-implementation
scoped to the sharpa-learning contract. Do not relabel it as full TD-MPC2 or
Dreamer in reports.

Contract scope (see outputs/sharpa-learning-20261118-v001/CONTRACT.md):
- Actor input ONLY: screen_rgb [128,128,3] uint8, wrist_rgb [128,128,3] uint8,
  proprio float32 [58], touch float32 [5], touch_valid uint8 [5].
  No cursor/target coordinates, no oracle reward/distance, no teacher action.
- Action float32 [29] normalized [-1, 1], joint position increments.
- observation[t] PRECEDES action[t]; observation[t+1] follows 8 physics substeps.
- Reward/continuation targets come from valid real transitions only; failure
  episodes may be used for dynamics when their transitions are valid.
- Episode boundaries are never crossed by training windows or model rollouts.
- The learned model actually chooses motor actions via receding-horizon CEM;
  image reconstruction is only an auxiliary representation loss.

CLI:
    python -m embodied_trackpad.sharpa_learning.world_model train \
        --dataset dataset.json --output run_dir --steps 20000 --seed 0
    python -m embodied_trackpad.sharpa_learning.world_model evaluate \
        --env-factory embodied_trackpad.sharpa_learning.env:SharpaEnv \
        --checkpoint run_dir/checkpoint_final.pt --episodes 100
    # or open-loop model validation against a dataset split (no env needed):
    python -m embodied_trackpad.sharpa_learning.world_model evaluate \
        --dataset dataset.json --checkpoint run_dir/checkpoint_final.pt \
        --openloop-split test
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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - only without torch installed
    raise ModuleNotFoundError(
        "embodied_trackpad.sharpa_learning.world_model requires PyTorch."
    ) from exc

from embodied_trackpad.sharpa_learning.bc_ppo import (
    ACTION_DIM,
    ACTOR_OBS_KEYS,
    DATASET_SCHEMA,
    IMAGE_SIZE,
    PROPRIO_DIM,
    TOUCH_DIM,
    TOUCH_VALID_DIM,
    ContractError as _BCContractError,
    DatasetBundle as _BCDatasetBundle,
    Episode as _BCEpisode,
    load_dataset as _bc_load_dataset,
)

ARCHITECTURE_ID = "proto_sharpa_wm_v1"
CHECKPOINT_FORMAT = "sharpa-wm-checkpoint-v1"
RUN_SCHEMA = "sharpa-wm-run-v1"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorldModelConfig:
    """Frozen first-prototype architecture.

    Any change to capacity, horizon, or input/output contract must bump
    ``architecture_revision`` and be reported. Do not silently retrain a
    different architecture under the same id.
    """

    architecture_id: str = ARCHITECTURE_ID
    architecture_revision: str = "r0"

    # observation dims (frozen by contract)
    image_size: int = IMAGE_SIZE
    proprio_dim: int = PROPRIO_DIM
    touch_dim: int = TOUCH_DIM
    touch_valid_dim: int = TOUCH_DIM
    action_dim: int = ACTION_DIM

    # encoder
    cnn_channels: Tuple[int, ...] = (32, 64, 128, 128)
    view_embed_dim: int = 128
    proprio_embed_dim: int = 128
    fused_dim: int = 384          # 2 * view_embed + proprio_embed
    latent_dim: int = 256         # encoder output / GRU input

    # recurrent latent dynamics
    gru_dim: int = 256

    # ensemble (disagreement -> exposed uncertainty + rollout rejection)
    ensemble_size: int = 5

    # planning (receding-horizon CEM-MPC)
    horizon: int = 5
    cem_iterations: int = 6
    cem_samples: int = 256
    cem_elites: int = 16
    cem_init_std: float = 0.6
    cem_momentum: float = 0.1
    discount: float = 0.99

    # training
    window: int = 8               # training window length inside one episode
    batch_size: int = 64
    learning_rate: float = 3e-4
    grad_clip: float = 20.0
    recon_weight: float = 0.1     # auxiliary visual reconstruction weight
    log_every: int = 200
    eval_every: int = 1000

    # rollout validity (ensemble disagreement rejection thresholds)
    reject_reward_std: float = 0.5
    reject_latent_std: float = 1.0

    seed: int = 0
    device: str = "auto"               # auto|cpu|cuda


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Dataset loading (reuses the published BC loader validation)
# --------------------------------------------------------------------------- #
class DatasetError(ValueError):
    """Raised when the dataset index or an episode violates the fixed schema."""


class Episode:
    """World-model view of a BC-validated immutable episode.

    Wraps ``bc_ppo.Episode`` so the same SHA, provenance, split, source-kind,
    action-scale, and backend checks apply to dynamics data, while exposing the
    path as a string and adding task/outcome helpers used by the WM reports.
    """

    def __init__(self, bc_ep: _BCEpisode):
        self._ep = bc_ep

    @property
    def path(self) -> str:
        return str(self._ep.path)

    @property
    def episode_id(self) -> str:
        return self._ep.episode_id

    @property
    def split(self) -> str:
        return self._ep.split

    @property
    def source_kind(self) -> str:
        return self._ep.source_kind

    @property
    def task(self) -> str:
        return str(self._ep.manifest.get("task", "unknown"))

    @property
    def outcome(self) -> str:
        return str(self._ep.manifest.get("outcome", "unknown"))

    @property
    def training_eligible_for_bc(self) -> bool:
        return self._ep.training_eligible_for_bc

    @property
    def manifest(self) -> Dict[str, Any]:
        return self._ep.manifest

    @property
    def arrays(self) -> Dict[str, np.ndarray]:
        return self._ep.arrays

    @property
    def T(self) -> int:
        return self._ep.T

    @property
    def dynamics_eligible(self) -> bool:
        return self._ep.dynamics_eligible

    def actor_obs(self, t: int) -> Dict[str, np.ndarray]:
        return self._ep.actor_obs(t)

    @staticmethod
    def load(path: str) -> "Episode":
        try:
            bc_ep = _BCEpisode.load(path)
        except _BCContractError as exc:
            raise DatasetError(str(exc)) from exc
        return Episode(bc_ep)


class SharpaDataset:
    """Index (dataset.json) backed collection of immutable episodes.

    Wraps ``bc_ppo.load_dataset`` so formal world-model admission enforces the
    same whole-trajectory SHA, array SHA, PhysX backend, source kind, fixed
    splits, action names, and scales as the BC path. Valid failure transitions
    are accepted; mock fixtures must not reach the formal CLI.
    """

    def __init__(self, index_path: str, root: Optional[str] = None):
        # ``root`` is kept for backward compatibility; BC loader resolves
        # relative paths against the index directory.
        try:
            self.bundle = _bc_load_dataset(index_path)
        except _BCContractError as exc:
            raise DatasetError(str(exc)) from exc
        self.index_path = str(self.bundle.path)
        self.action_names = self.bundle.action_names
        self.delta_scale = self.bundle.delta_scale
        self.episodes: List[Episode] = [Episode(ep) for ep in self.bundle.episodes]

    def splits(self) -> Dict[str, List[Episode]]:
        out: Dict[str, List[Episode]] = {}
        for ep in self.episodes:
            out.setdefault(ep.split, []).append(ep)
        return out

    def dynamics_data_report(self) -> Dict[str, Any]:
        """Honest per-split provenance/count report for dynamics training."""
        report: Dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "report": "dynamics_data_distribution",
            "index_path": self.index_path,
            "splits": {},
            "totals": {"episodes": len(self.episodes), "transitions": 0,
                       "bc_ineligible_transitions_used": 0},
        }
        for split, eps in sorted(self.splits().items()):
            bucket: Dict[str, Any] = {}
            transitions = 0
            for ep in eps:
                if not ep.dynamics_eligible:
                    continue
                key = (f"{ep.source_kind}|outcome={ep.outcome}"
                       f"|bc_eligible={ep.training_eligible_for_bc}")
                b = bucket.setdefault(key, {"episodes": 0, "transitions": 0})
                b["episodes"] += 1
                b["transitions"] += ep.T
                transitions += ep.T
                if not ep.training_eligible_for_bc:
                    report["totals"]["bc_ineligible_transitions_used"] += ep.T
            report["splits"][split] = {"transitions": transitions, "buckets": bucket}
            report["totals"]["transitions"] += transitions
        return report


class SequenceBatcher:
    """Sample fixed-length windows strictly inside single episodes.

    Guarantees:
    - a window uses observations/actions from exactly one episode;
    - obs[t] precedes action[t]; targets are one step ahead within the same
      episode (largest target index is T);
    - no window crosses a reset boundary (episodes are never concatenated).
    """

    def __init__(self, episodes: Sequence[Episode], window: int, batch_size: int,
                 seed: int, device: torch.device):
        self.episodes = [ep for ep in episodes if ep.T >= 2]
        if not self.episodes:
            raise DatasetError("no episodes with >=2 transitions for batching")
        self.window = max(2, min(window, min(ep.T for ep in self.episodes)))
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.device = device

    def get_state(self) -> Dict[str, Any]:
        """Picklable sampler state for checkpoint resume."""
        return {
            "window": self.window,
            "batch_size": self.batch_size,
            "bit_generator_state": self.rng.bit_generator.state,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.window = int(state["window"])
        self.batch_size = int(state["batch_size"])
        self.rng.bit_generator.state = state["bit_generator_state"]

    def sample_batch(self) -> Dict[str, torch.Tensor]:
        W = self.window
        dev = self.device
        parts: Dict[str, List[torch.Tensor]] = {
            "screen": [], "wrist": [], "proprio": [], "touch": [], "touch_valid": [],
            "screen_next": [], "wrist_next": [], "proprio_next": [],
            "touch_next": [], "touch_valid_next": [],
            "action": [], "reward": [], "continuation": [],
        }
        for _ in range(self.batch_size):
            ep = self.episodes[int(self.rng.integers(len(self.episodes)))]
            hi = max(0, ep.T - W)
            start = int(self.rng.integers(0, hi + 1))
            a = ep.arrays
            idx = np.arange(start, start + W)
            nxt = idx + 1
            parts["screen"].append(torch.from_numpy(a["screen_rgb"][idx]).to(dev))
            parts["wrist"].append(torch.from_numpy(a["wrist_rgb"][idx]).to(dev))
            parts["proprio"].append(torch.from_numpy(a["proprio"][idx]).to(dev))
            parts["touch"].append(torch.from_numpy(a["touch"][idx]).to(dev))
            parts["touch_valid"].append(torch.from_numpy(a["touch_valid"][idx]).to(dev))
            parts["screen_next"].append(torch.from_numpy(a["screen_rgb"][nxt]).to(dev))
            parts["wrist_next"].append(torch.from_numpy(a["wrist_rgb"][nxt]).to(dev))
            parts["proprio_next"].append(torch.from_numpy(a["proprio"][nxt]).to(dev))
            parts["touch_next"].append(torch.from_numpy(a["touch"][nxt]).to(dev))
            parts["touch_valid_next"].append(torch.from_numpy(a["touch_valid"][nxt]).to(dev))
            parts["action"].append(torch.from_numpy(a["action"][idx]).to(dev))
            parts["reward"].append(torch.from_numpy(a["reward"][idx]).to(dev))
            parts["continuation"].append(
                torch.from_numpy((1.0 - a["terminated"][idx]).astype(np.float32)).to(dev))
        return {
            "screen": torch.stack(parts["screen"], 0),
            "wrist": torch.stack(parts["wrist"], 0),
            "proprio": torch.stack(parts["proprio"], 0),
            "touch": torch.stack(parts["touch"], 0),
            "touch_valid": torch.stack(parts["touch_valid"], 0),
            "screen_next": torch.stack(parts["screen_next"], 0),
            "wrist_next": torch.stack(parts["wrist_next"], 0),
            "proprio_next": torch.stack(parts["proprio_next"], 0),
            "touch_next": torch.stack(parts["touch_next"], 0),
            "touch_valid_next": torch.stack(parts["touch_valid_next"], 0),
            "action": torch.stack(parts["action"], 0),
            "reward": torch.stack(parts["reward"], 0),
            "continuation": torch.stack(parts["continuation"], 0),
        }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _conv_stack(channels: Sequence[int], embed_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_c = 3
    size = IMAGE_SIZE
    for c in channels:
        layers.append(nn.Conv2d(in_c, c, kernel_size=4, stride=2, padding=1))
        layers.append(nn.SiLU(inplace=True))
        in_c = c
        size //= 2
    layers.append(nn.Flatten())
    layers.append(nn.Linear(in_c * size * size, embed_dim))
    return nn.Sequential(*layers)


class _ViewDecoder(nn.Module):
    """Decoder reconstructing one RGB view at 1/4 resolution (auxiliary)."""

    def __init__(self, state_dim: int, channels: Sequence[int], image_size: int):
        super().__init__()
        self.out_size = image_size // 4
        ch = list(channels)
        self.proj = nn.Linear(state_dim, ch[-1] * 8 * 8)
        mods: List[nn.Module] = []
        while len(ch) > 1:
            mods += [nn.ConvTranspose2d(ch[-1], ch[-2], 4, 2, 1), nn.SiLU(inplace=True)]
            ch.pop()
        mods.append(nn.ConvTranspose2d(ch[0], 3, 4, 2, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.proj(h).view(h.shape[0], -1, 8, 8)
        x = self.net(x)
        return x[..., : self.out_size, : self.out_size]


class _EnsembleHeads(nn.Module):
    """K independent small heads over [state, action] (disagreement = uncertainty)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int, k: int,
                 out_dim: int):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden), nn.SiLU(inplace=True),
                nn.Linear(hidden, hidden), nn.SiLU(inplace=True),
                nn.Linear(hidden, out_dim),
            ) for _ in range(k)
        ])

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, a], dim=-1)
        return torch.stack([net(x) for net in self.nets], dim=0)  # [K, B, out]


class SharpaWorldModel(nn.Module):
    """Recurrent latent world model with ensemble reward/continuation/dynamics.

    Latent state h_t = GRUCell(e_t, h_{t-1}) with e_t encoded from whitelisted
    actor observations only. Heads over [h_t, a_t] predict reward, continuation
    logit, and the next observation embedding, so open-loop rollouts consume
    actions only -- action dependence is preserved and no future frames leak.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.screen_enc = _conv_stack(cfg.cnn_channels, cfg.view_embed_dim)
        self.wrist_enc = _conv_stack(cfg.cnn_channels, cfg.view_embed_dim)
        self.proprio_enc = nn.Sequential(
            nn.Linear(cfg.proprio_dim + cfg.touch_dim + cfg.touch_valid_dim,
                      cfg.proprio_embed_dim), nn.SiLU(inplace=True),
            nn.Linear(cfg.proprio_embed_dim, cfg.proprio_embed_dim), nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Linear(2 * cfg.view_embed_dim + cfg.proprio_embed_dim, cfg.fused_dim),
            nn.SiLU(inplace=True),
        )
        self.to_latent = nn.Linear(cfg.fused_dim, cfg.latent_dim)
        self.gru = nn.GRUCell(cfg.latent_dim, cfg.gru_dim)
        self.reward_heads = _EnsembleHeads(cfg.gru_dim, cfg.action_dim, 256,
                                           cfg.ensemble_size, 1)
        self.cont_heads = _EnsembleHeads(cfg.gru_dim, cfg.action_dim, 256,
                                         cfg.ensemble_size, 1)
        self.next_emb_heads = _EnsembleHeads(cfg.gru_dim, cfg.action_dim, 256,
                                             cfg.ensemble_size, cfg.latent_dim)
        self.screen_dec = _ViewDecoder(cfg.gru_dim, cfg.cnn_channels, cfg.image_size)
        self.wrist_dec = _ViewDecoder(cfg.gru_dim, cfg.cnn_channels, cfg.image_size)
        self.proprio_dec = nn.Sequential(
            nn.Linear(cfg.gru_dim, 256), nn.SiLU(inplace=True),
            nn.Linear(256, cfg.proprio_dim + cfg.touch_dim),
        )

    # -- encoding -------------------------------------------------------- #
    def encode_obs_arrays(self, screen: torch.Tensor, wrist: torch.Tensor,
                          proprio: torch.Tensor, touch: torch.Tensor,
                          touch_valid: torch.Tensor) -> torch.Tensor:
        if screen.dtype == torch.uint8:
            screen = screen.float() / 255.0
        if wrist.dtype == torch.uint8:
            wrist = wrist.float() / 255.0
        e_s = self.screen_enc(screen.permute(0, 3, 1, 2))
        e_w = self.wrist_enc(wrist.permute(0, 3, 1, 2))
        pv = torch.cat([proprio.float(), touch.float(), touch_valid.float()], dim=-1)
        e_p = self.proprio_enc(pv)
        e = self.fuse(torch.cat([e_s, e_w, e_p], dim=-1))
        return self.to_latent(e)  # [B, latent_dim]

    def encode_obs_dict(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        keys = set(obs.keys())
        extra = keys - set(ACTOR_OBS_KEYS)
        if extra:
            raise DatasetError(
                f"actor observation contains non-whitelisted keys: {sorted(extra)}")
        dev = next(self.parameters()).device
        to = lambda k, dt: torch.as_tensor(np.asarray(obs[k]), dtype=dt, device=dev)
        return self.encode_obs_arrays(
            to("screen_rgb", torch.uint8).unsqueeze(0),
            to("wrist_rgb", torch.uint8).unsqueeze(0),
            to("proprio", torch.float32).unsqueeze(0),
            to("touch", torch.float32).unsqueeze(0),
            to("touch_valid", torch.float32).unsqueeze(0),
        )

    def init_state(self, batch: int = 1) -> torch.Tensor:
        dev = next(self.parameters()).device
        return torch.zeros(batch, self.cfg.gru_dim, device=dev)

    def advance(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.gru(e, h)

    def predict(self, h: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One-step ensemble prediction from latent state h and action."""
        r = self.reward_heads(h, action).squeeze(-1)          # [K, B]
        c = self.cont_heads(h, action).squeeze(-1)            # [K, B]
        de = self.next_emb_heads(h, action)                   # [K, B, latent]
        return {
            "reward": r,
            "continuation_logit": c,
            "continuation": torch.sigmoid(c),
            "next_emb": de,
            "reward_std": r.std(dim=0),                       # [B]
            "latent_std": de.std(dim=0).mean(dim=-1) / (
                de.mean(dim=0).abs().mean(dim=-1) + 1e-6),    # normalized [B]
        }

    def reconstruct(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "screen_small": self.screen_dec(h),
            "wrist_small": self.wrist_dec(h),
            "proprio_touch": self.proprio_dec(h),
        }

    def imagine(self, h: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Roll the latent state forward under an action sequence [B, H, A].

        Purely action-conditioned: consumes only the actions after the start
        frame. Returns ensemble-mean per-step reward/continuation plus
        disagreement used for uncertainty exposure and rejection.
        """
        B, H, _ = actions.shape
        rewards, conts, reward_stds, latent_stds = [], [], [], []
        hh = h
        for t in range(H):
            pred = self.predict(hh, actions[:, t])
            rewards.append(pred["reward"].mean(dim=0))
            conts.append(pred["continuation"].mean(dim=0))
            reward_stds.append(pred["reward_std"])
            latent_stds.append(pred["latent_std"])
            hh = self.advance(hh, pred["next_emb"].mean(dim=0))
        return {
            "reward": torch.stack(rewards, dim=1),            # [B, H]
            "continuation": torch.stack(conts, dim=1),
            "reward_std": torch.stack(reward_stds, dim=1),
            "latent_std": torch.stack(latent_stds, dim=1),
            "final_state": hh,
        }

    def rollout_valid(self, imagined: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-sample validity: reject when ensemble disagreement is too high."""
        bad_r = imagined["reward_std"].max(dim=1).values > self.cfg.reject_reward_std
        bad_z = imagined["latent_std"].max(dim=1).values > self.cfg.reject_latent_std
        return ~(bad_r | bad_z)


class CEMPlanner:
    """Receding-horizon CEM over learned latent dynamics (prototype MPC).

    Scores sampled action sequences with discounted predicted reward gated by
    predicted continuation, using ensemble means. Returns the first action of
    the refined mean and whether the chosen action passed ensemble-disagreement
    checks. There is no privileged fallback controller: a rejected rollout is
    surfaced to the caller, which must act without model confidence (e.g. hold
    position) and count the rejection.
    """

    def __init__(self, model: SharpaWorldModel, cfg: WorldModelConfig):
        self.model = model
        self.cfg = cfg

    @torch.no_grad()
    def plan(self, h: torch.Tensor, generator: Optional[torch.Generator] = None
             ) -> Tuple[torch.Tensor, bool]:
        cfg = self.cfg
        B = h.shape[0]
        dev = h.device
        mean = torch.zeros(B, cfg.horizon, cfg.action_dim, device=dev)
        std = torch.full((B, cfg.horizon, cfg.action_dim), cfg.cem_init_std, device=dev)
        for _ in range(cfg.cem_iterations):
            noise = torch.randn(B, cfg.cem_samples, cfg.horizon, cfg.action_dim,
                                device=dev, generator=generator)
            samples = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-1.0, 1.0)
            flat = samples.reshape(B * cfg.cem_samples, cfg.horizon, cfg.action_dim)
            h_rep = h.repeat_interleave(cfg.cem_samples, dim=0)
            imagined = self.model.imagine(h_rep, flat)
            valid = self.model.rollout_valid(imagined)
            disc = cfg.discount ** torch.arange(cfg.horizon, device=dev, dtype=torch.float32)
            cont_gate = torch.cumprod(
                torch.cat([torch.ones_like(imagined["continuation"][:, :1]),
                           imagined["continuation"][:, :-1]], dim=1), dim=1)
            ret = (imagined["reward"] * cont_gate * disc).sum(dim=1)
            ret = torch.where(valid, ret, torch.full_like(ret, -1e9))
            ret = ret.view(B, cfg.cem_samples)
            vals, idx = torch.topk(ret, k=cfg.cem_elites, dim=1)
            chosen = torch.gather(
                samples, 1,
                idx.view(B, cfg.cem_elites, 1, 1).expand(-1, -1, cfg.horizon, cfg.action_dim))
            mean = cfg.cem_momentum * mean + (1 - cfg.cem_momentum) * chosen.mean(dim=1)
            std = chosen.std(dim=1).clamp_min(1e-3)
        action = mean[:, 0].clamp(-1.0, 1.0)
        one = self.model.imagine(h, action.unsqueeze(1))
        valid = self.model.rollout_valid(one)
        return action, bool(valid.all().item())


# --------------------------------------------------------------------------- #
# Loss / training
# --------------------------------------------------------------------------- #
def compute_losses(model: SharpaWorldModel, batch: Dict[str, torch.Tensor]
                   ) -> Dict[str, torch.Tensor]:
    """Autoregressive latent-world-model loss over a [B, W] window.

    The latent state h_t is rolled forward with the model's own predicted next
    embeddings; real next embeddings are used only as targets. Reconstruction
    decodes the *current* latent state back to current proprio/touch. Per-finger
    touch valid masks are applied before averaging; ensemble head losses are
    computed independently.
    """
    cfg = model.cfg
    B, W = batch["action"].shape[:2]
    h = model.init_state(B)
    latent_loss = reward_loss = cont_loss = recon_loss = torch.tensor(0.0,
                                                                       device=h.device)
    for t in range(W):
        e = model.encode_obs_arrays(
            batch["screen"][:, t], batch["wrist"][:, t], batch["proprio"][:, t],
            batch["touch"][:, t], batch["touch_valid"][:, t])
        h = model.advance(h, e)
        pred = model.predict(h, batch["action"][:, t])
        e_next = model.encode_obs_arrays(
            batch["screen_next"][:, t], batch["wrist_next"][:, t],
            batch["proprio_next"][:, t], batch["touch_next"][:, t],
            batch["touch_valid_next"][:, t])
        # Independent head losses over the ensemble.
        latent_loss = latent_loss + F.smooth_l1_loss(
            pred["next_emb"], e_next.unsqueeze(0).expand_as(pred["next_emb"]))
        reward_loss = reward_loss + F.mse_loss(
            pred["reward"], batch["reward"][:, t].unsqueeze(0).expand_as(pred["reward"]))
        cont_loss = cont_loss + F.binary_cross_entropy_with_logits(
            pred["continuation_logit"],
            batch["continuation"][:, t].unsqueeze(0).expand_as(pred["continuation_logit"]))
        # Reconstruct current latent state against current proprio/touch.
        recon = model.reconstruct(h)
        rec_img_s = F.mse_loss(
            recon["screen_small"],
            F.avg_pool2d(batch["screen"][:, t].permute(0, 3, 1, 2).float() / 255.0, 4))
        rec_img_w = F.mse_loss(
            recon["wrist_small"],
            F.avg_pool2d(batch["wrist"][:, t].permute(0, 3, 1, 2).float() / 255.0, 4))
        rec_p = F.mse_loss(recon["proprio_touch"][:, :cfg.proprio_dim],
                           batch["proprio"][:, t])
        t_mask = batch["touch_valid"][:, t].float()
        touch_pred = recon["proprio_touch"][:, cfg.proprio_dim:]
        touch_true = batch["touch"][:, t]
        rec_t = ((touch_pred - touch_true) ** 2 * t_mask).sum() / (t_mask.sum() + 1e-8)
        recon_loss = recon_loss + rec_img_s + rec_img_w + rec_p + rec_t
        # Autoregressive roll: predicted next embedding becomes next latent state.
        h = model.advance(h, pred["next_emb"].mean(dim=0))
    latent_loss = latent_loss / W
    reward_loss = reward_loss / W
    cont_loss = cont_loss / W
    recon_loss = recon_loss / W
    total = latent_loss + reward_loss + cont_loss + cfg.recon_weight * recon_loss
    return {"loss": total, "latent": latent_loss, "reward": reward_loss,
            "continuation": cont_loss, "reconstruction": recon_loss}


# --------------------------------------------------------------------------- #
# Multi-step open-loop validation vs persistence baseline
# --------------------------------------------------------------------------- #
@torch.no_grad()
def open_loop_eval(model: SharpaWorldModel, episodes: Sequence[Episode],
                   horizons: Sequence[int] = (1, 2, 4, 8),
                   samples_per_episode: int = 4) -> Dict[str, Any]:
    """Multi-step action-conditioned prediction error vs persistence baseline.

    Persistence predicts proprio/touch/reward stay at their rollout-start
    values (zero delta). The model roll consumes only actions after the start
    frame. Reports per-horizon MSE on proprio, touch (contact), reward and
    1-step latent embedding.
    """
    model.eval()
    dev = next(model.parameters()).device
    rng = np.random.default_rng(1234)
    acc: Dict[int, Dict[str, List[float]]] = {
        k: {"proprio": [], "touch": [], "reward": [], "latent": [],
            "persistence_proprio": [], "persistence_touch": [],
            "persistence_reward": []}
        for k in horizons}
    for ep in episodes:
        if ep.T < max(horizons) + 2:
            continue
        for _ in range(samples_per_episode):
            start = int(rng.integers(0, ep.T - max(horizons)))
            obs0 = ep.actor_obs(start)
            e0 = model.encode_obs_dict(obs0)
            h = model.advance(model.init_state(1), e0)
            max_h = max(horizons)
            acts = torch.as_tensor(
                ep.arrays["action"][start:start + max_h],
                dtype=torch.float32, device=dev)
            imagined = model.imagine(h, acts.unsqueeze(0))
            for k in horizons:
                pred_reward = imagined["reward"][0, k - 1].item()
                hh = h
                for t in range(k):
                    pred = model.predict(hh, acts[t].unsqueeze(0))
                    hh = model.advance(hh, pred["next_emb"].mean(dim=0))
                recon = model.reconstruct(hh)["proprio_touch"][0]
                pred_proprio = recon[:PROPRIO_DIM].cpu().numpy()
                pred_touch = recon[PROPRIO_DIM:PROPRIO_DIM + TOUCH_DIM].cpu().numpy()
                true_proprio = ep.arrays["proprio"][start + k]
                true_touch = ep.arrays["touch"][start + k]
                true_reward = float(ep.arrays["reward"][start + k - 1])
                base_proprio = obs0["proprio"]
                base_touch = obs0["touch"]
                base_reward = float(ep.arrays["reward"][start])
                acc[k]["proprio"].append(float(np.mean((pred_proprio - true_proprio) ** 2)))
                acc[k]["touch"].append(float(np.mean((pred_touch - true_touch) ** 2)))
                acc[k]["reward"].append((pred_reward - true_reward) ** 2)
                acc[k]["persistence_proprio"].append(
                    float(np.mean((base_proprio - true_proprio) ** 2)))
                acc[k]["persistence_touch"].append(
                    float(np.mean((base_touch - true_touch) ** 2)))
                acc[k]["persistence_reward"].append((base_reward - true_reward) ** 2)
                e_true_next = model.encode_obs_dict(ep.actor_obs(start + k))
                pred_emb = model.predict(h, acts[0].unsqueeze(0))["next_emb"].mean(dim=0)
                acc[k]["latent"].append(float(F.mse_loss(pred_emb, e_true_next).item()))
    out: Dict[str, Any] = {"horizons": {}}
    for k in horizons:
        out["horizons"][str(k)] = {m: float(np.mean(v)) for m, v in acc[k].items() if v}
    model.train()
    return out


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: SharpaWorldModel,
                    optimizer: Optional[torch.optim.Optimizer], step: int,
                    metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None
                    ) -> None:
    payload: Dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "architecture_id": model.cfg.architecture_id,
        "architecture_revision": model.cfg.architecture_revision,
        "config": dataclasses.asdict(model.cfg),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": int(step),
        "metrics": metrics,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_id": (extra or {}).get("run_id", str(uuid.uuid4())),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: Optional[str] = None
                    ) -> Tuple[SharpaWorldModel, WorldModelConfig, Dict[str, Any]]:
    payload = torch.load(path, map_location=map_location or "cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"not a {CHECKPOINT_FORMAT}: {path}")
    cfg = WorldModelConfig(**payload["config"])
    model = SharpaWorldModel(cfg)
    model.load_state_dict(payload["model_state"])
    return model, cfg, payload


# --------------------------------------------------------------------------- #
# Training entry
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    run_id = str(uuid.uuid4())
    os.makedirs(args.output, exist_ok=True)
    device = resolve_device(args.device)
    cfg = WorldModelConfig(seed=args.seed, device=str(device))
    if getattr(args, "compact", False):
        cfg = dataclasses.replace(
            cfg,
            cnn_channels=(8, 16, 32, 32),
            view_embed_dim=16,
            proprio_embed_dim=16,
            fused_dim=48,
            latent_dim=32,
            gru_dim=32,
            ensemble_size=3,
            window=4,
            batch_size=4,
            log_every=2,
            eval_every=0,
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dataset = SharpaDataset(args.dataset)
    report = dataset.dynamics_data_report()
    with open(os.path.join(args.output, "dynamics_data_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    splits = dataset.splits()
    train_eps = splits.get("train", [])
    val_eps = splits.get("val") or splits.get("validation") or []
    if not train_eps:
        raise DatasetError("no train split episodes")
    if not val_eps:
        raise DatasetError("no validation split episodes; refusing to use train data for validation")

    model = SharpaWorldModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    batcher = SequenceBatcher(train_eps, cfg.window, cfg.batch_size, args.seed, device)
    log_path = os.path.join(args.output, "train_log.jsonl")
    metrics: Dict[str, Any] = {
        "updates": 0, "run_id": run_id,
        "architecture_id": cfg.architecture_id,
        "architecture_revision": cfg.architecture_revision,
        "parameters": parameter_count(model),
        "device": str(device), "seed": args.seed,
        "time_budget_s": args.time_budget, "stopped_reason": None,
    }
    stop_reason = "steps"
    t_deadline = (started + args.time_budget
                  if args.time_budget and args.time_budget > 0 else None)
    for step in range(1, args.steps + 1):
        if t_deadline is not None and time.time() > t_deadline:
            stop_reason = "time_budget"
            break
        batch = batcher.sample_batch()
        losses = compute_losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        metrics["updates"] = step
        if step == 1 or step % cfg.log_every == 0:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps({
                    "step": step,
                    "losses": {k: round(v.item(), 6) for k, v in losses.items()},
                    "elapsed_s": round(time.time() - started, 1)}) + "\n")
        if step == args.steps or (cfg.eval_every > 0 and step % cfg.eval_every == 0):
            ol = open_loop_eval(model, val_eps)
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps({"step": step, "open_loop_val": ol,
                                      "elapsed_s": round(time.time() - started, 1)}) + "\n")
            save_checkpoint(os.path.join(args.output, "checkpoint_latest.pt"),
                            model, optimizer, step, metrics,
                            extra={"run_id": run_id, "open_loop_val": ol})
    metrics["stopped_reason"] = stop_reason
    metrics["wallclock_s"] = round(time.time() - started, 1)
    ol = open_loop_eval(model, val_eps)
    for name in ("checkpoint_latest.pt", "checkpoint_final.pt"):
        save_checkpoint(os.path.join(args.output, name), model, optimizer,
                        metrics["updates"], metrics,
                        extra={"run_id": run_id, "open_loop_val": ol})
    with open(os.path.join(args.output, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"schema": RUN_SCHEMA, "metrics": metrics,
                   "config": dataclasses.asdict(cfg)}, fh, indent=2)
    return metrics


# --------------------------------------------------------------------------- #
# Evaluation entry
# --------------------------------------------------------------------------- #
def _load_env_factory(spec: str):
    if ":" not in spec:
        raise ValueError("--env-factory must be 'module:ClassName'")
    mod_name, attr = spec.split(":", 1)
    import importlib

    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = resolve_device(args.device)
    model, cfg, payload = load_checkpoint(args.checkpoint, map_location=str(device))
    model.to(device).eval()
    planner = CEMPlanner(model, cfg)

    if args.dataset:
        dataset = SharpaDataset(args.dataset)
        eps = dataset.splits().get(args.openloop_split, [])
        if not eps:
            raise DatasetError(f"no episodes in split {args.openloop_split}")
        ol = open_loop_eval(model, eps, samples_per_episode=args.openloop_samples)
        result = {"schema": RUN_SCHEMA, "mode": "open_loop",
                  "split": args.openloop_split, "episodes": len(eps),
                  "checkpoint": args.checkpoint, "payload_step": payload.get("step"),
                  "open_loop": ol}
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
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(args.seed)
    episodes_out: List[Dict[str, Any]] = []
    rejected_rollouts = 0
    total_steps = 0
    try:
        for ep_i in range(args.episodes):
            options: Dict[str, Any] = {"split": "heldout"}
            if args.task:
                options["task_id"] = args.task
            obs, info = env.reset(seed=args.seed * 100003 + ep_i, options=options)
            h = model.init_state(1).to(device)
            steps = 0
            ep_return = 0.0
            ep_rejected = 0
            terminated = truncated = False
            done = False
            while not done:
                e = model.encode_obs_dict(obs)
                h = model.advance(h, e)
                action_t, valid = planner.plan(h, generator=torch_gen)
                if not valid:
                    # No privileged fallback: expose rejection, hold position.
                    rejected_rollouts += 1
                    ep_rejected += 1
                    action = np.zeros(ACTION_DIM, dtype=np.float32)
                else:
                    action = action_t[0].detach().cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                total_steps += 1
                ep_return += float(reward)
                terminated = bool(terminated)
                truncated = bool(truncated)
                if steps >= args.max_steps and not (terminated or truncated):
                    truncated = True  # evaluator-side timeout handling
                done = terminated or truncated
            success = None
            if isinstance(info, dict):  # privileged evaluator fields, actor never reads
                for key in (args.success_key, "success", "task_success"):
                    if key in info:
                        success = bool(info[key])
                        break
            episodes_out.append({
                "episode_index": ep_i, "steps": steps, "return": ep_return,
                "terminated": terminated, "truncated": truncated,
                "success": success, "rejected_rollouts": ep_rejected})
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    n_succ = sum(1 for e in episodes_out if e["success"] is True)
    result = {"schema": RUN_SCHEMA, "mode": "closed_loop",
              "env_factory": args.env_factory,
              "checkpoint": args.checkpoint, "episodes": len(episodes_out),
              "total_steps": total_steps,
              "success_rate": n_succ / max(1, len(episodes_out)),
              "mean_return": (float(np.mean([e["return"] for e in episodes_out]))
                              if episodes_out else 0.0),
              "rejected_rollouts": rejected_rollouts,
              "episode_records": episodes_out}
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
        prog="sharpa-world-model",
        description="Compact recurrent latent world model + CEM-MPC prototype "
                    f"({ARCHITECTURE_ID}). Not official TD-MPC2/DreamerV3.")
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="train on a sharpa-learning dataset index")
    tr.add_argument("--dataset", required=True, help="path to dataset.json")
    tr.add_argument("--output", required=True, help="run output directory")
    tr.add_argument("--steps", type=int, default=20000)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--time-budget", type=float, default=0.0,
                    help="wall-clock seconds; <=0 disables. Checkpoint+exit on timeout.")
    tr.add_argument("--device", default="auto")
    tr.add_argument("--compact", action="store_true",
                    help="tiny model + small batch for fast software testing only")
    tr.set_defaults(func=train)

    ev = sub.add_parser("evaluate", help="closed-loop env eval or open-loop dataset eval")
    ev.add_argument("--env-factory", default=None,
                    help="e.g. embodied_trackpad.sharpa_learning.env:SharpaEnv")
    ev.add_argument("--dataset", default=None)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--episodes", type=int, default=100)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--max-steps", type=int, default=600)
    ev.add_argument("--task", default=None)
    ev.add_argument("--env-config", default=None)
    ev.add_argument("--success-key", default="success")
    ev.add_argument("--openloop-split", default="test")
    ev.add_argument("--openloop-samples", type=int, default=4)
    ev.add_argument("--output", default=None)
    ev.add_argument("--device", default="auto")
    ev.set_defaults(func=evaluate)
    return p


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_argparser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    json.dump(main(), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
