"""Visual recurrent BC + student-state corrections + PPO (Sharpa trackpad).

Owned by the bcppo campaign worker. This is the learning half of the
``embodied_trackpad.sharpa_learning`` namespace; the environment worker owns
``sharpa_learning/env.py`` (``SharpaEnv``) and the data manifest. This module
therefore imports the environment only lazily through the frozen factory
string ``embodied_trackpad.sharpa_learning.env:SharpaEnv``.

Campaign contract: ``outputs/sharpa-learning-20261118-v001/CONTRACT.md``
(schema ``sharpa-learning-v1``). Data is an index (``dataset.json``) pointing
at immutable episode directories (``manifest.json`` + ``trajectory.npz``);
there is no nested-JSON step format. Observations are length T+1, actions /
reward / terminated / truncated are length T; ``observation[t]`` PRECEDES
``action[t]``. Actor observations are exactly the five approved keys below;
privileged ``oracle_*`` / ``teacher_*`` arrays may exist in an NPZ but are
never read into the actor path, and the environment ``info`` mapping is never
read by the actor.

Pipeline: BC checkpoint -> heldout eval -> student roll-in corrections
(expert_action at the *actual* visited student state) -> correction
retraining -> heldout eval -> PPO with a BC anchor on real transitions ->
heldout eval. Stage checkpoints are preserved, never overwritten. Optimizer
steps, real env transitions, data counts, lr and losses are logged
separately from task success.

The tanh-Gaussian math, GAE, and PPO clipped surrogate follow the standard
formulas (see docs/research PPO references; no third-party code is copied).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Frozen contracts
# ---------------------------------------------------------------------------

DATASET_SCHEMA = "sharpa-learning-v1"
DEFAULT_ENV_FACTORY = "embodied_trackpad.sharpa_learning.env:SharpaEnv"

# Exact actor observation whitelist from the frozen contract. An environment
# observation must carry exactly these keys; an episode NPZ may additionally
# carry privileged oracle_* arrays and executed_q_ref/teacher_* label arrays,
# which the loaders below never concatenate into actor features.
ACTOR_OBS_KEYS = ("screen_rgb", "wrist_rgb", "proprio", "touch", "touch_valid")

# Arrays every trajectory.npz must contain (observations T+1, the rest T).
NPZ_REQUIRED_KEYS = ACTOR_OBS_KEYS + ("action", "reward", "terminated", "truncated", "sim_time", "executed_q_ref")
# Optional per-contract label/provenance arrays (also length T).
NPZ_OPTIONAL_KEYS = ("teacher_action", "teacher_valid")
# Extra NPZ arrays are only tolerated under this privileged prefix.
PRIVILEGED_NPZ_PREFIX = "oracle_"

# Privileged/legacy keys that must never appear in an actor observation.
REJECTED_OBS_KEYS = frozenset(
    {
        "teacher_action", "teacher_valid", "executed_q_ref", "expert_action",
        "student_action", "force", "jacobian", "reward", "outcome", "seed",
        "class", "split", "target_class", "episode_id", "frame_id", "oracle",
        "cursor_truth", "gui_state", "cursor_xy_px", "target_xy_px",
        "gui_buttons", "privileged", "pad_contact_truth", "device_truth",
        "click_truth",
    }
)

# Frozen actor dimensions.
ACTION_DIM = 29          # 7 Panda arm + 22 Sharpa hand channels.
PROPRIO_DIM = 58         # q (29) then qd (29), canonical order.
TOUCH_DIM = 5            # per-finger simulated normal force.
TOUCH_VALID_DIM = 5      # per-finger contact-valid flags.
IMAGE_SIZE = 128         # both cameras, uint8 HWC.

# Episode source kinds accepted by the loader. Old LEAP / raw source-video /
# mock / proxy material is rejected at validation time.
ALLOWED_SOURCE_KINDS = frozenset(
    {"sim_geometric_expert", "video_retargeted_sim", "feedback_expert_v1",
     "student_rollin_correction", "policy_rollout"}
)
ALLOWED_SPLITS = frozenset({"train", "val", "test"})

CHECKPOINT_FORMAT = "sharpa-bcppo-checkpoint-v1"
LOG_SCHEMA = "sharpa-bcppo-log-v1"

MANIFEST_REQUIRED_KEYS = {
    "schema", "episode_id", "source_kind", "action_names", "delta_scale",
    "joint_limits", "config", "seed", "split", "task", "outcome",
    "training_eligible_for_bc", "measured_contact_backend", "array_sha256",
    "trajectory_sha256",
}
INDEX_EPISODE_REQUIRED_KEYS = {
    "path", "episode_id", "split", "source_kind", "training_eligible_for_bc",
}

# Old geometric gap-spring proxy is not a measured contact backend.
_CONTACT_BACKEND_BANNED_MARKERS = ("gap", "spring", "proxy", "geometric")


class ContractError(ValueError):
    """Dataset, checkpoint, or environment contract violation."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def append_log(output_dir: Path, record: dict) -> None:
    """Append one JSONL training log line (losses/lr/steps) to output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    line = dict(record)
    line.setdefault("schema", LOG_SCHEMA)
    line.setdefault("utc", _utc_now())
    with open(output_dir / "train_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")


class _StopRequested:
    """Raised internally when SIGTERM or the wall-clock deadline fires."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__()


# ---------------------------------------------------------------------------
# RNG state capture / restore (full reproducibility across checkpoints)
# ---------------------------------------------------------------------------

def capture_rng_state() -> dict:
    """Snapshot python / numpy / torch CPU+CUDA RNG state into a picklable dict."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# tanh-Gaussian policy math
# ---------------------------------------------------------------------------

LOG_2 = math.log(2.0)
TANH_EPS = 1e-6


def tanh_log_det_jacobian(raw_action: torch.Tensor) -> torch.Tensor:
    """Sum log|d tanh(u)/du| = sum log(1 - tanh(u)^2), computed stably.

    Uses log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u)) so no atanh()
    or clamp near |a|=1 is needed on the raw (pre-squash) path.
    """
    return (2.0 * (LOG_2 - raw_action - F.softplus(-2.0 * raw_action))).sum(dim=-1)


def squashed_log_prob(
    mean: torch.Tensor, log_std: torch.Tensor, raw_action: torch.Tensor
) -> torch.Tensor:
    """log pi(a) with a = tanh(u), u ~ N(mean, std); raw_action is u."""
    var = (2.0 * log_std).exp()
    raw_lp = -0.5 * ((raw_action - mean) ** 2 / var + 2.0 * log_std + math.log(2.0 * math.pi)).sum(dim=-1)
    return raw_lp - tanh_log_det_jacobian(raw_action)


def sample_squashed(
    mean: torch.Tensor, log_std: torch.Tensor, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample (squashed_action, raw_action); raw_action is what log-probs use."""
    std = log_std.exp()
    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    raw = mean + std * noise
    return torch.tanh(raw), raw


def deterministic_squashed(mean: torch.Tensor) -> torch.Tensor:
    return torch.tanh(mean)


def probability_ratio(
    mean_new: torch.Tensor,
    log_std_new: torch.Tensor,
    mean_old: torch.Tensor,
    log_std_old: torch.Tensor,
    raw_action: torch.Tensor,
) -> torch.Tensor:
    """PPO ratio pi_new(a)/pi_old(a) for the *same* sampled squashed action.

    Both log-probs are evaluated at the stored raw (pre-tanh) action, so the
    tanh Jacobian term cancels algebraically but is kept for numerical parity
    with the standalone log-prob path.
    """
    lp_new = squashed_log_prob(mean_new, log_std_new, raw_action)
    lp_old = squashed_log_prob(mean_old, log_std_old, raw_action)
    return torch.exp(lp_new - lp_old)


# ---------------------------------------------------------------------------
# GAE with terminal-vs-timeout distinction
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation.

    ``values`` has length len(rewards) + 1 (bootstrap value of the final
    observation). ``terminated`` marks environment-decided ends (success /
    failure): the bootstrap is cut there (true terminal -> zero). ``truncated``
    marks time-limit cuts (including an evaluator-imposed step cap): the
    bootstrap value of the final observation *is* used, so a timeout does not
    zero the return. A step that is both is terminal.
    """
    steps = len(rewards)
    if not (len(values) == steps + 1 and len(terminated) == steps and len(truncated) == steps):
        raise ContractError("GAE input length mismatch")
    adv = np.zeros(steps, dtype=np.float64)
    last = 0.0
    for t in reversed(range(steps)):
        nonterminal = 0.0 if terminated[t] else 1.0
        next_value = values[t + 1] if not terminated[t] else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + np.asarray(values[:steps], dtype=np.float64)
    return adv, returns


# ---------------------------------------------------------------------------
# Normalization (images: fixed per-channel; proprio/touch: running Welford)
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Running mean/std normalizer for proprio + touch vectors.

    State lives in registered buffers so it is saved/restored inside the
    model state_dict (and therefore inside every checkpoint). Refitting on
    resume is opt-in (see train_bc ``refit_normalizer``) and versioned.
    """

    def __init__(self, dim: int, eps: float = 1e-4, std_floor: float = 1e-3):
        super().__init__()
        self.register_buffer("count", torch.tensor(float(eps)))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))
        self.std_floor = std_floor

    @property
    def std(self) -> torch.Tensor:
        var = self.m2 / torch.clamp(self.count, min=1.0)
        return torch.sqrt(var + self.std_floor * self.std_floor)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).double()
        batch_count = x.shape[0]
        if batch_count == 0:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean.double()
        tot = self.count + batch_count
        self.mean.copy_(((self.count * self.mean.double() + batch_count * batch_mean) / tot).float())
        self.m2.copy_((self.m2.double() + batch_var * batch_count + delta.pow(2) * self.count * batch_count / tot).float())
        self.count.copy_(torch.tensor(float(tot)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    action_dim: int = ACTION_DIM
    proprio_dim: int = PROPRIO_DIM
    touch_dim: int = TOUCH_DIM
    touch_valid_dim: int = TOUCH_VALID_DIM
    image_hw: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
    conv_channels: tuple[int, ...] = (16, 32, 32)
    conv_feature_dim: int = 128
    vector_hidden: int = 128
    fused_hidden: int = 256
    gru_hidden: int = 256
    gru_layers: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    init_log_std: float = -0.5

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["image_hw"] = list(self.image_hw)
        d["conv_channels"] = list(self.conv_channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        if "image_hw" in d:
            d["image_hw"] = tuple(d["image_hw"])
        if "conv_channels" in d:
            d["conv_channels"] = tuple(d["conv_channels"])
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Encoders + recurrent tanh-Gaussian actor-critic
# ---------------------------------------------------------------------------

def _conv_encoder(channels: Sequence[int], feature_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = 3
    for ch in channels:
        layers += [nn.Conv2d(in_ch, ch, kernel_size=3, stride=2, padding=1), nn.ReLU()]
        in_ch = ch
    layers += [nn.Flatten(), nn.LazyLinear(feature_dim), nn.ReLU()]
    return nn.Sequential(*layers)


class ConvEncoder(nn.Module):
    """Small CNN for one RGB camera; input (B, 3, H, W) floats in [0, 1]."""

    def __init__(self, channels: Sequence[int], feature_dim: int, image_hw: tuple[int, int]):
        super().__init__()
        self.net = _conv_encoder(channels, feature_dim)
        self.image_hw = image_hw
        with torch.no_grad():
            self.net(torch.zeros(1, 3, *image_hw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HistoryActorCritic(nn.Module):
    """GRU-history tanh-Gaussian actor + value critic.

    Observations per step: screen_rgb, wrist_rgb (uint8 (H, W, 3) or float in
    [0, 1]), proprio (58,), touch (5,), touch_valid (5,). Images are scaled to
    [0, 1] (when uint8) and normalized with the same fixed per-channel
    statistics at train and eval time; proprio/touch go through running
    normalizers whose state lives in the checkpoint.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        h, w = cfg.image_hw
        self.register_buffer("screen_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("screen_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.screen_encoder = ConvEncoder(cfg.conv_channels, cfg.conv_feature_dim, (h, w))
        self.wrist_encoder = ConvEncoder(cfg.conv_channels, cfg.conv_feature_dim, (h, w))
        self.proprio_norm = Normalizer(cfg.proprio_dim)
        self.touch_norm = Normalizer(cfg.touch_dim)
        self.vector_mlp = nn.Sequential(
            nn.Linear(cfg.proprio_dim + cfg.touch_dim + cfg.touch_valid_dim, cfg.vector_hidden), nn.ReLU()
        )
        fused_in = 2 * cfg.conv_feature_dim + cfg.vector_hidden
        self.fuse = nn.Sequential(nn.Linear(fused_in, cfg.fused_hidden), nn.ReLU())
        self.gru = nn.GRU(cfg.fused_hidden, cfg.gru_hidden, num_layers=cfg.gru_layers, batch_first=True)
        self.actor_mean = nn.Linear(cfg.gru_hidden, cfg.action_dim)
        self.actor_log_std = nn.Parameter(
            torch.full((cfg.action_dim,), float(cfg.init_log_std))
        )
        self.critic = nn.Linear(cfg.gru_hidden, 1)

    # -- helpers ------------------------------------------------------------
    def _images(self, key: str, obs: dict, encoder: ConvEncoder) -> torch.Tensor:
        x = obs[key]
        if not torch.is_tensor(x):
            x = torch.as_tensor(np.asarray(x))
        dev = next(self.parameters()).device
        x = x.to(device=dev)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if float(x.max()) > 1.5:  # tolerate unscaled float images
                x = x / 255.0
        if x.dim() != 5:
            raise ContractError(f"{key} must be (B,T,H,W,C), got {tuple(x.shape)}")
        b, t, h, w, c = x.shape
        if c != 3:
            raise ContractError(f"{key} channel dim must be 3, got {c}")
        x = x.permute(0, 1, 4, 2, 3).reshape(b * t, 3, h, w)
        y = encoder((x - self.screen_mean) / self.screen_std)
        return y.reshape(b, t, -1)

    def features(self, obs: dict) -> torch.Tensor:
        """Per-step fused features; obs maps may be (B, ...) or (B, T, ...)."""
        dev = next(self.parameters()).device
        screen = self._images("screen_rgb", obs, self.screen_encoder)
        wrist = self._images("wrist_rgb", obs, self.wrist_encoder)
        proprio = self.proprio_norm(obs["proprio"].to(device=dev, dtype=torch.float32))
        touch = self.touch_norm(obs["touch"].to(device=dev, dtype=torch.float32))
        touch_valid = obs["touch_valid"].to(device=dev, dtype=torch.float32)
        if proprio.dim() == 2:
            proprio = proprio.unsqueeze(1)
            touch = touch.unsqueeze(1)
            touch_valid = touch_valid.unsqueeze(1)
            screen = screen.unsqueeze(1)
            wrist = wrist.unsqueeze(1)
        vec = self.vector_mlp(torch.cat([proprio, touch, touch_valid], dim=-1))
        return self.fuse(torch.cat([screen, wrist, vec], dim=-1))

    def forward(self, obs: dict, hidden: torch.Tensor | None = None):
        feat = self.features(obs)
        out, hidden = self.gru(feat, hidden)
        mean = self.actor_mean(out)
        log_std = torch.clamp(self.actor_log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        value = self.critic(out).squeeze(-1)
        return mean, log_std, value, hidden

    def act(self, obs: dict, hidden: torch.Tensor | None, deterministic: bool = False):
        """Single-step interface for env interaction.

        ``obs`` must already be batched to (B, T, ...) by ``_obs_to_tensors``;
        for env interaction B=T=1. Returns ``action`` and ``raw`` with shape
        ``(B, action_dim)`` so callers can index ``[0]`` to obtain the full
        action vector. ``logp`` and ``value`` are scalar for B=T=1.
        """
        mean, log_std, value, hidden = self.forward(obs, hidden)
        # forward returns mean/log_std (B, T, A) and a shared parameter (A,).
        # Keep the batch dim so downstream action[0] is the full action vector.
        mean = mean[:, 0, :]  # (B, A)
        if log_std.dim() == 1:
            log_std = log_std.unsqueeze(0).expand(mean.shape[0], -1)  # (B, A)
        elif log_std.dim() >= 2:
            log_std = log_std[:, 0, :]  # (B, A)
        if deterministic:
            raw = mean
            action = deterministic_squashed(mean)
        else:
            action, raw = sample_squashed(mean, log_std)
        lp = squashed_log_prob(mean, log_std, raw)  # (B,)
        return action, raw, lp.reshape(-1)[0], value[0, 0], hidden

    def log_prob_of_raw(self, obs: dict, raw_action: torch.Tensor) -> torch.Tensor:
        mean, log_std, _, _ = self.forward(obs)
        return squashed_log_prob(mean, log_std, raw_action)


# ---------------------------------------------------------------------------
# Dataset: index (dataset.json) + immutable episode directories (NPZ)
# ---------------------------------------------------------------------------
#
# dataset.json index (schema "sharpa-learning-v1"):
#   {
#     "schema": "sharpa-learning-v1",
#     "action_names": [str x 29],
#     "delta_scale": [float x 29],
#     "episodes": [ {
#        "path": "episodes/<id>",          # relative to the index, or absolute
#        "episode_id": str,
#        "split": "train"|"val"|"test",
#        "source_kind": "sim_geometric_expert"|"video_retargeted_sim"|
#                       "student_rollin_correction"|"policy_rollout",
#        "training_eligible_for_bc": bool  # must be present and explicit
#     } ]
#   }
#
# Each episode directory holds manifest.json + trajectory.npz (no pickle).
# NPZ: screen_rgb/wrist_rgb uint8 [T+1,128,128,3]; proprio float32 [T+1,58];
# touch float32 [T+1,5]; touch_valid uint8 [T+1,5]; action float32 [T,29];
# reward/terminated/truncated [T]; sim_time [T+1]; optional executed_q_ref
# [T,29], teacher_action [T,29], teacher_valid [T]; privileged oracle_* arrays
# are tolerated but never loaded into the actor.

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _check_finite(arr: np.ndarray, what: str, where: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ContractError(f"{what} contains non-finite values: {where}")


def _fingerprint(config: Any) -> str:
    return json.dumps(config, sort_keys=True, default=str)


@dataclass
class Episode:
    """One immutable episode directory (manifest.json + trajectory.npz)."""

    path: Path
    manifest: dict
    arrays: dict[str, np.ndarray]

    @property
    def episode_id(self) -> str:
        return str(self.manifest["episode_id"])

    @property
    def split(self) -> str:
        return str(self.manifest["split"])

    @property
    def source_kind(self) -> str:
        return str(self.manifest["source_kind"])

    @property
    def training_eligible_for_bc(self) -> bool:
        return self.manifest["training_eligible_for_bc"] is True

    @property
    def T(self) -> int:
        return int(self.arrays["action"].shape[0])

    @property
    def dynamics_eligible(self) -> bool:
        return self.T > 0 and self.arrays["action"].shape[1] == ACTION_DIM

    @staticmethod
    def load(path: str | Path) -> "Episode":
        path = Path(path)
        manifest_path = path / "manifest.json"
        npz_path = path / "trajectory.npz"
        _require(manifest_path.is_file(), f"episode missing manifest.json: {path}")
        _require(npz_path.is_file(), f"episode missing trajectory.npz: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ep = Episode(path=path, manifest=manifest, arrays={})
        ep._validate_manifest()
        with np.load(npz_path, allow_pickle=False) as npz:
            ep.arrays = {k: npz[k] for k in npz.files}
        ep._validate_arrays()
        return ep

    def _validate_manifest(self) -> None:
        m = self.manifest
        missing = MANIFEST_REQUIRED_KEYS - set(m)
        _require(not missing, f"manifest missing keys {sorted(missing)}: {self.path}")
        _require(m["schema"] == DATASET_SCHEMA, f"episode schema mismatch: {self.path}")
        _require(len(m["action_names"]) == ACTION_DIM, f"action_names length != {ACTION_DIM}: {self.path}")
        delta = np.asarray(m["delta_scale"], dtype=np.float64)
        _require(delta.shape == (ACTION_DIM,), f"delta_scale must be ({ACTION_DIM},): {self.path}")
        _check_finite(delta, "delta_scale", str(self.path))
        limits = m["joint_limits"]
        _require(isinstance(limits, (list, tuple)) and len(limits) == 2, f"joint_limits must be (lo, hi): {self.path}")
        lo = np.asarray(limits[0], dtype=np.float64)
        hi = np.asarray(limits[1], dtype=np.float64)
        _require(lo.shape == (ACTION_DIM,) and hi.shape == (ACTION_DIM,), f"joint_limits must be ({ACTION_DIM},) each: {self.path}")
        _check_finite(lo, "joint_limits lo", str(self.path))
        _check_finite(hi, "joint_limits hi", str(self.path))
        _require(bool(np.all(lo < hi)), f"joint_limits must satisfy lo < hi: {self.path}")
        _require(m["split"] in ALLOWED_SPLITS, f"split must be one of {sorted(ALLOWED_SPLITS)}: {self.path}")
        _require(m["source_kind"] in ALLOWED_SOURCE_KINDS, f"source_kind {m['source_kind']!r} not allowed: {self.path}")
        _require(isinstance(m["training_eligible_for_bc"], bool), f"training_eligible_for_bc must be an explicit bool: {self.path}")
        backend = str(m["measured_contact_backend"]).lower()
        _require(bool(m["measured_contact_backend"]), f"measured_contact_backend missing: {self.path}")
        _require("physx" in backend, f"measured_contact_backend must be a measured PhysX backend: {self.path}")
        _require(not any(b in backend for b in _CONTACT_BACKEND_BANNED_MARKERS),
                 f"measured_contact_backend looks like the banned geometric proxy: {self.path}")
        _require(isinstance(m["config"], dict), f"manifest config must be a mapping: {self.path}")
        # Verify whole-trajectory digest against the immutable NPZ file bytes.
        npz_path = self.path / "trajectory.npz"
        recorded_traj = str(m.get("trajectory_sha256", ""))
        _require(recorded_traj, f"trajectory_sha256 missing: {self.path}")
        actual_traj = _sha256_file(npz_path)
        _require(recorded_traj == actual_traj,
                 f"trajectory_sha256 mismatch for {npz_path.name}: expected {recorded_traj[:16]}... got {actual_traj[:16]}...")

    def _validate_arrays(self) -> None:
        a = self.arrays
        where = str(self.path)
        missing = [k for k in NPZ_REQUIRED_KEYS if k not in a]
        _require(not missing, f"episode missing npz keys {missing}: {where}")
        for k in a:
            if k in NPZ_REQUIRED_KEYS or k in NPZ_OPTIONAL_KEYS:
                continue
            _require(k.startswith(PRIVILEGED_NPZ_PREFIX), f"unexpected npz key {k!r}: {where}")
        T = a["action"].shape[0]
        _require(a["action"].ndim == 2 and a["action"].shape[1] == ACTION_DIM, f"action must be [T,{ACTION_DIM}]: {where}")
        expected = {
            "screen_rgb": (T + 1, IMAGE_SIZE, IMAGE_SIZE, 3),
            "wrist_rgb": (T + 1, IMAGE_SIZE, IMAGE_SIZE, 3),
            "proprio": (T + 1, PROPRIO_DIM),
            "touch": (T + 1, TOUCH_DIM),
            "touch_valid": (T + 1, TOUCH_VALID_DIM),
            "reward": (T,),
            "terminated": (T,),
            "truncated": (T,),
            "sim_time": (T + 1,),
            "executed_q_ref": (T, ACTION_DIM),
        }
        for key, shape in expected.items():
            _require(tuple(a[key].shape) == shape, f"{key} shape {tuple(a[key].shape)} != {shape}: {where}")
        _require(a["screen_rgb"].dtype == np.uint8 and a["wrist_rgb"].dtype == np.uint8, f"rgb arrays must be uint8: {where}")
        _require(a["proprio"].dtype == np.float32, f"proprio must be float32: {where}")
        _require(a["touch"].dtype == np.float32, f"touch must be float32: {where}")
        _require(a["touch_valid"].dtype == np.uint8, f"touch_valid must be uint8: {where}")
        _require(a["action"].dtype == np.float32, f"action must be float32: {where}")
        _require(a["executed_q_ref"].dtype == np.float32, f"executed_q_ref must be float32: {where}")
        tv = a["touch_valid"]
        _require(bool(np.all((tv == 0) | (tv == 1))), f"touch_valid must be 0/1: {where}")
        if self.manifest["training_eligible_for_bc"] is True and bool(np.all(tv == 0)):
            raise ContractError(f"training_eligible_for_bc=True but all touch_valid are zero: {where}")
        st = a["sim_time"]
        _check_finite(st, "sim_time", where)
        _require(bool(np.all(np.diff(st) > 0)), f"sim_time must be strictly increasing: {where}")
        for key in ("proprio", "touch", "action", "reward", "executed_q_ref"):
            _check_finite(a[key], key, where)
        for key in NPZ_OPTIONAL_KEYS:
            if key not in a:
                continue
            want = (T,) if key == "teacher_valid" else (T, ACTION_DIM)
            _require(tuple(a[key].shape) == want, f"{key} shape {tuple(a[key].shape)} != {want}: {where}")
            if a[key].dtype.kind == "f":
                _check_finite(a[key], key, where)
        if self.source_kind == "student_rollin_correction":
            for key in ("teacher_action", "teacher_valid"):
                _require(key in a, f"correction episode missing {key}: {where}")
            ta = a["teacher_action"]
            tv = a["teacher_valid"]
            _check_finite(ta, "teacher_action", where)
            _require(bool(np.all(ta >= -1.0) and np.all(ta <= 1.0)), f"teacher_action must be in [-1,1]: {where}")
            _require(tv.dtype == np.uint8, f"teacher_valid must be uint8: {where}")
            _require(bool(np.all((tv == 0) | (tv == 1))), f"teacher_valid must be 0/1: {where}")
            if self.manifest["training_eligible_for_bc"] is True:
                _require(bool(np.any(tv == 1)), f"correction episode eligible but no teacher_valid==1: {where}")
        # Verify per-array content digests recorded in the manifest.
        recorded = self.manifest["array_sha256"]
        _require(isinstance(recorded, dict), f"array_sha256 must be a mapping: {where}")
        for key in a:
            digest = recorded.get(key)
            _require(isinstance(digest, str), f"array_sha256 missing entry for {key}: {where}")
            actual = _sha256_array(a[key])
            _require(digest == actual, f"npz sha mismatch for {key}: {where}")

    def actor_obs(self, t: int) -> dict[str, np.ndarray]:
        """Fresh actor-whitelisted observation dict at frame index t."""
        if not 0 <= t < self.T + 1:
            raise IndexError(t)
        return {k: self.arrays[k][t] for k in ACTOR_OBS_KEYS}

    def bc_batch_tensors(self, cfg: ModelConfig, device: torch.device) -> dict[str, torch.Tensor]:
        """(obs[0:T], action) tensors for BC; obs[t] precedes action[t].

        For ``student_rollin_correction`` episodes the executed ``action`` is
        the student's own output and must not be imitated; the BC target is
        ``teacher_action`` and ``teacher_valid`` selects which steps are usable.
        Ordinary expert episodes continue to use ``action`` with an all-ones
        validity mask.
        """
        a = self.arrays
        T = self.T
        if self.source_kind == "student_rollin_correction":
            for key in ("teacher_action", "teacher_valid"):
                _require(key in a, f"correction episode missing {key}: {self.path}")
            target = a["teacher_action"]
            valid = a["teacher_valid"]
            _require(target.shape == (T, ACTION_DIM), f"teacher_action shape {tuple(target.shape)} != [T,{ACTION_DIM}]: {self.path}")
            _require(valid.shape == (T,), f"teacher_valid shape {tuple(valid.shape)} != [T]: {self.path}")
            _require(valid.dtype == np.uint8, f"teacher_valid must be uint8: {self.path}")
            _require(bool(np.all((valid == 0) | (valid == 1))), f"teacher_valid must be 0/1: {self.path}")
        else:
            target = a["action"]
            valid = np.ones(T, dtype=np.uint8)
        return {
            "screen_rgb": torch.as_tensor(a["screen_rgb"][:T], device=device),
            "wrist_rgb": torch.as_tensor(a["wrist_rgb"][:T], device=device),
            "proprio": torch.as_tensor(a["proprio"][:T], device=device),
            "touch": torch.as_tensor(a["touch"][:T], device=device),
            "touch_valid": torch.as_tensor(a["touch_valid"][:T], device=device),
            "action": torch.as_tensor(target, device=device),
            "teacher_valid": torch.as_tensor(valid, device=device),
        }


@dataclass
class DatasetBundle:
    """Index (dataset.json) backed collection of immutable episodes."""

    path: Path
    schema: str
    action_names: list[str]
    delta_scale: np.ndarray
    episodes: list[Episode]

    def splits(self) -> dict[str, list[Episode]]:
        out: dict[str, list[Episode]] = {}
        for ep in self.episodes:
            out.setdefault(ep.split, []).append(ep)
        return out

    def require_split(self, name: str) -> list[Episode]:
        """Explicit split access: never fall back to train for a missing split."""
        eps = self.splits().get(name, [])
        _require(bool(eps), f"dataset has no explicit '{name}' split episodes: {self.path}")
        return eps

    def bc_train_pool(self) -> list[Episode]:
        """Eligible BC pool: split=='train' AND training_eligible_for_bc is True."""
        return [ep for ep in self.episodes
                if ep.split == "train" and ep.training_eligible_for_bc]

    def episode_index(self) -> dict[str, Episode]:
        return {ep.episode_id: ep for ep in self.episodes}


def load_dataset(
    path: str | Path,
    action_dim: int = ACTION_DIM,
    split_seed: int | None = None,
    auto_split_if_missing: bool = False,
) -> DatasetBundle:
    """Load + validate a dataset index and every referenced episode.

    Splits are frozen in the index before any optimizer runs: this loader
    never auto-splits (``auto_split_if_missing`` stays False; a missing split
    is an explicit error at point of use, never a silent train fallback).
    """
    _require(action_dim == ACTION_DIM, f"contract action dim is {ACTION_DIM}, got {action_dim}")
    if split_seed is not None or auto_split_if_missing:
        raise ContractError("splits are frozen in the index; auto-split is not permitted")
    path = Path(path)
    _require(path.is_file(), f"dataset index not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    _require(doc.get("schema") == DATASET_SCHEMA, f"schema must be {DATASET_SCHEMA!r}, got {doc.get('schema')!r}")
    action_names = list(doc.get("action_names", []))
    _require(len(action_names) == ACTION_DIM, f"index action_names length != {ACTION_DIM}")
    delta_scale = np.asarray(doc.get("delta_scale"), dtype=np.float64)
    _require(delta_scale.shape == (ACTION_DIM,), f"index delta_scale must be ({ACTION_DIM},)")
    _check_finite(delta_scale, "index delta_scale", str(path))
    records = doc.get("episodes")
    _require(isinstance(records, list) and records, "dataset index has no episodes")

    base = path.parent
    episodes: list[Episode] = []
    seen_ids: set[str] = set()
    group_split: dict[str, str] = {}
    for rec in records:
        missing = INDEX_EPISODE_REQUIRED_KEYS - set(rec)
        _require(not missing, f"index episode record missing keys {sorted(missing)}: {path}")
        eid = str(rec["episode_id"])
        _require(eid not in seen_ids, f"duplicate episode_id across splits: {eid}")
        seen_ids.add(eid)
        _require(isinstance(rec["training_eligible_for_bc"], bool),
                 f"index training_eligible_for_bc must be an explicit bool: {eid}")
        _require(rec["split"] in ALLOWED_SPLITS, f"index split invalid for {eid}: {rec['split']!r}")
        _require(rec["source_kind"] in ALLOWED_SOURCE_KINDS, f"index source_kind invalid for {eid}: {rec['source_kind']!r}")
        ep_path = Path(rec["path"])
        if not ep_path.is_absolute():
            ep_path = base / ep_path
        ep = Episode.load(ep_path)
        # Index <-> manifest identity, split, action names, scales must agree.
        _require(ep.episode_id == eid, f"index/manifest episode_id mismatch: {eid} vs {ep.episode_id}")
        _require(ep.split == rec["split"], f"index/manifest split mismatch for {eid}")
        _require(ep.source_kind == rec["source_kind"], f"index/manifest source_kind mismatch for {eid}")
        _require(ep.training_eligible_for_bc == rec["training_eligible_for_bc"],
                 f"index/manifest eligibility mismatch for {eid}")
        _require(list(ep.manifest["action_names"]) == action_names, f"index/manifest action_names mismatch for {eid}")
        _require(np.allclose(np.asarray(ep.manifest["delta_scale"], dtype=np.float64), delta_scale),
                 f"index/manifest delta_scale mismatch for {eid}")
        # Config groups must live in exactly one split (never duplicated across).
        fp = _fingerprint(ep.manifest["config"])
        prev = group_split.setdefault(fp, ep.split)
        _require(prev == ep.split, f"config group appears in both '{prev}' and '{ep.split}': {eid}")
        episodes.append(ep)
    return DatasetBundle(
        path=path,
        schema=str(doc["schema"]),
        action_names=action_names,
        delta_scale=delta_scale,
        episodes=episodes,
    )


def dataset_summary(bundle: DatasetBundle) -> dict:
    def _count(split: str) -> dict:
        eps = bundle.splits().get(split, [])
        ok = [e for e in eps if e.training_eligible_for_bc]
        return {
            "episodes": len(eps),
            "eligible_episodes": len(ok),
            "steps": sum(e.T for e in ok),
        }

    return {
        "schema": bundle.schema,
        "path": str(bundle.path),
        "action_dim_ok": True,
        "splits": {s: _count(s) for s in sorted(ALLOWED_SPLITS)},
    }


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def pad_episodes(
    episodes: list[Episode], cfg: ModelConfig, device: torch.device
) -> dict:
    """Pad episodes into one batched, masked dict of tensors (obs[0:T], action).

    The BC loss mask is ``teacher_valid`` for correction episodes and all ones
    for ordinary expert episodes, so invalid correction labels are ignored.
    """
    tensors = [ep.bc_batch_tensors(cfg, device) for ep in episodes]
    maxlen = max(t["action"].shape[0] for t in tensors)
    b = len(tensors)
    h, w = cfg.image_hw
    batch = {
        "screen_rgb": torch.zeros(b, maxlen, h, w, 3, dtype=torch.uint8, device=device),
        "wrist_rgb": torch.zeros(b, maxlen, h, w, 3, dtype=torch.uint8, device=device),
        "proprio": torch.zeros(b, maxlen, cfg.proprio_dim, device=device),
        "touch": torch.zeros(b, maxlen, cfg.touch_dim, device=device),
        "touch_valid": torch.zeros(b, maxlen, cfg.touch_valid_dim, dtype=torch.uint8, device=device),
        "action": torch.zeros(b, maxlen, cfg.action_dim, device=device),
        "mask": torch.zeros(b, maxlen, device=device),
    }
    for i, t in enumerate(tensors):
        n = t["action"].shape[0]
        for key in ("screen_rgb", "wrist_rgb", "proprio", "touch", "touch_valid", "action"):
            batch[key][i, :n] = t[key]
        batch["mask"][i, :n] = t["teacher_valid"].to(dtype=batch["mask"].dtype)
    return batch


# ---------------------------------------------------------------------------
# Checkpoints: model + optimizer + RNG + normalization + provenance
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: HistoryActorCritic,
    optimizer: torch.optim.Optimizer | None,
    cfg: ModelConfig,
    stage: str,
    global_step: int,
    provenance: dict,
    rng_state: dict | None = None,
    sampler_state: tuple | None = None,
    env_transitions: int = 0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "utc": _utc_now(),
        "model_config": cfg.to_dict(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "rng_state": rng_state if rng_state is not None else capture_rng_state(),
        "sampler_state": sampler_state,
        "stage": stage,
        "global_step": int(global_step),
        "env_transitions": int(env_transitions),
        "provenance": provenance,
    }
    # Atomic publish: readers and crash recovery never observe a torn file.
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(
    path: str | Path, device: torch.device | str = "cpu", restore_rng: bool = True
) -> tuple[HistoryActorCritic, dict]:
    """Restore model (+ optimizer state + RNG + normalizers). Returns (model, payload)."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ContractError(f"not a {CHECKPOINT_FORMAT}: {path}")
    cfg = ModelConfig.from_dict(payload["model_config"])
    model = HistoryActorCritic(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return model, payload


def load_optimizer_state(
    optimizer: torch.optim.Optimizer, payload: dict
) -> None:
    if payload.get("optimizer_state") is None:
        raise ContractError("checkpoint carries no optimizer state")
    optimizer.load_state_dict(payload["optimizer_state"])


# ---------------------------------------------------------------------------
# BC training
# ---------------------------------------------------------------------------

def bc_loss(
    model: HistoryActorCritic, batch: dict
) -> tuple[torch.Tensor, dict]:
    """Behavior-cloning NLL of expert raw actions (atanh of label actions)."""
    mean, log_std, _, _ = model.forward(batch)
    # Expert labels are stored squashed in [-1, 1]; invert tanh to raw space.
    actions = batch["action"].clamp(-1.0 + TANH_EPS, 1.0 - TANH_EPS)
    raw = torch.atanh(actions)
    nll = -squashed_log_prob(mean, log_std, raw)
    mask = batch["mask"]
    loss = (nll * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, {"bc_nll": float(loss.detach().cpu())}


def _install_sigterm_flag() -> tuple[Callable[[], bool], Callable[[], None]]:
    """Return (stop_requested, restore) handling SIGTERM during training."""
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


def train_bc(
    dataset: DatasetBundle,
    output: Path,
    steps: int,
    seed: int,
    cfg: ModelConfig | None = None,
    init_from: Path | None = None,
    corrections: list[Path] | None = None,
    lr: float = 3e-4,
    batch_episodes: int = 8,
    device: str = "cpu",
    stage: str = "bc",
    grad_clip: float = 1.0,
    checkpoint_every: int = 500,
    time_budget_s: float | None = None,
    refit_normalizer: bool = False,
    log_every: int = 25,
) -> Path:
    """Train (or resume) BC on the frozen train split (+ optional corrections).

    Only split=="train" episodes with explicit training_eligible_for_bc=True
    are consumed; heldout episodes are never used. Resume restores model,
    optimizer, RNG and normalizer state from the checkpoint; the normalizer is
    NOT refit on resume unless ``refit_normalizer=True`` (the refit is logged
    and versioned). Checkpoints are written periodically and on deadline /
    SIGTERM so a bounded campaign never relies on an end-only save.
    """
    dev = torch.device(device)
    seed_everything(seed)
    cfg = cfg or ModelConfig()
    stop_requested, restore_sigterm = _install_sigterm_flag()
    started = time.time()
    model: HistoryActorCritic
    optimizer: torch.optim.Adam
    start_step = 0
    normalizer_version = 0
    provenance: dict = {
        "dataset_sha256": _sha256_file(dataset.path),
        "dataset_path": str(dataset.path),
        "seed": seed,
        "corrections": [],
        "parent_checkpoint": None,
        "normalizer_version": normalizer_version,
    }
    if init_from is not None:
        model, payload = load_checkpoint(init_from, dev)
        cfg = ModelConfig.from_dict(payload["model_config"])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        load_optimizer_state(optimizer, payload)
        start_step = int(payload["global_step"])
        normalizer_version = int(payload.get("provenance", {}).get("normalizer_version", 0))
        provenance["parent_checkpoint"] = str(init_from)
        provenance.setdefault("stage_history", []).append(
            {"stage": payload["stage"], "global_step": start_step, "from": str(init_from)}
        )
    else:
        model = HistoryActorCritic(cfg).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Prime lazy layers with the configured image size so state_dict is complete.
        h0, w0 = cfg.image_hw
        model.forward(
            {
                "screen_rgb": torch.zeros(1, 1, h0, w0, 3, dtype=torch.uint8, device=dev),
                "wrist_rgb": torch.zeros(1, 1, h0, w0, 3, dtype=torch.uint8, device=dev),
                "proprio": torch.zeros(1, 1, cfg.proprio_dim, device=dev),
                "touch": torch.zeros(1, 1, cfg.touch_dim, device=dev),
                "touch_valid": torch.zeros(1, 1, cfg.touch_valid_dim, dtype=torch.uint8, device=dev),
            }
        )

    # ---- assemble training episode pool (train split only) --------------
    pool: list[Episode] = dataset.bc_train_pool()
    data_counts = {"train_episodes": len(pool), "train_steps": 0, "correction_steps": 0,
                   "heldout_steps_used": 0}
    data_counts["train_steps"] = sum(ep.T for ep in pool)
    corr_bundles: list[DatasetBundle] = []
    seen_ids = {ep.episode_id for ep in pool}
    seen_shas = {ep.manifest.get("trajectory_sha256") for ep in pool if ep.manifest.get("trajectory_sha256")}
    for cpath in corrections or []:
        cb = load_dataset(cpath)
        cpool = cb.bc_train_pool()
        for ep in cpool:
            if ep.source_kind != "student_rollin_correction":
                raise ContractError(
                    f"corrections must be student_rollin_correction, got {ep.source_kind!r}: {ep.episode_id}"
                )
            if ep.episode_id in seen_ids:
                raise ContractError(f"correction episode_id {ep.episode_id!r} duplicates an already-used episode")
            sha = ep.manifest.get("trajectory_sha256")
            if sha and sha in seen_shas:
                raise ContractError(f"correction episode {ep.episode_id!r} duplicates an already-used trajectory")
            seen_ids.add(ep.episode_id)
            if sha:
                seen_shas.add(sha)
        pool.extend(cpool)
        valid_steps = sum(int(ep.arrays["teacher_valid"].sum()) for ep in cpool)
        data_counts["correction_steps"] += valid_steps
        provenance["corrections"].append(
            {"path": str(cpath), "sha256": _sha256_file(Path(cpath)),
             "episodes": len(cpool), "valid_steps": valid_steps}
        )
        corr_bundles.append(cb)
    if not pool:
        restore_sigterm()
        raise ContractError("no eligible training episodes (split=train, training_eligible_for_bc=true)")
    # Fit proprio/touch normalizers on the training pool only. On resume the
    # checkpoint's normalizer state is kept unless an explicit refit is asked.
    if init_from is None or refit_normalizer:
        with torch.no_grad():
            for ep in pool:
                t = ep.bc_batch_tensors(cfg, dev)
                model.proprio_norm.update(t["proprio"])
                model.touch_norm.update(t["touch"])
        if refit_normalizer and init_from is not None:
            normalizer_version += 1
            append_log(Path(output), {"event": "normalizer_refit", "stage": stage,
                                      "normalizer_version": normalizer_version})
    provenance["normalizer_version"] = normalizer_version

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if init_from is not None:
        # Continue the episode sampler exactly where the checkpoint left off.
        rng = random.Random()
        rng.setstate(payload.get("sampler_state") or random.getstate())
    else:
        rng = random.Random(seed)
    append_log(
        output,
        {
            "event": "train_begin",
            "stage": stage,
            "optimizer_step": start_step,
            "data_counts": data_counts,
            "lr": lr,
            "device": device,
            "seed": seed,
            "normalizer_version": normalizer_version,
        },
    )
    step = start_step
    last_ckpt = output / f"{stage}_step{start_step}.pt"
    model.train()
    stop_reason = "steps"
    try:
        while step < start_step + steps:
            if time_budget_s is not None and time.time() - started > time_budget_s:
                stop_reason = "time_budget"
                break
            if stop_requested():
                stop_reason = "sigterm"
                break
            chosen = [pool[rng.randrange(len(pool))] for _ in range(min(batch_episodes, len(pool)))]
            batch = pad_episodes(chosen, cfg, dev)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = bc_loss(model, batch)
            if not torch.isfinite(loss):
                raise ContractError(f"non-finite BC loss at optimizer step {step + 1}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            if step % log_every == 0 or step == start_step + steps:
                append_log(
                    output,
                    {
                        "event": "optimizer_step",
                        "stage": stage,
                        "optimizer_step": step,
                        "data_counts": data_counts,
                        "lr": optimizer.param_groups[0]["lr"],
                        "losses": metrics,
                    },
                )
            if step % checkpoint_every == 0:
                save_checkpoint(output / f"{stage}_step{step}.pt", model, optimizer, cfg,
                                stage, step, provenance, sampler_state=rng.getstate(), env_transitions=0)
    finally:
        last_ckpt = output / f"{stage}_step{step}.pt"
        save_checkpoint(last_ckpt, model, optimizer, cfg, stage, step, provenance, sampler_state=rng.getstate(), env_transitions=0)
        # latest.pt is a *copy*: stage files are never overwritten.
        save_checkpoint(output / "latest.pt", model, optimizer, cfg, stage, step, provenance, sampler_state=rng.getstate(), env_transitions=0)
        append_log(
            output,
            {
                "event": "train_end",
                "stage": stage,
                "optimizer_step": step,
                "data_counts": data_counts,
                "checkpoint": str(last_ckpt),
                "stop_reason": stop_reason,
            },
        )
        restore_sigterm()
    return last_ckpt


# ---------------------------------------------------------------------------
# Environment factory (env worker owns the env module)
# ---------------------------------------------------------------------------

def load_env_factory(spec: str) -> Callable[..., Any]:
    """Resolve 'pkg.mod:Class' into a callable. Import happens lazily here."""
    if ":" not in spec:
        raise ContractError(f"env factory must be 'module:Name', got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    module = importlib.import_module(mod_name)
    factory = getattr(module, attr)
    if not callable(factory):
        raise ContractError(f"env factory {spec!r} is not callable")
    return factory


def parse_env_kwargs(items: Sequence[str] | None) -> dict:
    kwargs: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ContractError(f"--env-kwarg expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            kwargs[k] = json.loads(v)
        except json.JSONDecodeError:
            kwargs[k] = v
    return kwargs


def _validate_factory_signature(factory: Callable[..., Any], kwargs: dict) -> None:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    bad = [k for k in kwargs if k not in params]
    if bad:
        raise ContractError(f"env factory does not accept kwargs {bad}; signature {sig}")


def make_env(factory: Callable[..., Any], env_kwargs: dict | None, require_expert: bool = False) -> Any:
    """Validate the factory config signature and instantiate the environment.

    The returned env is verified to expose the frozen SharpaEnv surface
    (reset/step/close, plus expert_action when labels are required). The
    caller must close() the env only AFTER all reports/checkpoints are
    written (SimulationApp.close is a fast shutdown that exits Python).
    """
    kwargs = dict(env_kwargs or {})
    _validate_factory_signature(factory, kwargs)
    env = factory(**kwargs)
    for meth in ("reset", "step", "close"):
        if not callable(getattr(env, meth, None)):
            raise ContractError(f"env is missing required method {meth}()")
    if require_expert and not callable(getattr(env, "expert_action", None)):
        raise ContractError("env provides no expert_action(); refusing to fabricate labels")
    return env


def _env_reset(env: Any, seed: int, options: dict | None = None):
    """Reset through the frozen contract, preserving reset options."""
    out = env.reset(seed=seed, options=options)
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def _env_step(env: Any, action: Any):
    out = env.step(action)
    if len(out) == 5:
        return out
    obs, reward, done, info = out  # legacy 4-tuple: done == terminal
    return obs, reward, bool(done), False, info


def _check_exact_obs_keys(obs: dict) -> None:
    keys = set(obs)
    missing = set(ACTOR_OBS_KEYS) - keys
    extra = keys - set(ACTOR_OBS_KEYS)
    if missing or extra:
        raise ContractError(f"env obs must be exactly {sorted(ACTOR_OBS_KEYS)}; "
                            f"missing={sorted(missing)} extra={sorted(extra)}")


def _obs_to_tensors(obs: dict, cfg: ModelConfig, device: torch.device) -> dict:
    """Whitelist-check and batch an env observation; the actor path never reads info."""
    if not isinstance(obs, dict):
        raise ContractError("env obs must be a mapping")
    leaked = REJECTED_OBS_KEYS & set(obs)
    if leaked:
        raise ContractError(f"env obs leaks rejected teacher/oracle keys: {sorted(leaked)}")
    _check_exact_obs_keys(obs)
    _require(tuple(np.asarray(obs["screen_rgb"]).shape) == (cfg.image_hw[0], cfg.image_hw[1], 3),
             f"screen_rgb must be uint8 [{cfg.image_hw[0]},{cfg.image_hw[1]},3]")
    _require(tuple(np.asarray(obs["wrist_rgb"]).shape) == (cfg.image_hw[0], cfg.image_hw[1], 3),
             f"wrist_rgb must be uint8 [{cfg.image_hw[0]},{cfg.image_hw[1]},3]")
    return {
        "screen_rgb": torch.as_tensor(np.asarray(obs["screen_rgb"]), device=device).unsqueeze(0).unsqueeze(0),
        "wrist_rgb": torch.as_tensor(np.asarray(obs["wrist_rgb"]), device=device).unsqueeze(0).unsqueeze(0),
        "proprio": torch.as_tensor(np.asarray(obs["proprio"], dtype=np.float32), device=device).reshape(1, 1, -1),
        "touch": torch.as_tensor(np.asarray(obs["touch"], dtype=np.float32), device=device).reshape(1, 1, -1),
        "touch_valid": torch.as_tensor(np.asarray(obs["touch_valid"]), device=device).reshape(1, 1, -1),
    }


# ---------------------------------------------------------------------------
# Student roll-in corrections (DAgger-style, expert_action at visited state)
# ---------------------------------------------------------------------------

def _env_scalar_list(env: Any, attr: str, what: str) -> list:
    val = getattr(env, attr, None)
    _require(val is not None, f"env does not expose {attr}; cannot write {what}")
    return [float(v) for v in np.asarray(val, dtype=np.float64).reshape(-1)]


def _teacher_label_valid(env: Any, info: dict | None, expert: np.ndarray) -> bool:
    """Return False if the env signals the current expert label is unusable.

    A label can be rejected because it is numerically illegal, because the
    env's teacher is in a failure phase, or because the env/info explicitly
    flags ``expert_label_valid=False``. Callable flags are invoked (e.g.
    provider properties); non-callable flags are cast with bool().
    """
    ok = bool(np.all(np.isfinite(expert)) and np.all(expert >= -1.0) and np.all(expert <= 1.0))
    teacher = getattr(env, "teacher", None)
    if teacher is not None:
        phase = getattr(teacher, "_phase", None)
        if isinstance(phase, str) and "fail" in phase.lower():
            ok = False
    explicit = getattr(env, "expert_label_valid", None)
    if explicit is not None:
        ok = ok and bool(explicit() if callable(explicit) else explicit)
    info_valid = (info or {}).get("expert_label_valid")
    if info_valid is not None:
        ok = ok and bool(info_valid() if callable(info_valid) else info_valid)
    return ok


def gather_corrections(
    env_factory: Callable[..., Any],
    checkpoint: Path,
    output: Path,
    rollouts: int,
    seed: int,
    max_steps: int = 400,
    env_kwargs: dict | None = None,
    stage: str = "corrections",
    task_id: str = "tap",
) -> Path:
    """Run the student, record expert_action at each *actual visited* state.

    Every rollout resets the env with options {"split": "train"} — an explicit
    train configuration, never a heldout one. Labels are the env's
    expert_action at the student-visited state; per-step teacher_valid records
    whether that label is usable for BC, and episodes carry the student's
    actual executed_q_ref (from info; the actor itself never reads info).
    Output uses the same manifest.json + trajectory.npz format as every other
    episode, plus a dataset.json index. There is no teacher-action fallback
    anywhere in evaluation or training.

    All filesystem artifacts (episodes, index, log) are finalized before the
    env is closed, because some simulators exit the process on close.
    """
    checkpoint = Path(checkpoint)
    dev = torch.device("cpu")
    model, payload = load_checkpoint(checkpoint, dev)
    model.eval()
    cfg = ModelConfig.from_dict(payload["model_config"])
    env: Any = None
    provenance = {}
    action_names: list[str] = []
    delta_scale: list[float] = []
    limits: Any = None
    config: dict = {}
    ckpt_stage = str(payload.get("stage", "unknown"))
    contact_backend = ""
    out = Path(output)
    episode_dirs: list[tuple[str, Path, bool]] = []
    index_path = out / f"{stage}.json"

    def _write_index() -> Path:
        """Write dataset.json index and the gathered log for completed episodes."""
        index_doc = {
            "schema": DATASET_SCHEMA,
            "action_names": action_names,
            "delta_scale": delta_scale,
            "created_utc": _utc_now(),
            "episodes": [
                {
                    "path": str(ep_dir.relative_to(out)),
                    "episode_id": ep_id,
                    "split": "train",
                    "source_kind": "student_rollin_correction",
                    "training_eligible_for_bc": eligible,
                }
                for ep_id, ep_dir, eligible in episode_dirs
            ],
            "provenance": {
                "generator": "bc_ppo.gather_corrections",
                "student_checkpoint": checkpoint.name,
                "student_checkpoint_sha256": config.get("student_checkpoint_sha256", ""),
                "student_checkpoint_stage": ckpt_stage,
                "seed": seed,
                "rollouts": rollouts,
                "split_options": "train",
            },
        }
        index_path = out / f"{stage}.json"
        index_path.write_text(json.dumps(index_doc, indent=2, sort_keys=True), encoding="utf-8")
        valid_steps = sum(
            int(np.load(d / "trajectory.npz")["teacher_valid"].sum()) for _, d, _ in episode_dirs
        )
        append_log(
            out,
            {
                "event": "corrections_gathered",
                "stage": stage,
                "data_counts": {
                    "rollouts": rollouts,
                    "correction_steps": valid_steps,
                },
                "student_checkpoint": checkpoint.name,
                "output": str(index_path),
            },
        )
        return index_path

    try:
        env = make_env(env_factory, env_kwargs, require_expert=True)
        provenance = getattr(env, "provenance", {}) or {}
        contact_backend = str(provenance.get("contact_backend", ""))
        _require("physx" in contact_backend.lower() and
                 not any(b in contact_backend.lower() for b in _CONTACT_BACKEND_BANNED_MARKERS),
                 f"env contact backend is not a measured PhysX backend: {contact_backend!r}")
        action_names = list(getattr(env, "action_names", None) or [])
        _require(len(action_names) == cfg.action_dim, "env action_names length mismatch")
        delta_scale = _env_scalar_list(env, "delta_scale", "manifest delta_scale")
        _require(len(delta_scale) == cfg.action_dim, "env delta_scale length mismatch")
        limits = getattr(env, "joint_limits", None)
        _require(limits is not None, "env does not expose joint_limits")
        config = dict(getattr(env, "config", None) or {})
        config.setdefault("env_kwargs", dict(env_kwargs or {}))
        config["student_checkpoint"] = checkpoint.name
        config["student_checkpoint_sha256"] = _sha256_file(checkpoint)

        out.mkdir(parents=True, exist_ok=True)
        _require(rollouts >= 1, "gather_corrections needs rollouts >= 1")
        _require(max_steps >= 1, "gather_corrections needs max_steps >= 1")
        rng = random.Random(seed)
        for ep_idx in range(rollouts):
            ep_seed = rng.randrange(2**31)
            obs, _info = _env_reset(env, seed=ep_seed, options={"split": "train", "task_id": task_id})
            _check_exact_obs_keys(obs)
            hidden = None
            obs_frames: dict[str, list[np.ndarray]] = {k: [] for k in ACTOR_OBS_KEYS}
            actions: list[np.ndarray] = []
            teacher: list[np.ndarray] = []
            teacher_valid: list[int] = []
            exec_refs: list[np.ndarray] = []
            rewards: list[float] = []
            terminated_flags: list[bool] = []
            truncated_flags: list[bool] = []
            sim_times: list[float] = []
            terminated = truncated = False
            sim_times.append(float(_info.get("sim_time", math.nan)))
            _require(math.isfinite(sim_times[-1]), "env reset info must provide finite sim_time")
            for t in range(max_steps):
                for k in ACTOR_OBS_KEYS:
                    obs_frames[k].append(np.array(obs[k], copy=True))
                expert = np.asarray(env.expert_action(), dtype=np.float64).reshape(-1)
                _require(expert.shape == (cfg.action_dim,), "expert_action must be [29]")
                pre_ok = _teacher_label_valid(env, None, expert)
                obs_t = _obs_to_tensors(obs, cfg, dev)
                with torch.no_grad():
                    action, _raw, _lp, _v, hidden = model.act(obs_t, hidden)
                student_action = action[0].cpu().numpy().astype(np.float32)
                obs, reward, terminated, truncated, info = _env_step(env, student_action)
                _check_exact_obs_keys(obs)
                # Re-check label validity after the step and combine with pre-step.
                # A label that was invalid before the student moved cannot become
                # valid just because the state changed.
                post_ok = _teacher_label_valid(env, info, expert)
                label_ok = pre_ok and post_ok
                sim_time = float((info or {}).get("sim_time", math.nan))
                _require(math.isfinite(sim_time), "env step info must provide finite sim_time")
                sim_times.append(sim_time)
                exec_ref = np.asarray((info or {}).get("executed_q_ref"), dtype=np.float64).reshape(-1)
                _require(exec_ref.shape == (cfg.action_dim,),
                         "env info must provide executed_q_ref [29] for corrections")
                actions.append(student_action)
                teacher.append(expert.astype(np.float32))
                teacher_valid.append(1 if label_ok else 0)
                exec_refs.append(exec_ref.astype(np.float32))
                rewards.append(float(reward))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(bool(truncated))
                if terminated or truncated:
                    break
            # Evaluator-imposed cap: an unfinished rollout is a timeout and the
            # final observation is preserved as observation[T].
            cap_hit = not (terminated_flags[-1] or truncated_flags[-1])
            if cap_hit:
                truncated_flags[-1] = True
            for k in ACTOR_OBS_KEYS:
                obs_frames[k].append(np.array(obs[k], copy=True))
            T = len(actions)
            outcome = "success" if terminated_flags[-1] else ("timeout" if truncated_flags[-1] else "incomplete")
            teacher_valid_arr = np.asarray(teacher_valid, dtype=np.uint8)
            eligible = bool(teacher_valid_arr.sum() > 0)
            arrays_np: dict[str, np.ndarray] = {
                "screen_rgb": np.stack(obs_frames["screen_rgb"]).astype(np.uint8),
                "wrist_rgb": np.stack(obs_frames["wrist_rgb"]).astype(np.uint8),
                "proprio": np.stack(obs_frames["proprio"]).astype(np.float32),
                "touch": np.stack(obs_frames["touch"]).astype(np.float32),
                "touch_valid": np.stack(obs_frames["touch_valid"]).astype(np.uint8),
                "action": np.stack(actions).astype(np.float32),
                "executed_q_ref": np.stack(exec_refs).astype(np.float32),
                "teacher_action": np.stack(teacher).astype(np.float32),
                "teacher_valid": teacher_valid_arr,
                "reward": np.asarray(rewards, dtype=np.float32),
                "terminated": np.asarray(terminated_flags, dtype=np.uint8),
                "truncated": np.asarray(truncated_flags, dtype=np.uint8),
                "sim_time": np.asarray(sim_times, dtype=np.float64),
            }
            _require(arrays_np["screen_rgb"].shape == (T + 1, IMAGE_SIZE, IMAGE_SIZE, 3),
                     "screen_rgb frames must be [128,128,3]")
            ep_id = f"{stage}-{ep_idx:04d}"
            ep_dir = out / "episodes" / ep_id
            ep_dir.mkdir(parents=True, exist_ok=True)
            npz_path = ep_dir / "trajectory.npz"
            manifest_path = ep_dir / "manifest.json"
            _require(not npz_path.exists(), f"refusing to overwrite existing trajectory.npz: {npz_path}")
            _require(not manifest_path.exists(), f"refusing to overwrite existing manifest.json: {manifest_path}")
            np.savez(npz_path, **arrays_np)
            manifest = {
                "schema": DATASET_SCHEMA,
                "episode_id": ep_id,
                "source_kind": "student_rollin_correction",
                "source_checkpoint": checkpoint.name,
                "source_checkpoint_sha256": config["student_checkpoint_sha256"],
                "source_checkpoint_stage": ckpt_stage,
                "action_names": action_names,
                "delta_scale": delta_scale,
                "joint_limits": [[float(v) for v in np.asarray(limits[0], dtype=np.float64).reshape(-1)],
                                 [float(v) for v in np.asarray(limits[1], dtype=np.float64).reshape(-1)]],
                "config": config,
                "seed": int(ep_seed),
                "split": "train",
                "task": task_id,
                "outcome": outcome,
                "training_eligible_for_bc": eligible,
                "measured_contact_backend": contact_backend,
                "timestamps": {"created_utc": _utc_now(), "decision_hz": 15.0, "physics_hz": 120.0},
                "array_sha256": {k: _sha256_array(v) for k, v in arrays_np.items()},
                "trajectory_sha256": _sha256_file(npz_path),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            episode_dirs.append((ep_id, ep_dir, eligible))
        index_path = _write_index()
    except Exception:
        if episode_dirs:
            _write_index()
        raise
    finally:
        # Env is closed only after every episode/index/log artifact is written.
        if env is not None:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    return index_path


# ---------------------------------------------------------------------------
# PPO (on-policy, recurrent, BC-anchored)
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    bc_anchor_coef: float = 0.05
    epochs_per_cycle: int = 4
    max_episode_steps: int = 400
    lr: float = 3e-4
    grad_clip: float = 1.0
    checkpoint_every_transitions: int = 50000


def ppo_update(
    model: HistoryActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict,
    demo_batch: dict | None,
    ppo_cfg: PPOConfig,
) -> dict:
    """One PPO pass over a padded on-policy rollout.

    rollout keys: obs tensors (B, T, ...), raw_action (B, T, A), old_logp
    (B, T), value (B, T+1), returns (B, T), adv (B, T), mask (B, T). raw
    actions are the stored pre-tanh samples, so the ratio uses the exact same
    sampled actions; for an unchanged model the ratio is exactly 1. The GRU
    history is recomputed deterministically from the stored observations, so
    the recurrent history matches the collection-time history step for step.
    """
    mean, log_std, value, _ = model.forward(rollout)
    log_std = torch.clamp(log_std, model.cfg.log_std_min, model.cfg.log_std_max)
    new_logp = squashed_log_prob(mean, log_std, rollout["raw_action"])
    ratio = torch.exp(new_logp - rollout["old_logp"])
    adv = rollout["adv"]
    mask = rollout["mask"]
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - ppo_cfg.clip_eps, 1.0 + ppo_cfg.clip_eps) * adv
    pg_loss = -(torch.min(unclipped, clipped) * mask).sum() / mask.sum().clamp(min=1.0)
    value_loss = (0.5 * (value - rollout["returns"]).pow(2) * mask).sum() / mask.sum().clamp(min=1.0)
    # Entropy bonus: recomputed under the NEW policy as the masked mean
    # negative log-prob of the stored pre-tanh samples. This is an explicit
    # squashed-space proxy (exact tanh-Gaussian entropy has no closed form),
    # labelled as such here and in all logs; it is never taken from old_logp.
    entropy_proxy = (-new_logp * mask).sum() / mask.sum().clamp(min=1.0)
    anchor = torch.zeros((), device=pg_loss.device)
    if demo_batch is not None and ppo_cfg.bc_anchor_coef > 0:
        dmean, dlog_std, _, _ = model.forward(demo_batch)
        dlog_std = torch.clamp(dlog_std, model.cfg.log_std_min, model.cfg.log_std_max)
        draw = torch.atanh(demo_batch["action"].clamp(-1.0 + TANH_EPS, 1.0 - TANH_EPS))
        anchor = -(squashed_log_prob(dmean, dlog_std, draw) * demo_batch["mask"]).sum() / demo_batch["mask"].sum().clamp(min=1.0)
    loss = pg_loss + ppo_cfg.value_coef * value_loss - ppo_cfg.entropy_coef * entropy_proxy + ppo_cfg.bc_anchor_coef * anchor
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.grad_clip)
    optimizer.step()
    return {
        "loss_total": float(loss.detach().cpu()),
        "pg_loss": float(pg_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy_proxy": float(entropy_proxy.detach().cpu()),
        "bc_anchor_nll": float(anchor.detach().cpu()),
        "ratio_mean": float((ratio * mask).sum().detach().cpu() / mask.sum().clamp(min=1.0).cpu()),
        "clip_fraction": float(
            ((ratio - 1.0).abs() > ppo_cfg.clip_eps).float().mul(mask).sum().detach().cpu()
            / mask.sum().clamp(min=1.0).cpu()
        ),
    }


def collect_ppo_rollout(
    model: HistoryActorCritic,
    env: Any,
    cfg: ModelConfig,
    device: torch.device,
    max_steps: int,
    seed: int,
    gamma: float,
    gae_lambda: float,
    split: str = "train",
) -> dict:
    """One on-policy episode; stores raw actions + old logps for exact ratios.

    Timeouts (including the evaluator-imposed step cap) bootstrap from the
    value of the actual final observation; true terminals cut the bootstrap
    to zero. ``gamma``/``gae_lambda`` come from the PPO config (never
    hardcoded).
    """
    obs, _info = _env_reset(env, seed=seed, options={"split": split})
    _check_exact_obs_keys(obs)
    hidden = None
    chunks: list[dict] = []
    terminated = truncated = False
    for t in range(max_steps):
        obs_t = _obs_to_tensors(obs, cfg, device)
        with torch.no_grad():
            action, raw, lp, value, hidden = model.act(obs_t, hidden)
        chunk = {
            "obs": {k: np.array(obs[k], copy=True) for k in ACTOR_OBS_KEYS},
            "raw": raw[0].detach().cpu(),
            "logp": float(lp.cpu()),
            "value": float(value.cpu()),
            "action": action[0].detach().cpu().numpy(),
        }
        obs, reward, terminated, truncated, _info = _env_step(env, chunk["action"])
        _check_exact_obs_keys(obs)
        chunk["reward"] = float(reward)
        chunk["terminated"] = bool(terminated)
        chunk["truncated"] = bool(truncated)
        chunks.append(chunk)
        if terminated or truncated:
            break
    if not chunks:
        raise ContractError("env produced zero transitions in a PPO rollout")
    cap_hit = not (chunks[-1]["terminated"] or chunks[-1]["truncated"])
    if cap_hit:
        chunks[-1]["truncated"] = True  # evaluator-imposed cap: bootstrap applies
    # Bootstrap value of the actual final observation (used for timeouts).
    final_obs = {k: np.array(obs[k], copy=True) for k in ACTOR_OBS_KEYS}
    obs_t = _obs_to_tensors(final_obs, cfg, device)
    with torch.no_grad():
        _m, _ls, v_final, _h = model.forward(obs_t, hidden)
    n = len(chunks)
    ep = {
        "screen_rgb": torch.stack([torch.as_tensor(c["obs"]["screen_rgb"], device=device) for c in chunks]),
        "wrist_rgb": torch.stack([torch.as_tensor(c["obs"]["wrist_rgb"], device=device) for c in chunks]),
        "proprio": torch.stack([torch.as_tensor(c["obs"]["proprio"], dtype=torch.float32, device=device) for c in chunks]),
        "touch": torch.stack([torch.as_tensor(c["obs"]["touch"], dtype=torch.float32, device=device) for c in chunks]),
        "touch_valid": torch.stack([torch.as_tensor(c["obs"]["touch_valid"], device=device) for c in chunks]),
        "raw_action": torch.stack([c["raw"] for c in chunks]),
        "old_logp": torch.tensor([c["logp"] for c in chunks], device=device),
        "value": torch.tensor([c["value"] for c in chunks] + [float(v_final[0, 0].cpu())], device=device),
        "reward": [c["reward"] for c in chunks],
        "terminated": [c["terminated"] for c in chunks],
        "truncated": [c["truncated"] for c in chunks],
        "mask": torch.ones(n, device=device),
        "transitions": n,
    }
    adv, ret = compute_gae(ep["reward"], [float(v) for v in ep["value"]],
                           ep["terminated"], ep["truncated"], gamma, gae_lambda)
    ep["adv"] = torch.as_tensor(adv, dtype=torch.float32, device=device)
    ep["returns"] = torch.as_tensor(ret, dtype=torch.float32, device=device)
    return ep


def _pad_rollouts(episodes: list[dict], cfg: ModelConfig, device: torch.device) -> dict:
    maxlen = max(e["raw_action"].shape[0] for e in episodes)
    b = len(episodes)

    def pad(key):
        first = episodes[0][key]
        out = torch.zeros(b, maxlen, *first.shape[1:], device=device, dtype=first.dtype)
        for i, e in enumerate(episodes):
            n = e["raw_action"].shape[0]
            out[i, :n] = e[key]
        return out

    batch = {
        "screen_rgb": pad("screen_rgb"),
        "wrist_rgb": pad("wrist_rgb"),
        "proprio": pad("proprio"),
        "touch": pad("touch"),
        "touch_valid": pad("touch_valid"),
        "raw_action": pad("raw_action"),
        "old_logp": pad("old_logp"),
        "value": torch.zeros(b, maxlen + 1, device=device),
        "adv": pad("adv"),
        "returns": pad("returns"),
        "mask": torch.zeros(b, maxlen, device=device),
    }
    for i, e in enumerate(episodes):
        n = e["raw_action"].shape[0]
        batch["value"][i, : n + 1] = e["value"]
        batch["mask"][i, :n] = 1.0
    return batch


def train_ppo(
    env_factory: Callable[..., Any],
    dataset: DatasetBundle,
    checkpoint: Path,
    output: Path,
    transitions: int,
    seed: int,
    env_kwargs: dict | None = None,
    ppo_cfg: PPOConfig | None = None,
    device: str = "cpu",
    rollout_episodes_per_cycle: int = 4,
    demo_episodes_per_update: int = 4,
    time_budget_s: float | None = None,
) -> Path:
    """PPO with a BC anchor, starting from a BC checkpoint.

    ``transitions`` counts real completed env transitions consumed (never
    optimizer iterations). Resumed runs continue the transition counter from
    the parent checkpoint so the campaign budget is honest. Checkpoints are
    named by completed transitions and written periodically; the env is
    closed only after the final checkpoint and log are written.
    """
    ppo_cfg = ppo_cfg or PPOConfig()
    dev = torch.device(device)
    seed_everything(seed)
    stop_requested, restore_sigterm = _install_sigterm_flag()
    started = time.time()
    model, payload = load_checkpoint(checkpoint, dev)
    cfg = ModelConfig.from_dict(payload["model_config"])
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.lr)
    try:
        load_optimizer_state(optimizer, payload)
    except ContractError:
        pass  # fresh optimizer for the new stage is acceptable and logged
    env = make_env(env_factory, env_kwargs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    consumed = int(payload.get("env_transitions", 0))
    global_step = int(payload["global_step"])
    provenance = {
        "parent_checkpoint": str(checkpoint),
        "dataset_sha256": _sha256_file(dataset.path),
        "seed": seed,
        "stage_history": [{"stage": payload["stage"], "from": str(checkpoint)}],
    }
    demo_pool = dataset.bc_train_pool()
    if not demo_pool:
        restore_sigterm()
        raise ContractError("PPO BC anchor needs eligible train episodes")
    rng = random.Random(seed)
    cycle = 0
    append_log(
        output,
        {
            "event": "ppo_begin",
            "stage": "ppo",
            "optimizer_step": global_step,
            "env_transitions": consumed,
            "data_counts": {"target_transitions": transitions, "anchor_train_episodes": len(demo_pool)},
            "lr": ppo_cfg.lr,
            "seed": seed,
        },
    )
    model.train()
    stop_reason = "transitions"
    try:
        while consumed < transitions:
            if time_budget_s is not None and time.time() - started > time_budget_s:
                stop_reason = "time_budget"
                break
            if stop_requested():
                stop_reason = "sigterm"
                break
            episodes = [
                collect_ppo_rollout(model, env, cfg, dev, ppo_cfg.max_episode_steps,
                                    rng.randrange(2**31), ppo_cfg.gamma, ppo_cfg.gae_lambda)
                for _ in range(rollout_episodes_per_cycle)
            ]
            consumed += sum(int(e["transitions"]) for e in episodes)
            batch = _pad_rollouts(episodes, cfg, dev)
            metrics: dict = {}
            for _ in range(ppo_cfg.epochs_per_cycle):
                demo = pad_episodes(
                    [demo_pool[rng.randrange(len(demo_pool))] for _ in range(demo_episodes_per_update)],
                    cfg,
                    dev,
                )
                metrics = ppo_update(model, optimizer, batch, demo, ppo_cfg)
                global_step += 1
            cycle += 1
            append_log(
                output,
                {
                    "event": "ppo_cycle",
                    "stage": "ppo",
                    "optimizer_step": global_step,
                    "env_transitions": consumed,
                    "data_counts": {
                        "env_transitions_consumed": consumed,
                        "target_transitions": transitions,
                        "anchor_train_episodes": len(demo_pool),
                    },
                    "lr": optimizer.param_groups[0]["lr"],
                    "losses": metrics,
                },
            )
            if consumed % max(1, ppo_cfg.checkpoint_every_transitions) < sum(
                int(e["transitions"]) for e in episodes
            ):
                save_checkpoint(output / f"ppo_transitions{consumed}.pt", model, optimizer, cfg,
                                "ppo", global_step, provenance, env_transitions=consumed)
    finally:
        last = output / f"ppo_transitions{consumed}.pt"
        save_checkpoint(last, model, optimizer, cfg, "ppo", global_step, provenance, env_transitions=consumed)
        save_checkpoint(output / "latest.pt", model, optimizer, cfg, "ppo", global_step, provenance,
                        env_transitions=consumed)
        append_log(
            output,
            {
                "event": "ppo_end",
                "stage": "ppo",
                "optimizer_step": global_step,
                "env_transitions": consumed,
                "data_counts": {"env_transitions_consumed": consumed, "target_transitions": transitions},
                "checkpoint": str(last),
                "stop_reason": stop_reason,
            },
        )
        # The env is closed only after every checkpoint/log above is written.
        close = getattr(env, "close", None)
        if callable(close):
            close()
        restore_sigterm()
    return last


# ---------------------------------------------------------------------------
# Autonomous evaluation (closed-loop, heldout-only, no teacher fallback)
# ---------------------------------------------------------------------------

def _encode_video_from_array(arr: np.ndarray, out_path: Path, fps: int = 15) -> dict:
    """Encode one MP4 from a (T+1,H,W,C) uint8 array; no PNG copies kept."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim != 4 or arr.shape[0] < 4:
        return {"returncode": None, "frames": int(arr.shape[0]) if arr.ndim else 0,
                "error": "too few frames"}
    frames_dir = out_path.parent / f".{out_path.stem}_frames_tmp"
    if frames_dir.exists():
        shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    try:
        from PIL import Image
    except ImportError as exc:  # encoder missing: raw NPZ stays the evidence
        return {"returncode": None, "frames": int(arr.shape[0]),
                "error": f"encoder unavailable: {exc}"}
    for i, img in enumerate(arr):
        Image.fromarray(img).save(frames_dir / f"f_{i:05d}.png")
    try:
        rc = subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(fps), "-i",
             str(frames_dir / "f_%05d.png"), "-pix_fmt", "yuv420p", str(out_path)],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    except OSError as exc:  # e.g. ffmpeg absent: record, do not crash the eval
        shutil.rmtree(frames_dir, ignore_errors=True)
        return {"returncode": None, "frames": int(arr.shape[0]),
                "error": f"encoder unavailable: {exc}"}
    shutil.rmtree(frames_dir, ignore_errors=True)
    return {"returncode": int(rc), "frames": int(arr.shape[0])}


def evaluate_policy(
    env_factory: Callable[..., Any],
    checkpoint: Path,
    episodes: int,
    seed: int,
    env_kwargs: dict | None = None,
    max_steps: int = 400,
    output: Path | None = None,
    task_id: str = "tap",
    seeds: Sequence[int] | None = None,
    trace_dir: Path | None = None,
    trace_episodes: int = 10,
    time_budget_s: float | None = None,
) -> dict:
    """Closed-loop evaluation on the explicit heldout configuration.

    Every reset passes options {"split": "heldout"}; the actor reads only the
    whitelisted observation keys and never the privileged info mapping. The
    env is closed only after the report is written (SimulationApp.close is a
    fast shutdown). No teacher action is ever substituted for the policy.

    ``seeds`` optionally pins the exact per-episode reset seeds (fixed nominal
    target coordinates). For the first ``trace_episodes`` episodes the full
    raw observation/action stream plus privileged contacts/cursor/target are
    saved under ``trace_dir`` (never overwriting existing files), with a
    paired screen video encoded from the saved arrays. ``time_budget_s``
    stops the campaign at an episode boundary; SIGTERM does the same, and the
    report is always written before the env is closed.
    """
    dev = torch.device("cpu")
    model, payload = load_checkpoint(checkpoint, dev)
    model.eval()
    cfg = ModelConfig.from_dict(payload["model_config"])
    env = make_env(env_factory, env_kwargs)
    rng = random.Random(seed)
    if seeds is not None:
        seeds = [int(s) for s in seeds]
        if len(seeds) < episodes:
            raise ContractError(
                f"explicit seeds list shorter than episodes: {len(seeds)} < {episodes}")
    stop_requested, restore_sigterm = _install_sigterm_flag()
    started = time.time()
    rows: list[dict] = []
    event_counts: dict[str, int] = {}
    trace_records: list[dict] = []

    def _persist_trace(ep: int, ep_seed: int, data: dict) -> dict:
        """Write one episode's raw trace + paired videos immediately.

        Never overwrites existing raw files; returns the record metadata or a
        skip marker. Full observation arrays are copied at collection time so
        a backend reusing buffers cannot corrupt earlier steps.
        """
        tdir = Path(trace_dir)
        tdir.mkdir(parents=True, exist_ok=True)
        ep_dir = tdir / f"episode_{ep:03d}_seed{ep_seed}"
        if (ep_dir / "raw_trace.npz").exists() or (ep_dir / "trace_meta.json").exists():
            return {"episode": ep, "seed": int(ep_seed), "skipped_existing": True}
        ep_dir.mkdir(parents=True, exist_ok=True)
        meta = data.pop("_meta")
        arrays = {k: np.stack(v) for k, v in data.items()}
        np.savez_compressed(ep_dir / "raw_trace.npz", **arrays)
        (ep_dir / "trace_meta.json").write_text(
            json.dumps({
                "episode": ep, "seed": int(ep_seed),
                "checkpoint": str(checkpoint),
                "max_steps": int(max_steps),
                "privileged_evaluator_trace": True,
                "actor_observation_keys": list(ACTOR_OBS_KEYS),
                "steps": meta,
            }, indent=2), encoding="utf-8")
        v_screen = _encode_video_from_array(arrays["screen_rgb"], ep_dir / "screen_rgb.mp4")
        v_wrist = _encode_video_from_array(arrays["wrist_rgb"], ep_dir / "wrist_rgb.mp4")
        return {"episode": ep, "seed": int(ep_seed),
                "raw": str(ep_dir / "raw_trace.npz"),
                "video_screen": str(ep_dir / "screen_rgb.mp4"),
                "video_wrist": str(ep_dir / "wrist_rgb.mp4"),
                "video_screen_encode": v_screen,
                "video_wrist_encode": v_wrist}

    try:
        for ep in range(episodes):
            if stop_requested():
                break
            if time_budget_s is not None and time.time() - started > time_budget_s:
                break
            ep_seed = seeds[ep] if seeds is not None else rng.randrange(2**31)
            obs, info = _env_reset(env, seed=ep_seed,
                                   options={"split": "heldout", "task_id": task_id})
            _check_exact_obs_keys(obs)
            hidden = None
            steps = 0
            trace_obs: dict[str, list] | None = None
            if trace_dir is not None and ep < int(trace_episodes):
                trace_obs = {k: [np.array(obs[k], copy=True)] for k in ACTOR_OBS_KEYS}
                trace_obs["_meta"] = []
            outcome: dict[str, Any] = {"episode": ep, "seed": int(ep_seed)}
            for t in range(max_steps):
                obs_t = _obs_to_tensors(obs, cfg, dev)
                with torch.no_grad():
                    action, _raw, _lp, _v, hidden = model.act(obs_t, hidden, deterministic=True)
                act = action[0].cpu().numpy()
                obs, reward, terminated, truncated, info = _env_step(env, act)
                _check_exact_obs_keys(obs)
                steps = t + 1
                is_last = terminated or truncated or (t == max_steps - 1)
                # The evaluator-imposed step cap applies ONLY when the env did
                # not end the episode itself: a success exactly on the last
                # step is terminated=True, truncated=False, no cap.
                cap_hit = (t == max_steps - 1) and not terminated and not truncated
                if trace_obs is not None:
                    for k in ACTOR_OBS_KEYS:
                        trace_obs[k].append(np.array(obs[k], copy=True))
                    trace_obs["_meta"].append({
                        "t": t,
                        "action": [float(v) for v in act],
                        "executed_q_ref": [float(v) for v in (
                            np.asarray((info or {}).get("executed_q_ref"),
                                       dtype=np.float64).tolist()
                            if (info or {}).get("executed_q_ref") is not None else [])],
                        "reward": float(reward),
                        "sim_time": float((info or {}).get("sim_time", float("nan"))),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated) or cap_hit,
                        "evaluator_step_cap": bool(cap_hit),
                        "failure_reason": ((info or {}).get("failure_reason")
                                           or ("eval_step_limit" if cap_hit else None)),
                        "cursor_px": (info or {}).get("oracle_cursor_px"),
                        "target_px": (info or {}).get("oracle_target_px"),
                        "cursor_over_target": bool((info or {}).get("oracle_cursor_over_target", False)),
                        "click_registered": int((info or {}).get("oracle_click_registered", 0)),
                        "click_on_target": int((info or {}).get("oracle_click_on_target", 0)),
                        "contacts": (info or {}).get("contacts", {}),
                        "events": (info or {}).get("events", []) or [],
                    })
                for ev in (info or {}).get("events", []) or []:
                    event_counts[str(ev)] = event_counts.get(str(ev), 0) + 1
                if is_last:
                    outcome.update(
                        {
                            "success": bool((info or {}).get("success", False)) and bool(terminated),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated) or cap_hit,
                            "steps": steps,
                            "failure_reason": ((info or {}).get("failure_reason")
                                               or ("eval_step_limit" if cap_hit else None)),
                        }
                    )
                    break
            rows.append(outcome)
            if trace_obs is not None:
                trace_records.append(_persist_trace(ep, int(ep_seed), trace_obs))
    finally:
        report = {
            "schema": LOG_SCHEMA,
            "checkpoint": str(checkpoint),
            "stage": payload["stage"],
            "split_options": "heldout",
            "episodes_attempted": len(rows),
            "episodes_requested": int(episodes),
            "stop_reason": ("sigterm" if stop_requested() else
                            ("time_budget" if (time_budget_s is not None
                                               and time.time() - started > time_budget_s
                                               and len(rows) < episodes)
                             else "completed")),
            "successes": sum(1 for r in rows if r.get("success")),
            "success_rate": (sum(1 for r in rows if r.get("success")) / len(rows)) if rows else 0.0,
            "mean_steps": sum(r["steps"] for r in rows) / len(rows) if rows else 0.0,
            "failure_reasons": {k: sum(1 for r in rows if r.get("failure_reason") == k)
                                for k in {r.get("failure_reason") for r in rows}},
            "event_counts": event_counts,
            "seeds": [r.get("seed") for r in rows],
            "rows": rows,
            "traces": trace_records,
        }
        # Raw traces are persisted per episode during the loop; the aggregate
        # report is written here, AFTER trace metadata exists and BEFORE the
        # env is closed (Isaac fast_shutdown exits the process).
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            (output / f"eval_{int(time.time())}.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
        close = getattr(env, "close", None)
        if callable(close):
            close()
        restore_sigterm()
    return report


# ---------------------------------------------------------------------------
# GPU probe (labelled software check; verifies visible-device UUID mapping)
# ---------------------------------------------------------------------------

def os_environ_cuda() -> str | None:
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def probe_gpus() -> dict:
    """Report CUDA visibility vs physical UUID mapping.

    Labeled software check only: it inspects driver visibility, it does not
    allocate or train. When CUDA_VISIBLE_DEVICES restricts visibility, the
    torch index -> physical UUID mapping is cross-checked against nvidia-smi.
    """
    visible = [v for v in (os_environ_cuda() or "").split(",") if v.strip()]
    result: dict[str, Any] = {
        "label": "software-check-only-no-training",
        "cuda_visible_devices": os_environ_cuda(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "visible_index_to_uuid": {},
        "physical_index_to_uuid": {},
        "warnings": [],
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0 and out.stdout.strip():
            for line in out.stdout.strip().splitlines():
                idx, uuid, name = [p.strip() for p in line.split(",", 2)]
                result["physical_index_to_uuid"][idx] = {"uuid": uuid, "name": name}
        else:
            result["warnings"].append("nvidia-smi unavailable; UUID mapping unverified")
    except (OSError, subprocess.SubprocessError):
        result["warnings"].append("nvidia-smi unavailable; UUID mapping unverified")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                p = torch.cuda.get_device_properties(i)
                result["visible_index_to_uuid"][str(i)] = {"name": p.name, "total_mb": p.total_memory // (1 << 20)}
            except Exception as exc:  # pragma: no cover - hardware dependent
                result["warnings"].append(f"torch device {i} probe failed: {exc}")
        if result["physical_index_to_uuid"]:
            order = visible
            if order and all(v.isdigit() for v in order):
                result["visible_index_to_uuid_mapping"] = {
                    f"torch:{vi}": result["physical_index_to_uuid"].get(pi, {}) for vi, pi in enumerate(order)
                }
                for vi, pi in enumerate(order):
                    if pi not in result["physical_index_to_uuid"]:
                        result["warnings"].append(f"CUDA_VISIBLE_DEVICES entry {pi} not present on this host")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--action-dim", type=int, default=ACTION_DIM)
    p.add_argument("--proprio-dim", type=int, default=PROPRIO_DIM)
    p.add_argument("--touch-dim", type=int, default=TOUCH_DIM)
    p.add_argument("--image-hw", type=int, nargs=2, default=[IMAGE_SIZE, IMAGE_SIZE])
    p.add_argument("--device", default="cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sharpa-bcppo",
        description="Visual recurrent BC + corrections + PPO for the Sharpa trackpad.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-dataset", help="Validate a dataset index + episodes.")
    p.add_argument("--dataset", required=True)

    p = sub.add_parser("train-bc", help="Train BC on the frozen train split (+ corrections).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--init-from", default=None, help="Resume from a checkpoint (correction retraining).")
    p.add_argument("--corrections", action="append", default=None, help="Correction dataset index; repeatable.")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-episodes", type=int, default=8)
    p.add_argument("--stage", default="bc")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--time-budget", type=float, default=None, help="Wall-clock seconds; checkpoint+exit on timeout.")
    p.add_argument("--refit-normalizer", action="store_true",
                   help="Explicitly refit proprio/touch normalizers on resume (versioned).")
    _add_common_model_args(p)

    p = sub.add_parser("correct", help="Student roll-in correction gathering (split='train' options).")
    p.add_argument("--env-factory", default=DEFAULT_ENV_FACTORY)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rollouts", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--task-id", default="tap")
    p.add_argument("--env-kwarg", action="append", default=None)

    p = sub.add_parser("retrain-bc", help="Alias of train-bc --init-from for correction retraining.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--corrections", action="append", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-episodes", type=int, default=8)
    p.add_argument("--stage", default="bc+corrections")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--refit-normalizer", action="store_true")
    _add_common_model_args(p)

    p = sub.add_parser("train-ppo", help="PPO with BC anchor from a BC checkpoint.")
    p.add_argument("--env-factory", default=DEFAULT_ENV_FACTORY)
    p.add_argument("--dataset", required=True, help="BC dataset index for the anchor.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--transitions", type=int, required=True, help="Real env transition budget (not optimizer steps).")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--env-kwarg", action="append", default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bc-anchor-coef", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--rollout-episodes-per-cycle", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=400)
    p.add_argument("--checkpoint-every-transitions", type=int, default=50000)
    p.add_argument("--time-budget", type=float, default=None)
    _add_common_model_args(p)

    p = sub.add_parser("eval", help="Autonomous closed-loop evaluation (split='heldout' options).")
    p.add_argument("--env-factory", default=DEFAULT_ENV_FACTORY)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default=None,
                   help="Comma-separated explicit per-episode reset seeds (fixed nominal targets).")
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--task-id", default="tap")
    p.add_argument("--output", default=None)
    p.add_argument("--trace-dir", default=None,
                    help="Save full raw obs/action/contact/cursor traces + paired video here.")
    p.add_argument("--trace-episodes", type=int, default=10)
    p.add_argument("--time-budget", type=float, default=None)
    p.add_argument("--env-kwarg", action="append", default=None)

    sub.add_parser("probe-gpu", help="Labelled software check of CUDA/UUID visibility.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.command == "validate-dataset":
        bundle = load_dataset(args.dataset)
        summary = dataset_summary(bundle)
        print(json.dumps(summary, indent=2, sort_keys=True))
        ok = all(s["eligible_episodes"] > 0 for s in summary["splits"].values())
        return 0 if ok else 2

    if args.command in ("train-bc", "retrain-bc"):
        cfg = ModelConfig(
            action_dim=args.action_dim,
            proprio_dim=args.proprio_dim,
            touch_dim=args.touch_dim,
            image_hw=(int(args.image_hw[0]), int(args.image_hw[1])),
        )
        dataset = load_dataset(args.dataset)
        init_from = Path(args.checkpoint) if args.command == "retrain-bc" else (
            Path(args.init_from) if args.init_from else None)
        ckpt = train_bc(
            dataset=dataset,
            output=Path(args.output),
            steps=args.steps,
            seed=args.seed,
            cfg=cfg,
            init_from=init_from,
            corrections=[Path(c) for c in (args.corrections or [])] or None,
            lr=args.lr,
            batch_episodes=args.batch_episodes,
            device=args.device,
            stage=args.stage,
            grad_clip=args.grad_clip,
            checkpoint_every=args.checkpoint_every,
            time_budget_s=args.time_budget,
            refit_normalizer=args.refit_normalizer,
        )
        print(json.dumps({"checkpoint": str(ckpt), "stage": args.stage}, sort_keys=True))
        return 0

    if args.command == "correct":
        factory = load_env_factory(args.env_factory)
        path = gather_corrections(
            factory,
            Path(args.checkpoint),
            Path(args.output),
            rollouts=args.rollouts,
            seed=args.seed,
            max_steps=args.max_steps,
            env_kwargs=parse_env_kwargs(args.env_kwarg),
            task_id=args.task_id,
        )
        print(json.dumps({"corrections": str(path)}, sort_keys=True))
        return 0

    if args.command == "train-ppo":
        factory = load_env_factory(args.env_factory)
        dataset = load_dataset(args.dataset)
        ckpt = train_ppo(
            factory,
            dataset,
            Path(args.checkpoint),
            Path(args.output),
            transitions=args.transitions,
            seed=args.seed,
            env_kwargs=parse_env_kwargs(args.env_kwarg),
            ppo_cfg=PPOConfig(
                lr=args.lr,
                bc_anchor_coef=args.bc_anchor_coef,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                max_episode_steps=args.max_episode_steps,
                checkpoint_every_transitions=args.checkpoint_every_transitions,
            ),
            device=args.device,
            rollout_episodes_per_cycle=args.rollout_episodes_per_cycle,
            time_budget_s=args.time_budget,
        )
        print(json.dumps({"checkpoint": str(ckpt), "stage": "ppo"}, sort_keys=True))
        return 0

    if args.command == "eval":
        factory = load_env_factory(args.env_factory)
        explicit_seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
                          if args.seeds else None)
        report = evaluate_policy(
            factory,
            Path(args.checkpoint),
            episodes=args.episodes,
            seed=args.seed,
            env_kwargs=parse_env_kwargs(args.env_kwarg),
            max_steps=args.max_steps,
            output=Path(args.output) if args.output else None,
            task_id=args.task_id,
            seeds=explicit_seeds,
            trace_dir=Path(args.trace_dir) if args.trace_dir else None,
            trace_episodes=args.trace_episodes,
            time_budget_s=args.time_budget,
        )
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, sort_keys=True))
        return 0

    if args.command == "probe-gpu":
        print(json.dumps(probe_gpus(), indent=2, sort_keys=True))
        return 0

    raise ContractError(f"unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
