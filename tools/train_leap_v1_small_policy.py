"""CLI trainer for the LEAP v1 Full small-policy baseline.

PyTorch is imported lazily inside training functions so that the file can be
syntax-checked and its non-training paths inspected on a NumPy-only runtime.
The trainer refuses to optimize without a coordinator-approved permit, a CUDA
H800 GPU, and a SLURM_JOB_ID environment variable; there is no CPU fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# NumPy-only module: safe to import on CPU-only runtimes.
from embodied_trackpad.leap_v1.small_policy import (
    ACTION_CAP_RAD,
    BLOCK_CAP_RAD,
    HIDDEN_DIM,
    HORIZON,
    INPUT_DIM,
    MOTION4_INDICES,
    NUM_CHANNELS,
    OUTPUT_DIM,
    TEST_TARGET_CLASSES,
    SmallPolicyError,
    SmallPolicyNormalizer,
    decode_controls_batched,
    extract_features,
    extract_label_delta,
    forward_numpy,
    load_sample_index,
    sha256_file,
    split_by_target_class,
    validate_permit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / ".local" / "small-policy-source" / "checkpoints"
DEFAULT_REPORT_PATH = REPO_ROOT / ".local" / "small-policy-source" / "report.json"

# Safety margins from docs/tasks/v1-small-policy-smoke.zh-CN.md
MIN_REMAINING_S = 60
OUTPUT_BUDGET_BYTES = 512 * 1024 * 1024
RESUME_CHECK_TOLERANCE = 1e-5


def _get_git_commit() -> str:
    """Best-effort current source commit; empty string if not a git checkout."""
    try:
        import subprocess

        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write a JSON file; useful for progress receipts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)


def _require_gpu_and_slurm() -> tuple[str, str]:
    """Abort unless CUDA is available, SLURM allocation holds, and GPU is H800."""
    import torch

    if not torch.cuda.is_available():
        raise SmallPolicyError("CUDA is not available; CPU training is not allowed")
    slurm_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_id:
        raise SmallPolicyError("SLURM_JOB_ID is not set; training only under SLURM allocation")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.count(",") != 0:
        raise SmallPolicyError("CUDA_VISIBLE_DEVICES must name exactly one GPU")

    device_count = torch.cuda.device_count()
    if device_count != 1:
        raise SmallPolicyError(f"expected exactly 1 CUDA device, found {device_count}")

    device_name = torch.cuda.get_device_name(0)
    if "H800" not in device_name:
        raise SmallPolicyError(f"GPU must be H800, found {device_name}")

    return slurm_id, device_name


def set_seed(seed: int) -> dict[str, Any]:
    """Seed Python, NumPy, and both CPU/CUDA torch RNGs; return state dict."""
    random.seed(seed)
    np.random.seed(seed)
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng_state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return rng_state


def get_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    try:
        import torch

        state["torch"] = torch.get_rng_state()
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    try:
        import torch

        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except ImportError:
        pass


def build_model(device: str) -> Any:
    """Construct 50->128->128->400 MLP with SiLU activations."""
    import torch
    import torch.nn as nn

    class SmallPolicyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
            self.fc2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
            self.fc3 = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.nn.functional.silu(self.fc1(x))
            x = torch.nn.functional.silu(self.fc2(x))
            return self.fc3(x)

    model = SmallPolicyMLP().to(device)
    return model


def model_hash(model: Any) -> str:
    """SHA-256 of model parameters for before/after comparisons."""
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def model_forward_torch_batched(model: Any, features: np.ndarray) -> np.ndarray:
    """Run a batched forward pass and return Bx25x16 delta predictions."""
    import torch

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(features).float()
        if next(model.parameters()).is_cuda:
            x = x.cuda()
        logits = model(x)
    return (ACTION_CAP_RAD * torch.tanh(logits)).view(-1, HORIZON, NUM_CHANNELS).cpu().numpy()


def compute_loss_torch(
    logits: Any,
    target_deltas: Any,
    previous_ctrl: Any,
) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    """Normalized increment + cumulative MSE loss in torch (batched)."""
    import torch

    pred_deltas = ACTION_CAP_RAD * torch.tanh(logits.view(-1, HORIZON, NUM_CHANNELS))

    inc_mse = torch.mean((pred_deltas - target_deltas) ** 2) / (ACTION_CAP_RAD**2)

    pred_ctrl = previous_ctrl[:, None, :] + torch.cumsum(pred_deltas, dim=1)
    target_ctrl = previous_ctrl[:, None, :] + torch.cumsum(target_deltas, dim=1)
    cum_mse = torch.mean((pred_ctrl - target_ctrl) ** 2) / (BLOCK_CAP_RAD**2)

    loss = inc_mse + 0.25 * cum_mse
    metrics = {
        "loss": loss.item(),
        "increment_mse_normalized": inc_mse.item(),
        "cumulative_mse_normalized": cum_mse.item(),
    }
    return loss, metrics, pred_deltas, pred_ctrl, target_ctrl


def evaluate_split(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    previous_ctrls: np.ndarray,
    device: str,
) -> dict[str, Any]:
    """Evaluate loss and cumulative control RMSE on a split."""
    import torch

    model.eval()
    n = features.shape[0]
    if n == 0:
        return {
            "loss": float("nan"),
            "increment_mse_normalized": float("nan"),
            "cumulative_mse_normalized": float("nan"),
            "control_rmse_rad": float("nan"),
            "count": 0,
        }

    with torch.no_grad():
        x = torch.from_numpy(features).float().to(device)
        target = torch.from_numpy(labels).float().to(device)
        prev = torch.from_numpy(previous_ctrls).float().to(device)
        logits = model(x)
        loss, metrics, _, pred_ctrl, target_ctrl = compute_loss_torch(logits, target, prev)
        rmse = torch.sqrt(torch.mean((pred_ctrl - target_ctrl) ** 2))

    return {
        "loss": metrics["loss"],
        "increment_mse_normalized": metrics["increment_mse_normalized"],
        "cumulative_mse_normalized": metrics["cumulative_mse_normalized"],
        "control_rmse_rad": rmse.item(),
        "count": n,
    }


def _compute_channel_metrics_batched(
    pred_controls: np.ndarray, target_controls: np.ndarray
) -> dict[str, Any]:
    """Per-channel and motion-4 RMSEs across a batch of control blocks."""
    diff = pred_controls - target_controls
    per_channel = np.sqrt(np.mean(diff**2, axis=(0, 1)))
    motion4_rmse = float(np.sqrt(np.mean(diff[:, :, MOTION4_INDICES] ** 2)))
    return {
        "rmse_per_channel_rad": per_channel.astype(np.float32),
        "rmse_motion4_rad": motion4_rmse,
        "rmse_all_channels_rad": float(np.sqrt(np.mean(diff**2))),
    }


def evaluate_split_with_channels(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    previous_ctrls: np.ndarray,
    device: str,
) -> dict[str, Any]:
    """Like ``evaluate_split`` but additionally returns per-channel motion-4 RMSEs."""
    import torch

    base = evaluate_split(model, features, labels, previous_ctrls, device)
    n = base["count"]
    if n == 0:
        return base

    with torch.no_grad():
        x = torch.from_numpy(features).float().to(device)
        target = torch.from_numpy(labels).float().to(device)
        prev = torch.from_numpy(previous_ctrls).float().to(device)
        logits = model(x)
        pred_deltas = ACTION_CAP_RAD * torch.tanh(logits.view(-1, HORIZON, NUM_CHANNELS))
        pred_ctrl = prev[:, None, :] + torch.cumsum(pred_deltas, dim=1)
        target_ctrl = prev[:, None, :] + torch.cumsum(target, dim=1)

    channels = _compute_channel_metrics_batched(pred_ctrl.cpu().numpy(), target_ctrl.cpu().numpy())
    base.update(channels)
    return base


def prepare_datasets(
    records: list[dict[str, Any]], normalizer: SmallPolicyNormalizer | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], SmallPolicyNormalizer]:
    """Extract raw features/labels and fit normalizer on training split only.

    Returns ``raw_features``, ``normalized_features``, ``labels``, ``normalizer``.
    The raw arrays are preserved for the NumPy inference parity check, while
    training uses the normalized arrays.
    """
    splits = split_by_target_class(records)

    raw_features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split_name, recs in splits.items():
        feat_list = []
        lab_list = []
        for rec in recs:
            feat_list.append(extract_features(rec))
            lab_list.append(extract_label_delta(rec))
        raw_features[split_name] = np.stack(feat_list) if feat_list else np.empty((0, INPUT_DIM), dtype=np.float32)
        labels[split_name] = np.stack(lab_list) if lab_list else np.empty((0, HORIZON, NUM_CHANNELS), dtype=np.float32)

    if normalizer is None:
        normalizer = SmallPolicyNormalizer().fit(raw_features["train"])

    normalized_features: dict[str, np.ndarray] = {}
    for split_name in raw_features:
        normalized_features[split_name] = normalizer.transform(raw_features[split_name])

    return raw_features, normalized_features, labels, normalizer


REQUIRED_SOURCE_FILES = frozenset(
    {
        "tools/train_leap_v1_small_policy.py",
        "src/embodied_trackpad/leap_v1/small_policy.py",
        "env/leap-v1-small-policy-contract.json",
    }
)


def _build_source_manifest() -> dict[str, str]:
    """SHA-256 per authorized source file; works without a .git bundle."""
    files = [
        Path(__file__).resolve(),
        REPO_ROOT / "src" / "embodied_trackpad" / "leap_v1" / "small_policy.py",
        REPO_ROOT / "env" / "leap-v1-small-policy-contract.json",
    ]
    manifest: dict[str, str] = {}
    for path in files:
        if path.exists():
            manifest[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
    return manifest


def _is_safe_relative_path(path: str) -> bool:
    """Reject absolute paths and any ``..`` escape attempts."""
    p = Path(path)
    if p.is_absolute():
        return False
    return ".." not in p.parts


def validate_source_manifest(
    manifest: dict[str, Any],
    permit: dict[str, Any],
    repo_root: Path,
    git_commit: str | None = None,
) -> None:
    """Validate that the supplied source manifest matches the permit and disk.

    Requires ``{source_commit, files: {relative_path: sha256}}``. The manifest
    source_commit must equal the permit source_commit and (when available) the
    current git HEAD. Every listed file must exist, be a safe relative path,
    and its on-disk SHA-256 must match both the manifest and the permit's
    source_hashes.
    """
    if not isinstance(manifest, dict):
        raise SmallPolicyError("source manifest must be a dict")
    if "source_commit" not in manifest or "files" not in manifest:
        raise SmallPolicyError("source manifest must contain source_commit and files")

    source_commit = str(manifest["source_commit"])
    if not source_commit:
        raise SmallPolicyError("source manifest source_commit is empty")

    expected_commit = str(permit.get("source_commit", ""))
    if not expected_commit:
        raise SmallPolicyError("permit source_commit is empty")
    if source_commit != expected_commit:
        raise SmallPolicyError(
            f"source manifest source_commit mismatch: manifest {source_commit[:12]} "
            f"vs permit {expected_commit[:12]}"
        )

    if git_commit is not None and git_commit and git_commit != source_commit:
        raise SmallPolicyError(
            f"git HEAD {git_commit[:12]} does not match source manifest {source_commit[:12]}"
        )

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise SmallPolicyError("source manifest files must be a dict")

    # Reject unsafe paths before checking exact file membership so that
    # path-escape attempts are caught with the correct error message.
    for rel_path in files.keys():
        if not _is_safe_relative_path(rel_path):
            raise SmallPolicyError(f"source manifest path is not safe: {rel_path}")

    if set(files.keys()) != REQUIRED_SOURCE_FILES:
        raise SmallPolicyError(
            f"source manifest files must be exactly {sorted(REQUIRED_SOURCE_FILES)}, got {sorted(files.keys())}"
        )

    expected_hashes = permit.get("source_hashes")
    if not isinstance(expected_hashes, dict):
        raise SmallPolicyError("permit source_hashes must be a dict")
    if set(expected_hashes.keys()) != REQUIRED_SOURCE_FILES:
        raise SmallPolicyError(
            f"permit source_hashes must be exactly {sorted(REQUIRED_SOURCE_FILES)}, got {sorted(expected_hashes.keys())}"
        )

    for rel_path in REQUIRED_SOURCE_FILES:
        full_path = repo_root / rel_path
        if not full_path.is_file():
            raise SmallPolicyError(f"source manifest claims missing file: {rel_path}")

        actual_hash = sha256_file(full_path)
        manifest_hash = str(files[rel_path])
        permit_hash = str(expected_hashes[rel_path])

        if len(manifest_hash) != 64 or not all(c in "0123456789abcdef" for c in manifest_hash.lower()):
            raise SmallPolicyError(f"source manifest hash for {rel_path} is not 64-character hex")
        if manifest_hash.lower() != actual_hash.lower():
            raise SmallPolicyError(f"source manifest hash mismatch for {rel_path}")
        if permit_hash.lower() != actual_hash.lower():
            raise SmallPolicyError(f"permit source_hashes mismatch for {rel_path}")


def validate_frozen_hyperparams(args: argparse.Namespace, permit: dict[str, Any]) -> dict[str, Any]:
    """Enforce the frozen baseline hyperparameters and permit limits.

    lr=1e-3, weight_decay=1e-4, batch_size=64, grad_clip=1.0, seed=17,
    val_every=100. max_updates must be <= 2000 and <= permit limits.
    """
    hyperparams = {
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "grad_clip": float(args.grad_clip),
        "seed": int(args.seed),
        "max_updates": int(args.max_updates),
        "val_every": int(args.val_every),
    }

    if hyperparams["lr"] != 1e-3:
        raise SmallPolicyError(f"lr must be 0.001, got {hyperparams['lr']}")
    if hyperparams["weight_decay"] != 1e-4:
        raise SmallPolicyError(f"weight_decay must be 0.0001, got {hyperparams['weight_decay']}")
    if hyperparams["batch_size"] != 64:
        raise SmallPolicyError(f"batch_size must be 64, got {hyperparams['batch_size']}")
    if hyperparams["grad_clip"] != 1.0:
        raise SmallPolicyError(f"grad_clip must be 1.0, got {hyperparams['grad_clip']}")
    if hyperparams["seed"] != 17:
        raise SmallPolicyError(f"seed must be 17, got {hyperparams['seed']}")
    if hyperparams["val_every"] != 100:
        raise SmallPolicyError(f"val_every must be 100, got {hyperparams['val_every']}")
    if hyperparams["max_updates"] > 2000:
        raise SmallPolicyError(f"max_updates must be <= 2000, got {hyperparams['max_updates']}")
    if hyperparams["max_updates"] < 101:
        raise SmallPolicyError(
            f"max_updates must be >= 101 to allow the step-100 resume determinism check, "
            f"got {hyperparams['max_updates']}"
        )

    permit_limits = permit.get("limits", {})
    permit_max_updates = permit_limits.get("max_updates")
    if isinstance(permit_max_updates, int) and hyperparams["max_updates"] > permit_max_updates:
        raise SmallPolicyError(
            f"max_updates {hyperparams['max_updates']} exceeds permit limit {permit_max_updates}"
        )

    return hyperparams


def save_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    step: int,
    normalizer: SmallPolicyNormalizer,
    contract: dict[str, Any],
    indices_state: dict[str, Any],
    rng_state: dict[str, Any],
    metrics: dict[str, Any],
    source_manifest: dict[str, str],
    hyperparams: dict[str, Any],
    best_model_state_dict: dict[str, Any] | None = None,
    best_metric: float | None = None,
    best_checkpoint_path: Path | None = None,
) -> None:
    """Atomic write of a fully resumable checkpoint."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer": normalizer.state_dict(),
        "contract": contract,
        "indices_state": indices_state,
        "rng_state": rng_state,
        "metrics": metrics,
        "source_manifest": source_manifest,
        "hyperparams": hyperparams,
        "best_model_state_dict": best_model_state_dict,
        "best_metric": best_metric,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: str) -> dict[str, Any]:
    """Load a checkpoint and validate contract/normalizer shapes."""
    import torch

    # Always load checkpoints on CPU; model/optimizer states are moved to the
    # target device by their own load_state_dict calls. This keeps CPU RNG byte
    # tensors off CUDA and avoids tensor aliasing during resume checks.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "normalizer",
        "contract",
        "indices_state",
        "rng_state",
        "source_manifest",
    }
    missing = required - ckpt.keys()
    if missing:
        raise SmallPolicyError(f"checkpoint missing keys: {sorted(missing)}")

    norm = SmallPolicyNormalizer()
    norm.load_state_dict(ckpt["normalizer"])
    return ckpt


def export_npz(
    path: Path,
    model: Any,
    normalizer: SmallPolicyNormalizer,
    contract: dict[str, Any],
) -> None:
    """Export a pickle-free NPZ with weights/biases/norm for NumPy inference."""
    import torch

    state = model.state_dict()
    payload: dict[str, np.ndarray] = {}
    layer_map = {"fc1.weight": "W1", "fc1.bias": "b1", "fc2.weight": "W2", "fc2.bias": "b2", "fc3.weight": "W3", "fc3.bias": "b3"}
    for torch_name, np_name in layer_map.items():
        arr = state[torch_name].detach().cpu().numpy()
        payload[np_name] = arr

    # Verify shapes.
    assert payload["W1"].shape == (HIDDEN_DIM, INPUT_DIM), payload["W1"].shape
    assert payload["b1"].shape == (HIDDEN_DIM,)
    assert payload["W2"].shape == (HIDDEN_DIM, HIDDEN_DIM)
    assert payload["b2"].shape == (HIDDEN_DIM,)
    assert payload["W3"].shape == (OUTPUT_DIM, HIDDEN_DIM)
    assert payload["b3"].shape == (OUTPUT_DIM,)

    norm_state = normalizer.state_dict()
    payload["feature_mean"] = norm_state["feature_mean"]
    payload["feature_std"] = norm_state["feature_std"]

    # Non-array metadata stored as UTF-8 bytes so allow_pickle=False is valid.
    payload["contract_json"] = np.asarray(
        bytes(json.dumps(contract, sort_keys=True), "utf-8")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def verify_numpy_torch_match(
    model: Any,
    npz_path: Path,
    raw_features: np.ndarray,
    normalized_features: np.ndarray,
    previous_ctrls: np.ndarray,
) -> dict[str, Any]:
    """Check Torch and NumPy forward outputs agree within 1e-8 rad delta.

    ``normalized_features`` is fed to the Torch model (already normalized once),
    while ``raw_features`` is fed to the NumPy forward, which applies the stored
    normalizer exactly once. This prevents double-normalization mismatches.
    """
    npz = np.load(npz_path, allow_pickle=False)
    torch_deltas = model_forward_torch_batched(model, normalized_features)
    numpy_deltas = forward_numpy(npz, raw_features)
    max_delta = float(np.max(np.abs(torch_deltas - numpy_deltas)))

    # Also check decoded controls.
    torch_ctrl = decode_controls_batched(torch_deltas, previous_ctrls)
    numpy_ctrl = decode_controls_batched(numpy_deltas, previous_ctrls)
    max_ctrl = float(np.max(np.abs(torch_ctrl - numpy_ctrl)))

    return {
        "max_abs_delta_rad": max_delta,
        "max_abs_control_rad": max_ctrl,
        "within_tolerance": max_delta <= 1e-8,
    }


def compute_zero_delta_baseline(
    labels: np.ndarray, previous_ctrls: np.ndarray
) -> dict[str, Any]:
    """Hold-zero-delta baseline: controls stay at previous_ctrl."""
    pred_ctrl = np.repeat(previous_ctrls[:, None, :], HORIZON, axis=1)
    target_ctrl = previous_ctrls[:, None, :] + np.cumsum(labels, axis=1)
    rmse = float(np.sqrt(np.mean((pred_ctrl - target_ctrl) ** 2)))
    return {"rmse_rad": rmse, "description": "controls held at previous_ctrl"}


def _state_dict_to_numpy(obj: Any) -> Any:
    """Recursively convert tensors in a state dict to NumPy arrays."""
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().copy()
    if isinstance(obj, dict):
        return {k: _state_dict_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_state_dict_to_numpy(v) for v in obj]
    return obj


def _max_state_diff(state_a: Any, state_b: Any) -> float:
    """Max absolute difference between two (possibly nested) state structures."""
    if isinstance(state_a, np.ndarray) and isinstance(state_b, np.ndarray):
        if state_a.shape != state_b.shape:
            return float("inf")
        return float(np.max(np.abs(state_a - state_b)))
    if isinstance(state_a, dict) and isinstance(state_b, dict):
        if state_a.keys() != state_b.keys():
            return float("inf")
        return max((_max_state_diff(state_a[k], state_b[k]) for k in state_a), default=0.0)
    if isinstance(state_a, (list, tuple)) and isinstance(state_b, (list, tuple)):
        if len(state_a) != len(state_b):
            return float("inf")
        return max((_max_state_diff(a, b) for a, b in zip(state_a, state_b)), default=0.0)
    return 0.0 if state_a == state_b else float("inf")


class _BatchSampler:
    """Pure NumPy sampler shared by the main loop and the restored branch.

    Uses the global ``np.random`` stream so that saving/restoring the NumPy RNG
    state (via ``get_rng_state``/``set_rng_state``) reproduces the same shuffle.
    The first epoch starts with the identity permutation; when the next batch
    would run past the end of the dataset the order is shuffled and the position
    rolls over to zero.
    """

    def __init__(self, n: int, batch_size: int) -> None:
        if not isinstance(n, int) or n <= 0:
            raise SmallPolicyError(f"sampler n must be a positive integer, got {n}")
        if not isinstance(batch_size, int) or batch_size <= 0 or batch_size > n:
            raise SmallPolicyError(
                f"sampler batch_size must be in [1, n], got {batch_size} (n={n})"
            )
        self.n = n
        self.batch_size = batch_size
        self.epoch = 0
        self.order = np.arange(n)
        self.position = 0

    def next_batch(self) -> np.ndarray:
        """Return the next batch indices and advance the cursor."""
        if self.position + self.batch_size > self.n:
            np.random.shuffle(self.order)
            self.position = 0
            self.epoch += 1
        batch = self.order[self.position : self.position + self.batch_size].copy()
        self.position += self.batch_size
        return batch

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "order": self.order.copy(),
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sampler state and validate shapes/bounds."""
        epoch = int(state["epoch"])
        position = int(state["position"])
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != (self.n,):
            raise SmallPolicyError(
                f"sampler order shape mismatch: got {order.shape}, want ({self.n},)"
            )
        if not set(order.tolist()) == set(range(self.n)):
            raise SmallPolicyError("sampler order is not a permutation of 0..n-1")
        if position < 0 or position > self.n:
            raise SmallPolicyError(f"sampler position out of bounds: {position}")
        if position % self.batch_size != 0:
            raise SmallPolicyError(
                f"sampler position {position} is not on a batch boundary"
            )
        self.epoch = epoch
        self.position = position
        self.order = order.copy()


def _run_restored_step(
    checkpoint_path: Path,
    features_train: np.ndarray,
    labels_train: np.ndarray,
    previous_ctrls_train: np.ndarray,
    device: str,
    grad_clip: float,
    current_source_manifest: dict[str, str],
    current_contract: dict[str, Any],
    current_hyperparams: dict[str, Any],
    main_rng_state: dict[str, Any],
) -> dict[str, Any]:
    """Load a checkpoint from disk and run exactly one restored update.

    This branch is independent of the live model/optimizer: it builds a fresh
    model and AdamW optimizer, loads saved states with strict validation, and
    restores the checkpoint's RNG/order state. The main RNG is preserved and
    restored after the branch so the training loop is not advanced.
    """
    import torch

    if not checkpoint_path.is_file():
        raise SmallPolicyError(f"resume determinism checkpoint not found: {checkpoint_path}")

    # Load checkpoint onto CPU to avoid CUDA tensor aliasing and to keep the
    # branch independent from live GPU tensors.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "normalizer",
        "contract",
        "indices_state",
        "rng_state",
        "source_manifest",
        "hyperparams",
    }
    missing = required - ckpt.keys()
    if missing:
        raise SmallPolicyError(f"resume checkpoint missing keys: {sorted(missing)}")

    # Strict validation: the checkpoint must match the current source identity,
    # contract, and frozen hyperparameters.
    ckpt_manifest = ckpt["source_manifest"]
    if not isinstance(ckpt_manifest, dict) or set(ckpt_manifest.keys()) != REQUIRED_SOURCE_FILES:
        raise SmallPolicyError("resume checkpoint source manifest has wrong file set")
    for rel_path in REQUIRED_SOURCE_FILES:
        if ckpt_manifest.get(rel_path) != current_source_manifest.get(rel_path):
            raise SmallPolicyError(
                f"resume checkpoint source manifest mismatch for {rel_path}"
            )

    ckpt_contract = ckpt["contract"]
    if not isinstance(ckpt_contract, dict):
        raise SmallPolicyError("resume checkpoint contract is not a dict")
    for key in ("source_commit", "dataset_schema", "dataset_index_sha256", "split_counts_observed"):
        if ckpt_contract.get(key) != current_contract.get(key):
            raise SmallPolicyError(f"resume checkpoint contract mismatch on {key}")

    ckpt_hyperparams = ckpt["hyperparams"]
    if not isinstance(ckpt_hyperparams, dict):
        raise SmallPolicyError("resume checkpoint hyperparams is not a dict")
    for key in ("lr", "weight_decay", "batch_size", "grad_clip", "seed", "val_every"):
        if ckpt_hyperparams.get(key) != current_hyperparams.get(key):
            raise SmallPolicyError(f"resume checkpoint hyperparams mismatch on {key}")

    norm = SmallPolicyNormalizer()
    norm.load_state_dict(ckpt["normalizer"])

    branch_model = build_model(device)
    branch_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    branch_optimizer = torch.optim.AdamW(
        branch_model.parameters(),
        lr=current_hyperparams["lr"],
        weight_decay=current_hyperparams["weight_decay"],
    )
    branch_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    branch_model.train()

    sampler = _BatchSampler(features_train.shape[0], current_hyperparams["batch_size"])
    sampler.load_state_dict(ckpt["indices_state"])

    # Preserve the main RNG, switch to the checkpoint's RNG for the branch, then
    # restore the main RNG so the next main-loop step is not mutated.
    saved_main_rng = main_rng_state
    set_rng_state(ckpt["rng_state"])

    try:
        batch_idx = sampler.next_batch()

        x = torch.from_numpy(features_train[batch_idx]).float().to(device)
        y = torch.from_numpy(labels_train[batch_idx]).float().to(device)
        prev = torch.from_numpy(previous_ctrls_train[batch_idx]).float().to(device)

        branch_optimizer.zero_grad()
        logits = branch_model(x)
        pred_deltas = ACTION_CAP_RAD * torch.tanh(logits.view(-1, HORIZON, NUM_CHANNELS))
        loss, metrics, _, _, _ = compute_loss_torch(logits, y, prev)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(branch_model.parameters(), grad_clip)
        branch_optimizer.step()

        return {
            "loss": loss.item(),
            "metrics": metrics,
            "logits": logits.detach().cpu().numpy().copy(),
            "deltas": pred_deltas.detach().cpu().numpy().copy(),
            "model_state_dict": _state_dict_to_numpy(branch_model.state_dict()),
            "optimizer_state_dict": _state_dict_to_numpy(branch_optimizer.state_dict()),
            "batch_idx": batch_idx.copy(),
        }
    finally:
        set_rng_state(saved_main_rng)


def _directory_size(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total


def _output_size_bytes(checkpoint_dir: Path, report_path: Path) -> int:
    total = _directory_size(checkpoint_dir)
    if report_path.exists():
        total += report_path.stat().st_size
    return total


def _save_latest_checkpoint(
    path: Path,
    model: Any,
    optimizer: Any,
    step: int,
    normalizer: SmallPolicyNormalizer,
    contract: dict[str, Any],
    sampler: _BatchSampler,
    source_manifest: dict[str, str],
    hyperparams: dict[str, Any],
    best_model_state_dict: dict[str, Any] | None,
    best_metric: float | None,
    best_checkpoint_path: Path | None,
) -> None:
    """Persist the latest resumable state (overwrites previous latest)."""
    save_checkpoint(
        path,
        model,
        optimizer,
        step,
        normalizer,
        contract,
        sampler.state_dict(),
        get_rng_state(),
        {"best_val_rmse": best_metric if best_metric is not None else float("inf")},
        source_manifest,
        hyperparams,
        best_model_state_dict,
        best_metric,
        best_checkpoint_path,
    )


def _validate_resume_checkpoint(
    ckpt: dict[str, Any],
    source_manifest: dict[str, str],
    contract: dict[str, Any],
    hyperparams: dict[str, Any],
    raw_features_train: np.ndarray,
) -> None:
    """Validate that a checkpoint is safe to resume from.

    Checks source manifest, contract, hyperparameters, normalizer agreement,
    and step bounds. Raises SmallPolicyError on any mismatch.
    """
    ckpt_manifest = ckpt.get("source_manifest")
    if not isinstance(ckpt_manifest, dict) or set(ckpt_manifest.keys()) != REQUIRED_SOURCE_FILES:
        raise SmallPolicyError("resume checkpoint source manifest has wrong file set")
    for rel_path in REQUIRED_SOURCE_FILES:
        if ckpt_manifest.get(rel_path) != source_manifest.get(rel_path):
            raise SmallPolicyError(
                f"resume checkpoint source manifest mismatch for {rel_path}"
            )

    ckpt_contract = ckpt.get("contract")
    if not isinstance(ckpt_contract, dict):
        raise SmallPolicyError("resume checkpoint contract is not a dict")
    for key in ("source_commit", "dataset_schema", "dataset_index_sha256", "split_counts_observed"):
        if ckpt_contract.get(key) != contract.get(key):
            raise SmallPolicyError(f"resume checkpoint contract mismatch on {key}")

    ckpt_hyperparams = ckpt.get("hyperparams")
    if not isinstance(ckpt_hyperparams, dict):
        raise SmallPolicyError("resume checkpoint hyperparams is not a dict")
    for key in ("lr", "weight_decay", "batch_size", "grad_clip", "seed", "val_every"):
        if ckpt_hyperparams.get(key) != hyperparams.get(key):
            raise SmallPolicyError(f"resume checkpoint hyperparams mismatch on {key}")

    ckpt_norm = SmallPolicyNormalizer()
    ckpt_norm.load_state_dict(ckpt["normalizer"])
    recomputed_norm = SmallPolicyNormalizer().fit(raw_features_train)
    if not np.allclose(ckpt_norm.state_dict()["feature_mean"], recomputed_norm.state_dict()["feature_mean"]):
        raise SmallPolicyError(
            "resume checkpoint normalizer mean does not match recomputed train-only normalizer"
        )
    if not np.allclose(ckpt_norm.state_dict()["feature_std"], recomputed_norm.state_dict()["feature_std"]):
        raise SmallPolicyError(
            "resume checkpoint normalizer std does not match recomputed train-only normalizer"
        )

    start_step = int(ckpt["step"])
    if not (0 <= start_step < hyperparams["max_updates"]):
        raise SmallPolicyError(f"resume checkpoint step {start_step} is out of bounds")


def train(args: argparse.Namespace) -> int:
    """Run the small-policy training loop under SLURM/CUDA."""
    import torch

    slurm_id, device_name = _require_gpu_and_slurm()

    permit_path = Path(args.permit)
    with open(permit_path, "r", encoding="utf-8") as f:
        permit = json.load(f)

    source_manifest_path = Path(args.source_manifest)
    with open(source_manifest_path, "r", encoding="utf-8") as f:
        source_manifest = json.load(f)
    validate_source_manifest(source_manifest, permit, REPO_ROOT, _get_git_commit())
    source_commit = source_manifest["source_commit"]

    hyperparams = validate_frozen_hyperparams(args, permit)
    if hyperparams["max_updates"] < 101:
        raise SmallPolicyError(
            f"max_updates must be >= 101 to allow the step-100 resume determinism check, "
            f"got {hyperparams['max_updates']}"
        )

    sample_index_path = Path(args.sample_index) if args.sample_index else None
    contract = validate_permit(
        permit,
        sample_index_path=sample_index_path,
        source_commit=source_commit,
        dataset_schema=args.dataset_schema,
    ).to_dict()

    if sample_index_path is None:
        raise SmallPolicyError("--sample-index is required to bind the permit to data")
    if not sample_index_path.exists():
        raise SmallPolicyError(f"sample index not found: {sample_index_path}")

    dataset_sha = _sha256_bytes(sample_index_path.read_bytes())
    records = load_sample_index(sample_index_path)
    split_counts = {k: len(v) for k, v in split_by_target_class(records).items()}

    expected_counts = contract.get("split_counts") or contract.get("expected_split_counts", {})
    for split, expected in expected_counts.items():
        if split_counts.get(split) != expected:
            raise SmallPolicyError(
                f"observed {split} count {split_counts.get(split)} does not match permit {expected}"
            )

    contract["dataset_index_sha256"] = dataset_sha
    contract["split_counts_observed"] = split_counts

    # Canonical source identity is always the current disk state, never a
    # checkpoint's stale manifest.
    source_manifest = _build_source_manifest()

    device = torch.device("cuda:0")
    torch_dtype = torch.float32

    # Prefer deterministic FP32 behavior; TF32 is disabled for the parity check.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None:
        warnings.warn(
            "CUBLAS_WORKSPACE_CONFIG is not set; set it to ':4096:8' or ':16:8' for fully deterministic cuBLAS."
        )

    seed = hyperparams["seed"]
    rng_state = set_seed(seed)

    raw_features, features, labels, normalizer = prepare_datasets(records)

    # Previous controls for cumulative reconstruction; fitted only on train for norm.
    previous_ctrls: dict[str, np.ndarray] = {}
    splits = split_by_target_class(records)
    for split_name in features:
        prev_list = []
        for rec in splits[split_name]:
            prev_list.append(np.asarray(rec["actor_input"]["previous_ctrl_rad16"], dtype=np.float32))
        previous_ctrls[split_name] = np.stack(prev_list) if prev_list else np.empty((0, NUM_CHANNELS), dtype=np.float32)

    zero_baseline = compute_zero_delta_baseline(labels["train"], previous_ctrls["train"])

    model = build_model(str(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
    )

    start_step = 0
    best_val_rmse = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else DEFAULT_REPORT_PATH

    best_model_state_dict: dict[str, Any] | None = None
    best_metric: float | None = None
    best_checkpoint_path: Path | None = None

    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), str(device))

        # Strict identity validation: do not trust a checkpoint from a different
        # source commit, contract, or hyperparameter set.
        _validate_resume_checkpoint(ckpt, source_manifest, contract, hyperparams, raw_features["train"])

        start_step = int(ckpt["step"])

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        normalizer.load_state_dict(ckpt["normalizer"])
        set_rng_state(ckpt["rng_state"])

        # Restore prior best validation state; never drop it just because no new
        # validation run has happened yet.
        best_metric = ckpt.get("best_metric")
        best_val_rmse = float(best_metric) if best_metric is not None else float("inf")
        best_model_state_dict = ckpt.get("best_model_state_dict")
        loaded_best_path = ckpt.get("best_checkpoint_path")
        if isinstance(loaded_best_path, str) and loaded_best_path:
            best_checkpoint_path = Path(loaded_best_path)

    batch_size = hyperparams["batch_size"]
    max_updates = min(hyperparams["max_updates"], contract["limits"]["max_updates"])
    grad_clip = hyperparams["grad_clip"]
    val_every = hyperparams["val_every"]
    N_train = features["train"].shape[0]

    if N_train == 0:
        raise SmallPolicyError("training split is empty")

    sampler = _BatchSampler(N_train, batch_size)
    if args.resume:
        sampler.load_state_dict(ckpt["indices_state"])

    report = {
        "contract": contract,
        "source_manifest": source_manifest,
        "device": str(device),
        "device_name": device_name,
        "slurm_job_id": slurm_id,
        "torch_dtype": str(torch_dtype),
        "seed": seed,
        "source_commit": source_commit,
        "dataset_index_sha256": dataset_sha,
        "split_counts": split_counts,
        "zero_delta_baseline": zero_baseline,
        "training_started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "weight_hash_before": model_hash(model),
        "updates": [],
    }

    model.train()
    step = start_step

    pending_resume_check: dict[str, Any] | None = None
    comparison_updates = 0
    train_start = time.monotonic()
    max_wall_s = contract["limits"]["max_wall_s"]
    failed = False
    failed_progress_checkpoint: Path | None = None

    while step < max_updates:
        record_appended = False
        elapsed = time.monotonic() - train_start
        if elapsed + MIN_REMAINING_S > max_wall_s:
            report["stopped_by_wall"] = True
            report["failure_reason"] = "wall clock budget exhausted before completion"
            report["failed_at_step"] = step
            failed = True
            failed_progress_checkpoint = checkpoint_dir / "latest.pt"
            _save_latest_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                step,
                normalizer,
                contract,
                sampler,
                source_manifest,
                hyperparams,
                best_model_state_dict,
                best_metric,
                best_checkpoint_path,
            )
            break

        batch_idx = sampler.next_batch()

        x = torch.from_numpy(features["train"][batch_idx]).float().to(device)
        y = torch.from_numpy(labels["train"][batch_idx]).float().to(device)
        prev = torch.from_numpy(previous_ctrls["train"][batch_idx]).float().to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss, train_metrics, _, _, _ = compute_loss_torch(logits, y, prev)

        # Reject non-finite loss before any update.
        if not torch.isfinite(loss):
            failed_path = checkpoint_dir / f"failed_loss_step_{step + 1:06d}.pt"
            save_checkpoint(
                failed_path,
                model,
                optimizer,
                step,
                normalizer,
                contract,
                sampler.state_dict(),
                get_rng_state(),
                {"best_val_rmse": best_val_rmse, "failed": True},
                source_manifest,
                hyperparams,
                best_model_state_dict,
                best_metric,
                best_checkpoint_path,
            )
            report["failed_at_step"] = step + 1
            report["failed_progress_checkpoint"] = str(failed_path)
            report["failure_reason"] = "non-finite loss before update"
            failed = True
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Reject non-finite gradients.
        grads_finite = all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        )
        if not grads_finite:
            failed_path = checkpoint_dir / f"failed_grad_step_{step + 1:06d}.pt"
            save_checkpoint(
                failed_path,
                model,
                optimizer,
                step,
                normalizer,
                contract,
                sampler.state_dict(),
                get_rng_state(),
                {"best_val_rmse": best_val_rmse, "failed": True},
                source_manifest,
                hyperparams,
                best_model_state_dict,
                best_metric,
                best_checkpoint_path,
            )
            report["failed_at_step"] = step + 1
            report["failed_progress_checkpoint"] = str(failed_path)
            report["failure_reason"] = "non-finite gradient before update"
            failed = True
            break

        optimizer.step()
        step += 1

        # Reject non-finite weights after update.
        weights_finite = all(torch.isfinite(p).all().item() for p in model.parameters())
        if not weights_finite:
            failed_path = checkpoint_dir / f"failed_weights_step_{step:06d}.pt"
            save_checkpoint(
                failed_path,
                model,
                optimizer,
                step,
                normalizer,
                contract,
                sampler.state_dict(),
                get_rng_state(),
                {"best_val_rmse": best_val_rmse, "failed": True},
                source_manifest,
                hyperparams,
                best_model_state_dict,
                best_metric,
                best_checkpoint_path,
            )
            report["failed_at_step"] = step
            report["failed_progress_checkpoint"] = str(failed_path)
            report["failure_reason"] = "non-finite weights after update"
            failed = True
            break

        record = {
            "step": step,
            "train_loss": train_metrics["loss"],
            "train_increment_mse": train_metrics["increment_mse_normalized"],
            "train_cumulative_mse": train_metrics["cumulative_mse_normalized"],
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0,
        }

        is_validation_step = (step % val_every == 0) or (step == max_updates)
        if is_validation_step:
            val_metrics = evaluate_split(
                model, features["validation"], labels["validation"], previous_ctrls["validation"], str(device)
            )
            record["val_loss"] = val_metrics["loss"]
            record["val_increment_mse"] = val_metrics["increment_mse_normalized"]
            record["val_cumulative_mse"] = val_metrics["cumulative_mse_normalized"]
            record["val_rmse_rad"] = val_metrics["control_rmse_rad"]

            # Select the checkpoint with the lowest validation cumulative block RMSE.
            if val_metrics["control_rmse_rad"] < best_val_rmse:
                best_val_rmse = val_metrics["control_rmse_rad"]
                best_path = checkpoint_dir / f"best_step_{step:06d}.pt"
                best_model_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metric = best_val_rmse
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    step,
                    normalizer,
                    contract,
                    sampler.state_dict(),
                    get_rng_state(),
                    {"best_val_rmse": best_val_rmse},
                    source_manifest,
                    hyperparams,
                    best_model_state_dict,
                    best_metric,
                    best_path,
                )
                best_checkpoint_path = best_path
                record["new_best"] = True

            # Always persist a resumable latest checkpoint at validation intervals.
            _save_latest_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                step,
                normalizer,
                contract,
                sampler,
                source_manifest,
                hyperparams,
                best_model_state_dict,
                best_metric,
                best_checkpoint_path,
            )

            # Resume determinism check at step 100 (only if a next step will exist).
            if step == 100 and step < max_updates:
                check_path = checkpoint_dir / "resume_determinism_check.pt"
                saved_rng = get_rng_state()
                save_checkpoint(
                    check_path,
                    model,
                    optimizer,
                    step,
                    normalizer,
                    contract,
                    sampler.state_dict(),
                    saved_rng,
                    {"best_val_rmse": best_val_rmse},
                    source_manifest,
                    hyperparams,
                    best_model_state_dict,
                    best_metric,
                    best_checkpoint_path,
                )
                # Persist a progress receipt before running the expensive restored branch.
                record["resume_check_checkpoint"] = str(check_path)
                report["updates"].append(record)
                report["actual_updates"] = step - start_step
                report["comparison_updates"] = comparison_updates
                _atomic_write_json(report_path, report)
                record_appended = True
                # Run a restored branch for one update without consuming the main RNG stream.
                expected = _run_restored_step(
                    check_path,
                    features["train"],
                    labels["train"],
                    previous_ctrls["train"],
                    str(device),
                    grad_clip,
                    source_manifest,
                    contract,
                    hyperparams,
                    saved_rng,
                )
                comparison_updates += 1
                pending_resume_check = {
                    "expected": expected,
                    "checked_at_step": step,
                    "checkpoint_path": str(check_path),
                }
                # Restore RNG so the main loop's next step sees exactly the saved state.
                set_rng_state(saved_rng)

        # Compare the uninterrupted next step against the restored branch.
        if pending_resume_check and step == pending_resume_check["checked_at_step"] + 1:
            expected = pending_resume_check["expected"]
            actual_model_np = _state_dict_to_numpy(model.state_dict())
            actual_opt_np = _state_dict_to_numpy(optimizer.state_dict())
            actual_logits_np = logits.detach().cpu().numpy().copy()
            actual_deltas_np = (
                ACTION_CAP_RAD * torch.tanh(logits.view(-1, HORIZON, NUM_CHANNELS))
            ).detach().cpu().numpy().copy()
            model_diff = _max_state_diff(actual_model_np, expected["model_state_dict"])
            opt_diff = _max_state_diff(actual_opt_np, expected["optimizer_state_dict"])
            logits_diff = _max_state_diff(actual_logits_np, expected["logits"])
            deltas_diff = _max_state_diff(actual_deltas_np, expected["deltas"])
            loss_diff = abs(record["train_loss"] - expected["loss"])
            batch_idx_equal = np.array_equal(batch_idx, expected["batch_idx"])
            record["resume_determinism"] = {
                "model_max_diff": model_diff,
                "optimizer_max_diff": opt_diff,
                "logits_max_diff": logits_diff,
                "deltas_max_diff": deltas_diff,
                "loss_diff": loss_diff,
                "batch_idx_equal": bool(batch_idx_equal),
                "tolerance": RESUME_CHECK_TOLERANCE,
                "passed": model_diff <= RESUME_CHECK_TOLERANCE
                and opt_diff <= RESUME_CHECK_TOLERANCE
                and logits_diff <= RESUME_CHECK_TOLERANCE
                and deltas_diff <= RESUME_CHECK_TOLERANCE
                and loss_diff <= RESUME_CHECK_TOLERANCE
                and batch_idx_equal,
            }
            if not record["resume_determinism"]["passed"]:
                failed_path = checkpoint_dir / f"failed_resume_determinism_step_{step:06d}.pt"
                save_checkpoint(
                    failed_path,
                    model,
                    optimizer,
                    step,
                    normalizer,
                    contract,
                    sampler.state_dict(),
                    get_rng_state(),
                    {"best_val_rmse": best_val_rmse, "failed": True},
                    source_manifest,
                    hyperparams,
                    best_model_state_dict,
                    best_metric,
                    best_checkpoint_path,
                )
                report["failed_at_step"] = step
                report["failed_progress_checkpoint"] = str(failed_path)
                report["failure_reason"] = "resume determinism check failed"
                report["resume_determinism_failure"] = record["resume_determinism"]
                failed = True
                break
            pending_resume_check = None

        if not record_appended:
            report["updates"].append(record)

        # Write an atomic progress receipt at every validation checkpoint.
        if is_validation_step:
            _atomic_write_json(report_path, report)

        # Output budget guard.
        if _output_size_bytes(checkpoint_dir, report_path) > OUTPUT_BUDGET_BYTES:
            report["stopped_by_output_budget"] = True
            report["failure_reason"] = "output size budget exhausted before completion"
            report["failed_at_step"] = step
            failed = True
            failed_progress_checkpoint = checkpoint_dir / "latest.pt"
            _save_latest_checkpoint(
                checkpoint_dir / "latest.pt",
                model,
                optimizer,
                step,
                normalizer,
                contract,
                sampler,
                source_manifest,
                hyperparams,
                best_model_state_dict,
                best_metric,
                best_checkpoint_path,
            )
            break

    report["actual_updates"] = step - start_step
    report["comparison_updates"] = comparison_updates
    report["weight_hash_after"] = model_hash(model)

    # Final NumPy export / Torch parity check and held-out test (only when the
    # run did not fail earlier and a best model is available).
    if not failed and best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        optimizer.zero_grad(set_to_none=True)

        npz_path = Path(args.export_npz) if args.export_npz else checkpoint_dir / "small_policy.npz"
        export_npz(npz_path, model, normalizer, contract)
        report["npz_export_path"] = str(npz_path)

        all_raw_features = np.concatenate([raw_features["train"], raw_features["validation"]], axis=0)
        all_features = np.concatenate([features["train"], features["validation"]], axis=0)
        all_prev = np.concatenate([previous_ctrls["train"], previous_ctrls["validation"]], axis=0)
        consistency = verify_numpy_torch_match(model, npz_path, all_raw_features, all_features, all_prev)
        report["torch_numpy_consistency"] = consistency

        if not consistency["within_tolerance"]:
            report["failed_at_step"] = step
            report["failed_progress_checkpoint"] = str(npz_path)
            report["failure_reason"] = f"Torch/NumPy parity failed: max delta {consistency['max_abs_delta_rad']} rad > 1e-8"
            failed = True
        elif comparison_updates == 0:
            report["failure_reason"] = "resume determinism disk check not completed before held-out test"
            report["failed_progress_checkpoint"] = str(checkpoint_dir / "latest.pt")
            failed = True
        else:
            test_metrics = evaluate_split_with_channels(
                model, features["test"], labels["test"], previous_ctrls["test"], str(device)
            )
            report["test_metrics"] = test_metrics

            # Per-target-class test breakdown.
            per_class: dict[str, Any] = {}
            test_indices = {tc: [i for i, r in enumerate(splits["test"]) if r["target_class"] == tc] for tc in TEST_TARGET_CLASSES}
            for tc, idx in test_indices.items():
                if idx:
                    per_class[str(tc)] = evaluate_split_with_channels(
                        model,
                        features["test"][idx],
                        labels["test"][idx],
                        previous_ctrls["test"][idx],
                        str(device),
                    )
            report["test_per_class"] = per_class
            report["best_val_rmse_rad"] = best_val_rmse
            report["best_checkpoint_path"] = str(best_checkpoint_path) if best_checkpoint_path is not None else None

    report["optimization_started"] = report["actual_updates"] > 0
    report["training_finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_write_json(report_path, report)

    if failed:
        print(
            f"Training failed: {report['failure_reason']} at step {report.get('failed_at_step', step)}",
            file=sys.stderr,
        )
        return 1

    print(f"Training finished: {step} updates, best val RMSE {best_val_rmse:.6f} rad")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LEAP v1 small-policy baseline")
    parser.add_argument("--permit", required=True, help="Path to approved optimization permit JSON")
    parser.add_argument("--source-manifest", required=True, help="Path to source manifest JSON with source_commit and file hashes")
    parser.add_argument("--sample-index", required=True, help="Path to dataset sample index JSONL")
    parser.add_argument("--dataset-schema", default="v1-small-policy-smoke-v1", help="Dataset schema identifier")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    parser.add_argument("--export-npz", default=None, help="Path for NumPy export")
    parser.add_argument("--report", default=None, help="Path for JSON report")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-updates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=100)
    args = parser.parse_args(argv)

    try:
        return train(args)
    except SmallPolicyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        report_path = Path(args.report) if args.report else DEFAULT_REPORT_PATH
        # Do not overwrite a partial progress report already written by train().
        if report_path.exists() and report_path.stat().st_size > 0:
            try:
                existing = json.loads(report_path.read_text(encoding="utf-8"))
                if "actual_updates" in existing:
                    return 1
            except Exception:
                pass
        report_path.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "error": str(exc),
            "training_finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "optimization_started": False,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(failure, f, indent=2, default=str)
        return 1


if __name__ == "__main__":
    sys.exit(main())
