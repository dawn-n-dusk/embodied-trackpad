"""LEAP v1 Full small-policy baseline helpers (NumPy only).

This module contains the deterministic feature extraction, normalization,
forward pass, and action decoding for the first v1 small-policy learning
baseline. It deliberately imports neither PyTorch nor MuJoCo so that the
inference path can run on the NumPy-only CPU runtime.

The 16-channel motor order is copied from ``model.py`` (MOTOR_JOINTS) to avoid
the MuJoCo import; it must stay byte-for-byte identical to that list.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1 import pilot_dataset

REPO_ROOT = Path(__file__).resolve().parents[3]

# Frozen constants from docs/tasks/v1-small-policy-smoke.zh-CN.md
INPUT_FEATURE_NAMES = ("error_xy_px2", "q_rad16", "qvel_rad_s16", "previous_ctrl_rad16")
APPROVED_ACTOR_FIELDS = (
    "error_xy_px",
    "q_rad16",
    "qvel_rad_s16",
    "previous_ctrl_rad16",
)
# Non-numeric metadata keys permitted in actor_input but never concatenated.
ALLOWED_NON_FEATURE_ACTOR_KEYS = frozenset(
    {
        "png_ref",
        "png_sha256",
        "frame_monotonic_time_s",
        "valid",
        "fsr_N4",
        "motor_current_A16",
        "wrist",
    }
)
# Explicitly rejected actor_input keys that would leak privileged or metadata.
REJECTED_ACTOR_INPUT_KEYS = frozenset(
    {
        "force",
        "jacobian",
        "reward",
        "outcome",
        "seed",
        "class",
        "split",
        "target_class",
        "episode_id",
        "frame_id",
        "oracle",
        "cursor_truth",
        "gui_state",
    }
)
INPUT_DIM = 50
HIDDEN_DIM = 128
HORIZON = 25
NUM_CHANNELS = 16
OUTPUT_DIM = HORIZON * NUM_CHANNELS  # 400

# Action / block caps
ACTION_CAP_RAD = 2e-4
BLOCK_CAP_RAD = 0.005
DT_S = 0.002

# Training split by target class
TRAIN_TARGET_CLASSES = frozenset({0, 2, 3, 5, 6, 8, 9, 11})
VAL_TARGET_CLASSES = frozenset({1, 7})
TEST_TARGET_CLASSES = frozenset({4, 10})

# Scoped (near_diagonal_v1) identity keys; legacy records must not carry them.
SCOPED_IDENTITY_KEYS = ("collection_id", "target_key", "episode_key")

# 16-channel order copied from model.py::MOTOR_JOINTS (no MuJoCo import).
FINGERS = ("index", "middle", "ring", "thumb")
FINGER_JOINTS = {
    "index": ["if_mcp", "if_rot", "if_pip", "if_dip"],
    "middle": ["mf_mcp", "mf_rot", "mf_pip", "mf_dip"],
    "ring": ["rf_mcp", "rf_rot", "rf_pip", "rf_dip"],
    "thumb": ["th_cmc", "th_axl", "th_mcp", "th_ipl"],
}
MOTOR_JOINTS = [j for f in FINGERS for j in FINGER_JOINTS[f]]

# The four index-finger joints used for coarse motion reporting.
MOTION4_INDICES = [0, 1, 2, 3]

REQUIRED_SAMPLE_KEYS = {"episode_id", "frame_id", "split", "target_class", "actor_input", "labels", "action_window"}
REQUIRED_ACTOR_INPUT_KEYS = {"error_xy_px", "q_rad16", "qvel_rad_s16", "previous_ctrl_rad16"}
REQUIRED_LABEL_KEYS = {"executed_ctrl_rad"}
REQUIRED_ACTION_WINDOW_KEYS = {"dt_s", "first_step_index", "end_exclusive_step_index"}

STD_FLOOR = 1e-3


class SmallPolicyError(ValueError):
    """Raised when a sample, permit, or contract check fails."""


@dataclass(frozen=True)
class SmallPolicyContract:
    """In-memory contract binding an approved dataset index to this baseline."""

    status: str
    allow_optimizer: bool
    sample_index_sha256: str
    source_commit: str
    dataset_schema: str
    split_counts: dict[str, int]
    limits: dict[str, Any]
    approved_actor_fields: tuple[str, ...]
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "allow_optimizer": self.allow_optimizer,
            "sample_index_sha256": self.sample_index_sha256,
            "source_commit": self.source_commit,
            "dataset_schema": self.dataset_schema,
            "split_counts": dict(self.split_counts),
            "limits": dict(self.limits),
            "approved_actor_fields": list(self.approved_actor_fields),
            "purpose": self.purpose,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SmallPolicyContract":
        return cls(
            status=str(data["status"]),
            allow_optimizer=bool(data["allow_optimizer"]),
            sample_index_sha256=str(data["sample_index_sha256"]),
            source_commit=str(data["source_commit"]),
            dataset_schema=str(data["dataset_schema"]),
            split_counts=dict(data["split_counts"]),
            limits=dict(data["limits"]),
            approved_actor_fields=tuple(str(x) for x in data["approved_actor_fields"]),
            purpose=str(data.get("purpose", "small_policy_smoke")),
        )


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_array(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Convert ``value`` to a finite float32 ndarray with the requested shape."""
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise SmallPolicyError(f"{name} is not convertible to float32 ndarray: {exc}") from exc
    if not np.isfinite(arr).all():
        raise SmallPolicyError(f"{name} contains non-finite values")
    if shape is not None and arr.shape != shape:
        raise SmallPolicyError(f"{name} shape mismatch: got {arr.shape}, want {shape}")
    return arr


def extract_features(sample: dict[str, Any]) -> np.ndarray:
    """Return the 50-dim actor-feature vector for ``sample``.

    Only ``error_xy_px``, ``q_rad16``, ``qvel_rad_s16`` and
    ``previous_ctrl_rad16`` are concatenated. Known non-feature metadata keys
    (PNG references, masks, clocks) are permitted but ignored. Force/Jacobian,
    oracle, reward/outcome, seed/class/split metadata in actor_input are
    explicitly rejected.
    """
    if not isinstance(sample, dict):
        raise SmallPolicyError("sample must be a dict")
    missing = REQUIRED_SAMPLE_KEYS - sample.keys()
    if missing:
        raise SmallPolicyError(f"sample missing keys: {sorted(missing)}")

    actor = sample["actor_input"]
    if not isinstance(actor, dict):
        raise SmallPolicyError("actor_input must be a dict")

    rejected = set(actor.keys()) & REJECTED_ACTOR_INPUT_KEYS
    if rejected:
        raise SmallPolicyError(f"actor_input contains rejected fields: {sorted(rejected)}")

    allowed_only = set(actor.keys()) - set(APPROVED_ACTOR_FIELDS) - ALLOWED_NON_FEATURE_ACTOR_KEYS
    if allowed_only:
        raise SmallPolicyError(f"actor_input contains unknown fields: {sorted(allowed_only)}")

    if actor.get("wrist") is not None:
        raise SmallPolicyError("actor_input['wrist'] must be null when present; image features are not actor inputs")

    missing_actor = set(APPROVED_ACTOR_FIELDS) - actor.keys()
    if missing_actor:
        raise SmallPolicyError(f"actor_input missing required fields: {sorted(missing_actor)}")

    error_xy = _as_array(actor["error_xy_px"], "error_xy_px", (2,))
    q = _as_array(actor["q_rad16"], "q_rad16", (NUM_CHANNELS,))
    qvel = _as_array(actor["qvel_rad_s16"], "qvel_rad_s16", (NUM_CHANNELS,))
    prev = _as_array(actor["previous_ctrl_rad16"], "previous_ctrl_rad16", (NUM_CHANNELS,))

    features = np.concatenate([error_xy, q, qvel, prev]).astype(np.float32)
    if features.shape != (INPUT_DIM,):
        raise SmallPolicyError(f"feature shape {features.shape} != ({INPUT_DIM},)")
    return features


def extract_label_delta(sample: dict[str, Any]) -> np.ndarray:
    """Return the 25x16 per-substep control delta label.

    Per-step increments are always derived from ``executed_ctrl_rad``:
      delta[0] = executed[0] - previous_ctrl
      delta[t] = executed[t] - executed[t-1]   for t > 0

    If the sample also contains ``delta_from_frame_previous_ctrl_rad`` (the
    cumulative offset from the frame's previous control), it is validated
    against ``executed[t] - previous_ctrl`` but never used as the increment.
    """
    if not isinstance(sample, dict):
        raise SmallPolicyError("sample must be a dict")
    labels = sample.get("labels")
    if not isinstance(labels, dict):
        raise SmallPolicyError("labels must be a dict")
    missing = REQUIRED_LABEL_KEYS - labels.keys()
    if missing:
        raise SmallPolicyError(f"labels missing keys: {sorted(missing)}")

    # Keep original float64 until after subtraction to avoid cancellation.
    executed_64 = np.asarray(labels["executed_ctrl_rad"], dtype=np.float64)
    if executed_64.shape != (HORIZON, NUM_CHANNELS):
        raise SmallPolicyError(f"executed_ctrl_rad shape mismatch: {executed_64.shape}")
    if not np.isfinite(executed_64).all():
        raise SmallPolicyError("executed_ctrl_rad contains non-finite values")

    previous_ctrl_64 = np.asarray(
        sample["actor_input"]["previous_ctrl_rad16"], dtype=np.float64
    )
    if previous_ctrl_64.shape != (NUM_CHANNELS,):
        raise SmallPolicyError(f"previous_ctrl_rad16 shape mismatch: {previous_ctrl_64.shape}")
    if not np.isfinite(previous_ctrl_64).all():
        raise SmallPolicyError("previous_ctrl_rad16 contains non-finite values")

    delta = np.empty((HORIZON, NUM_CHANNELS), dtype=np.float64)
    delta[0] = executed_64[0] - previous_ctrl_64
    delta[1:] = executed_64[1:] - executed_64[:-1]

    # Validate optional cumulative-offset field, if present.
    cum_key = "delta_from_frame_previous_ctrl_rad"
    if cum_key in labels:
        cumulative_64 = np.asarray(labels[cum_key], dtype=np.float64)
        if cumulative_64.shape != (HORIZON, NUM_CHANNELS):
            raise SmallPolicyError(f"{cum_key} shape mismatch: {cumulative_64.shape}")
        expected_cumulative = executed_64 - previous_ctrl_64
        if not np.allclose(cumulative_64, expected_cumulative, atol=1e-9, rtol=1e-9):
            raise SmallPolicyError(
                f"{cum_key} is inconsistent with executed_ctrl_rad - previous_ctrl_rad16"
            )

    return delta.astype(np.float32)


def extract_action_window(sample: dict[str, Any]) -> dict[str, Any]:
    """Return validated action-window metadata (dt, step range, horizon check)."""
    window = sample.get("action_window")
    if not isinstance(window, dict):
        raise SmallPolicyError("action_window must be a dict")
    missing = REQUIRED_ACTION_WINDOW_KEYS - window.keys()
    if missing:
        raise SmallPolicyError(f"action_window missing keys: {sorted(missing)}")

    dt = float(window["dt_s"])
    if not (dt > 0.0 and np.isclose(dt, DT_S, atol=1e-6)):
        raise SmallPolicyError(f"action_window dt_s must be {DT_S}, got {dt}")

    first = int(window["first_step_index"])
    end = int(window["end_exclusive_step_index"])
    if end - first != HORIZON:
        raise SmallPolicyError(f"action_window horizon mismatch: {end - first} != {HORIZON}")
    return {"dt_s": dt, "first_step_index": first, "end_exclusive_step_index": end}


def decode_deltas(logits: np.ndarray) -> np.ndarray:
    """Map 400-dim network logits to 25x16 bounded per-substep deltas."""
    logits = _as_array(logits, "logits", (OUTPUT_DIM,))
    return (ACTION_CAP_RAD * np.tanh(logits)).reshape(HORIZON, NUM_CHANNELS)


def decode_controls(deltas: np.ndarray, previous_ctrl: np.ndarray) -> np.ndarray:
    """Cumulatively add per-substep deltas to ``previous_ctrl``.

    Returns the 25x16 reconstructed absolute control references.
    """
    deltas = _as_array(deltas, "deltas", (HORIZON, NUM_CHANNELS))
    previous_ctrl = _as_array(previous_ctrl, "previous_ctrl", (NUM_CHANNELS,))
    return previous_ctrl + np.cumsum(deltas, axis=0)


def decode_controls_batched(deltas: np.ndarray, previous_ctrl: np.ndarray) -> np.ndarray:
    """Batched cumulative control reconstruction.

    ``deltas`` is ``(B, HORIZON, NUM_CHANNELS)`` and ``previous_ctrl`` is
    ``(B, NUM_CHANNELS)``; returns ``(B, HORIZON, NUM_CHANNELS)`` absolute refs.
    """
    deltas = _as_array(deltas, "deltas")
    previous_ctrl = _as_array(previous_ctrl, "previous_ctrl")
    if deltas.ndim != 3 or deltas.shape[1:] != (HORIZON, NUM_CHANNELS):
        raise SmallPolicyError(f"deltas shape must be (B, {HORIZON}, {NUM_CHANNELS}), got {deltas.shape}")
    if previous_ctrl.ndim != 2 or previous_ctrl.shape[1:] != (NUM_CHANNELS,):
        raise SmallPolicyError(f"previous_ctrl shape must be (B, {NUM_CHANNELS}), got {previous_ctrl.shape}")
    if deltas.shape[0] != previous_ctrl.shape[0]:
        raise SmallPolicyError(
            f"deltas batch {deltas.shape[0]} != previous_ctrl batch {previous_ctrl.shape[0]}"
        )
    return previous_ctrl[:, None, :] + np.cumsum(deltas, axis=1)


def _silu(x: np.ndarray) -> np.ndarray:
    """Numerically stable SiLU / Swish activation.

    The positive and negative branches are evaluated on disjoint index sets so
    that ``exp`` is never applied to a value with the wrong sign; this avoids
    overflow in the unused branch.
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0.0
    neg = ~pos
    # x >= 0: x / (1 + exp(-x)); exp(-x) is safe (underflows toward 0).
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    # x < 0: x * exp(x) / (1 + exp(x)); exp(x) is safe (underflows toward 0).
    out[neg] = x[neg] * np.exp(x[neg]) / (1.0 + np.exp(x[neg]))
    return out


def forward_numpy(npz: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """Run the MLP forward pass using a NumPy-loaded NPZ weight dictionary.

    ``features`` is the raw 50-dim actor vector, or a batch ``(B, 50)``. The
    stored ``feature_mean`` and ``feature_std`` are applied exactly once before
    the first layer. Weight shapes are validated; all values must be finite.
    A malformed (negative or sub-floor) stored standard deviation is rejected.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim == 1 and features.shape == (INPUT_DIM,):
        batch = False
    elif features.ndim == 2 and features.shape[1] == INPUT_DIM:
        batch = True
    else:
        raise SmallPolicyError(
            f"features shape must be ({INPUT_DIM},) or (B, {INPUT_DIM}), got {features.shape}"
        )
    if not np.isfinite(features).all():
        raise SmallPolicyError("features contains non-finite values")

    expected_shapes = {
        "W1": (HIDDEN_DIM, INPUT_DIM),
        "b1": (HIDDEN_DIM,),
        "W2": (HIDDEN_DIM, HIDDEN_DIM),
        "b2": (HIDDEN_DIM,),
        "W3": (OUTPUT_DIM, HIDDEN_DIM),
        "b3": (OUTPUT_DIM,),
        "feature_mean": (INPUT_DIM,),
        "feature_std": (INPUT_DIM,),
    }
    for name, shape in expected_shapes.items():
        if name not in npz:
            raise SmallPolicyError(f"NPZ missing key: {name}")
        arr = np.asarray(npz[name])
        if arr.shape != shape:
            raise SmallPolicyError(f"NPZ {name} shape mismatch: got {arr.shape}, want {shape}")
        if not np.isfinite(arr).all():
            raise SmallPolicyError(f"NPZ {name} contains non-finite values")

    mean = np.asarray(npz["feature_mean"], dtype=np.float32)
    std = np.asarray(npz["feature_std"], dtype=np.float32)
    if (std < STD_FLOOR).any():
        raise SmallPolicyError(
            f"NPZ feature_std contains malformed values below STD_FLOOR ({STD_FLOOR})"
        )
    x = normalize_features(features, mean, std)

    for i in range(1, 4):  # three layers: fc1, fc2, fc3
        w = np.asarray(npz[f"W{i}"], dtype=np.float32)
        b = np.asarray(npz[f"b{i}"], dtype=np.float32)
        x = np.dot(x, w.T) + b
        if i < 3:
            x = _silu(x)

    deltas = ACTION_CAP_RAD * np.tanh(x)
    if batch:
        return deltas.reshape(-1, HORIZON, NUM_CHANNELS)
    return deltas.reshape(HORIZON, NUM_CHANNELS)


def normalize_features(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-normalize features; std is clamped to at least ``STD_FLOOR``."""
    std_safe = np.maximum(std, STD_FLOOR)
    return (features - mean) / std_safe


def denormalize_features(norm_features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std_safe = np.maximum(std, STD_FLOOR)
    return norm_features * std_safe + mean


class SmallPolicyNormalizer:
    """Train-only normalizer for inputs and (optionally) delta targets."""

    def __init__(self) -> None:
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "SmallPolicyNormalizer":
        """Fit on training features only; std is floored at ``STD_FLOOR``."""
        features = _as_array(features, "features")
        if features.ndim != 2 or features.shape[1] != INPUT_DIM:
            raise SmallPolicyError(f"features must be (N, {INPUT_DIM}), got {features.shape}")
        self.feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_std = features.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_std = np.maximum(self.feature_std, STD_FLOOR)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise SmallPolicyError("normalizer has not been fit")
        return normalize_features(features, self.feature_mean, self.feature_std)

    def state_dict(self) -> dict[str, np.ndarray]:
        if self.feature_mean is None or self.feature_std is None:
            raise SmallPolicyError("normalizer has not been fit")
        return {
            "feature_mean": self.feature_mean.copy(),
            "feature_std": self.feature_std.copy(),
        }

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        self.feature_mean = np.asarray(state["feature_mean"], dtype=np.float32).copy()
        self.feature_std = np.asarray(state["feature_std"], dtype=np.float32).copy()
        if self.feature_mean.shape != (INPUT_DIM,) or self.feature_std.shape != (INPUT_DIM,):
            raise SmallPolicyError("normalizer state shape mismatch")


def compute_loss(pred_deltas: np.ndarray, target_deltas: np.ndarray, previous_ctrl: np.ndarray) -> dict[str, float]:
    """Normalized increment MSE + 0.25 * cumulative control MSE (normalized by 0.005)."""
    pred_deltas = _as_array(pred_deltas, "pred_deltas", (HORIZON, NUM_CHANNELS))
    target_deltas = _as_array(target_deltas, "target_deltas", (HORIZON, NUM_CHANNELS))
    previous_ctrl = _as_array(previous_ctrl, "previous_ctrl", (NUM_CHANNELS,))

    inc_mse = float(np.mean((pred_deltas - target_deltas) ** 2) / (ACTION_CAP_RAD**2))

    pred_ctrl = decode_controls(pred_deltas, previous_ctrl)
    target_ctrl = decode_controls(target_deltas, previous_ctrl)
    cum_mse = float(np.mean((pred_ctrl - target_ctrl) ** 2) / (BLOCK_CAP_RAD**2))

    loss = inc_mse + 0.25 * cum_mse
    return {
        "loss": loss,
        "increment_mse_normalized": inc_mse,
        "cumulative_mse_normalized": cum_mse,
    }


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root-mean-square error between two equally-shaped finite arrays."""
    pred = _as_array(pred, "pred")
    target = _as_array(target, "target")
    if pred.shape != target.shape:
        raise SmallPolicyError(f"shape mismatch in RMSE: {pred.shape} vs {target.shape}")
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_channel_metrics(
    pred_controls: np.ndarray, target_controls: np.ndarray
) -> dict[str, Any]:
    """Return per-channel and motion-4 RMSEs in radians."""
    pred_controls = _as_array(pred_controls, "pred_controls", (HORIZON, NUM_CHANNELS))
    target_controls = _as_array(target_controls, "target_controls", (HORIZON, NUM_CHANNELS))

    per_channel = np.sqrt(np.mean((pred_controls - target_controls) ** 2, axis=0))
    motion4_pred = pred_controls[:, MOTION4_INDICES]
    motion4_target = target_controls[:, MOTION4_INDICES]
    motion4_rmse = float(np.sqrt(np.mean((motion4_pred - motion4_target) ** 2)))

    return {
        "rmse_per_channel_rad": per_channel.astype(np.float32),
        "rmse_motion4_rad": motion4_rmse,
        "rmse_all_channels_rad": float(np.sqrt(np.mean((pred_controls - target_controls) ** 2))),
    }


def expected_split_for_target_class(tc: int) -> str:
    if tc in TRAIN_TARGET_CLASSES:
        return "train"
    if tc in VAL_TARGET_CLASSES:
        return "validation"
    if tc in TEST_TARGET_CLASSES:
        return "test"
    raise SmallPolicyError(f"target_class {tc} is outside the approved 0..11 set")


def _strict_int(value: Any, name: str, prefix: str) -> int:
    """Exact integer ids only; bools and numeric strings are never coerced."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SmallPolicyError(f"{prefix}{name} must be an exact integer, got {value!r}")
    return value


def _record_identity(rec: dict[str, Any], line_no: int | None = None) -> tuple[str, str | None]:
    """Return ``(task_profile, collection_id)`` for one record.

    An absent task_profile means the legacy ``near_pointing_v1`` profile; an
    unknown profile is rejected, never silently treated as axial. Scoped
    (``near_diagonal_v1``) records must carry matching ``collection_id``,
    ``target_key`` and ``episode_key`` (seed == target_class in 0..7); legacy
    records must not carry scoped identity keys.
    """
    prefix = f"line {line_no}: " if line_no is not None else ""
    profile = rec.get("task_profile", pilot_dataset.DEFAULT_TASK_PROFILE)
    if profile not in pilot_dataset.KNOWN_TASK_PROFILES:
        raise SmallPolicyError(f"{prefix}unknown task_profile {profile!r}")
    if profile == pilot_dataset.DEFAULT_TASK_PROFILE:
        scoped = sorted(k for k in SCOPED_IDENTITY_KEYS if k in rec)
        if scoped:
            raise SmallPolicyError(f"{prefix}legacy record carries scoped identity keys: {scoped}")
        return profile, None

    seed = _strict_int(rec["episode_id"], "episode_id", prefix)
    tc = _strict_int(rec["target_class"], "target_class", prefix)
    if not 0 <= seed <= 7:
        raise SmallPolicyError(f"{prefix}near_diagonal_v1 seed {seed} is outside 0..7")
    if tc != seed:
        raise SmallPolicyError(f"{prefix}target_class {tc} != seed {seed}")
    collection_id = rec.get("collection_id")
    if not isinstance(collection_id, str) or not pilot_dataset.COLLECTION_ID_RE.fullmatch(collection_id):
        raise SmallPolicyError(
            f"{prefix}near_diagonal_v1 requires a nonempty collection_id matching [A-Za-z0-9_-]+"
        )
    expected_target_key = f"{profile}:{seed}"
    if rec.get("target_key") != expected_target_key:
        raise SmallPolicyError(
            f"{prefix}target_key mismatch: expected {expected_target_key!r}, got {rec.get('target_key')!r}"
        )
    expected_episode_key = f"{collection_id}:{profile}:seed-{seed:03d}"
    if rec.get("episode_key") != expected_episode_key:
        raise SmallPolicyError(
            f"{prefix}episode_key mismatch: expected {expected_episode_key!r}, got {rec.get('episode_key')!r}"
        )
    return profile, collection_id


def _check_homogeneity(identities: list[tuple[str, str | None]]) -> None:
    """One index/call holds a single profile and (when scoped) one collection."""
    profiles = {p for p, _ in identities}
    if len(profiles) > 1:
        raise SmallPolicyError(f"mixed task_profile in one index: {sorted(profiles)}")
    collections = {c for _, c in identities if c is not None}
    if len(collections) > 1:
        raise SmallPolicyError(f"mixed collection_id in one index: {sorted(collections)}")


def validate_record_split(rec: dict[str, Any], line_no: int | None = None) -> None:
    """Validate task_profile / target_class / split / episode consistency.

    Legacy ``near_pointing_v1`` records keep the historical axial checks
    (target_class 0..11, ``target_class == episode_id % 12``, axial split).
    Scoped ``near_diagonal_v1`` records split whole-episode by seed via
    ``pilot_dataset.split_of`` (train 0..3, validation 4/6, test 5/7).
    """
    prefix = f"line {line_no}: " if line_no is not None else ""
    profile, _ = _record_identity(rec, line_no=line_no)
    split = str(rec["split"])
    if profile == pilot_dataset.DEFAULT_TASK_PROFILE:
        tc = int(rec["target_class"])
        eid = int(rec["episode_id"])
        if not (0 <= tc <= 11):
            raise SmallPolicyError(f"{prefix}target_class {tc} is outside 0..11")
        if tc != eid % 12:
            raise SmallPolicyError(f"{prefix}target_class {tc} != episode_id {eid} % 12")
        expected = expected_split_for_target_class(tc)
        if split != expected:
            raise SmallPolicyError(f"{prefix}split {split!r} does not match target_class {tc} (expected {expected})")
        return
    seed = int(rec["episode_id"])
    expected = pilot_dataset.split_of(seed, task_profile=profile)
    if expected == "rejected":
        raise SmallPolicyError(f"{prefix}seed {seed} outside {profile} split")
    if split != expected:
        raise SmallPolicyError(
            f"{prefix}split {split!r} does not match {profile} seed {seed} (expected {expected})"
        )


def split_by_target_class(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Partition ``records`` into train/validation/test by task profile/class."""
    _check_homogeneity([_record_identity(rec) for rec in records])
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for rec in records:
        validate_record_split(rec)
        splits[str(rec["split"])].append(rec)
    return splits


def compute_split_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {k: len(v) for k, v in split_by_target_class(records).items()}


def load_sample_index(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL sample index and verify structural/split/feature invariants."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    seen_legacy: set[tuple[int, int]] = set()
    seen_scoped: set[tuple[str, int]] = set()
    index_identity: tuple[str, str | None] | None = None
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SmallPolicyError(f"line {line_no}: invalid JSON: {exc}") from exc

            for key in REQUIRED_SAMPLE_KEYS:
                if key not in rec:
                    raise SmallPolicyError(f"line {line_no}: missing key {key}")

            identity = _record_identity(rec, line_no=line_no)
            if index_identity is None:
                index_identity = identity
            elif identity != index_identity:
                raise SmallPolicyError(
                    f"line {line_no}: mixed task_profile/collection_id in one index: "
                    f"{identity} vs {index_identity}"
                )

            if identity[0] == pilot_dataset.DEFAULT_TASK_PROFILE:
                eid = int(rec["episode_id"])
                fid = int(rec["frame_id"])
                if (eid, fid) in seen_legacy:
                    raise SmallPolicyError(f"line {line_no}: duplicate episode/frame id ({eid}, {fid})")
                seen_legacy.add((eid, fid))
            else:
                ekey = rec["episode_key"]
                fid = _strict_int(rec["frame_id"], "frame_id", f"line {line_no}: ")
                if (ekey, fid) in seen_scoped:
                    raise SmallPolicyError(f"line {line_no}: duplicate episode/frame id ({ekey}, {fid})")
                seen_scoped.add((ekey, fid))

            validate_record_split(rec, line_no=line_no)

            try:
                extract_action_window(rec)
                extract_features(rec)
                extract_label_delta(rec)
            except SmallPolicyError as exc:
                raise SmallPolicyError(f"line {line_no}: {exc}") from exc

            records.append(rec)
    return records


def validate_permit(
    permit: dict[str, Any],
    *,
    sample_index_path: str | Path | None = None,
    sample_index_sha256: str | None = None,
    source_commit: str | None = None,
    dataset_schema: str | None = None,
    split_counts: dict[str, int] | None = None,
) -> SmallPolicyContract:
    """Validate an optimization permit against the contract schema.

    Either ``sample_index_path`` (hashed here) or ``sample_index_sha256`` must
    be supplied. Missing, mismatched, or disallowed fields abort with
    ``SmallPolicyError``.
    """
    if not isinstance(permit, dict):
        raise SmallPolicyError("permit must be a dict")

    required = {
        "status",
        "allow_optimizer",
        "sample_index_sha256",
        "source_commit",
        "dataset_schema",
        "split_counts",
        "limits",
        "approved_actor_fields",
    }
    missing = required - permit.keys()
    if missing:
        raise SmallPolicyError(f"permit missing keys: {sorted(missing)}")

    status = str(permit["status"])
    if status != "accepted_for_small_policy_smoke":
        raise SmallPolicyError(f"permit status {status!r} is not accepted_for_small_policy_smoke")

    if permit["allow_optimizer"] is not True:
        raise SmallPolicyError("permit allow_optimizer must be boolean True")

    expected_index_sha = str(permit["sample_index_sha256"])
    if len(expected_index_sha) != 64 or not all(c in "0123456789abcdef" for c in expected_index_sha.lower()):
        raise SmallPolicyError("sample_index_sha256 is not a 64-character hex string")
    if sample_index_path is not None:
        sample_index_sha256 = sha256_file(sample_index_path)
    if sample_index_sha256 is None:
        raise SmallPolicyError("sample_index_sha256 or sample_index_path required")
    if len(sample_index_sha256) != 64 or not all(c in "0123456789abcdef" for c in sample_index_sha256.lower()):
        raise SmallPolicyError("provided sample_index_sha256 is not a 64-character hex string")
    if sample_index_sha256.lower() != expected_index_sha.lower():
        raise SmallPolicyError(
            f"sample_index_sha256 mismatch: permit {expected_index_sha[:16]}... vs "
            f"provided {sample_index_sha256[:16]}..."
        )

    expected_source = str(permit["source_commit"])
    if not expected_source:
        raise SmallPolicyError("permit source_commit is empty")
    if source_commit is not None and source_commit != expected_source:
        raise SmallPolicyError(
            f"source_commit mismatch: permit {expected_source[:12]} vs provided {source_commit[:12]}"
        )

    expected_schema = str(permit["dataset_schema"])
    if not expected_schema:
        raise SmallPolicyError("permit dataset_schema is empty")
    if dataset_schema is not None and dataset_schema != expected_schema:
        raise SmallPolicyError(
            f"dataset_schema mismatch: permit {expected_schema} vs provided {dataset_schema}"
        )

    permit_counts = dict(permit["split_counts"])
    for split in ("train", "validation", "test"):
        if split not in permit_counts:
            raise SmallPolicyError(f"permit split_counts missing {split}")
        if not isinstance(permit_counts[split], int) or permit_counts[split] < 0:
            raise SmallPolicyError(f"permit split_counts[{split}] must be a non-negative integer")
    if split_counts is not None:
        for split in ("train", "validation", "test"):
            if permit_counts.get(split) != split_counts.get(split):
                raise SmallPolicyError(
                    f"split_counts[{split}] mismatch: permit {permit_counts.get(split)} "
                    f"vs provided {split_counts.get(split)}"
                )

    limits = dict(permit["limits"])
    for key in ("max_updates", "max_gpu", "max_wall_s"):
        if key not in limits:
            raise SmallPolicyError(f"permit limits missing {key}")
        val = limits[key]
        if not isinstance(val, int) or val <= 0:
            raise SmallPolicyError(f"permit limits[{key}] must be a positive integer")
    if limits["max_updates"] > 2000:
        raise SmallPolicyError("permit max_updates exceeds baseline cap of 2000")
    if limits["max_gpu"] > 1:
        raise SmallPolicyError("permit max_gpu exceeds baseline cap of 1")
    if limits["max_wall_s"] > 1800:
        raise SmallPolicyError("permit max_wall_s exceeds baseline cap of 1800")

    approved_fields = tuple(str(x) for x in permit["approved_actor_fields"])
    if approved_fields != APPROVED_ACTOR_FIELDS:
        raise SmallPolicyError(
            f"approved_actor_fields mismatch: permit {approved_fields} vs required {APPROVED_ACTOR_FIELDS}"
        )

    return SmallPolicyContract(
        status=status,
        allow_optimizer=True,
        sample_index_sha256=expected_index_sha,
        source_commit=expected_source,
        dataset_schema=expected_schema,
        split_counts=permit_counts,
        limits=limits,
        approved_actor_fields=approved_fields,
        purpose=str(permit.get("purpose", "small_policy_smoke")),
    )
