"""Frozen learned-policy evaluator guards for the LEAP v1 Full small-policy baseline.

This module is intentionally NumPy-only and does not import PyTorch or MuJoCo.
It holds the permit/checkpoint binding, actor-input construction, deterministic
MLP inference via :mod:`embodied_trackpad.leap_v1.small_policy`, deployment
cumulative-control arithmetic, action-cap validation, and screen-dependence
counterfactuals.  No privileged physics/Jacobian/planner helpers are used here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from embodied_trackpad.leap_v1 import small_policy as sp

# Re-export for callers that need the same SHA routine.
sha256_file = sp.sha256_file

# Source files whose hashes must be frozen in an evaluation permit.
# The permit is issued *after* coordinator review of these exact files.
# The action chain depends on more than the three evaluator files: physics,
# scene, model, visual servo, and the sandbox/near-probe client must also be
# frozen for a source-bound permit to mean anything.
REQUIRED_SOURCE_FILES = (
    "src/embodied_trackpad/leap_v1/small_policy.py",
    "src/embodied_trackpad/leap_v1/learned_evaluator.py",
    "tools/evaluate_leap_v1_small_policy.py",
    "src/embodied_trackpad/leap_v1/visual_physics.py",
    "src/embodied_trackpad/leap_v1/normal_priority.py",
    "src/embodied_trackpad/leap_v1/pad_env.py",
    "src/embodied_trackpad/leap_sim/pad_env.py",
    "src/embodied_trackpad/leap_v1/model.py",
    "src/embodied_trackpad/leap_v1/visual_servo.py",
    "tools/mac_sandbox_client.py",
    "tools/leap_v1_near_probe.py",
)

# Hard caps from the v1 small-policy smoke contract.
ALLOWED_TARGET_CLASSES = frozenset({4, 10})
MAX_ACTION_IMAGES = 120
MAX_WALL_S = 120
MAX_SIM_S = 20
MAX_DIR_BYTES = 60 * 1024 * 1024
MAX_BATCH_BYTES = 160 * 1024 * 1024

# Tiny tolerance for float round-off inside the deployment arithmetic only.
_DEPLOY_TOL_RAD = 1e-12

# Small deterministic perturbations used for screen-dependence counterfactuals.
_COUNTERFACTUAL_PERTURBATIONS = (
    ("plus_error", (1.0, 1.0)),   # add the observed error vector
    ("minus_error", (-1.0, -1.0)),  # subtract the observed error vector
    ("plus_x_1px", (1.0, 0.0)),
    ("minus_x_1px", (-1.0, 0.0)),
    ("plus_y_1px", (0.0, 1.0)),
    ("minus_y_1px", (0.0, -1.0)),
)


class EvaluatorError(ValueError):
    """Raised when permit, checkpoint, or action validation fails."""


@dataclass(frozen=True)
class EvaluationContract:
    """Validated in-memory contract binding a frozen checkpoint to one evaluation."""

    status: str
    allow_optimizer: bool
    sample_index_sha256: str
    source_commit: str
    dataset_schema: str
    checkpoint_sha256: str
    checkpoint_identity: str
    target_class: int
    approved_actor_fields: tuple[str, ...]
    source_file_hashes: dict[str, str]
    limits: dict[str, Any]
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "allow_optimizer": self.allow_optimizer,
            "sample_index_sha256": self.sample_index_sha256,
            "source_commit": self.source_commit,
            "dataset_schema": self.dataset_schema,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_identity": self.checkpoint_identity,
            "target_class": self.target_class,
            "approved_actor_fields": list(self.approved_actor_fields),
            "source_file_hashes": dict(self.source_file_hashes),
            "limits": dict(self.limits),
            "purpose": self.purpose,
        }


def _hex64(value: Any, name: str) -> str:
    s = str(value).lower()
    if len(s) != 64 or not all(c in "0123456789abcdef" for c in s):
        raise EvaluatorError(f"{name} is not a 64-character lowercase hex SHA-256")
    return s


def compute_source_file_hashes(
    repo_root: str | Path, paths: tuple[str, ...] = REQUIRED_SOURCE_FILES
) -> dict[str, str]:
    """Return a dict of repository-relative path -> SHA-256 for required source files."""
    root = Path(repo_root)
    out: dict[str, str] = {}
    for p in paths:
        full = root / p
        if not full.is_file():
            raise EvaluatorError(f"required source file missing: {p}")
        out[p] = sha256_file(full)
    return out


def validate_evaluation_permit(
    permit: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    sample_index_path: str | Path | None = None,
    sample_index_sha256: str | None = None,
    source_commit: str | None = None,
    dataset_schema: str | None = None,
    target_class: int | None = None,
    source_file_hashes: dict[str, str] | None = None,
) -> EvaluationContract:
    """Validate an evaluation permit and bind it to current source/checkpoint hashes.

    Raises ``EvaluatorError`` on any mismatch or unsupported limit.  The permit
    must explicitly disallow optimization, target only class 4 or 10, and bind
    exact hashes for the required source files, checkpoint, and sample index.
    """
    if not isinstance(permit, dict):
        raise EvaluatorError("permit must be a dict")

    required = {
        "status",
        "allow_optimizer",
        "sample_index_sha256",
        "source_commit",
        "dataset_schema",
        "checkpoint_sha256",
        "checkpoint_identity",
        "target_class",
        "approved_actor_fields",
        "source_file_hashes",
        "limits",
    }
    missing = required - permit.keys()
    if missing:
        raise EvaluatorError(f"permit missing keys: {sorted(missing)}")

    status = str(permit["status"])
    if status != "accepted_for_leap_v1_small_policy_evaluation":
        raise EvaluatorError(f"permit status {status!r} is not accepted")

    if permit["allow_optimizer"] is not False:
        raise EvaluatorError("permit allow_optimizer must be boolean False for evaluation")

    tc = int(permit["target_class"])
    if tc not in ALLOWED_TARGET_CLASSES:
        raise EvaluatorError(f"permit target_class {tc} not in {sorted(ALLOWED_TARGET_CLASSES)}")
    if target_class is not None and tc != target_class:
        raise EvaluatorError(f"permit target_class {tc} != requested {target_class}")

    expected_index_sha = _hex64(permit["sample_index_sha256"], "permit sample_index_sha256")
    if sample_index_path is not None:
        sample_index_sha256 = sha256_file(sample_index_path)
    if sample_index_sha256 is not None:
        sample_index_sha256 = _hex64(sample_index_sha256, "provided sample_index_sha256")
        if sample_index_sha256 != expected_index_sha:
            raise EvaluatorError("sample_index_sha256 mismatch")

    expected_checkpoint_sha = _hex64(permit["checkpoint_sha256"], "permit checkpoint_sha256")
    if checkpoint_path is not None:
        checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 is not None:
        checkpoint_sha256 = _hex64(checkpoint_sha256, "provided checkpoint_sha256")
        if checkpoint_sha256 != expected_checkpoint_sha:
            raise EvaluatorError("checkpoint_sha256 mismatch")

    expected_source = str(permit["source_commit"])
    if not expected_source:
        raise EvaluatorError("permit source_commit is empty")
    if source_commit is not None and source_commit != expected_source:
        raise EvaluatorError(
            f"source_commit mismatch: permit {expected_source[:12]} vs provided {source_commit[:12]}"
        )

    expected_schema = str(permit["dataset_schema"])
    if not expected_schema:
        raise EvaluatorError("permit dataset_schema is empty")
    if dataset_schema is not None and dataset_schema != expected_schema:
        raise EvaluatorError(
            f"dataset_schema mismatch: permit {expected_schema} vs provided {dataset_schema}"
        )

    approved_fields = tuple(str(x) for x in permit["approved_actor_fields"])
    if approved_fields != sp.APPROVED_ACTOR_FIELDS:
        raise EvaluatorError(
            f"approved_actor_fields mismatch: permit {approved_fields} vs required {sp.APPROVED_ACTOR_FIELDS}"
        )

    permit_hashes = dict(permit["source_file_hashes"])
    for p in REQUIRED_SOURCE_FILES:
        if p not in permit_hashes:
            raise EvaluatorError(f"permit source_file_hashes missing {p}")
    if source_file_hashes is not None:
        for p in REQUIRED_SOURCE_FILES:
            if source_file_hashes.get(p) != permit_hashes.get(p):
                raise EvaluatorError(f"source_file_hashes mismatch for {p}")

    limits = dict(permit["limits"])
    for key in ("max_action_images", "max_wall_s", "max_sim_s", "dir_cap_bytes", "batch_cap_bytes"):
        if key not in limits:
            raise EvaluatorError(f"permit limits missing {key}")
        val = limits[key]
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            raise EvaluatorError(f"permit limits[{key}] must be a positive integer")
    if limits["max_action_images"] > MAX_ACTION_IMAGES:
        raise EvaluatorError(f"permit max_action_images exceeds cap {MAX_ACTION_IMAGES}")
    if limits["max_wall_s"] > MAX_WALL_S:
        raise EvaluatorError(f"permit max_wall_s exceeds cap {MAX_WALL_S}")
    if limits["max_wall_s"] <= 10:
        raise EvaluatorError("permit max_wall_s must leave the declared 10 s teardown reserve")
    if limits["max_sim_s"] > MAX_SIM_S:
        raise EvaluatorError(f"permit max_sim_s exceeds cap {MAX_SIM_S}")
    if limits["dir_cap_bytes"] > MAX_DIR_BYTES:
        raise EvaluatorError(f"permit dir_cap_bytes exceeds cap {MAX_DIR_BYTES}")
    if limits["batch_cap_bytes"] > MAX_BATCH_BYTES:
        raise EvaluatorError(f"permit batch_cap_bytes exceeds cap {MAX_BATCH_BYTES}")

    return EvaluationContract(
        status=status,
        allow_optimizer=False,
        sample_index_sha256=expected_index_sha,
        source_commit=expected_source,
        dataset_schema=expected_schema,
        checkpoint_sha256=expected_checkpoint_sha,
        checkpoint_identity=str(permit["checkpoint_identity"]),
        target_class=tc,
        approved_actor_fields=approved_fields,
        source_file_hashes=permit_hashes,
        limits=limits,
        purpose=str(permit.get("purpose", "small_policy_evaluation")),
    )


def load_and_validate_checkpoint(path: str | Path) -> tuple[dict[str, np.ndarray], str]:
    """Load an NPZ checkpoint, hash it, and validate its schema/finiteness.

    Validation runs one dummy forward pass through :func:`sp.forward_numpy` so
    every weight shape, normalizer parameter, and finite-ness check is applied
    before any episode work begins.
    """
    path = Path(path)
    if not path.is_file():
        raise EvaluatorError(f"checkpoint not found: {path}")
    sha = sha256_file(path)
    try:
        with np.load(path) as data:
            npz = {k: np.asarray(data[k]) for k in data.files}
    except Exception as exc:
        raise EvaluatorError(f"failed to load checkpoint NPZ: {exc}") from exc

    dummy = np.zeros(sp.INPUT_DIM, dtype=np.float32)
    try:
        deltas = sp.forward_numpy(npz, dummy)
    except sp.SmallPolicyError as exc:
        raise EvaluatorError(f"checkpoint validation failed: {exc}") from exc

    if deltas.shape != (sp.HORIZON, sp.NUM_CHANNELS):
        raise EvaluatorError(f"checkpoint forward shape {deltas.shape} is invalid")
    if not np.all(np.isfinite(deltas)):
        raise EvaluatorError("checkpoint forward produced non-finite deltas")
    return npz, sha


class PolicyInference:
    """Frozen small-policy inference with deployment control reconstruction."""

    def __init__(self, checkpoint_path: str | Path, contract: EvaluationContract) -> None:
        self.contract = contract
        self.npz, self.checkpoint_sha256 = load_and_validate_checkpoint(checkpoint_path)
        if self.checkpoint_sha256 != contract.checkpoint_sha256:
            raise EvaluatorError(
                "checkpoint SHA mismatch between loaded file and validated contract"
            )

    def predict(self, actor_input: dict[str, Any]) -> dict[str, Any]:
        """Return prediction metadata including float32 deltas and float64 controls.

        The actor input is validated against the exact approved feature whitelist,
        the 50-dim feature vector is built, the frozen NPZ produces 25×16 deltas,
        and deployment arithmetic reconstructs absolute references in float64.
        """
        features = build_feature_vector(actor_input)
        pred_deltas = sp.forward_numpy(self.npz, features)
        pred_deltas = np.asarray(pred_deltas, dtype=np.float32)
        if pred_deltas.shape != (sp.HORIZON, sp.NUM_CHANNELS):
            raise EvaluatorError(f"predicted deltas shape {pred_deltas.shape} is invalid")
        if not np.all(np.isfinite(pred_deltas)):
            raise EvaluatorError("predicted deltas contain non-finite values")

        previous_ctrl = np.asarray(actor_input["previous_ctrl_rad16"], dtype=np.float64)
        controls = decode_controls_deployment(pred_deltas, previous_ctrl)

        return {
            "features": features.astype(np.float32),
            "previous_ctrl_rad16": previous_ctrl.astype(np.float64),
            "predicted_deltas_rad": pred_deltas,
            "requested_controls_rad": controls,
        }


def build_actor_input(
    error_xy_px: tuple[float, float] | list[float] | np.ndarray,
    q_rad16: np.ndarray | list[float],
    qvel_rad_s16: np.ndarray | list[float],
    previous_ctrl_rad16: np.ndarray | list[float],
    *,
    png_ref: str | None = None,
    png_sha256: str | None = None,
    frame_monotonic_time_s: float | None = None,
    valid: bool = True,
) -> dict[str, Any]:
    """Build an actor_input dict that passes the small-policy feature guard.

    The previous control anchor is kept in ``float64`` so cumulative deployment
    arithmetic does not lose the actual anchor.  The feature vector alone is
    cast to ``float32`` for the network.  Sensor inputs that are not installed
    (FSR, motor current, wrist image) are explicitly ``None`` / ``False`` so
    they cannot be silently treated as valid.  Invalid observations are
    rejected at construction time.
    """
    if valid is not True:
        raise EvaluatorError("actor_input valid must be True; invalid observations are rejected")
    error_xy = _finite_f32(error_xy_px, "error_xy_px", (2,))
    q = _finite_f32(q_rad16, "q_rad16", (sp.NUM_CHANNELS,))
    qvel = _finite_f32(qvel_rad_s16, "qvel_rad_s16", (sp.NUM_CHANNELS,))
    prev = _finite_f64(previous_ctrl_rad16, "previous_ctrl_rad16", (sp.NUM_CHANNELS,))

    return {
        "error_xy_px": error_xy,
        "q_rad16": q,
        "qvel_rad_s16": qvel,
        "previous_ctrl_rad16": prev,
        # Non-feature metadata: absent sensors / missing image features.
        "fsr_N4": None,
        "motor_current_A16": None,
        "wrist": None,
        "valid": True,
        "png_ref": png_ref,
        "png_sha256": png_sha256,
        "frame_monotonic_time_s": frame_monotonic_time_s,
    }


def _finite_f32(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise EvaluatorError(f"{name} shape {arr.shape} != {shape}")
    if not np.all(np.isfinite(arr)):
        raise EvaluatorError(f"{name} contains non-finite values")
    return arr


def _finite_f64(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise EvaluatorError(f"{name} shape {arr.shape} != {shape}")
    if not np.all(np.isfinite(arr)):
        raise EvaluatorError(f"{name} contains non-finite values")
    return arr


def build_feature_vector(actor_input: dict[str, Any]) -> np.ndarray:
    """Concatenate the exact 50-dim input vector from a validated actor_input."""
    if not isinstance(actor_input, dict):
        raise EvaluatorError("actor_input must be a dict")

    rejected = set(actor_input.keys()) & sp.REJECTED_ACTOR_INPUT_KEYS
    if rejected:
        raise EvaluatorError(f"actor_input contains rejected fields: {sorted(rejected)}")

    allowed = set(sp.APPROVED_ACTOR_FIELDS) | sp.ALLOWED_NON_FEATURE_ACTOR_KEYS
    unknown = set(actor_input.keys()) - allowed
    if unknown:
        raise EvaluatorError(f"actor_input contains unknown fields: {sorted(unknown)}")

    if actor_input.get("wrist") is not None:
        raise EvaluatorError("actor_input['wrist'] must be None; image features are not actor inputs")

    if actor_input.get("valid") is not True:
        raise EvaluatorError("actor_input['valid'] must be True")

    if actor_input.get("fsr_N4") is not None:
        raise EvaluatorError("actor_input['fsr_N4'] must be None; FSR sensors are not installed")

    if actor_input.get("motor_current_A16") is not None:
        raise EvaluatorError(
            "actor_input['motor_current_A16'] must be None; motor current sensors are not installed"
        )

    for key in sp.APPROVED_ACTOR_FIELDS:
        if key not in actor_input:
            raise EvaluatorError(f"actor_input missing required field {key}")

    error_xy = _finite_f32(actor_input["error_xy_px"], "error_xy_px", (2,))
    q = _finite_f32(actor_input["q_rad16"], "q_rad16", (sp.NUM_CHANNELS,))
    qvel = _finite_f32(actor_input["qvel_rad_s16"], "qvel_rad_s16", (sp.NUM_CHANNELS,))
    prev = _finite_f64(actor_input["previous_ctrl_rad16"], "previous_ctrl_rad16", (sp.NUM_CHANNELS,))

    features = np.concatenate([error_xy, q, qvel, prev.astype(np.float32)])
    if features.shape != (sp.INPUT_DIM,):
        raise EvaluatorError(f"feature vector shape {features.shape} != ({sp.INPUT_DIM},)")
    return features


def decode_controls_deployment(
    deltas: np.ndarray, previous_ctrl: np.ndarray
) -> np.ndarray:
    """Cumulatively reconstruct absolute controls using float64 anchors.

    The network emits ``float32`` deltas.  Those deltas are kept unchanged, but
    are cast to ``float64`` *before* cumulative summation against the actual
    ``float64`` previous control reference.  This avoids the cancellation/round-off
    that can occur when a large absolute reference (~0.6 rad) plus many small
    increments is accumulated in ``float32``, which could otherwise push a
    substep or block envelope just outside the approved caps.

    The reconstructed controls are then validated against:
      * per-substep delta <= 2e-4 rad
      * per-block delta from the previous reference <= 0.005 rad
      * finite values
    """
    deltas = np.asarray(deltas, dtype=np.float32)
    if deltas.shape != (sp.HORIZON, sp.NUM_CHANNELS):
        raise EvaluatorError(f"deltas shape {deltas.shape} != ({sp.HORIZON}, {sp.NUM_CHANNELS})")
    if not np.all(np.isfinite(deltas)):
        raise EvaluatorError("deltas contain non-finite values")

    previous_ctrl = np.asarray(previous_ctrl, dtype=np.float64)
    if previous_ctrl.shape != (sp.NUM_CHANNELS,):
        raise EvaluatorError(f"previous_ctrl shape {previous_ctrl.shape} != ({sp.NUM_CHANNELS},)")
    if not np.all(np.isfinite(previous_ctrl)):
        raise EvaluatorError("previous_ctrl contains non-finite values")

    deltas_64 = deltas.astype(np.float64)
    controls = previous_ctrl + np.cumsum(deltas_64, axis=0)

    # Per-substep cap: the network's tanh already bounds this, but we verify
    # the deployment reconstruction explicitly and abort on violation.
    step0 = controls[0] - previous_ctrl
    step_rest = controls[1:] - controls[:-1]
    max_substep = float(max(np.max(np.abs(step0)), np.max(np.abs(step_rest))))
    if max_substep > sp.ACTION_CAP_RAD + _DEPLOY_TOL_RAD:
        raise EvaluatorError(
            f"predicted substep delta {max_substep:.6e} rad exceeds cap {sp.ACTION_CAP_RAD}"
        )

    # Block cap from the action-window anchor.
    max_block = float(np.max(np.abs(controls - previous_ctrl)))
    if max_block > sp.BLOCK_CAP_RAD + _DEPLOY_TOL_RAD:
        raise EvaluatorError(
            f"predicted block delta {max_block:.6e} rad exceeds cap {sp.BLOCK_CAP_RAD}"
        )

    if not np.all(np.isfinite(controls)):
        raise EvaluatorError("reconstructed controls contain non-finite values")

    return controls


def validate_controls_against_ranges(
    controls: np.ndarray, ranges: Sequence[tuple[float, float]]
) -> None:
    """Abort (never clip) if any requested control lies outside joint ranges."""
    if len(ranges) != sp.NUM_CHANNELS:
        raise EvaluatorError(f"ranges length {len(ranges)} != {sp.NUM_CHANNELS}")
    controls = np.asarray(controls, dtype=np.float64)
    if controls.shape != (sp.HORIZON, sp.NUM_CHANNELS):
        raise EvaluatorError(f"controls shape {controls.shape} is invalid")
    for t in range(sp.HORIZON):
        for i, (lo, hi) in enumerate(ranges):
            v = float(controls[t, i])
            if not (lo - _DEPLOY_TOL_RAD <= v <= hi + _DEPLOY_TOL_RAD):
                raise EvaluatorError(
                    f"control t={t} channel={i} ({sp.MOTOR_JOINTS[i]}) "
                    f"value {v:.6e} outside range [{lo}, {hi}]"
                )


def compute_screen_dependence(
    actor_input: dict[str, Any], npz: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Counterfactual screen-dependence check: vary only the pixel error.

    For the same frozen q, qvel, and previous_ctrl, this evaluates how much the
    predicted delta block changes when the screen error is perturbed.  No
    physical step is executed.  The differences are recorded, not used to
    correct the action.
    """
    base_features = build_feature_vector(actor_input)
    base_deltas = np.asarray(sp.forward_numpy(npz, base_features), dtype=np.float32)

    error_xy = np.asarray(actor_input["error_xy_px"], dtype=np.float32)
    perturbations: dict[str, dict[str, Any]] = {}

    for name, coeff in _COUNTERFACTUAL_PERTURBATIONS:
        if name in ("plus_error", "minus_error"):
            perturbed_error = error_xy + coeff[0] * error_xy
        else:
            perturbed_error = error_xy + np.array(coeff, dtype=np.float32)
        alt_input = dict(actor_input)
        alt_input["error_xy_px"] = perturbed_error
        alt_features = build_feature_vector(alt_input)
        alt_deltas = np.asarray(sp.forward_numpy(npz, alt_features), dtype=np.float32)
        diff = np.abs(alt_deltas - base_deltas)
        perturbations[name] = {
            "perturbed_error_xy_px": perturbed_error.tolist(),
            "max_abs_delta_rad": float(np.max(diff)),
            "max_per_channel_abs_delta_rad": np.max(diff, axis=0).astype(np.float32).tolist(),
        }

    return {
        "base_error_xy_px": error_xy.tolist(),
        "base_predicted_deltas_rad": base_deltas.tolist(),
        "perturbations": perturbations,
    }


def stop_and_reap_client(client: Any, bounds: dict[str, float] | None = None) -> dict[str, Any]:
    """Bounded software teardown of a sandbox client.

    Restores the original ``_send`` if the runner wrapped it with a budgeted
    version, then issues ``{"cmd":"stop"}`` capped at 1 s and calls
    ``client.quit(timeout=1)``.  The native :class:`SandboxClient.quit` already
    runs ``_teardown`` (terminate 3 s / kill 5 s) if the child does not exit on
    EOF, so this helper does NOT add a second terminate/kill sequence.  The
    worst-case bound is therefore stop 1 s + quit 1 s + terminate 3 s + kill 5 s
    = 10 s.

    Records the requested timeouts and any errors.  This is pure lifecycle
    cleanup and does not perform any privileged rescue of the episode.
    """
    bounds = bounds or {}
    requested = {
        "stop": float(bounds.get("stop", 1.0)),
        "quit": float(bounds.get("quit", 1.0)),
    }
    # Hard cap at the reviewed contract: 1 s for the stop request and 1 s for
    # the graceful quit.  Any additional teardown time comes from the client's
    # own _teardown, not from this helper.
    stop_timeout = min(requested["stop"], 1.0)
    quit_timeout = min(requested["quit"], 1.0)
    teardown: dict[str, Any] = {
        "requested_bounds_s": {"stop": stop_timeout, "quit": quit_timeout},
        "stop_error": None,
        "quit_error": None,
    }
    child_pid, returncode, reaped = None, None, False
    if client is not None:
        # Remove the runner's _send_budgeted wrapper before issuing teardown IO.
        if hasattr(client, "client_orig_send"):
            try:
                client._send = client.client_orig_send
            except Exception as exc:
                teardown["restore_send_error"] = f"{type(exc).__name__}:{exc}"
        child_pid = getattr(getattr(client, "proc", None), "pid", None)
        try:
            ack = client._send({"cmd": "stop"}, timeout=stop_timeout)
            if not isinstance(ack, dict) or ack.get("type") != "stopped":
                raise RuntimeError(f"unexpected stop ack: {ack!r}")
        except Exception as exc:
            teardown["stop_error"] = f"{type(exc).__name__}:{exc}"
        # Always attempt quit, even if stop failed.
        try:
            returncode = client.quit(timeout=quit_timeout)
        except Exception as exc:
            teardown["quit_error"] = f"{type(exc).__name__}:{exc}"
            proc = getattr(client, "proc", None)
            returncode = getattr(proc, "returncode", None)
        # A normal reap means the child returned a code AND quit succeeded.
        # A quit error is still reported as a failure even if the child process
        # was reaped by the client's internal _teardown.
        reaped = returncode is not None and teardown["quit_error"] is None
    return {
        "child_pid": child_pid,
        "child_returncode": returncode,
        "child_reaped": bool(reaped),
        "teardown": teardown,
    }
