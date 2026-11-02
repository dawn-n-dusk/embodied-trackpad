"""Raw pi0.5 window-input repacker for LEAP v1 20 ms coarse control references.

Repacks one validated
:class:`~embodied_trackpad.leap_v1.temporal_windows.WindowObservation` into the
raw pi0.5 model-input layout WITHOUT normalization, resize, tokenizer, 32-dim
padding or loss application: the result is deliberately NOT model-ready. This
is a standalone type — NOT a ``Pi05NativeSample`` subclass — because the native
chain is 25x2 ms while this window chain is 25x20 ms; identical tensor shapes
must not let one time semantics pass as the other.

Action-name mapping (vocabulary only, no numeric transform; delta semantics
rejected): the temporal source corpus names these 16 channels
``absolute_executed_actuator_command_rad`` while the accepted normalization
statistics use ``absolute_executed_joint_reference_rad``. Both name the same
16 absolute executed position references in radians with identical channel
order and values. ``provenance["window_contract"]`` records both strings
explicitly; only the exact temporal source string is accepted here.

Absent sensors (FSR, motor current, wrist) stay null with all-false boolean
masks, carried in provenance only. ``qvel_rad_s16`` and
``previous_ctrl_rad16`` are explicitly recorded as unused and never enter the
model input. No teacher configuration, target_class or error_xy enters
``raw_model_data`` (keys: state, image, image_mask, actions, prompt).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from embodied_trackpad.leap_v1.action_resampling import COARSE_DT_S
from embodied_trackpad.leap_v1.pi05_inputs import ACTION_DIM, DEFAULT_PROMPT
from embodied_trackpad.leap_v1.temporal_windows import (
    ACTION_SEMANTICS as SOURCE_ACTION_SEMANTICS,
)
from embodied_trackpad.leap_v1.temporal_windows import (
    FORMAT_VERSION,
    WINDOW_HORIZON_S,
    WINDOW_REFS,
    WINDOW_ROWS,
    TemporalWindowError,
    WindowObservation,
)
from embodied_trackpad.leap_v1.visual_dataset import (
    ACTION_SEMANTICS,
    DT_S,
    HORIZON,
    HORIZON_S,
    NUM_CHANNELS,
    PNG_HEIGHT,
    PNG_WIDTH,
)

SAMPLE_FORMAT_VERSION = "leap-v1-pi05-window-sample-v1"

_PENDING_TRANSFORMS = (
    "normalization",
    "padding",
    "resize",
    "tokenization",
    "loss_application",
)
# Carried in the window observation but deliberately not mapped into the model
# input by this first window repacker.
_UNUSED_OBSERVATION_KEYS = ("qvel_rad_s16", "previous_ctrl_rad16")

_RGB_SHAPE = (PNG_HEIGHT, PNG_WIDTH, 3)
_NATIVE_TIME_TOL = 1e-6
_COARSE_TIME_TOL_S = 1e-9
_SPLITS = frozenset({"train", "validation", "test"})
_CORPUS_ID_RE = re.compile(r"[A-Za-z0-9_-]+")
_INDEX_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class Pi05WindowSample:
    """Raw 25x20 ms window repack; NOT a Pi05NativeSample; NOT model-ready."""

    raw_model_data: dict[str, Any]  # keys: state, image, image_mask, actions, prompt
    loss_mask32: np.ndarray  # boolean[32]; kept outside raw_model_data
    provenance: dict[str, Any]
    coarse_actions_rad64: np.ndarray  # float64 (25,16) independent label copy


def _finite_f32(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape or arr.dtype != np.float32:
        raise TemporalWindowError(
            "invalid_shape",
            f"{name} must be float32{shape}, got {arr.dtype}{tuple(arr.shape)}",
        )
    if not np.isfinite(arr).all():
        raise TemporalWindowError("invalid_units", f"{name} contains non-finite values")
    return arr


def _finite_f64(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.float64 or arr.shape != shape:
        raise TemporalWindowError(
            "invalid_units",
            f"{name} must be float64{shape}, got {arr.dtype}{tuple(arr.shape)}",
        )
    if not np.isfinite(arr).all():
        raise TemporalWindowError(
            "invalid_units", f"{name} must be finite absolute radians"
        )
    return np.array(arr, dtype=np.float64, copy=True)


def _false_bool_flags(value: Any, count: int, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise TemporalWindowError(
            "invalid_shape", f"{name} must be {count} false booleans (sensor absent)"
        )
    if any(type(v) is not bool or v for v in value):
        raise TemporalWindowError(
            "invalid_shape", f"{name} must be boolean-typed false, not numeric (sensor absent)"
        )


def _require_int(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TemporalWindowError(
            "invalid_shape", f"{name} must be int >= {minimum}, got {value!r}"
        )
    return value


def _check_row_indices(value: Any) -> None:
    if not isinstance(value, (list, tuple)) or len(value) < WINDOW_ROWS:
        raise TemporalWindowError(
            "invalid_shape",
            f"row_indices must be a list/tuple of >= {WINDOW_ROWS} ints, got "
            f"{type(value).__name__} of {len(value) if hasattr(value, '__len__') else '?'}",
        )
    prev = -1
    for i, item in enumerate(value):
        if type(item) is not int or item < 0:
            raise TemporalWindowError(
                "invalid_shape", f"row_indices[{i}] must be a nonnegative int, got {item!r}"
            )
        if item <= prev:
            raise TemporalWindowError(
                "nonmonotone_row_order", f"row_indices[{i}]={item} follows {prev}; keep index order"
            )
        prev = item


def _check_absent_sensors(observation: Mapping[str, Any]) -> None:
    # Absent sensors must stay absent; true or numeric masks count as
    # fabrication of data this embodiment never recorded.
    if observation.get("fsr_N4") is not None or observation.get("motor_current_A16") is not None:
        raise TemporalWindowError(
            "invalid_units", "absent sensor data (fsr/motor_current) must stay null"
        )
    if observation.get("wrist") is not None:
        raise TemporalWindowError("invalid_units", "absent wrist data must stay null")
    _false_bool_flags(observation.get("fsr_valid4"), 4, "fsr_valid4")
    _false_bool_flags(
        observation.get("motor_current_valid16"), NUM_CHANNELS, "motor_current_valid16"
    )
    if type(observation.get("wrist_valid")) is not bool or observation["wrist_valid"]:
        raise TemporalWindowError(
            "invalid_shape", "wrist_valid must be boolean False (sensor absent)"
        )


def _check_native_action_window(provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    window = provenance.get("action_window")
    if not isinstance(window, dict):
        raise TemporalWindowError("invalid_shape", "provenance.action_window missing")
    if (
        window.get("dt_s") != DT_S
        or window.get("horizon") != HORIZON
        or window.get("horizon_s") != HORIZON_S
    ):
        raise TemporalWindowError(
            "invalid_dt", "action_window is not the native dt=0.002/horizon=25 window"
        )
    if window.get("action_semantics") != ACTION_SEMANTICS:
        raise TemporalWindowError(
            "invalid_units",
            "action_window action_semantics is not absolute_executed_joint_reference_rad",
        )
    try:
        first = float(window["first_sim_time_s"])
        end = float(window["end_sim_time_s"])
        prev = float(window["consumed_prev_sim_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TemporalWindowError("invalid_units", f"action_window sim times invalid: {exc}") from exc
    if not np.isfinite([first, end, prev]).all():
        raise TemporalWindowError("invalid_units", "action_window sim times must be finite")
    if (
        abs((first - prev) - DT_S) > _NATIVE_TIME_TOL
        or abs((end - first) - (HORIZON - 1) * DT_S) > _NATIVE_TIME_TOL
    ):
        raise TemporalWindowError(
            "invalid_units", "action_window sim times inconsistent with dt/horizon"
        )
    return window


def prepare_pi05_window_sample(
    window: WindowObservation,
    prompt: str = DEFAULT_PROMPT,
) -> Pi05WindowSample:
    """Validate and repack ``window`` into the raw 25x20 ms pi0.5 input layout."""
    if not isinstance(window, WindowObservation):
        raise TemporalWindowError(
            "invalid_input_type",
            "window must be a temporal_windows.WindowObservation "
            f"(VisualExample/native samples are not accepted), got {type(window).__name__}",
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise TemporalWindowError("invalid_shape", "prompt must be a nonempty string")

    observation = window.observation
    target = window.target
    provenance = window.provenance
    for name, field in (("observation", observation), ("target", target),
                        ("provenance", provenance)):
        if not isinstance(field, Mapping):
            raise TemporalWindowError("invalid_shape", f"window.{name} must be a mapping")

    rgb = np.asarray(observation.get("screen_rgb"))
    if rgb.shape != _RGB_SHAPE or rgb.dtype != np.uint8:
        raise TemporalWindowError(
            "invalid_shape",
            f"screen_rgb must be uint8{_RGB_SHAPE}, got {rgb.dtype}{tuple(rgb.shape)}",
        )
    q = _finite_f32(observation.get("q_rad16"), "q_rad16", (NUM_CHANNELS,))
    actions64 = _finite_f64(
        target.get("target_ctrl_rad"), "target_ctrl_rad", (WINDOW_REFS, NUM_CHANNELS)
    )
    _check_absent_sensors(observation)

    # Temporal provenance contract (copied, never rewritten).
    if provenance.get("format_version") != FORMAT_VERSION:
        raise TemporalWindowError(
            "invalid_format", f"format_version must be {FORMAT_VERSION!r}"
        )
    corpus_id = provenance.get("corpus_id")
    if not isinstance(corpus_id, str) or not _CORPUS_ID_RE.fullmatch(corpus_id):
        raise TemporalWindowError(
            "invalid_shape", "corpus_id must match [A-Za-z0-9_-]+"
        )
    index_sha256 = provenance.get("index_sha256")
    if not isinstance(index_sha256, str) or not _INDEX_SHA256_RE.fullmatch(index_sha256):
        raise TemporalWindowError(
            "invalid_shape", "index_sha256 must be 64 hexadecimal characters"
        )
    if provenance.get("split") not in _SPLITS:
        raise TemporalWindowError(
            "invalid_split", f"split must be one of train/validation/test, got {provenance.get('split')!r}"
        )
    _require_int(provenance.get("episode_id"), "episode_id")
    _require_int(provenance.get("frame_id"), "frame_id")
    _check_row_indices(provenance.get("row_indices"))
    native_window = _check_native_action_window(provenance)

    # Coarse 20 ms target contract: accept ONLY the exact temporal source
    # semantics string; any other vocabulary (including delta actions) and any
    # wrong dt/horizon/horizon_s or non-finite value is rejected.
    if (
        target.get("dt_s") != COARSE_DT_S
        or target.get("horizon") != WINDOW_REFS
        or target.get("horizon_s") != WINDOW_HORIZON_S
    ):
        raise TemporalWindowError(
            "invalid_dt", "target is not the coarse dt=0.02/horizon=25/horizon_s=0.5 window"
        )
    if target.get("action_semantics") != SOURCE_ACTION_SEMANTICS:
        raise TemporalWindowError(
            "invalid_units",
            "target action_semantics must be exactly "
            f"{SOURCE_ACTION_SEMANTICS!r} (same values as {ACTION_SEMANTICS!r}; "
            "vocabulary mapping only, no numeric transform, deltas rejected)",
        )
    endpoints = _finite_f64(
        target.get("coarse_endpoint_sim_time_s"),
        "coarse_endpoint_sim_time_s",
        (WINDOW_REFS,),
    )
    prev_sim_time = float(native_window["consumed_prev_sim_time_s"])
    expected = prev_sim_time + (
        np.arange(1, WINDOW_REFS + 1, dtype=np.float64) * COARSE_DT_S
    )
    if float(np.max(np.abs(endpoints - expected))) > _COARSE_TIME_TOL_S:
        raise TemporalWindowError(
            "invalid_units",
            "coarse_endpoint_sim_time_s does not equal consumed_prev_sim_time_s "
            "+ (1..25)*0.02 within 1e-9",
        )

    # Explicit float64 -> float32 action cast with overflow and rounding record.
    # Overflow is reported by our own finite check, not by a cast warning.
    with np.errstate(over="ignore"):
        actions_f32 = actions64.astype(np.float32)
    if not np.isfinite(actions_f32).all():
        raise TemporalWindowError(
            "invalid_units", "target_ctrl_rad overflows or becomes non-finite in float32"
        )
    rounding_err = float(
        np.max(np.abs(actions_f32.astype(np.float64) - actions64))
    )

    loss_mask32 = np.zeros(ACTION_DIM, dtype=bool)
    loss_mask32[:NUM_CHANNELS] = True

    raw_model_data = {
        "state": q.copy(),
        "image": {
            "base_0_rgb": np.array(rgb, dtype=np.uint8, copy=True),
            "left_wrist_0_rgb": np.zeros(_RGB_SHAPE, dtype=np.uint8),
            "right_wrist_0_rgb": np.zeros(_RGB_SHAPE, dtype=np.uint8),
        },
        "image_mask": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": False,
            "right_wrist_0_rgb": False,
        },
        "actions": actions_f32,
        "prompt": prompt,
    }

    out_provenance = copy.deepcopy(dict(provenance))
    # Absence flags ride along in provenance only (validated null/all-false
    # above); they never enter raw_model_data.
    out_provenance["absent_sensors"] = {
        key: copy.deepcopy(observation[key])
        for key in (
            "fsr_N4",
            "motor_current_A16",
            "wrist",
            "fsr_valid4",
            "motor_current_valid16",
            "wrist_valid",
        )
    }
    out_provenance["model_contract"] = {
        "pi05": True,
        "action_dim": ACTION_DIM,
        "action_horizon": WINDOW_REFS,
    }
    out_provenance["sample_format_version"] = SAMPLE_FORMAT_VERSION
    out_provenance["window_contract"] = {
        "native_dt_s": DT_S,
        "target_dt_s": COARSE_DT_S,
        "horizon": WINDOW_REFS,
        "horizon_s": WINDOW_HORIZON_S,
        "source_action_semantics": SOURCE_ACTION_SEMANTICS,
        "action_semantics": ACTION_SEMANTICS,
        "coarse_endpoint_sim_time_s": endpoints.copy(),
    }
    out_provenance["pending_transforms"] = {name: "PENDING" for name in _PENDING_TRANSFORMS}
    out_provenance["unused_observation_keys"] = list(_UNUSED_OBSERVATION_KEYS)
    out_provenance["actions_f64_to_f32_max_abs_error_rad"] = rounding_err

    return Pi05WindowSample(
        raw_model_data=raw_model_data,
        loss_mask32=loss_mask32,
        provenance=out_provenance,
        coarse_actions_rad64=actions64.copy(),
    )
