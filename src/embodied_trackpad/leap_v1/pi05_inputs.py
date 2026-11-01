"""Pure-NumPy native OpenPI-input repacker for the LEAP v1 pi0.5 mainline.

Repacks one accepted ``VisualExample`` into the raw pi0.5 model-input layout
WITHOUT importing OpenPI/Torch/JAX and WITHOUT applying any official
normalization, tokenizer, 32-dim padding, image resize or loss: the result is
deliberately NOT yet model-ready. Native data units stay absolute radians,
pixels stay native 800x600 uint8, and raw state/actions stay 16-dim.

Downstream contract markers only: the 32-dim OpenPI action layout and the
[32] loss vector appear solely as ``loss_mask32`` (first 16 true, rest false,
kept OUTSIDE ``raw_model_data``) plus ``provenance["model_contract"]``; a
PyTorch trainer must apply that mask to its unreduced [B, H, 32] loss itself.

Absent sensors (FSR, motor current, wrist) stay null with all-false boolean
masks; the flags are copied into provenance only. ``qvel_rad_s16`` and
``previous_ctrl_rad16`` are explicitly recorded as unused in this first
native representation rather than silently consumed. No wrist images are
fabricated: missing wrist views are real zeros with false image masks.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1.visual_dataset import (
    ACTION_SEMANTICS,
    DT_S,
    HORIZON,
    HORIZON_S,
    NUM_CHANNELS,
    PNG_HEIGHT,
    PNG_WIDTH,
    VisualCorpusError,
    VisualExample,
)

ACTION_DIM = 32  # downstream OpenPI action layout; raw state/actions stay 16
DEFAULT_PROMPT = "Move the on-screen cursor to the target without clicking."

_PENDING_TRANSFORMS = (
    "normalization",
    "padding",
    "resize",
    "tokenization",
    "loss_application",
)
# Carried in the example observation but deliberately not mapped into the
# native OpenPI input by this first repacker.
_UNUSED_OBSERVATION_KEYS = ("qvel_rad_s16", "previous_ctrl_rad16")

_RGB_SHAPE = (PNG_HEIGHT, PNG_WIDTH, 3)
_TIME_TOL = 1e-6


@dataclass(frozen=True)
class Pi05NativeSample:
    """Native pi0.5 repack result; raw data is NOT yet model-ready."""

    raw_model_data: dict[str, Any]
    loss_mask32: np.ndarray  # boolean[32]; kept outside raw_model_data
    provenance: dict[str, Any]


def _finite_f32(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape or arr.dtype != np.float32:
        raise VisualCorpusError(
            f"{name} must be float32{shape}, got {arr.dtype}{tuple(arr.shape)}"
        )
    if not np.isfinite(arr).all():
        raise VisualCorpusError(f"{name} contains non-finite values")
    return arr


def _false_bool_flags(value: Any, count: int, name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise VisualCorpusError(f"{name} must be {count} false booleans (sensor absent)")
    if any(type(v) is not bool or v for v in value):
        raise VisualCorpusError(f"{name} must be boolean-typed false, not numeric (sensor absent)")


def prepare_pi05_native_sample(
    example: VisualExample,
    prompt: str = DEFAULT_PROMPT,
) -> Pi05NativeSample:
    """Validate and repack ``example`` into the raw native pi0.5 input layout."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise VisualCorpusError("prompt must be a nonempty string")

    observation = example.observation
    rgb = np.asarray(observation.get("screen_rgb"))
    if rgb.shape != _RGB_SHAPE or rgb.dtype != np.uint8:
        raise VisualCorpusError(
            f"screen_rgb must be uint8{_RGB_SHAPE}, got {rgb.dtype}{tuple(rgb.shape)}"
        )
    q = _finite_f32(observation.get("q_rad16"), "q_rad16", (NUM_CHANNELS,))
    actions = _finite_f32(
        example.target.get("target_ctrl_rad"), "target_ctrl_rad", (HORIZON, NUM_CHANNELS)
    )

    # Absent sensors must stay absent; true or numeric masks count as
    # fabrication of data this embodiment never recorded.
    if observation.get("fsr_N4") is not None or observation.get("motor_current_A16") is not None:
        raise VisualCorpusError("absent sensor data (fsr/motor_current) must stay null")
    if observation.get("wrist") is not None:
        raise VisualCorpusError("absent wrist data must stay null")
    _false_bool_flags(observation.get("fsr_valid4"), 4, "fsr_valid4")
    _false_bool_flags(
        observation.get("motor_current_valid16"), NUM_CHANNELS, "motor_current_valid16"
    )
    if type(observation.get("wrist_valid")) is not bool or observation["wrist_valid"]:
        raise VisualCorpusError("wrist_valid must be boolean False (sensor absent)")

    window = example.provenance.get("action_window")
    if not isinstance(window, dict):
        raise VisualCorpusError("provenance.action_window missing")
    if (
        window.get("dt_s") != DT_S
        or window.get("horizon") != HORIZON
        or window.get("horizon_s") != HORIZON_S
    ):
        raise VisualCorpusError("action_window is not the native dt=0.002/horizon=25 window")
    if window.get("action_semantics") != ACTION_SEMANTICS:
        raise VisualCorpusError("action_semantics is not absolute_executed_joint_reference_rad")
    try:
        first = float(window["first_sim_time_s"])
        end = float(window["end_sim_time_s"])
        prev = float(window["consumed_prev_sim_time_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VisualCorpusError(f"action_window sim times invalid: {exc}") from exc
    if not np.isfinite([first, end, prev]).all():
        raise VisualCorpusError("action_window sim times must be finite")
    if (
        abs((first - prev) - DT_S) > _TIME_TOL
        or abs((end - first) - (HORIZON - 1) * DT_S) > _TIME_TOL
    ):
        raise VisualCorpusError("action_window sim times inconsistent with dt/horizon")

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
        "actions": actions.copy(),
        "prompt": prompt,
    }

    provenance = copy.deepcopy(example.provenance)
    # Absence flags ride along in provenance only (validated null/all-false
    # above); they never enter raw_model_data.
    provenance["absent_sensors"] = {
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
    provenance["model_contract"] = {
        "pi05": True,
        "action_dim": ACTION_DIM,
        "action_horizon": HORIZON,
    }
    provenance["native_dt_s"] = DT_S
    provenance["pending_transforms"] = {name: "PENDING" for name in _PENDING_TRANSFORMS}
    provenance["unused_observation_keys"] = list(_UNUSED_OBSERVATION_KEYS)
    return Pi05NativeSample(
        raw_model_data=raw_model_data, loss_mask32=loss_mask32, provenance=provenance
    )
