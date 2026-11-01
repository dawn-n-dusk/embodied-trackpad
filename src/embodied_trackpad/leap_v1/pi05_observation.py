"""Official pi0.5 Observation adapter for one accepted LEAP v1 training sample.

Turns one already-accepted native sample (uint8 RGB[600,800,3] main view,
already quantile-normalized float32 state[16], already normalized float32
actions[25,16], fixed nonempty prompt) into the official batch-size-1 OpenPI
``Observation`` plus ``actions[1,25,32]`` and a separate ``loss_mask32``
boolean mask (first 16 true, rest false; kept outside the Observation, aligned
with the accepted native pack and the valid-16 loss API).

Pipeline order is fixed by the accepted contract and mirrors the task doc:
official ``ResizeImages(224,224)`` -> ``TokenizePrompt(discrete_state_input=True)``
on the normalized 16-dim state (BEFORE any 32-dim padding) ->
``PadStatesAndActions(32)`` -> ``Observation.from_dict`` with the official
uint8 -> [-1,1] float32 scaling. Inputs arrive already normalized and are
never normalized again. No augmentations, no oracles. Missing wrist views are
real zero images with false masks; only the main screen view is valid. Inputs
are validated for shape/dtype/finiteness and are never mutated.

Tokenizer asset policy: no implicit downloads. ``LocalPaligemmaTokenizer``
subclasses the official ``PaligemmaTokenizer`` and only overrides
``__init__`` (max_len=200, ``SentencePieceProcessor`` from verified local
bytes); the official ``tokenize`` is inherited unchanged. Heavy imports
(openpi/sentencepiece) are deferred into function bodies so this module
imports without them, and all pure validation (including the tokenizer asset
SHA256 check) happens before any heavy import.

Attribution: pipeline semantics adapted from Physical Intelligence OpenPI,
Apache-2.0, commit ``215abfb217dbac7d5f1273282331b9b1866c0479``,
``src/openpi/transforms.py`` (ResizeImages, TokenizePrompt,
PadStatesAndActions), ``src/openpi/models/model.py`` (Observation.from_dict)
and ``src/openpi/models/tokenizer.py`` (PaligemmaTokenizer). Local
modifications: deferred imports, local-asset tokenizer subclass, explicit
input validation, valid-16 loss mask plumbing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
from typing import Any

import numpy as np

__all__ = [
    "PINNED_TOKENIZER_SHA256",
    "DEFAULT_MAX_TOKEN_LEN",
    "IMAGE_RESOLUTION",
    "MODEL_ACTION_DIM",
    "MAIN_VIEW_KEY",
    "WRIST_VIEW_KEYS",
    "STATE_DIM",
    "ACTION_HORIZON",
    "Pi05ObservationBundle",
    "build_pi05_observation",
]

# SHA256 of the pinned official gs://big_vision/paligemma_tokenizer.model asset.
PINNED_TOKENIZER_SHA256 = (
    "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
)
DEFAULT_MAX_TOKEN_LEN = 200
IMAGE_RESOLUTION = (224, 224)
MODEL_ACTION_DIM = 32
STATE_DIM = 16
ACTION_HORIZON = 25
MAIN_VIEW_KEY = "base_0_rgb"
WRIST_VIEW_KEYS = ("left_wrist_0_rgb", "right_wrist_0_rgb")
_RGB_SHAPE = (600, 800, 3)


@dataclasses.dataclass(frozen=True)
class Pi05ObservationBundle:
    """Batch-size-1 official Observation plus padded actions and loss mask."""

    observation: Any  # official ``Observation`` (batch 1); type depends on openpi
    actions: np.ndarray  # float32 [1, 25, 32]; trailing 16 channels zero-padded
    loss_mask32: np.ndarray  # bool [32]; first 16 true, rest false; outside Observation
    diagnostics: dict[str, Any]  # input-transform diagnostics (see builder)


def _local_tokenizer_class():
    """Build (once) the local-asset ``PaligemmaTokenizer`` subclass.

    Defined lazily so importing this module never triggers the heavy
    openpi/sentencepiece imports; the official ``tokenize`` is inherited
    unchanged and only ``__init__`` is overridden.
    """
    cached = globals().get("_LocalPaligemmaTokenizer")
    if cached is not None:
        return cached

    import sentencepiece  # noqa: PLC0415 - deferred heavy import
    from openpi.models import tokenizer as _tokenizer  # noqa: PLC0415

    class LocalPaligemmaTokenizer(_tokenizer.PaligemmaTokenizer):
        """Official PaliGemma tokenizer initialized from verified local bytes.

        The official constructor hardcodes a ``gs://`` download; this subclass
        replaces only initialization: ``_max_len`` is set to the pi0.5 budget
        and the ``SentencePieceProcessor`` is built from asset bytes that the
        caller already SHA256-verified against the pinned official asset hash.
        No download, no global monkeypatching, no copied tokenize logic.
        """

        def __init__(
            self,
            verified_model_proto: bytes,
            max_len: int = DEFAULT_MAX_TOKEN_LEN,
        ) -> None:
            self._max_len = max_len
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=verified_model_proto
            )

    globals()["_LocalPaligemmaTokenizer"] = LocalPaligemmaTokenizer
    return LocalPaligemmaTokenizer


def _checked(value: Any, name: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(arr.shape)}")
    if arr.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def build_pi05_observation(
    *,
    rgb_uint8: Any,
    state16: Any,
    actions: Any,
    prompt: str,
    tokenizer_path: str | pathlib.Path,
    expected_sha256: str = PINNED_TOKENIZER_SHA256,
    image_size: tuple[int, int] = IMAGE_RESOLUTION,
    model_action_dim: int = MODEL_ACTION_DIM,
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
) -> Pi05ObservationBundle:
    """Build the batch-size-1 official pi0.5 Observation from one native sample.

    Args:
        rgb_uint8: uint8 RGB ``[600, 800, 3]`` main screen view (unchanged).
        state16: already quantile-normalized float32 ``[16]`` (unchanged,
            never normalized again).
        actions: already normalized float32 ``[25, 16]`` (unchanged, never
            normalized again).
        prompt: fixed nonempty prompt string.
        tokenizer_path: local path to the pinned official tokenizer asset;
            its SHA256 must match ``expected_sha256`` (no implicit download).
        expected_sha256: expected tokenizer asset hash (defaults to the pin).
        image_size: must be the smoke-contract ``(224, 224)``.
        model_action_dim: must be the smoke-contract ``32``.
        max_token_len: must be the smoke-contract ``200``.

    Returns:
        ``Pi05ObservationBundle`` with the official ``Observation`` (batch 1),
        ``actions[1, 25, 32]`` and the separate ``loss_mask32`` mask, plus
        input-transform diagnostics. Inputs are never mutated.

    All validation — including the tokenizer asset hash — completes before
    any heavy OpenPI/sentencepiece import, so bad inputs fail fast and light.
    """
    # --- pure validation phase (no heavy imports) ---
    rgb = _checked(rgb_uint8, "rgb_uint8", _RGB_SHAPE, np.dtype(np.uint8))
    state = _checked(state16, "state16", (STATE_DIM,), np.dtype(np.float32))
    act = _checked(actions, "actions", (ACTION_HORIZON, STATE_DIM), np.dtype(np.float32))
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if tuple(image_size) != IMAGE_RESOLUTION:
        raise ValueError(f"image_size must be the smoke-contract {IMAGE_RESOLUTION}, got {image_size}")
    if model_action_dim != MODEL_ACTION_DIM:
        raise ValueError(f"model_action_dim must be {MODEL_ACTION_DIM}, got {model_action_dim}")
    if max_token_len != DEFAULT_MAX_TOKEN_LEN:
        raise ValueError(f"max_token_len must be {DEFAULT_MAX_TOKEN_LEN}, got {max_token_len}")
    tokenizer_file = pathlib.Path(tokenizer_path)
    if not tokenizer_file.is_file():
        raise ValueError(f"tokenizer asset not found: {tokenizer_file}")
    model_proto = tokenizer_file.read_bytes()
    actual_sha256 = hashlib.sha256(model_proto).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "tokenizer asset SHA256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256} "
            f"for {tokenizer_file}"
        )

    # --- heavy phase: official transforms in the accepted order ---
    from openpi import transforms as _transforms  # noqa: PLC0415 - deferred heavy import
    from openpi.models import model as _model  # noqa: PLC0415

    tokenizer = _local_tokenizer_class()(model_proto, max_token_len)

    # Unbatched transform dict; tokenization consumes the normalized 16-dim
    # state BEFORE PadStatesAndActions zero-pads state/actions to 32.
    data: dict[str, Any] = {
        "image": {
            MAIN_VIEW_KEY: np.array(rgb, copy=True),
            WRIST_VIEW_KEYS[0]: np.zeros(_RGB_SHAPE, dtype=np.uint8),
            WRIST_VIEW_KEYS[1]: np.zeros(_RGB_SHAPE, dtype=np.uint8),
        },
        "image_mask": {
            MAIN_VIEW_KEY: np.asarray(True),
            WRIST_VIEW_KEYS[0]: np.asarray(False),
            WRIST_VIEW_KEYS[1]: np.asarray(False),
        },
        "state": np.array(state, copy=True),
        "actions": np.array(act, copy=True),
        "prompt": prompt,
    }

    height, width = IMAGE_RESOLUTION
    data = _transforms.ResizeImages(height, width)(data)
    resized_shape = {k: tuple(v.shape) for k, v in data["image"].items()}
    data = _transforms.TokenizePrompt(tokenizer, discrete_state_input=True)(data)
    tokens = np.asarray(data["tokenized_prompt"])
    token_mask = np.asarray(data["tokenized_prompt_mask"])
    data = _transforms.PadStatesAndActions(MODEL_ACTION_DIM)(data)

    # Batch-size-1 expansion, then official dict -> Observation mapping with
    # the official uint8 -> [-1, 1] float32 image scaling.
    batched: dict[str, Any] = {
        "image": {k: v[None, ...] for k, v in data["image"].items()},
        "image_mask": {k: np.asarray(v)[None] for k, v in data["image_mask"].items()},
        "state": np.asarray(data["state"])[None, ...],
        "actions": np.asarray(data["actions"])[None, ...],
        "tokenized_prompt": tokens[None, ...],
        "tokenized_prompt_mask": token_mask[None, ...],
    }
    observation = _model.Observation.from_dict(batched)

    actions_b32 = np.array(batched["actions"], dtype=np.float32, copy=True)
    loss_mask32 = np.zeros(MODEL_ACTION_DIM, dtype=bool)
    loss_mask32[:STATE_DIM] = True

    diagnostics: dict[str, Any] = {
        "image_input_shape": _RGB_SHAPE,
        "image_resized_shape": resized_shape,
        "image_output_shape": {k: tuple(v.shape) for k, v in batched["image"].items()},
        "image_output_dtype": {k: str(v.dtype) for k, v in batched["image"].items()},
        "token_len": int(tokens.shape[-1]),
        "token_mask_true": int(np.count_nonzero(token_mask)),
        "state_dim_in": STATE_DIM,
        "state_dim_padded": int(batched["state"].shape[-1]),
        "action_dim_in": STATE_DIM,
        "action_dim_padded": int(actions_b32.shape[-1]),
        "tokenizer_sha256": actual_sha256,
        "image_mask": {k: bool(np.asarray(v).all()) for k, v in batched["image_mask"].items()},
        "transform_order": [
            "ResizeImages",
            "TokenizePrompt(discrete_state_input=True)",
            "PadStatesAndActions",
            "Observation.from_dict",
        ],
    }

    return Pi05ObservationBundle(
        observation=observation,
        actions=actions_b32,
        loss_mask32=loss_mask32,
        diagnostics=diagnostics,
    )
