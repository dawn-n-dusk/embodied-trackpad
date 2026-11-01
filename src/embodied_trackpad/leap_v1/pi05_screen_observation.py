"""Explicit two-screen-view pi0.5 Observation adapter (screen pair layout).

SOURCE-ONLY adapter for the paused pi0.5 CPU-restore line. This module does NOT
change the existing ``build_pi05_observation`` 3-camera layout (main + two
zero wrist views); it builds the coordinator-frozen SCREEN PAIR layout instead:

* ``SCREEN_PAIR_KEYS = ("base_0_rgb", "screen_local_0_rgb")`` — the full
  800x600 screen view (officially resized to 224x224 by the reused builder)
  plus a NATIVE-PIXEL 1:1 224x224 local view centred on the current cursor,
  produced by :func:`embodied_trackpad.leap_v1.screen_views.prepare_screen_views`.
  These are two explicit screen views, NOT wrist aliases; no real or zero
  wrist image slots exist in this layout.

:func:`build_pi05_screen_observation` takes exactly the same keyword args as
:func:`embodied_trackpad.leap_v1.pi05_observation.build_pi05_observation`
(``rgb_uint8, state16, actions, prompt, tokenizer_path`` and the fixed
optional pins) and REUSES that builder for validation, tokenizer asset hash
check, transform order (ResizeImages -> TokenizePrompt(discrete_state_input=
True) -> PadStatesAndActions), full-RGB handling and padding. Only after the
old builder validates does any heavy OpenPI/sentencepiece import happen (the
heavy imports live inside the reused builder and in
:func:`preprocess_pi05_screen_observation`, never at module import).

The returned official Flax ``Observation`` is derived from the old one with
``dataclasses.replace`` (works on the frozen Flax struct): the image dict is
replaced by exactly the two screen keys, the local uint8 view is scaled with
the official ``/255*2-1`` rule (the rule ``Observation.from_dict`` applies to
uint8), and state/tokens/actions/loss-mask values are preserved unchanged. The
local view is always a fresh array; carried-over arrays are copied so no
returned tensor aliases another bundle or an input. When the cursor is
missing/spread-out/low-confidence, the local view is official-scaling zeros
(=-1) with a False mask; the full view stays valid. The image-padding mask
``local_pixel_valid`` (real vs zero-padded pixels) is carried in diagnostics
— it is NOT an attention mask.

CRITICAL limitation: the existing valid-16 loss (``pi05_jax_loss``) and the
upstream ``sample_actions`` still invoke the DEFAULT 3-camera preprocessing
and therefore CANNOT be used with this layout. This module only provides the
accepted Observation construction plus the explicit preprocessing entry
:func:`preprocess_pi05_screen_observation`; routing the loss/sampler through
it is a separate, later reviewed change. No parameter counts, weights,
training or compute are touched here.

HOST-SIDE ONLY: :func:`preprocess_pi05_screen_observation` is host-side
admission + preprocessing. It uses ``np.asarray`` / ``np.isfinite`` and Python
``bool`` on concrete values, so it is NOT callable inside ``jax.jit`` /
``jax.value_and_grad`` with tracer values. A future compiled-loss route needs
static layout dispatch (choosing this screen-pair layout outside the traced
function) plus a different, trace-safe inner preprocessing; the legacy
loss/core/sampler are intentionally left unchanged in this task.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1.pi05_observation import (
    DEFAULT_MAX_TOKEN_LEN,
    IMAGE_RESOLUTION,
    MAIN_VIEW_KEY,
    PINNED_TOKENIZER_SHA256,
    Pi05ObservationBundle,
    _checked,
    build_pi05_observation,
)
from embodied_trackpad.leap_v1.screen_views import (
    LOCAL_SIZE,
    prepare_screen_views,
)

# Explicit screen-pair layout: full screen view + native-pixel cursor-local
# view. NOT wrist aliases; no wrist keys exist in this layout.
SCREEN_PAIR_KEYS = (MAIN_VIEW_KEY, "screen_local_0_rgb")
_LOCAL_KEY = SCREEN_PAIR_KEYS[1]
_WRIST_PROVENANCE_KEYS = ("left_wrist_0_rgb", "right_wrist_0_rgb")

__all__ = [
    "SCREEN_PAIR_KEYS",
    "build_pi05_screen_observation",
    "preprocess_pi05_screen_observation",
]


def _scale_uint8_to_float1(image: np.ndarray) -> np.ndarray:
    """Official uint8 -> [-1, 1] float32 scaling (Observation.from_dict rule)."""
    return image.astype(np.float32) / 255.0 * 2.0 - 1.0


def build_pi05_screen_observation(
    *,
    rgb_uint8: Any,
    state16: Any,
    actions: Any,
    prompt: str,
    tokenizer_path: Any,
    expected_sha256: str = PINNED_TOKENIZER_SHA256,
    image_size: tuple[int, int] = IMAGE_RESOLUTION,
    model_action_dim: int = 32,
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
) -> Pi05ObservationBundle:
    """Build the batch-size-1 official Observation in the screen-pair layout.

    Same args and pins as ``build_pi05_observation``. The old builder is
    reused unchanged for validation, tokenizer hash, transform order and
    padding; its Observation is then rewritten via ``dataclasses.replace`` to
    the two explicit screen views. Every field the new Observation keeps is
    freshly copied — images, masks, state, ``tokenized_prompt``,
    ``tokenized_prompt_mask`` and, when present and not None, the optional
    ``token_ar_mask`` / ``token_loss_mask`` — so nothing aliases the discarded
    base bundle or the caller's inputs (values unchanged). Diagnostics keep
    the old transform provenance explicitly separate under ``base_builder``,
    replace ``image_mask`` / ``image_resized_shape`` / ``image_output_shape``
    / ``image_output_dtype`` with exactly the two screen keys taken from the
    actual new Observation (the prior wrist masks are recorded under
    ``absent_sensor_masks`` as provenance of absent sensors, not renamed as
    valid views), and add ``view_roles`` / ``view_order`` / ``view_valid`` /
    crop-origin / cursor-confidence plus the ``local_pixel_valid``
    image-padding mask. Legacy valid-16 loss / upstream sampling cannot
    consume this layout.
    """
    # Import-time light; all validation (incl. tokenizer hash) happens inside
    # the reused builder BEFORE any heavy OpenPI/sentencepiece import.
    base = build_pi05_observation(
        rgb_uint8=rgb_uint8,
        state16=state16,
        actions=actions,
        prompt=prompt,
        tokenizer_path=tokenizer_path,
        expected_sha256=expected_sha256,
        image_size=image_size,
        model_action_dim=model_action_dim,
        max_token_len=max_token_len,
    )
    old = base.observation

    # Native-pixel local view from the CURRENT RGB only (no GUI/oracle truth).
    views = prepare_screen_views(np.asarray(rgb_uint8))
    local_scaled = _scale_uint8_to_float1(views.local_rgb)[None, ...]  # [1,224,224,3]

    # Fresh dicts/arrays only: dataclasses.replace alone would keep the
    # discarded base bundle's state/token arrays alive by reference, so every
    # carried field is copied explicitly (values unchanged). Works on the
    # official frozen Flax struct Observation.
    new_images = {
        SCREEN_PAIR_KEYS[0]: np.array(old.images[MAIN_VIEW_KEY], copy=True),
        _LOCAL_KEY: np.ascontiguousarray(local_scaled),
    }
    new_masks = {
        SCREEN_PAIR_KEYS[0]: np.array(old.image_masks[MAIN_VIEW_KEY], copy=True),
        _LOCAL_KEY: np.asarray([bool(views.local_valid)]),
    }
    replace_kwargs: dict[str, Any] = {
        "images": new_images,
        "image_masks": new_masks,
        "state": np.array(old.state, copy=True),
        "tokenized_prompt": np.array(old.tokenized_prompt, copy=True),
        "tokenized_prompt_mask": np.array(old.tokenized_prompt_mask, copy=True),
    }
    for _optional in ("token_ar_mask", "token_loss_mask"):
        _value = getattr(old, _optional, None)
        if _value is not None:
            replace_kwargs[_optional] = np.array(_value, copy=True)
    observation = dataclasses.replace(old, **replace_kwargs)

    actions_b32 = np.array(base.actions, dtype=np.float32, copy=True)
    loss_mask32 = np.array(base.loss_mask32, dtype=bool, copy=True)

    diagnostics = dict(base.diagnostics)
    # Final-image diagnostics must describe the ACTUAL new two-view
    # Observation (exactly the two screen keys, valid or invalid local view);
    # the historical 3-camera transform outputs stay available, explicitly
    # separated under "base_builder".
    diagnostics["base_builder"] = {
        "image_resized_shape": dict(base.diagnostics["image_resized_shape"]),
        "image_output_shape": dict(base.diagnostics["image_output_shape"]),
        "image_output_dtype": dict(base.diagnostics["image_output_dtype"]),
        "transform_order": list(base.diagnostics["transform_order"]),
    }
    diagnostics["image_resized_shape"] = {
        SCREEN_PAIR_KEYS[0]: tuple(base.diagnostics["image_resized_shape"][MAIN_VIEW_KEY]),
        # Native-pixel crop from screen_views: never resized.
        _LOCAL_KEY: tuple(views.local_rgb.shape),
    }
    diagnostics["image_resized_shape_note"] = (
        f"{_LOCAL_KEY} is a native-pixel screen_views crop; no resize applied"
    )
    diagnostics["image_output_shape"] = {
        k: tuple(np.asarray(v).shape) for k, v in new_images.items()
    }
    diagnostics["image_output_dtype"] = {
        k: str(np.asarray(v).dtype) for k, v in new_images.items()
    }
    # The old 3-key image_mask would falsify this layout: publish exactly the
    # two screen keys; wrist absence is preserved separately as provenance.
    diagnostics["image_mask"] = {
        SCREEN_PAIR_KEYS[0]: bool(np.asarray(new_masks[SCREEN_PAIR_KEYS[0]]).all()),
        _LOCAL_KEY: bool(views.local_valid),
    }
    diagnostics["absent_sensor_masks"] = {k: False for k in _WRIST_PROVENANCE_KEYS}
    diagnostics["layout"] = "explicit_two_screen_views_no_wrist"
    diagnostics["screen_pair_keys"] = list(SCREEN_PAIR_KEYS)
    diagnostics["view_roles"] = {
        SCREEN_PAIR_KEYS[0]: "screen_full",
        _LOCAL_KEY: "screen_local",
    }
    diagnostics["view_order"] = list(SCREEN_PAIR_KEYS)
    diagnostics["view_valid"] = {
        SCREEN_PAIR_KEYS[0]: True,
        _LOCAL_KEY: bool(views.local_valid),
    }
    diagnostics["crop_origin_col_row"] = views.origin_col_row
    diagnostics["cursor_col_row"] = views.cursor_col_row
    diagnostics["cursor_confidence"] = float(views.cursor_confidence)
    # Image-padding mask (real vs zero-padded local pixels), NOT an attention
    # mask; kept outside the official Observation by design.
    diagnostics["local_pixel_valid"] = np.array(views.local_pixel_valid, copy=True)
    diagnostics["local_pixel_valid_meaning"] = "image-padding mask, not attention mask"
    diagnostics["local_scaling"] = "uint8->float32 /255*2-1 (official from_dict rule)"

    return Pi05ObservationBundle(
        observation=observation,
        actions=actions_b32,
        loss_mask32=loss_mask32,
        diagnostics=diagnostics,
    )


def _validate_pair_dict(
    mapping: Any,
    name: str,
) -> "dict[str, Any]":
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a dict, got {type(mapping).__name__}")
    keys = tuple(mapping.keys())
    if keys != SCREEN_PAIR_KEYS:
        raise ValueError(
            f"{name} keys must be exactly {SCREEN_PAIR_KEYS} in order, got {keys}"
        )
    return mapping


def preprocess_pi05_screen_observation(observation: Any) -> Any:
    """Run the official OpenPI preprocessing on a screen-pair Observation.

    Strictly validates the exact ``SCREEN_PAIR_KEYS`` image/mask key set and
    order (legacy wrist keys, unknown, extra or missing keys are REFUSED, not
    silently dropped), image shapes ``[1,224,224,3]`` float32 finite in
    ``[-1,1]``, bool masks of shape ``[1]`` with the full view valid, state
    ``[1,32]`` and fixed tokens ``[1,200]``. Only then invokes the official
    ``openpi.models.model.preprocess_observation`` with ``rng=None``,
    ``train=False`` and ``image_keys=SCREEN_PAIR_KEYS`` — never the default
    camera keys, never augmentation, no GUI state. Requires the heavy OpenPI
    runtime; the import is deferred until after validation.

    HOST-SIDE ONLY: this is admission + preprocessing on concrete host values
    (``np.asarray`` / ``np.isfinite`` / Python ``bool``) and is NOT callable
    inside ``jax.jit`` or ``jax.value_and_grad`` with tracer values. The
    future compiled-loss route needs static layout dispatch outside the trace
    plus a different, trace-safe inner preprocessing; the legacy
    loss/core/sampler are unchanged by this task.
    """
    images = _validate_pair_dict(getattr(observation, "images", None), "observation.images")
    image_masks = _validate_pair_dict(
        getattr(observation, "image_masks", None), "observation.image_masks"
    )
    expected_img_shape = (1,) + IMAGE_RESOLUTION + (3,)
    for key in SCREEN_PAIR_KEYS:
        img = np.asarray(images[key])
        if img.shape != expected_img_shape:
            raise ValueError(f"images[{key!r}] must have shape {expected_img_shape}, got {img.shape}")
        if img.dtype != np.float32:
            raise ValueError(f"images[{key!r}] must have dtype float32, got {img.dtype}")
        if not np.isfinite(img).all():
            raise ValueError(f"images[{key!r}] must be finite")
        if img.min() < -1.0 or img.max() > 1.0:
            raise ValueError(f"images[{key!r}] must lie in [-1, 1]")
        mask = np.asarray(image_masks[key])
        if mask.shape != (1,) or mask.dtype != bool:
            raise ValueError(f"image_masks[{key!r}] must be bool of shape (1,), got {mask.shape} {mask.dtype}")
    if not bool(np.asarray(image_masks[SCREEN_PAIR_KEYS[0]])[0]):
        raise ValueError(f"image_masks[{SCREEN_PAIR_KEYS[0]!r}] must be True (full view always valid)")
    _checked(np.asarray(observation.state), "state", (1, 32), np.dtype(np.float32))
    tokens = np.asarray(observation.tokenized_prompt)
    if tokens.shape != (1, DEFAULT_MAX_TOKEN_LEN) or tokens.dtype.kind not in "iu":
        raise ValueError(
            f"tokenized_prompt must have shape (1, {DEFAULT_MAX_TOKEN_LEN}) and integer dtype, got {tokens.shape} {tokens.dtype}"
        )
    token_mask = np.asarray(observation.tokenized_prompt_mask)
    if token_mask.shape != (1, DEFAULT_MAX_TOKEN_LEN) or token_mask.dtype != bool:
        raise ValueError(
            f"tokenized_prompt_mask must be bool of shape (1, {DEFAULT_MAX_TOKEN_LEN}), got {token_mask.shape} {token_mask.dtype}"
        )

    # Deferred heavy import: official preprocessing with the EXPLICIT screen
    # image keys only. No default camera keys, no augmentation (train=False),
    # no model weights import/init here.
    from openpi.models import model as _model  # noqa: PLC0415

    return _model.preprocess_observation(
        rng=None, observation=observation, train=False, image_keys=SCREEN_PAIR_KEYS
    )
