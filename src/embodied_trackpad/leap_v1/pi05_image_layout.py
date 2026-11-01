"""Shared trace-safe image-layout dispatch for pi0.5 preprocessing.

Two static layouts select how the official OpenPI ``preprocess_observation``
picks image fields from an ``Observation``:

* ``"legacy"`` (default) — the pinned official default: three camera fields
  ``base_0_rgb`` / ``left_wrist_0_rgb`` / ``right_wrist_0_rgb``. The official
  call is made WITHOUT ``image_keys`` so legacy behavior is unchanged.
* ``"screen_pair"`` — the coordinator-frozen explicit two-screen layout
  (full screen view + native-pixel cursor-local view). Requires the exact
  ``SCREEN_PAIR_KEYS`` tuple in both ``images`` and ``image_masks``, in order;
  no wrist aliases, no default fallback, no extra keys.

:func:`preprocess_observation_for_layout` is safe to call inside ``jax.jit``
/ ``jax.value_and_grad`` with tracer values: it uses ONLY static Python
metadata (the image dict key tuples and the closed-over layout string) and
never ``np.asarray`` / ``np.isfinite`` / Python ``bool`` on array values.
Finite/value validation stays in the host-side builders
(``build_pi05_screen_observation`` etc.) before compilation.

No import-time OpenPI/JAX here: layout validation is pure Python so unknown
layouts are rejected BEFORE any heavy import.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SCREEN_PAIR_KEYS",
    "LEGACY_LAYOUT",
    "SCREEN_PAIR_LAYOUT",
    "validate_image_layout",
    "preprocess_observation_for_layout",
]

SCREEN_PAIR_KEYS = ("base_0_rgb", "screen_local_0_rgb")
LEGACY_LAYOUT = "legacy"
SCREEN_PAIR_LAYOUT = "screen_pair"
_VALID_LAYOUTS = (LEGACY_LAYOUT, SCREEN_PAIR_LAYOUT)


def validate_image_layout(image_layout: Any) -> str:
    """Return the normalized layout string; reject unknown layouts early.

    Pure Python (no heavy imports), so bad layouts fail before OpenPI/JAX are
    imported and before any compilation. Accepted values are exactly
    ``"legacy"`` and ``"screen_pair"``.
    """
    if not isinstance(image_layout, str) or image_layout not in _VALID_LAYOUTS:
        raise ValueError(
            f"image_layout must be one of {_VALID_LAYOUTS}, got {image_layout!r}"
        )
    return image_layout


def _require_exact_keys(mapping: Any, name: str) -> None:
    # Dict key access is static Python metadata (tracer-safe): Observation
    # images/image_masks are plain dicts of str -> array; only the KEY TUPLE
    # is inspected here, never array values.
    keys = tuple(mapping.keys())
    if keys != SCREEN_PAIR_KEYS:
        raise ValueError(
            f"{name} keys must be exactly {SCREEN_PAIR_KEYS} in order, got {keys}"
        )


def preprocess_observation_for_layout(
    rng: Any,
    observation: Any,
    *,
    train: bool,
    image_layout: str,
) -> Any:
    """Official OpenPI preprocessing with static layout dispatch.

    ``legacy`` calls the official default 3-camera preprocessing unchanged
    (no ``image_keys`` argument). ``screen_pair`` validates the exact two
    screen keys on ``images`` and ``image_masks`` (static metadata only) and
    calls the official function with the explicit ``SCREEN_PAIR_KEYS``.
    ``rng`` is passed through unchanged. Layout dispatch is a closed-over
    Python string, static under ``jax.jit``; unknown layouts raise at trace
    time via :func:`validate_image_layout`.
    """
    layout = validate_image_layout(image_layout)
    from openpi.models import model as _model  # noqa: PLC0415 - deferred heavy import

    if layout == SCREEN_PAIR_LAYOUT:
        _require_exact_keys(observation.images, "observation.images")
        _require_exact_keys(observation.image_masks, "observation.image_masks")
        return _model.preprocess_observation(
            rng, observation, train=train, image_keys=SCREEN_PAIR_KEYS
        )
    return _model.preprocess_observation(rng, observation, train=train)
