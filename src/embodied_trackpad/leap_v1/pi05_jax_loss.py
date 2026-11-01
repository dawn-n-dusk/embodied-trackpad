"""Valid-16 flow-matching loss for Pi0(pi05=True) full fine-tuning.

Two small pieces:

1. ``valid16_mse`` — NumPy/JAX-agnostic mean squared error over the first 16
   action channels of ``[..., 32]`` predictions/targets. The trailing 16
   channels are the padded region of the LEAP v1 32-channel action layout and
   must not contribute to the loss.

2. ``pi05_valid16_flow_loss`` — replicates the official OpenPI Pi0
   ``compute_loss`` forward path (rng split, observation preprocessing with
   ``train=False``, beta time sampling, noise/target mixing, prefix+suffix
   embedding, attention mask, PaliGemma forward, action projection) and reduces
   with ``valid16_mse(..., xp=jnp)`` instead of the official last-axis mean
   over all 32 channels.

Attribution: the flow body is adapted line-for-line from
Physical Intelligence OpenPI, Apache-2.0, commit
``215abfb217dbac7d5f1273282331b9b1866c0479``, ``Pi0.compute_loss``:
https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py
Modifications: GUI augmentation disabled; first 16 action dimensions supervised. JAX/OpenPI imports are deferred into
the function body so the pure-NumPy helper stays importable without them.
"""

from __future__ import annotations

import numpy as np

from embodied_trackpad.leap_v1.pi05_image_layout import (
    preprocess_observation_for_layout,
    validate_image_layout,
)

__all__ = ["valid16_mse", "pi05_valid16_flow_loss"]

_VALID16 = 16
_ACTION_DIM = 32


def valid16_mse(predicted, target, *, xp=np):
    """Mean squared error over the first 16 channels of ``[..., 32]`` arrays.

    Reduces only the last axis, preserving any batch/time axes. The padded
    trailing 16 channels are ignored, so arbitrarily large errors there do
    not affect the result. ``xp`` selects the array backend (``numpy`` or
    ``jax.numpy``). Raises ``ValueError`` on mismatched shapes or a last
    dimension other than 32.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    if predicted.shape[-1] != _ACTION_DIM:
        raise ValueError(f"expected last dimension {_ACTION_DIM}, got {predicted.shape[-1]}")
    diff = predicted[..., :_VALID16] - target[..., :_VALID16]
    return xp.mean(xp.square(diff), axis=-1)


def pi05_valid16_flow_loss(model, rng, observation, actions, *, image_layout="legacy"):
    """Official Pi0 flow-matching loss path, reduced over valid 16 channels.

    ``model`` must be a fully constructed official ``Pi0`` instance with
    ``pi05=True`` (32-dim action noise/horizon preserved from the model's
    ``action_dim``/``action_horizon``). The official ``compute_loss`` is not
    called; its body is reproduced here so the final reduction can be the
    valid-16 MSE instead of the official 32-channel mean. Suitable for use
    with ``nnx.value_and_grad``.

    ``image_layout`` selects preprocessing statically (closed-over Python
    string, safe under ``jax.jit``): ``"legacy"`` keeps the official default
    3-camera preprocessing exactly; ``"screen_pair"`` dispatches through the
    shared trace-safe layout module with the explicit two screen keys. The
    rest of the flow body, RNG splitting, and the valid-16 reduction are
    unchanged. Unknown layouts are rejected before any heavy import.
    """
    layout = validate_image_layout(image_layout)  # pure Python, pre-import

    import jax  # noqa: PLC0415 - deferred: heavy JAX/OpenPI import
    import jax.numpy as jnp  # noqa: PLC0415
    from openpi.models import pi0 as _pi0  # noqa: PLC0415

    # --- official Pi0.compute_loss body (train=False; no GUI augmentation) ---
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = preprocess_observation_for_layout(
        preprocess_rng, observation, train=False, image_layout=layout
    )

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = _pi0.make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])
    # --- end official body ---

    return valid16_mse(v_t, u_t, xp=jnp)
