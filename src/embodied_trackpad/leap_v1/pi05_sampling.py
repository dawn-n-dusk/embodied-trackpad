"""Official Pi0 ``sample_actions`` adaptation with static image-layout routing.

:func:`pi05_sample_actions` faithfully reproduces the pinned OpenPI
``Pi0.sample_actions`` Euler-integration sampler, replacing ONLY the
observation preprocessing step with the shared trace-safe layout dispatch
(:mod:`embodied_trackpad.leap_v1.pi05_image_layout`). The upstream
``sample_actions`` is NOT called (it would apply the default 3-camera
preprocessing again) and nothing is monkeypatched.

Preserved from upstream: prefix KV-cache fill, attention masks and
positions, suffix adarms conditioning, noise initialization (or the
caller-provided noise), ``dt = -1.0 / num_steps``, the
``jax.lax.while_loop`` with threshold ``time >= -dt / 2`` (robust to
floating-point error), and the full 32-channel action output. The sampler
does NOT beta-sample time (only the training loss draws
``jax.random.beta``); time starts at 1.0 (noise) and integrates toward 0
(target) with negative ``dt``.

``num_steps`` must be a positive Python ``int`` (``bool`` rejected) and is
static under JIT via closure, together with the closed-over layout string;
both must be fixed before compilation. No control application happens here.

Local modifications vs upstream: preprocessing routed through the shared
layout module (``legacy`` = official default unchanged; ``screen_pair`` =
explicit two screen keys); the ``einops.repeat`` of the prefix mask is
written as ``jnp.broadcast_to`` (numerically identical) to keep the module
importable without einops. Everything else follows upstream line-for-line.

Attribution: adapted from Physical Intelligence OpenPI, Apache-2.0, commit
``215abfb217dbac7d5f1273282331b9b1866c0479``, ``Pi0.sample_actions``:
https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py
"""

from __future__ import annotations

from typing import Any

from embodied_trackpad.leap_v1.pi05_image_layout import (
    preprocess_observation_for_layout,
    validate_image_layout,
)

__all__ = ["pi05_sample_actions"]


def pi05_sample_actions(
    model: Any,
    rng: Any,
    observation: Any,
    *,
    num_steps: int = 10,
    noise: Any = None,
    image_layout: str = "legacy",
):
    """Euler-integrate the Pi0 action flow with static layout preprocessing.

    ``model`` is a fully constructed official ``Pi0`` (pi05=True). Returns
    the full ``[batch, action_horizon, 32]`` action array. ``image_layout``
    and ``num_steps`` are static Python values closed over at trace time;
    value/finite validation of the observation stays in the host-side
    builder before compilation.
    """
    layout = validate_image_layout(image_layout)  # pure Python, pre-import
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive Python int, got {num_steps!r}")

    import jax  # noqa: PLC0415 - deferred: heavy JAX/OpenPI import
    import jax.numpy as jnp  # noqa: PLC0415
    from openpi.models import pi0 as _pi0  # noqa: PLC0415

    make_attn_mask = _pi0.make_attn_mask

    # --- official Pi0.sample_actions body (preprocessing replaced only) ---
    # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    observation = preprocess_observation_for_layout(None, observation, train=False, image_layout=layout)
    dt = -1.0 / num_steps
    batch_size = observation.state.shape[0]
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

    # first fill KV cache with a forward pass of the prefix
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    def step(carry):
        x_t, time = carry
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
        # other
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
        # prefix tokens
        prefix_attn_mask = jnp.broadcast_to(
            prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1])
        )
        # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
        # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        assert full_attn_mask.shape == (
            batch_size,
            suffix_tokens.shape[1],
            prefix_tokens.shape[1] + suffix_tokens.shape[1],
        )
        # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

        (prefix_out, suffix_out), _ = model.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        assert prefix_out is None
        v_t = model.action_out_proj(suffix_out[:, -model.action_horizon :])

        return x_t + dt * v_t, time + dt

    def cond(carry):
        x_t, time = carry
        # robust to floating-point error
        return time >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
