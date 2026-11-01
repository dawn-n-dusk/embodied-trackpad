"""Metadata-only abstract update-state template for CPU checkpoint restore.

``build_abstract_update_state(create_model, seed=17)`` produces the exact
``jax.ShapeDtypeStruct`` template that ``pi05_checkpoint.restore_state``
needs, WITHOUT loading pretrained weights, allocating random full-size
tensors, or running any forward/backward pass. It traces the whole accepted
build path once under ``nnx.eval_shape``:

1. ``create_model(init_key)`` constructs the caller's model exactly as the
   pinned OpenPI load does (``cfg.create(rng) -> Pi0(cfg, rngs=nnx.Rngs(rng))``);
   under shape evaluation only abstract tracers exist.
2. ``nnx.split`` separates graphdef from state; every ``nnx.Param`` leaf is
   explicitly cast to FP32 to mirror the original pretrained FP32 load, and
   the untouched graphdef is merged back, preserving the full NNX static
   metadata/treedef.
3. ``pi05_update_core.build_full_update(model, seed=seed)`` builds the same
   four-key state (``params``/``opt_state``/``step``/``rng``) it builds for
   the real model; only ``state`` is returned (the compiled step function and
   census are dropped, not traced out).

The returned abstract state passes the template contract of
``pi05_checkpoint.restore_state`` (exactly four top-level keys, every leaf a
``jax.ShapeDtypeStruct``), so a future CPU-only allocation can restore the
preserved step-1 checkpoint and re-run the unchanged
``pi05_checkpoint.fingerprint_state`` full hash against it. This helper is
shape-only: it does not itself pin a backend, load weights, or run the model;
the only concrete array it ever creates is the tiny ``jax.random.key(0)``
trace seed, no full-size arrays are materialized, and CPU-only execution is
enforced by the caller's reviewed allocation, not by this module. If shape
tracing fails, the original exception propagates; nothing is retried and no
fallback allocation happens.

Pinned runtime APIs (source-attributed, matching pi05_checkpoint):
- JAX 0.5.3: ``jax.random.key``/``jax.tree.leaves``, ``jax.ShapeDtypeStruct``.
- Flax 0.10.2: ``nnx.eval_shape``, ``nnx.split``/``nnx.merge``,
  ``nnx.VariableState`` (``.type``/``.replace``), ``nnx.Param``.

Imports are deferred into the function body so the module imports on the
foundation-only runtime and fail-closed validation runs before any heavy
import. Caller-owned checks (e.g. 51 leaves / 3,353,433,872 params for the
full model, or the tiny-model equivalents) stay with the caller; this helper
stays generic.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["build_abstract_update_state"]

_STATE_KEYS = ("params", "opt_state", "step", "rng")


def _cast_param_leaves_fp32(state: Any) -> Any:
    """Cast every nnx.Param VariableState leaf to FP32, preserving metadata."""
    import jax.numpy as jnp  # noqa: PLC0415 - deferred heavy import
    from flax import nnx  # noqa: PLC0415

    def _cast(leaf: Any) -> Any:
        if isinstance(leaf, nnx.VariableState) and issubclass(leaf.type, nnx.Param):
            return leaf.replace(value=jnp.asarray(leaf.value, dtype=jnp.float32))
        return leaf

    import jax  # noqa: PLC0415

    # VariableState is itself a registered pytree node in pinned Flax 0.10.2, so
    # without an is_leaf predicate jax.tree.map would visit the inner .value
    # arrays and the isinstance check would never cast. Treat VariableState as
    # an opaque leaf so .replace preserves all variable metadata.
    return jax.tree.map(_cast, state, is_leaf=lambda x: isinstance(x, nnx.VariableState))


def _require_abstract_state(state: Any) -> None:
    """Fail closed: exactly the four accepted keys, all leaves abstract."""
    keys = tuple(state.keys()) if isinstance(state, dict) else ()
    if len(keys) != 4 or set(keys) != set(_STATE_KEYS):
        raise ValueError(f"abstract state must have exactly the four keys {_STATE_KEYS}, got {keys}")
    import jax  # noqa: PLC0415

    for leaf in jax.tree.leaves(state):
        if not isinstance(leaf, jax.ShapeDtypeStruct):
            raise ValueError(
                f"abstract state leaf is {type(leaf).__name__}, expected jax.ShapeDtypeStruct"
            )


def build_abstract_update_state(create_model: Callable[[Any], Any], seed: int = 17) -> Any:
    """Return the abstract four-key update state for ``create_model`` via shape tracing.

    ``create_model`` is a callable taking one PRNG key and returning a fully
    constructed NNX model (factory around the pinned OpenPI ``cfg.create``).
    Only shape/dtype/treedef information is computed; no weights are loaded
    and no full-size arrays are materialized. Raises on any tracing failure
    with no fallback.
    """
    if not callable(create_model):
        raise TypeError(f"create_model must be callable, got {type(create_model).__name__}")

    import jax  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415

    from embodied_trackpad.leap_v1 import pi05_update_core as core  # noqa: PLC0415

    def _build(init_key: Any) -> Any:
        model = create_model(init_key)
        graphdef, state = nnx.split(model)
        state = _cast_param_leaves_fp32(state)
        model = nnx.merge(graphdef, state)
        state_out, _step_fn, _census = core.build_full_update(model, seed=seed)
        return state_out

    abstract_state = nnx.eval_shape(_build, jax.random.key(0))
    _require_abstract_state(abstract_state)
    return abstract_state
