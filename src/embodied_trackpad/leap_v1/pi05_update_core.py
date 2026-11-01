"""Full-parameter single-step optimizer core for an already-loaded Pi0 (pi05=True).

``build_full_update(model, seed=17)`` takes a fully constructed official NNX
``Pi0`` (full pretrained weights already restored by the caller) and returns
``(state, step_fn, census)``:

- ``state``: plain dict ``{'params': nnx.State, 'opt_state': optax state,
  'step': int32 scalar, 'rng': PRNG key}``. The model ``graphdef`` and the
  optax ``tx`` live in step-function closures. The state dict is a pytree and
  is donated to each compiled step (callers must not reuse the old dict).
  Every leaf of ``params`` must be a ``nnx.Param``-typed ``nnx.VariableState``
  (verified at build against the pinned Flax 0.10.2 ``State.flat_state()``
  mapping); anything else is rejected. Every leaf must also be float32 and
  smaller than 2**31 elements so per-leaf int32 non-finite counts cannot
  overflow.
- ``step_fn``: ``jax.jit``-compiled ``(state, observation, actions) ->
  (new_state, metrics)`` with ``donate_argnums=(0,)``, following the pinned
  official OpenPI train-step pattern: ``model = nnx.merge(graphdef,
  state['params'])``, ``nnx.value_and_grad(loss_fn,
  argnums=nnx.DiffState(0, nnx.Param))(model, step_rng)`` with the immutable
  ``pi05_valid16_flow_loss`` (scalar loss = mean over the [1, 25] valid-16
  loss). Grad paths/shapes are verified against static build-time metadata.
  One AdamW update (lr=1e-5, b1=.9, b2=.95, eps=1e-8, weight_decay=0,
  global-norm clip 1.0) is applied functionally to the state dict with optax —
  no ``nnx.update`` model mutation inside ``lax.cond`` and no second retained
  model. One RNG split is stored per accepted step. No EMA, LoRA, freezing,
  or dtype reduction; params/opt_state stay FP32.
- ``census``: static Python dict with the full-tree parameter count (Python
  int, no int32 cast), leaf count, all-fp32 flag, and per-leaf
  path/shape/dtype entries.

Non-finite scalar loss, grad global norm, or any gradient element rejects the
step: ``jax.lax.cond`` returns the old params/opt_state/step/RNG unchanged with
``accepted=0`` and zero update/delta norms. Metrics are on-device scalars only:
loss, accepted, grad_norm, update_norm, param_delta_norm, step, plus per-leaf
``grad_norms`` and int32 ``grad_nonfinite_counts`` keyed by static param path
(each leaf reports its own count; totals are aggregated as Python ints by the
caller, never as a JAX int32 sum). JAX/Flax/Optax imports are deferred into
function bodies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["build_full_update"]

_ACTIONS_SHAPE = (1, 25, 32)
_INT32_LEAF_LIMIT = 2**31
_OPTIMIZER_KW = {"lr": 1e-5, "b1": 0.9, "b2": 0.95, "eps": 1e-8, "weight_decay": 0.0, "clip_norm": 1.0}


def _leaf_path(path: tuple) -> str:
    return "/".join(str(p) for p in path)


def build_full_update(model: Any, seed: int = 17):
    """Build ``(state, step_fn, census)`` for full fine-tuning of ``model``."""
    import jax  # noqa: PLC0415 - deferred heavy import
    import jax.numpy as jnp  # noqa: PLC0415
    import optax  # noqa: PLC0415
    from flax import nnx  # noqa: PLC0415

    from embodied_trackpad.leap_v1.pi05_jax_loss import pi05_valid16_flow_loss  # noqa: PLC0415

    graphdef, params = nnx.split(model)
    flat = params.flat_state()
    if not flat:
        raise ValueError("model state has no parameter leaves")
    static_shapes: dict = {}
    for path, var in flat.items():
        # Pinned Flax 0.10.2: flat_state() values are VariableState (.type, .value).
        if not (isinstance(var, nnx.VariableState) and issubclass(var.type, nnx.Param)):
            raise ValueError(
                f"state leaf {_leaf_path(path)} has type {getattr(var, 'type', type(var))!r}, "
                "expected an nnx.Param VariableState"
            )
        value = var.value
        if value.dtype != jnp.float32:
            raise ValueError(f"state leaf {_leaf_path(path)} is {value.dtype}, expected float32")
        if value.size >= _INT32_LEAF_LIMIT:
            raise ValueError(
                f"state leaf {_leaf_path(path)} has {value.size} elements >= 2**31; "
                "per-leaf int32 non-finite counts would overflow"
            )
        static_shapes[path] = tuple(value.shape)
    leaves = [
        {"path": _leaf_path(p), "shape": list(s), "dtype": "float32", "numel": int(np.prod(s, dtype=np.int64))}
        for p, s in static_shapes.items()
    ]
    census = {
        "total_params": sum(int(np.prod(s, dtype=np.int64)) for s in static_shapes.values()),
        "leaf_count": len(leaves),
        "all_fp32": True,
        "leaves": leaves,
    }

    tx = optax.chain(
        optax.clip_by_global_norm(_OPTIMIZER_KW["clip_norm"]),
        optax.adamw(
            learning_rate=_OPTIMIZER_KW["lr"],
            b1=_OPTIMIZER_KW["b1"],
            b2=_OPTIMIZER_KW["b2"],
            eps=_OPTIMIZER_KW["eps"],
            weight_decay=_OPTIMIZER_KW["weight_decay"],
        ),
    )
    state = {
        "params": params,
        "opt_state": tx.init(params),
        "step": jnp.zeros((), dtype=jnp.int32),
        "rng": jax.random.PRNGKey(int(seed)),
    }

    def step_fn(state, observation, actions):
        if tuple(actions.shape) != _ACTIONS_SHAPE:
            raise ValueError(f"actions must have shape {_ACTIONS_SHAPE}, got {tuple(actions.shape)}")

        def loss_fn(mdl, rng):
            return jnp.mean(pi05_valid16_flow_loss(mdl, rng, observation, actions))

        model = nnx.merge(graphdef, state["params"])
        new_rng, step_rng = jax.random.split(state["rng"])
        loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param))(model, step_rng)

        grad_flat = grads.flat_state()
        if set(grad_flat) != set(static_shapes) or any(
            tuple(grad_flat[p].value.shape) != static_shapes[p] for p in static_shapes
        ):
            raise ValueError("gradient tree structure/shape does not match the full parameter state")

        grad_norm = optax.global_norm(grads)
        grad_norms = {_leaf_path(p): jnp.linalg.norm(grad_flat[p].value) for p in grad_flat}
        grad_nonfinite_counts = {
            _leaf_path(p): jnp.count_nonzero(~jnp.isfinite(grad_flat[p].value)) for p in grad_flat
        }
        finite = jnp.isfinite(loss) & jnp.isfinite(grad_norm)

        def apply_update(st):
            updates, new_opt = tx.update(grads, st["opt_state"], st["params"])
            new_params = optax.apply_updates(st["params"], updates)
            delta_norm = optax.global_norm(jax.tree.map(lambda a, b: a - b, new_params, st["params"]))
            new_state = {"params": new_params, "opt_state": new_opt, "step": st["step"] + 1, "rng": new_rng}
            return new_state, optax.global_norm(updates), delta_norm

        def reject(st):
            return st, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)

        new_state, update_norm, delta_norm = jax.lax.cond(finite, apply_update, reject, operand=state)
        metrics = {
            "loss": loss,
            "accepted": finite.astype(jnp.float32),
            "grad_norm": grad_norm,
            "update_norm": update_norm,
            "param_delta_norm": delta_norm,
            "step": new_state["step"],
            "grad_norms": grad_norms,
            "grad_nonfinite_counts": grad_nonfinite_counts,
        }
        return new_state, metrics

    return state, jax.jit(step_fn, donate_argnums=(0,)), census
