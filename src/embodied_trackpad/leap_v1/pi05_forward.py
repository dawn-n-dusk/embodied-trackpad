"""Forward-only diagnostic for an already-loaded official NNX Pi0 (pi05=True).

``run_forward_diagnostic`` runs the valid-16 flow-matching loss in pure
inference mode (``nnx.jit``, no ``value_and_grad``) over already-accepted
``Pi05ObservationBundle`` inputs. It never constructs a model, never runs a
backward pass or optimizer step, never touches disk or the network, and never
mutates the supplied bundles. Model loading and GPU/CPU admission are the
caller's responsibility.

A single ``nnx.jit``-wrapped ``pi05_valid16_flow_loss`` is created once,
outside the sample loop, and reused for every bundle. JAX/Flax imports are
deferred into function bodies so the pure-NumPy ``summarize_forward_loss``
helper stays importable and testable without JAX installed.

Note: the JAX execution path is marked unverified on this host; only the
pure-NumPy helper paths are covered by local tests.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

__all__ = ["run_forward_diagnostic", "summarize_forward_loss"]

_LOSS_SHAPE = (1, 25)
_MASK_TRUE = 16
_MASK_TOTAL = 32

_jitted_valid16_loss = None


def _get_jitted_loss():
    """Create (exactly once) the single nnx.jit-wrapped valid-16 flow loss."""
    global _jitted_valid16_loss
    if _jitted_valid16_loss is None:
        from flax import nnx  # noqa: PLC0415 - deferred heavy import

        from embodied_trackpad.leap_v1.pi05_jax_loss import (  # noqa: PLC0415
            pi05_valid16_flow_loss,
        )

        _jitted_valid16_loss = nnx.jit(pi05_valid16_flow_loss)
    return _jitted_valid16_loss


def summarize_forward_loss(loss: Any) -> dict[str, float]:
    """Pure-NumPy shape/finiteness check plus mean/min/max aggregation.

    ``loss`` must be a finite float array of shape ``[1, 25]`` (batch 1,
    fixed action horizon 25). Raises ``ValueError`` on any other shape or on
    non-finite values. The input array is never modified.
    """
    arr = np.asarray(loss, dtype=np.float64)
    if arr.shape != _LOSS_SHAPE:
        raise ValueError(f"forward loss must have shape {_LOSS_SHAPE}, got {tuple(arr.shape)}")
    if not np.isfinite(arr).all():
        raise ValueError("forward loss must contain only finite values")
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def _check_loss_mask32(mask: Any) -> None:
    m = np.asarray(mask)
    if m.shape != (_MASK_TOTAL,) or m.dtype != np.dtype(bool):
        raise ValueError(f"loss_mask32 must be bool with shape ({_MASK_TOTAL},), got {m.shape} {m.dtype}")
    if not m[:_MASK_TRUE].all() or m[_MASK_TRUE:].any():
        raise ValueError("loss_mask32 must have the first 16 entries true and the rest false")


def run_forward_diagnostic(model: Any, bundles: Any, seeds: tuple[int, ...] = (17, 18)) -> dict[str, Any]:
    """Jit-forward the valid-16 loss for each accepted bundle and summarize.

    ``model`` must be an already constructed official NNX ``Pi0`` with
    ``pi05=True``; ``bundles`` an iterable of accepted ``Pi05ObservationBundle``
    (batch-1 official ``Observation``, ``actions[1,25,32]``, ``loss_mask32``).
    ``seeds`` supplies one JAX PRNG seed per bundle. Returns per-sample
    seed/mean/min/max, overall mean/min/max across all returned losses,
    ``n_samples``, elapsed wall seconds, and explicitly
    ``backward_count=optimizer_updates=0``.
    """
    bundle_list = list(bundles)
    seed_list = tuple(int(s) for s in seeds)
    if not bundle_list:
        raise ValueError("bundles must be nonempty")
    if len(seed_list) != len(bundle_list):
        raise ValueError(f"seeds length {len(seed_list)} must match bundles length {len(bundle_list)}")
    for bundle in bundle_list:
        _check_loss_mask32(bundle.loss_mask32)

    import jax  # noqa: PLC0415 - deferred heavy import

    forward = _get_jitted_loss()
    per_sample = []
    all_losses = []
    start = time.perf_counter()
    for bundle, seed in zip(bundle_list, seed_list):
        loss = forward(model, jax.random.PRNGKey(seed), bundle.observation, bundle.actions)
        loss = jax.block_until_ready(loss)
        stats = summarize_forward_loss(np.asarray(loss))
        per_sample.append({"seed": seed, **stats})
        all_losses.append(np.asarray(loss, dtype=np.float64).reshape(-1))
    elapsed = time.perf_counter() - start

    stacked = np.concatenate(all_losses)
    return {
        "n_samples": len(bundle_list),
        "elapsed_seconds": elapsed,
        "mean": float(stacked.mean()),
        "min": float(stacked.min()),
        "max": float(stacked.max()),
        "per_sample": per_sample,
        "backward_count": 0,
        "optimizer_updates": 0,
    }
