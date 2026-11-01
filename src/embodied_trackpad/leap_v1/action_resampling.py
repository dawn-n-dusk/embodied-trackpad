"""Pure NumPy control-trajectory resampling for LEAP v1 (PURE MATH).

Numerical helper ONLY: coarsens a fine native LEAP v1 control trajectory
(``N x 16`` absolute actuator-position commands at the 500 Hz native step,
``native_dt_s = 0.002``) to one-in-ten endpoint controls (``coarse_dt_s =
0.020``), and reconstructs a fine trajectory from a coarse one by per-segment
linear interpolation. This module carries NO physical-equivalence claim: a
resampled command trajectory is not a physically equivalent hand motion, and
the reconstruction error reported by :func:`coarsen_native_controls` is a pure
numerical mismatch, not a tracking/physics error. It defines no labels,
indices, actor APIs, source/corpus APIs, or OpenPI/training integration; the
full temporal-window sampler remains a separate pending task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHANNELS = 16  # LEAP v1 Full: four fingers x four independent channels
STEPS_PER_COARSE = 10
NATIVE_DT_S = 0.002
COARSE_DT_S = STEPS_PER_COARSE * NATIVE_DT_S  # 0.020

__all__ = [
    "CHANNELS",
    "COARSE_DT_S",
    "CoarseNativeResult",
    "NATIVE_DT_S",
    "STEPS_PER_COARSE",
    "coarsen_native_controls",
    "interpolate_coarse_controls",
]


@dataclass(frozen=True)
class CoarseNativeResult:
    """Coarse trajectory plus its pure numerical reconstruction metrics."""

    coarse: np.ndarray  # (N/10, 16) float64, fresh copy of fine[9::10]
    native_dt_s: float  # 0.002
    coarse_dt_s: float  # 0.020
    fine_steps: int  # N
    coarse_steps: int  # N / 10
    horizon_s: float  # N * native_dt_s
    rms_rad: float  # elementwise RMS of (fine - reconstructed) in rad
    max_abs_rad: float  # max abs of (fine - reconstructed) in rad


def _as_finite_float64(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value)
    kind = arr.dtype.kind
    if kind in "bc" or kind not in "iuf":
        raise TypeError(
            f"{name} must be a real numeric array, got dtype {arr.dtype!r}"
        )
    out = np.array(arr, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite, got NaN or Inf")
    return out


def _validate_previous_ctrl(previous_ctrl: object) -> np.ndarray:
    prev = _as_finite_float64(previous_ctrl, "previous_ctrl")
    if prev.shape != (CHANNELS,):
        raise ValueError(
            f"previous_ctrl must have shape ({CHANNELS},), got {prev.shape!r}"
        )
    return prev


def interpolate_coarse_controls(
    coarse: object, previous_ctrl: object
) -> np.ndarray:
    """Reconstruct a fine (M*10, 16) float64 trajectory from coarse controls.

    Row ``k*10 + j`` (j = 0..9) lies a fraction ``(j+1)/10`` of the way from
    the segment base (``previous_ctrl`` for k = 0, else ``coarse[k-1]``) to
    ``coarse[k]``. Inputs are not mutated and the returned array does not
    alias them.
    """
    c = _as_finite_float64(coarse, "coarse")
    if c.ndim != 2 or c.shape[1] != CHANNELS:
        raise ValueError(
            f"coarse must have shape (M, {CHANNELS}), got {c.shape!r}"
        )
    m = c.shape[0]
    if m < 1:
        raise ValueError("coarse must contain at least one step, got M=0")
    prev = _validate_previous_ctrl(previous_ctrl)
    fine = np.empty((m * STEPS_PER_COARSE, CHANNELS), dtype=np.float64)
    base = prev
    fractions = np.arange(1, STEPS_PER_COARSE + 1, dtype=np.float64) / float(
        STEPS_PER_COARSE
    )
    for k in range(m):
        target = c[k]
        fine[k * STEPS_PER_COARSE : (k + 1) * STEPS_PER_COARSE] = (
            base + fractions[:, None] * (target - base)
        )
        base = target
    return fine


def coarsen_native_controls(
    fine: object, previous_ctrl: object
) -> CoarseNativeResult:
    """Coarsen a fine (N, 16) native control trajectory to its 1-in-10 endpoints.

    Returns a fresh ``fine[9::10]`` copy plus pure numerical reconstruction
    metrics from :func:`interpolate_coarse_controls`. This is NOT a physical
    equivalence statement: the reported error is only the numerical mismatch
    between the input and the linear reconstruction.
    """
    f = _as_finite_float64(fine, "fine")
    if f.ndim != 2 or f.shape[1] != CHANNELS:
        raise ValueError(
            f"fine must have shape (N, {CHANNELS}), got {f.shape!r}"
        )
    n = f.shape[0]
    if n < 1 or n % STEPS_PER_COARSE != 0:
        raise ValueError(
            f"fine must have N > 0 steps divisible by {STEPS_PER_COARSE}, "
            f"got N={n}"
        )
    prev = _validate_previous_ctrl(previous_ctrl)
    coarse = np.array(f[STEPS_PER_COARSE - 1 :: STEPS_PER_COARSE], copy=True)
    reconstructed = interpolate_coarse_controls(coarse, prev)
    diff = f - reconstructed
    return CoarseNativeResult(
        coarse=coarse,
        native_dt_s=NATIVE_DT_S,
        coarse_dt_s=COARSE_DT_S,
        fine_steps=n,
        coarse_steps=n // STEPS_PER_COARSE,
        horizon_s=n * NATIVE_DT_S,
        rms_rad=float(np.sqrt(np.mean(diff * diff))),
        max_abs_rad=float(np.max(np.abs(diff))),
    )
