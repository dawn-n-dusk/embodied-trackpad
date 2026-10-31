"""Privileged teacher normal-force admittance increment for LEAP v1 (PURE MATH).

Coordinator-selected next candidate after the measured seed2 failure of the
kinematic planners (2026-10-31 evidence: commanded vs actual joint motion
disagreed even in sign; a kinematic planner is not a force/actual-motion
servo). This module implements ONLY the isolated, testable admittance
increment; it does NOT claim to solve dynamics and is NOT integrated with any
runner, ContactLossGuard, solver stepping, app, or oracle state.

    v_z = clip(gain_m_per_ns * (F - target), +-max_normal_speed_m_s)
    dz  = v_z * dt_s
    dq  = n * dz / (n @ n)                    # minimum-norm normal increment

Higher-than-target force yields an UPWARD (+z) commanded correction, lower-
than-target a DOWNWARD one. The whole 4-vector is scaled by ONE common factor
<= 1 (per-component clipping would change the commanded normal direction) for
the per-substep joint cap, the declared joint ranges at the current command,
and the per-cycle envelope |new_ctrl - cycle_anchor| <= max_cycle_delta_rad.
At an exhausted range/envelope the scale is 0 (or smaller) and the binding
limit is reported via ``bound_limited``; nothing is extrapolated.

ALL gains in :class:`NormalForceConfig` are PROPOSED and UNCALIBRATED. The
increment operates on commanded position references only:
``predicted_normal_dz_m`` is the first-order commanded-normal prediction
``n @ applied_dq``, NOT an actual fingertip height or force response. A later
reviewed runner may call this at the physics substep rate with the actual
contact-point normal row; no such integration exists in this task.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1.visual_physics import INDEX_JOINTS, NU, NamedJointMap

_DEGENERATE_NORM2 = 1e-24  # |n|^2 at or below this is a degenerate normal row
_MAX_DT_S = 0.01  # admittance increment is validated for substep-scale dt only
# Float roundoff tolerance for boundary VALIDATION ONLY (e.g. 0.605 - 0.6 ==
# 0.0050000000000000044 must not fail a 0.005 envelope). This is NOT an
# enlarged safety budget: headroom still clamps at max(0, ...), outward motion
# at an exhausted bound still gets scale 0, and meaningful violations (>= 1e-9
# rad) are rejected.
_BOUND_TOL_RAD = 1e-12


@dataclass(frozen=True)
class NormalForceConfig:
    """PROPOSED uncalibrated admittance gains (no hardware/sim tuning yet)."""

    target_force_n: float = 0.8
    gain_m_per_ns: float = 0.004  # metres per newton-second
    max_normal_speed_m_s: float = 0.002
    max_joint_step_rad: float = 0.0002  # per-substep per-joint delta cap
    max_cycle_delta_rad: float = 0.005  # |ctrl - cycle_anchor| envelope

    def __post_init__(self) -> None:
        for name in ("target_force_n", "gain_m_per_ns", "max_normal_speed_m_s",
                     "max_joint_step_rad", "max_cycle_delta_rad"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0):
                raise ValueError(
                    f"NormalForceConfig.{name} must be a finite positive number, got {v!r}")


def _finite_nonnegative_force(value: Any) -> float:
    if not (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))):
        raise ValueError(f"measured_force_n must be a real number, got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f"measured_force_n must be finite and >= 0, got {value!r}")
    return f


def normal_force_admittance(
    current_ctrl: np.ndarray,
    cycle_anchor: np.ndarray,
    normal_row: np.ndarray,
    measured_force_n: float,
    dt_s: float,
    mapping: NamedJointMap,
    cfg: NormalForceConfig = NormalForceConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """One bounded normal-force admittance increment on commanded references.

    ``normal_row`` is the finite contact-normal Jacobian row (shape (4,)) in
    ``mapping.joint_names`` column order, supplied by the caller (e.g. the
    actual contact-point translational z row at the current substep). Returns
    (full-16 ctrl, info); channels not in ``mapping.act_index`` are bitwise
    unchanged. Raises ValueError on any invalid input; nothing is imputed.
    """
    force = _finite_nonnegative_force(measured_force_n)
    if not (isinstance(dt_s, (int, float, np.integer, np.floating))
            and not isinstance(dt_s, (bool, np.bool_))
            and math.isfinite(float(dt_s)) and 0.0 < float(dt_s) <= _MAX_DT_S):
        raise ValueError(f"dt_s must be finite, positive and <= {_MAX_DT_S}, got {dt_s!r}")
    dt = float(dt_s)
    ctrl = np.asarray(current_ctrl, dtype=float)
    if ctrl.shape != (NU,) or not np.all(np.isfinite(ctrl)):
        raise ValueError(f"current_ctrl must be a finite ({NU},) vector")
    anchor = np.asarray(cycle_anchor, dtype=float)
    if anchor.shape != (NU,) or not np.all(np.isfinite(anchor)):
        raise ValueError(f"cycle_anchor must be a finite ({NU},) vector")
    if not (len(mapping.joint_names) == len(mapping.act_index) == len(mapping.ranges)
            == len(mapping.qposadr) == len(mapping.dofadr) == 4):
        raise ValueError("all mapping arrays must have length 4")
    if sorted(mapping.joint_names) != sorted(INDEX_JOINTS):
        raise ValueError(
            f"mapping joints must be a distinct permutation of {INDEX_JOINTS}, "
            f"got {mapping.joint_names}"
        )
    for a in mapping.act_index:
        if not (isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_))
                and 0 <= a < NU):
            raise ValueError(f"invalid actuator channel {a!r}")
    if len(set(mapping.act_index)) != 4:
        raise ValueError("mapping actuator channels duplicated")
    n = np.asarray(normal_row, dtype=float)
    if n.shape != (4,) or not np.all(np.isfinite(n)):
        raise ValueError(f"normal_row must be a finite (4,) array, got {n.shape}")
    nn = float(n @ n)
    if not math.isfinite(nn) or nn <= _DEGENERATE_NORM2:
        raise ValueError(f"degenerate normal row: |n|^2={nn}")
    # Current commanded reference must already be legal, even for a zero
    # increment: selected channels inside declared ranges and cycle envelope.
    for k, (lo, hi) in enumerate(mapping.ranges):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"invalid joint range ({lo}, {hi})")
        cur = float(ctrl[mapping.act_index[k]])
        if not (lo - _BOUND_TOL_RAD <= cur <= hi + _BOUND_TOL_RAD):
            raise ValueError(
                f"current ctrl for {mapping.joint_names[k]} is {cur}, outside its "
                f"declared range [{lo}, {hi}]; refusing to re-emit an invalid reference"
            )
        from_anchor = cur - float(anchor[mapping.act_index[k]])
        if not (abs(from_anchor) <= cfg.max_cycle_delta_rad + _BOUND_TOL_RAD):
            raise ValueError(
                f"current ctrl for {mapping.joint_names[k]} is {from_anchor} rad from "
                f"cycle_anchor, beyond max_cycle_delta_rad={cfg.max_cycle_delta_rad}"
            )

    force_error = force - cfg.target_force_n
    v_z = float(np.clip(cfg.gain_m_per_ns * force_error,
                        -cfg.max_normal_speed_m_s, cfg.max_normal_speed_m_s))
    dz = v_z * dt
    dq = n * (dz / nn)
    if not np.all(np.isfinite(dq)):
        raise ValueError("non-finite dq from admittance increment")

    # ONE common scale for the whole 4-vector; no per-component clipping.
    scale = 1.0
    candidates: list[tuple[str, float]] = []
    peak = float(np.abs(dq).max()) if dq.size else 0.0
    if peak > cfg.max_joint_step_rad:
        candidates.append(("max_joint_step_rad", cfg.max_joint_step_rad / peak))
    for k, act in enumerate(mapping.act_index):
        d = float(dq[k])
        if d == 0.0:
            continue
        lo, hi = mapping.ranges[k]
        cur = float(ctrl[act])
        d0 = cur - float(anchor[act])
        if d > 0.0:
            range_scale = (hi - cur) / d
            envelope_scale = (cfg.max_cycle_delta_rad - d0) / d
        else:
            range_scale = (cur - lo) / (-d)
            envelope_scale = (cfg.max_cycle_delta_rad + d0) / (-d)
        candidates.append((f"range:{mapping.joint_names[k]}", max(0.0, range_scale)))
        candidates.append((f"cycle_envelope:{mapping.joint_names[k]}",
                           max(0.0, envelope_scale)))
    if candidates:
        scale = min(1.0, min(s for _, s in candidates))
    bound_limited = sorted(label for label, s in candidates
                           if scale < 1.0 and s <= scale + 1e-15)
    applied = scale * dq
    out = ctrl.copy()
    for k, act in enumerate(mapping.act_index):
        out[act] = float(ctrl[act] + applied[k])
    if not np.all(np.isfinite(out)):
        raise ValueError("admittance increment produced a non-finite command")
    return out, {
        "force_error_n": float(force_error),
        "requested_normal_speed_m_s": v_z,
        "requested_dz_m": float(dz),
        "applied_dq": [float(v) for v in applied],
        "predicted_normal_dz_m": float(n @ applied),
        "common_scale": float(scale),
        "bound_limited": bound_limited,
        "teacher_only": True,
    }
