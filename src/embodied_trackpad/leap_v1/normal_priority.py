"""Normal-priority teacher planner candidate for the LEAP v1 visual task.

PURE MATH candidate, no solver stepping, no episode runner, no app coupling.

Motivation (measured failure, private seed2 evidence 2026-10-31): the scalar-
damped 3D DLS of :func:`visual_physics.plan_index_correction` traded the z-hold
row against planar error — at cycle 20 it was REQUESTED downward z
(-0.0002 m) but PREDICTED upward z (+0.000121 m), and z error drifted to
-11.9 mm by cycle 120 with contact lost. This candidate instead enforces the
first-order normal constraint EXACTLY and puts tangential motion in its null
space:

    n = J[2, :];  d_z = n * z_task / (n @ n)          # exact normal row
    P = I - outer(n, n) / (n @ n)                     # normal null space
    A = J[:2, :] @ P;  r = t_xy - J[:2, :] @ d_z
    d_t = P @ A.T @ solve(A @ A.T + damping^2 I2, r)  # damped tangential
    dq = d_z + d_t                                    # J[2,:] @ dq == z_task

Bounds NEVER clip components individually (that would break the normal
constraint): the entire 4-vector is scaled by ONE common factor <= 1 from the
joint-delta cap and the remaining joint range at the current command. At a
range limit the scale may be 0; it is reported via ``common_scale`` and never
silently extrapolated. Output is a teacher label, not a learned actor, and
this module claims NO contact/closed-loop performance.

Also provides :class:`ContactLossGuard`, a privileged diagnostic safety gate
(never an actor sensor) for a future reviewed runner.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1.visual_physics import (
    INDEX_JOINTS,
    NU,
    NamedJointMap,
    PlannerConfig,
)

_DEGENERATE_NORM2 = 1e-24  # |n|^2 below this is a degenerate normal row


def plan_normal_priority(
    error_px: tuple[float, float],
    z_error_m: float,
    jac_3x4: np.ndarray,
    current_ctrl: np.ndarray,
    mapping: NamedJointMap,
    cfg: PlannerConfig = PlannerConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact-normal-constraint correction; same signature as plan_index_correction.

    Jacobian column-order contract identical to the existing planner: column k
    belongs to ``mapping.joint_names[k]``. Returns (full-16 ctrl, info) with
    requested_task (post-clip), predicted_task = J @ applied_dq, common_scale,
    dq (applied, scaled), dq_unscaled, mapping and teacher_only=True.
    """
    ex, ey = float(error_px[0]), float(error_px[1])
    if not (np.isfinite(ex) and np.isfinite(ey) and np.isfinite(z_error_m)):
        raise ValueError("non-finite planner input")
    ctrl = np.asarray(current_ctrl, dtype=float)
    if ctrl.shape != (NU,) or not np.all(np.isfinite(ctrl)):
        raise ValueError(f"current_ctrl must be a finite ({NU},) vector")
    if not (len(mapping.joint_names) == len(mapping.act_index) == len(mapping.ranges)
            == len(mapping.qposadr) == len(mapping.dofadr) == 4):
        raise ValueError("all mapping arrays must have length 4")
    if sorted(mapping.joint_names) != sorted(INDEX_JOINTS):
        raise ValueError(
            f"mapping joints must be a distinct permutation of {INDEX_JOINTS}, "
            f"got {mapping.joint_names}"
        )
    for a in mapping.act_index:
        if not (isinstance(a, (int, np.integer)) and not isinstance(a, bool)
                and 0 <= a < NU):
            raise ValueError(f"invalid actuator channel {a!r}")
    if len(set(mapping.act_index)) != 4:
        raise ValueError("mapping actuator channels duplicated")
    for k, (lo, hi) in enumerate(mapping.ranges):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"invalid joint range ({lo}, {hi})")
        cur = float(ctrl[mapping.act_index[k]])
        if not (lo <= cur <= hi):
            raise ValueError(
                f"current ctrl for {mapping.joint_names[k]} is {cur}, outside its "
                f"declared range [{lo}, {hi}]; refusing to re-emit an invalid reference"
            )
    jac = np.asarray(jac_3x4, dtype=float)
    if jac.shape != (3, 4) or not np.all(np.isfinite(jac)):
        raise ValueError(f"jac_3x4 must be a finite (3, 4) array, got {jac.shape}")
    n = jac[2, :]
    nn = float(n @ n)
    if not math.isfinite(nn) or nn <= _DEGENERATE_NORM2:
        raise ValueError(f"degenerate normal row: |n|^2={nn}")

    t_xy = np.array([
        float(np.clip(ex / cfg.px_per_m, -cfg.max_planar_step_m, cfg.max_planar_step_m)),
        float(np.clip(ey / cfg.px_per_m, -cfg.max_planar_step_m, cfg.max_planar_step_m)),
    ])
    z_task = float(np.clip(z_error_m, -cfg.max_z_step_m, cfg.max_z_step_m))
    d_z = n * (z_task / nn)
    P = np.eye(4) - np.outer(n, n) / nn
    j_xy = jac[:2, :]
    A = j_xy @ P
    r = t_xy - j_xy @ d_z
    d_t = P @ A.T @ np.linalg.solve(A @ A.T + cfg.damping**2 * np.eye(2), r)
    dq = d_z + d_t
    if not np.all(np.isfinite(dq)):
        raise ValueError("non-finite dq from solve")

    # ONE common scale for the whole 4-vector (per-component clipping would
    # break the exact normal constraint).
    scale = 1.0
    peak = float(np.abs(dq).max())
    if peak > cfg.max_joint_delta_rad:
        scale = cfg.max_joint_delta_rad / peak
    for k, act in enumerate(mapping.act_index):
        if dq[k] == 0.0:
            continue
        lo, hi = mapping.ranges[k]
        avail = (hi - float(ctrl[act])) if dq[k] > 0 else (float(ctrl[act]) - lo)
        scale = min(scale, max(0.0, avail / abs(dq[k])))  # 0 at a range limit
    applied = scale * dq
    out = ctrl.copy()
    for k, act in enumerate(mapping.act_index):
        out[act] = float(ctrl[act] + applied[k])
    if not np.all(np.isfinite(out)):
        raise ValueError("planner produced a non-finite command")
    predicted = jac @ applied
    return out, {
        "requested_task": [float(t_xy[0]), float(t_xy[1]), z_task],
        "predicted_task": [float(v) for v in predicted],
        "common_scale": float(scale),
        "dq": [float(v) for v in applied],
        "dq_unscaled": [float(v) for v in dq],
        "mapping": list(mapping.joint_names),
        "teacher_only": True,
    }


@dataclass
class ContactLossGuard:
    """Privileged diagnostic safety gate (NEVER an actor sensor): accumulate
    visual-phase substep time with a missing index aggregate or force below
    ``force_min_n``; terminate after ``max_loss_s`` continuous loss. Any valid
    contact resets the accumulator; a non-finite force fails immediately."""

    force_min_n: float = 0.05
    max_loss_s: float = 0.1
    lost_s: float = 0.0
    terminated: bool = False

    def __post_init__(self) -> None:
        for name in ("force_min_n", "max_loss_s"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0):
                raise ValueError(f"ContactLossGuard.{name} must be finite positive, got {v!r}")
        if not (isinstance(self.lost_s, (int, float)) and not isinstance(self.lost_s, bool)
                and math.isfinite(self.lost_s) and self.lost_s >= 0):
            raise ValueError(f"ContactLossGuard.lost_s must be finite >= 0, got {self.lost_s!r}")
        if not isinstance(self.terminated, bool):
            raise ValueError(f"ContactLossGuard.terminated must be a bool, got {self.terminated!r}")

    def update(self, dt_s: float, contact_present: bool, force_n: float | None) -> bool:
        if self.terminated:
            return True
        if not (math.isfinite(dt_s) and dt_s >= 0.0):
            raise ValueError(f"bad dt_s {dt_s!r}")
        if contact_present and (force_n is None or not math.isfinite(force_n)):
            self.terminated = True  # non-finite device truth: fail at once
            return True
        valid = bool(contact_present) and float(force_n) >= self.force_min_n
        self.lost_s = 0.0 if valid else self.lost_s + dt_s
        if self.lost_s >= self.max_loss_s:
            self.terminated = True
        return self.terminated
