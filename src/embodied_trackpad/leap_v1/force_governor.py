"""Force-governed tangential-substep composer for LEAP v1 (PURE MATH).

This is a PROPOSED teacher helper, not a new physics safety layer and not a
replacement for the existing :mod:`force_servo` admittance increment. The
caller first computes a normal-only correction with
:func:`force_servo.normal_force_admittance`, then calls
:func:`compose_governed_substep` to add a tangential component that is gated by
the current measured normal force. The normal component has priority: the
tangential step is scaled by a single common factor so the total per-substep
delta, joint ranges and cycle envelope are respected, without ever scaling the
already-accepted normal part.

    gate = clip((tangential_stop_force_n - F) / (tangential_stop_force_n - target_force_n), 0, 1)
    requested_tangent = gate * nominal_frame_delta / substeps_per_frame
    new_ctrl = current_ctrl + normal_delta + alpha * requested_tangent

At F >= tangential_stop_force_n the tangential step is zero; at F <=
target_force_n the full (range/envelope/cap-limited) tangential step is
allowed. Skipped tangential fraction is not stored or carried forward.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1.visual_physics import INDEX_JOINTS, NU, NamedJointMap

_BOUND_TOL_RAD = 1e-12  # float roundoff tolerance for boundary validation only


@dataclass(frozen=True)
class ForceGovernorConfig:
    """PROPOSED uncalibrated force-governor gains."""

    substeps_per_frame: int = 25
    target_force_n: float = 0.8
    tangential_stop_force_n: float = 1.1
    per_substep_total_joint_delta_cap_rad: float = 0.0002
    max_cycle_delta_rad: float = 0.005

    def __post_init__(self) -> None:
        # substeps_per_frame must be a strict positive integer (not a float like 1.5).
        if not (isinstance(self.substeps_per_frame, (int, np.integer))
                and not isinstance(self.substeps_per_frame, (bool, np.bool_))
                and self.substeps_per_frame > 0):
            raise ValueError(
                f"ForceGovernorConfig.substeps_per_frame must be a positive integer, "
                f"got {self.substeps_per_frame!r}")
        for name in ("target_force_n", "tangential_stop_force_n",
                     "per_substep_total_joint_delta_cap_rad", "max_cycle_delta_rad"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float, np.integer, np.floating))
                    and not isinstance(v, (bool, np.bool_))
                    and math.isfinite(v) and v > 0):
                raise ValueError(
                    f"ForceGovernorConfig.{name} must be a finite positive number, got {v!r}")
        if self.tangential_stop_force_n <= self.target_force_n:
            raise ValueError(
                "tangential_stop_force_n must be greater than target_force_n")


def _finite_positive_force(value: Any) -> float:
    if not (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))):
        raise ValueError(f"measured_force_n must be a real number, got {value!r}")
    f = float(value)
    if not math.isfinite(f) or f <= 0.0:
        raise ValueError(f"measured_force_n must be finite and positive, got {value!r}")
    return f


def compose_governed_substep(
    current_ctrl: np.ndarray,
    normal_ctrl: np.ndarray,
    nominal_frame_delta: np.ndarray,
    cycle_anchor: np.ndarray,
    measured_force_n: float,
    mapping: NamedJointMap,
    cfg: ForceGovernorConfig = ForceGovernorConfig(),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add a force-gated tangential increment after an accepted normal increment.

    ``normal_ctrl`` is the already-computed force-admittance result; it is
    validated to be within per-substep, range and cycle-envelope bounds
    relative to ``current_ctrl``/``cycle_anchor``. ``nominal_frame_delta`` is
    the full-frame planned position delta (e.g. the z0 normal-priority plan).
    The tangential component is ``gate * nominal_frame_delta / substeps_per_frame``,
    scaled by one common ``alpha`` in [0, 1] so the total step stays inside the
    per-substep cap, joint ranges and cycle envelope. Non-index channels are
    bitwise unchanged.
    """
    force = _finite_positive_force(measured_force_n)
    ctrl = np.asarray(current_ctrl, dtype=float)
    normal = np.asarray(normal_ctrl, dtype=float)
    nominal = np.asarray(nominal_frame_delta, dtype=float)
    anchor = np.asarray(cycle_anchor, dtype=float)
    for arr, label in ((ctrl, "current_ctrl"), (normal, "normal_ctrl"),
                       (nominal, "nominal_frame_delta"), (anchor, "cycle_anchor")):
        if arr.shape != (NU,) or not np.all(np.isfinite(arr)):
            raise ValueError(f"{label} must be a finite ({NU},) vector")
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

    # Non-index channels must be unchanged in current/normal/anchor.
    for i in range(NU):
        if i not in mapping.act_index:
            if not (ctrl[i] == normal[i] == anchor[i]):
                raise ValueError(
                    f"non-index channel {i} differs across current/normal/anchor")
            if nominal[i] != 0.0:
                raise ValueError(
                    f"nominal_frame_delta must be zero on non-index channel {i}")

    # Validate normal_ctrl is within bounds relative to current and anchor.
    for k, (lo, hi) in enumerate(mapping.ranges):
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"invalid joint range ({lo}, {hi})")
        act = mapping.act_index[k]
        cur = float(ctrl[act])
        if not (lo - _BOUND_TOL_RAD <= cur <= hi + _BOUND_TOL_RAD):
            raise ValueError(f"current ctrl for {mapping.joint_names[k]} out of range")
        nrm = float(normal[act])
        if not (lo - _BOUND_TOL_RAD <= nrm <= hi + _BOUND_TOL_RAD):
            raise ValueError(f"normal_ctrl for {mapping.joint_names[k]} out of range")
        nd = nrm - cur
        if not (abs(nd) <= cfg.per_substep_total_joint_delta_cap_rad + _BOUND_TOL_RAD):
            raise ValueError(
                f"normal_ctrl delta for {mapping.joint_names[k]} exceeds "
                f"per-substep cap {cfg.per_substep_total_joint_delta_cap_rad}")
        d0_cur = cur - float(anchor[act])
        if not (abs(d0_cur) <= cfg.max_cycle_delta_rad + _BOUND_TOL_RAD):
            raise ValueError(
                f"current ctrl for {mapping.joint_names[k]} outside cycle envelope")
        d0_nrm = nrm - float(anchor[act])
        if not (abs(d0_nrm) <= cfg.max_cycle_delta_rad + _BOUND_TOL_RAD):
            raise ValueError(
                f"normal_ctrl for {mapping.joint_names[k]} outside cycle envelope")

    # Compute force gate and requested tangential step (only on index channels).
    gate = float(np.clip(
        (cfg.tangential_stop_force_n - force) /
        (cfg.tangential_stop_force_n - cfg.target_force_n),
        0.0, 1.0))
    requested_tangent = np.zeros(4)
    if gate > 0.0:
        for k, act in enumerate(mapping.act_index):
            requested_tangent[k] = gate * float(nominal[act]) / cfg.substeps_per_frame

    normal_delta = np.array([float(normal[mapping.act_index[k]]) - float(ctrl[mapping.act_index[k]])
                             for k in range(4)])

    # ONE common scale alpha for the whole 4-vector tangent increment.
    alpha = 1.0
    candidates: list[tuple[str, float]] = []
    cap = cfg.per_substep_total_joint_delta_cap_rad
    for k, act in enumerate(mapping.act_index):
        t = float(requested_tangent[k])
        if t == 0.0:
            continue
        lo, hi = mapping.ranges[k]
        n = float(normal[act])
        d0_n = n - float(anchor[act])
        # Per-substep cap: |normal_delta + alpha*t| <= cap.
        if t > 0.0:
            cap_scale = (cap - normal_delta[k]) / t
        else:
            cap_scale = (-cap - normal_delta[k]) / t
        # Joint range: lo <= n + alpha*t <= hi.
        if t > 0.0:
            range_scale = (hi - n) / t
        else:
            range_scale = (lo - n) / t
        # Cycle envelope: |d0_n + alpha*t| <= max_cycle_delta_rad.
        if t > 0.0:
            env_scale = (cfg.max_cycle_delta_rad - d0_n) / t
        else:
            env_scale = (-cfg.max_cycle_delta_rad - d0_n) / t
        candidates.append((f"per_step_cap:{mapping.joint_names[k]}", max(0.0, cap_scale)))
        candidates.append((f"range:{mapping.joint_names[k]}", max(0.0, range_scale)))
        candidates.append((f"cycle_envelope:{mapping.joint_names[k]}", max(0.0, env_scale)))
    if candidates:
        alpha = min(1.0, min(s for _, s in candidates))
    binding = sorted(label for label, s in candidates
                     if alpha < 1.0 and s <= alpha + 1e-15)
    applied_tangent = alpha * requested_tangent
    total_delta = normal_delta + applied_tangent
    out = ctrl.copy()
    for k, act in enumerate(mapping.act_index):
        out[act] = float(ctrl[act] + total_delta[k])
    if not np.all(np.isfinite(out)):
        raise ValueError("governed substep produced a non-finite command")
    return out, {
        "teacher_only": True,
        "gate": gate,
        "common_scale": float(alpha),
        "bound_limited": binding,
        "requested_tangent": [float(v) for v in requested_tangent],
        "applied_tangent": [float(v) for v in applied_tangent],
        "normal_delta": [float(v) for v in normal_delta],
        "total_delta": [float(v) for v in total_delta],
    }
