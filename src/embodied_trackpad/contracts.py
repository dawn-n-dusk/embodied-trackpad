"""Typed, versioned data contracts for protocol v2 (draft).

These messages describe the Embodied Trackpad action path:
policy -> actuator position references -> robot dynamics -> touch contacts
-> device behavior -> GUI. Continuous actions are always actuator
references; coordinate-click commands do not exist in this protocol.

Structural field checks live here (``__post_init__``); cross-message
runtime checks (embodiment/actuator match, expiry, goal versions) live in
``validation.py``. JSON Schema mirrors of these contracts are published
under ``schemas/`` (JSON Schema draft 2020-12).
"""

from __future__ import annotations

import hashlib
import math
import ntpath
from dataclasses import dataclass
from typing import Optional, Tuple

from .clock import NS_PER_MS, ClockStamp, elapsed_ns
from .errors import ValidationError

SCHEMA_VERSION = "trackpad.v2-draft"
JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

#: Action representations this package can execute semantics for.
ACTION_REPRESENTATIONS = ("normalized_actuator_position_reference",)

#: Continuous command types. Click/scroll-by-coordinate commands are
#: deliberately absent: GUI effects must emerge from contact physics.
COMMAND_TYPES = ("actuator_position_reference",)

INTERACTION_KINDS = (
    "move",
    "activate",
    "drag",
    "scroll",
    "pinch",
    "rotate",
    "release",
    "wait",
)

MOTION_STATUSES = ("attempted", "completed", "aborted", "rejected", "interrupted")

#: "attempted" never implies success; success requires explicit evidence.
VISIBLE_OUTCOMES = ("unknown", "observed_success", "observed_failure", "uncertain")


def _check_unknown(data: dict, allowed: set, owner: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValidationError(
            "bad_field", f"{owner}: unknown fields {sorted(unknown)} "
            f"(schema {SCHEMA_VERSION})"
        )


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError("bad_field", f"{name} must be a non-empty string")
    return value


def _require_int(value: object, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError("bad_field", f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError("bad_field", f"{name} must be >= {minimum}")
    return value


def _require_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("bad_field", f"{name} must be a number")
    out = float(value)
    if not math.isfinite(out):
        raise ValidationError("nonfinite", f"{name} must be finite, got {value!r}")
    return out


def _require_str_tuple(value: object, name: str) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValidationError("bad_field", f"{name} must be a non-empty list")
    items = tuple(_require_str(v, f"{name}[]") for v in value)
    if len(set(items)) != len(items):
        raise ValidationError("bad_field", f"{name} entries must be unique")
    return items


def _check_version(data: dict, owner: str) -> None:
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValidationError(
            "schema_version",
            f"{owner}: expected schema_version {SCHEMA_VERSION!r}, got {version!r}",
        )


def compute_actuator_layout_id(actuator_order: Tuple[str, ...]) -> str:
    """Canonical identity of an actuator ordering.

    Two configs with the same actuator names in a different order have
    different layout ids, so a chunk bound to one layout cannot be accepted
    against the other even when widths match.
    """

    digest = hashlib.sha256("\x1f".join(actuator_order).encode("utf-8")).hexdigest()
    return f"actlayout-{digest[:16]}"


def require_safe_path_component(value: object, name: str) -> str:
    """Require ``value`` to be one safe filesystem path component.

    Rejects separators, dot components, absolute paths, drive prefixes,
    and NUL on every platform, independent of the host filesystem.
    """

    value = _require_str(value, name)
    if value in {".", ".."}:
        raise ValidationError("unsafe_path", f"{name} must not be {value!r}")
    if any(sep in value for sep in ("/", "\\", "\x00")):
        raise ValidationError(
            "unsafe_path", f"{name} must be a single path component, got {value!r}"
        )
    if ntpath.isabs(value) or value.startswith("/") or ntpath.splitdrive(value)[0]:
        raise ValidationError(
            "unsafe_path", f"{name} must not be an absolute or drive path, got {value!r}"
        )
    return value


def _bbox(value: object, name: str) -> Tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValidationError("shape", f"{name} must have exactly 4 numbers")
    box = tuple(_require_finite(v, f"{name}[]") for v in value)
    for v in box:
        if not 0.0 <= v <= 1.0:
            raise ValidationError(
                "out_of_range", f"{name} values must lie in [0, 1], got {box}"
            )
    if not box[0] < box[2] or not box[1] < box[3]:
        raise ValidationError(
            "out_of_range", f"{name} must satisfy x0<x1 and y0<y1, got {box}"
        )
    return box  # type: ignore[return-value]


def _stamp(value: object, name: str) -> ClockStamp:
    if isinstance(value, ClockStamp):
        return value
    if isinstance(value, dict):
        try:
            return ClockStamp.from_dict(value)
        except ValidationError as exc:
            raise ValidationError(exc.code, f"{name}: {exc}") from exc
    raise ValidationError("bad_field", f"{name} must be a ClockStamp or object")


@dataclass(frozen=True)
class EpisodeConfig:
    """Immutable per-episode configuration and identity.

    ``episode_id`` must be a single safe path component (no separators,
    dot components, absolute or drive paths). ``actuator_layout_id`` is the
    canonical hash of ``actuator_order``; every ActionChunk must carry the
    matching value.
    """

    episode_id: str
    embodiment_id: str
    actuator_order: Tuple[str, ...]
    joint_order: Tuple[str, ...]
    action_representation: str
    units: dict
    calibration_id: str
    bridge_version: str
    cursor_profile: dict
    actuator_layout_id: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "episode_id", require_safe_path_component(self.episode_id, "episode_id")
        )
        object.__setattr__(self, "embodiment_id", _require_str(self.embodiment_id, "embodiment_id"))
        object.__setattr__(
            self, "actuator_order", _require_str_tuple(self.actuator_order, "actuator_order")
        )
        object.__setattr__(
            self, "joint_order", _require_str_tuple(self.joint_order, "joint_order")
        )
        computed_layout = compute_actuator_layout_id(self.actuator_order)
        if self.actuator_layout_id and self.actuator_layout_id != computed_layout:
            raise ValidationError(
                "layout_mismatch",
                f"actuator_layout_id {self.actuator_layout_id!r} does not match "
                f"actuator_order (expected {computed_layout!r})",
            )
        object.__setattr__(self, "actuator_layout_id", computed_layout)
        if self.action_representation not in ACTION_REPRESENTATIONS:
            raise ValidationError(
                "bad_field",
                f"action_representation must be one of {ACTION_REPRESENTATIONS}, "
                f"got {self.action_representation!r}",
            )
        if not isinstance(self.units, dict):
            raise ValidationError("bad_field", "units must be an object")
        for key in ("length", "angle", "time"):
            _require_str(self.units.get(key), f"units.{key}")
        object.__setattr__(self, "calibration_id", _require_str(self.calibration_id, "calibration_id"))
        object.__setattr__(self, "bridge_version", _require_str(self.bridge_version, "bridge_version"))
        if not isinstance(self.cursor_profile, dict):
            raise ValidationError("bad_field", "cursor_profile must be an object")
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                "schema_version", f"unsupported schema_version {self.schema_version!r}"
            )

    @property
    def action_dim(self) -> int:
        """Continuous action width; actions follow actuator_order, not joints."""

        return len(self.actuator_order)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "embodiment_id": self.embodiment_id,
            "actuator_order": list(self.actuator_order),
            "joint_order": list(self.joint_order),
            "action_representation": self.action_representation,
            "units": dict(self.units),
            "calibration_id": self.calibration_id,
            "bridge_version": self.bridge_version,
            "cursor_profile": dict(self.cursor_profile),
            "actuator_layout_id": self.actuator_layout_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeConfig":
        _check_version(data, "EpisodeConfig")
        _check_unknown(
            data,
            {
                "schema_version", "episode_id", "embodiment_id", "actuator_order",
                "joint_order", "action_representation", "units", "calibration_id",
                "bridge_version", "cursor_profile", "actuator_layout_id",
            },
            "EpisodeConfig",
        )
        return cls(
            episode_id=data["episode_id"],
            embodiment_id=data["embodiment_id"],
            actuator_order=tuple(data["actuator_order"]),
            joint_order=tuple(data["joint_order"]),
            action_representation=data["action_representation"],
            units=dict(data["units"]),
            calibration_id=data["calibration_id"],
            bridge_version=data["bridge_version"],
            cursor_profile=dict(data["cursor_profile"]),
            actuator_layout_id=data.get("actuator_layout_id", ""),
        )


@dataclass(frozen=True)
class FramePacket:
    """One captured frame with capture/receive timing identity.

    ``age_estimate_ms`` may only come from an explicit synchronization
    source; it is never derived across different clock domains here.
    """

    frame_id: str
    stream_id: str
    capture: ClockStamp
    receive: ClockStamp
    image_shape_px: Tuple[int, int]
    image_ref: str
    age_estimate_ms: Optional[float] = None
    age_uncertainty_ms: Optional[float] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _require_str(self.frame_id, "frame_id"))
        object.__setattr__(self, "stream_id", _require_str(self.stream_id, "stream_id"))
        object.__setattr__(self, "capture", _stamp(self.capture, "capture"))
        object.__setattr__(self, "receive", _stamp(self.receive, "receive"))
        shape = tuple(self.image_shape_px)
        if len(shape) != 2:
            raise ValidationError("shape", "image_shape_px must be [width, height]")
        object.__setattr__(
            self,
            "image_shape_px",
            tuple(_require_int(v, "image_shape_px[]", minimum=1) for v in shape),
        )
        object.__setattr__(self, "image_ref", _require_str(self.image_ref, "image_ref"))
        for name in ("age_estimate_ms", "age_uncertainty_ms"):
            value = getattr(self, name)
            if value is not None:
                value = _require_finite(value, name)
                if value < 0:
                    raise ValidationError("out_of_range", f"{name} must be >= 0")
                object.__setattr__(self, name, value)
        if self.age_estimate_ms is None and self.age_uncertainty_ms is not None:
            raise ValidationError(
                "bad_field", "age_uncertainty_ms requires age_estimate_ms"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                "schema_version", f"unsupported schema_version {self.schema_version!r}"
            )

    def measured_latency_ms(self) -> float:
        """Receive-minus-capture latency, only within one declared domain.

        Raises ClockDomainError when capture and receive clocks differ; this
        package never invents a cross-host frame age.
        """

        return elapsed_ns(self.capture, self.receive) / NS_PER_MS

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "stream_id": self.stream_id,
            "capture": self.capture.to_dict(),
            "receive": self.receive.to_dict(),
            "age_estimate_ms": self.age_estimate_ms,
            "age_uncertainty_ms": self.age_uncertainty_ms,
            "image_shape_px": list(self.image_shape_px),
            "image_ref": self.image_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FramePacket":
        _check_version(data, "FramePacket")
        _check_unknown(
            data,
            {
                "schema_version", "frame_id", "stream_id", "capture", "receive",
                "age_estimate_ms", "age_uncertainty_ms", "image_shape_px", "image_ref",
            },
            "FramePacket",
        )
        return cls(
            frame_id=data["frame_id"],
            stream_id=data["stream_id"],
            capture=ClockStamp.from_dict(data["capture"]),
            receive=ClockStamp.from_dict(data["receive"]),
            age_estimate_ms=data.get("age_estimate_ms"),
            age_uncertainty_ms=data.get("age_uncertainty_ms"),
            image_shape_px=tuple(data["image_shape_px"]),
            image_ref=data["image_ref"],
        )


@dataclass(frozen=True)
class MotorIntent:
    """A time-bounded visual interaction goal handed to the motor layer."""

    intent_id: str
    goal_id: str
    goal_version: int
    observation_id: str
    interaction_kind: str
    target_bbox_norm: Tuple[float, float, float, float]
    target_confidence: float
    calibration_id: str
    issued_at: ClockStamp
    valid_until: ClockStamp
    destination_bbox_norm: Optional[Tuple[float, float, float, float]] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _require_str(self.intent_id, "intent_id"))
        object.__setattr__(self, "goal_id", _require_str(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "goal_version", _require_int(self.goal_version, "goal_version", minimum=0)
        )
        object.__setattr__(
            self, "observation_id", _require_str(self.observation_id, "observation_id")
        )
        if self.interaction_kind not in INTERACTION_KINDS:
            raise ValidationError(
                "bad_field",
                f"interaction_kind must be one of {INTERACTION_KINDS}, "
                f"got {self.interaction_kind!r}",
            )
        object.__setattr__(
            self, "target_bbox_norm", _bbox(self.target_bbox_norm, "target_bbox_norm")
        )
        if self.destination_bbox_norm is not None:
            object.__setattr__(
                self,
                "destination_bbox_norm",
                _bbox(self.destination_bbox_norm, "destination_bbox_norm"),
            )
        confidence = _require_finite(self.target_confidence, "target_confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValidationError(
                "out_of_range", f"target_confidence must lie in [0, 1], got {confidence}"
            )
        object.__setattr__(self, "target_confidence", confidence)
        object.__setattr__(self, "calibration_id", _require_str(self.calibration_id, "calibration_id"))
        object.__setattr__(self, "issued_at", _stamp(self.issued_at, "issued_at"))
        object.__setattr__(self, "valid_until", _stamp(self.valid_until, "valid_until"))
        if self.issued_at.clock_id != self.valid_until.clock_id:
            raise ValidationError(
                "clock_domain",
                "issued_at and valid_until must share one clock domain",
            )
        if self.valid_until.time_ns <= self.issued_at.time_ns:
            raise ValidationError(
                "bad_field", "valid_until must be strictly after issued_at"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                "schema_version", f"unsupported schema_version {self.schema_version!r}"
            )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "goal_id": self.goal_id,
            "goal_version": self.goal_version,
            "observation_id": self.observation_id,
            "interaction_kind": self.interaction_kind,
            "target_bbox_norm": list(self.target_bbox_norm),
            "destination_bbox_norm": (
                list(self.destination_bbox_norm)
                if self.destination_bbox_norm is not None
                else None
            ),
            "target_confidence": self.target_confidence,
            "calibration_id": self.calibration_id,
            "issued_at": self.issued_at.to_dict(),
            "valid_until": self.valid_until.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MotorIntent":
        _check_version(data, "MotorIntent")
        _check_unknown(
            data,
            {
                "schema_version", "intent_id", "goal_id", "goal_version",
                "observation_id", "interaction_kind", "target_bbox_norm",
                "destination_bbox_norm", "target_confidence", "calibration_id",
                "issued_at", "valid_until",
            },
            "MotorIntent",
        )
        dest = data.get("destination_bbox_norm")
        return cls(
            intent_id=data["intent_id"],
            goal_id=data["goal_id"],
            goal_version=data["goal_version"],
            observation_id=data["observation_id"],
            interaction_kind=data["interaction_kind"],
            target_bbox_norm=tuple(data["target_bbox_norm"]),
            destination_bbox_norm=tuple(dest) if dest is not None else None,
            target_confidence=data["target_confidence"],
            calibration_id=data["calibration_id"],
            issued_at=ClockStamp.from_dict(data["issued_at"]),
            valid_until=ClockStamp.from_dict(data["valid_until"]),
        )


@dataclass(frozen=True)
class ActionChunk:
    """A short block of actuator position references.

    ``values`` has shape [horizon][len(actuator_order)] in the referenced
    embodiment's normalized units and follows exactly the actuator ordering
    identified by ``actuator_layout_id`` (the canonical hash of the episode
    config's ``actuator_order``). Zero rows can be legitimate references;
    executability is expressed only by ``valid_mask``.
    """

    policy_version: str
    embodiment_id: str
    actuator_layout_id: str
    goal_id: str
    goal_version: int
    observation_id: str
    command_type: str
    start: ClockStamp
    dt_seconds: float
    horizon: int
    values: Tuple[Tuple[float, ...], ...]
    valid_mask: Tuple[bool, ...]
    valid_until: Optional[ClockStamp] = None
    phase_estimate: Optional[float] = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_version", _require_str(self.policy_version, "policy_version"))
        object.__setattr__(self, "embodiment_id", _require_str(self.embodiment_id, "embodiment_id"))
        object.__setattr__(
            self,
            "actuator_layout_id",
            _require_str(self.actuator_layout_id, "actuator_layout_id"),
        )
        object.__setattr__(self, "goal_id", _require_str(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "goal_version", _require_int(self.goal_version, "goal_version", minimum=0)
        )
        object.__setattr__(
            self, "observation_id", _require_str(self.observation_id, "observation_id")
        )
        if self.command_type not in COMMAND_TYPES:
            raise ValidationError(
                "command_type",
                f"command_type must be one of {COMMAND_TYPES} (actuator references, "
                f"never coordinate-click commands), got {self.command_type!r}",
            )
        object.__setattr__(self, "start", _stamp(self.start, "start"))
        dt = _require_finite(self.dt_seconds, "dt_seconds")
        if dt <= 0:
            raise ValidationError("out_of_range", f"dt_seconds must be > 0, got {dt}")
        object.__setattr__(self, "dt_seconds", dt)
        object.__setattr__(self, "horizon", _require_int(self.horizon, "horizon", minimum=1))
        rows = tuple(tuple(_require_finite(v, "values[]") for v in row) for row in self.values)
        if len(rows) != self.horizon:
            raise ValidationError(
                "shape",
                f"values must have exactly horizon={self.horizon} rows, got {len(rows)}",
            )
        width = len(rows[0])
        if width == 0:
            raise ValidationError("shape", "values rows must be non-empty")
        for row in rows:
            if len(row) != width:
                raise ValidationError("shape", "values rows must share one width")
        object.__setattr__(self, "values", rows)
        mask = tuple(self.valid_mask)
        if len(mask) != self.horizon or any(not isinstance(m, bool) for m in mask):
            raise ValidationError(
                "shape", f"valid_mask must be {self.horizon} booleans"
            )
        object.__setattr__(self, "valid_mask", mask)
        if self.valid_until is not None:
            object.__setattr__(self, "valid_until", _stamp(self.valid_until, "valid_until"))
            if self.valid_until.clock_id != self.start.clock_id:
                raise ValidationError(
                    "clock_domain",
                    "start and valid_until must share one clock domain",
                )
            if self.valid_until.time_ns <= self.start.time_ns:
                raise ValidationError(
                    "bad_field", "valid_until must be strictly after start"
                )
        if self.phase_estimate is not None:
            phase = _require_finite(self.phase_estimate, "phase_estimate")
            if not 0.0 <= phase <= 1.0:
                raise ValidationError(
                    "out_of_range", f"phase_estimate must lie in [0, 1], got {phase}"
                )
            object.__setattr__(self, "phase_estimate", phase)
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                "schema_version", f"unsupported schema_version {self.schema_version!r}"
            )

    @property
    def action_width(self) -> int:
        return len(self.values[0])

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "embodiment_id": self.embodiment_id,
            "actuator_layout_id": self.actuator_layout_id,
            "goal_id": self.goal_id,
            "goal_version": self.goal_version,
            "observation_id": self.observation_id,
            "command_type": self.command_type,
            "start": self.start.to_dict(),
            "dt_seconds": self.dt_seconds,
            "horizon": self.horizon,
            "values": [list(row) for row in self.values],
            "valid_mask": list(self.valid_mask),
            "valid_until": self.valid_until.to_dict() if self.valid_until else None,
            "phase_estimate": self.phase_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ActionChunk":
        _check_version(data, "ActionChunk")
        _check_unknown(
            data,
            {
                "schema_version", "policy_version", "embodiment_id",
                "actuator_layout_id", "goal_id",
                "goal_version", "observation_id", "command_type", "start",
                "dt_seconds", "horizon", "values", "valid_mask", "valid_until",
                "phase_estimate",
            },
            "ActionChunk",
        )
        until = data.get("valid_until")
        return cls(
            policy_version=data["policy_version"],
            embodiment_id=data["embodiment_id"],
            actuator_layout_id=data["actuator_layout_id"],
            goal_id=data["goal_id"],
            goal_version=data["goal_version"],
            observation_id=data["observation_id"],
            command_type=data["command_type"],
            start=ClockStamp.from_dict(data["start"]),
            dt_seconds=data["dt_seconds"],
            horizon=data["horizon"],
            values=tuple(tuple(row) for row in data["values"]),
            valid_mask=tuple(data["valid_mask"]),
            valid_until=ClockStamp.from_dict(until) if until is not None else None,
            phase_estimate=data.get("phase_estimate"),
        )


@dataclass(frozen=True)
class AttemptOutcome:
    """What happened after one intent was attempted.

    ``motion_status`` reports the motion layer's own state only. A visible
    success claim requires ``evidence_frame_ids``; nothing here is inferred
    from the mere fact that the attempt was emitted.
    """

    intent_id: str
    motion_status: str
    visible_outcome: str
    evidence_frame_ids: Tuple[str, ...] = ()
    uncertainty: float = 1.0
    request_replan: bool = False
    note: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent_id", _require_str(self.intent_id, "intent_id"))
        if self.motion_status not in MOTION_STATUSES:
            raise ValidationError(
                "bad_field",
                f"motion_status must be one of {MOTION_STATUSES}, got {self.motion_status!r}",
            )
        if self.visible_outcome not in VISIBLE_OUTCOMES:
            raise ValidationError(
                "bad_field",
                f"visible_outcome must be one of {VISIBLE_OUTCOMES}, "
                f"got {self.visible_outcome!r}",
            )
        evidence = tuple(_require_str(v, "evidence_frame_ids[]") for v in self.evidence_frame_ids)
        if self.visible_outcome == "observed_success" and not evidence:
            raise ValidationError(
                "missing_evidence",
                "observed_success requires at least one evidence frame id",
            )
        object.__setattr__(self, "evidence_frame_ids", evidence)
        uncertainty = _require_finite(self.uncertainty, "uncertainty")
        if not 0.0 <= uncertainty <= 1.0:
            raise ValidationError(
                "out_of_range", f"uncertainty must lie in [0, 1], got {uncertainty}"
            )
        object.__setattr__(self, "uncertainty", uncertainty)
        if not isinstance(self.request_replan, bool):
            raise ValidationError("bad_field", "request_replan must be a boolean")
        if not isinstance(self.note, str):
            raise ValidationError("bad_field", "note must be a string")
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                "schema_version", f"unsupported schema_version {self.schema_version!r}"
            )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "motion_status": self.motion_status,
            "visible_outcome": self.visible_outcome,
            "evidence_frame_ids": list(self.evidence_frame_ids),
            "uncertainty": self.uncertainty,
            "request_replan": self.request_replan,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttemptOutcome":
        _check_version(data, "AttemptOutcome")
        _check_unknown(
            data,
            {
                "schema_version", "intent_id", "motion_status", "visible_outcome",
                "evidence_frame_ids", "uncertainty", "request_replan", "note",
            },
            "AttemptOutcome",
        )
        return cls(
            intent_id=data["intent_id"],
            motion_status=data["motion_status"],
            visible_outcome=data["visible_outcome"],
            evidence_frame_ids=tuple(data.get("evidence_frame_ids", ())),
            uncertainty=data.get("uncertainty", 1.0),
            request_replan=data.get("request_replan", False),
            note=data.get("note", ""),
        )


@dataclass(frozen=True)
class EvidencedFact:
    """A claimed fact plus the evidence that supports it."""

    fact: str
    evidence_frame_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact", _require_str(self.fact, "fact"))
        evidence = tuple(_require_str(v, "evidence_frame_ids[]") for v in self.evidence_frame_ids)
        if not evidence:
            raise ValidationError(
                "missing_evidence", "evidenced facts require at least one frame id"
            )
        object.__setattr__(self, "evidence_frame_ids", evidence)

    def to_dict(self) -> dict:
        return {"fact": self.fact, "evidence_frame_ids": list(self.evidence_frame_ids)}

    @classmethod
    def from_dict(cls, data: dict) -> "EvidencedFact":
        _check_unknown(data, {"fact", "evidence_frame_ids"}, "EvidencedFact")
        return cls(fact=data["fact"], evidence_frame_ids=tuple(data["evidence_frame_ids"]))


@dataclass(frozen=True)
class MemoryRevision:
    """A correction that supersedes an earlier memory entry."""

    supersedes: str
    fact: str
    evidence_frame_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "supersedes", _require_str(self.supersedes, "supersedes"))
        object.__setattr__(self, "fact", _require_str(self.fact, "fact"))
        object.__setattr__(
            self,
            "evidence_frame_ids",
            tuple(_require_str(v, "evidence_frame_ids[]") for v in self.evidence_frame_ids),
        )

    def to_dict(self) -> dict:
        return {
            "supersedes": self.supersedes,
            "fact": self.fact,
            "evidence_frame_ids": list(self.evidence_frame_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryRevision":
        _check_unknown(data, {"supersedes", "fact", "evidence_frame_ids"}, "MemoryRevision")
        return cls(
            supersedes=data["supersedes"],
            fact=data["fact"],
            evidence_frame_ids=tuple(data.get("evidence_frame_ids", ())),
        )


@dataclass(frozen=True)
class MemoryDelta:
    """Increment to long-term semantic memory.

    ``attempted`` entries record what was tried; ``observed_success``
    entries are separate and must carry observable evidence. A memory
    update never turns an attempt into a success by itself.
    """

    memory_version: int
    source_observation_id: str
    generated_at: ClockStamp
    attempted: Tuple[str, ...] = ()
    observed_success: Tuple[EvidencedFact, ...] = ()
    uncertain: Tuple[str, ...] = ()
    revisions: Tuple[MemoryRevision, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "memory_version", _require_int(self.memory_version, "memory_version", minimum=0)
        )
        object.__setattr__(
            self,
            "source_observation_id",
            _require_str(self.source_observation_id, "source_observation_id"),
        )
        object.__setattr__(self, "generated_at", _stamp(self.generated_at, "generated_at"))
        object.__setattr__(
            self, "attempted", tuple(_require_str(v, "attempted[]") for v in self.attempted)
        )
        object.__setattr__(
            self,
            "observed_success",
            tuple(
                f if isinstance(f, EvidencedFact) else EvidencedFact.from_dict(f)
                for f in self.observed_success
            ),
        )
        object.__setattr__(
            self, "uncertain", tuple(_require_str(v, "uncertain[]") for v in self.uncertain)
        )
        object.__setattr__(
            self,
            "revisions",
            tuple(
                r if isinstance(r, MemoryRevision) else MemoryRevision.from_dict(r)
                for r in self.revisions
            ),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ValidationError(
                "schema_version", f"unsupported schema_version {self.schema_version!r}"
            )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "memory_version": self.memory_version,
            "source_observation_id": self.source_observation_id,
            "generated_at": self.generated_at.to_dict(),
            "attempted": list(self.attempted),
            "observed_success": [f.to_dict() for f in self.observed_success],
            "uncertain": list(self.uncertain),
            "revisions": [r.to_dict() for r in self.revisions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryDelta":
        _check_version(data, "MemoryDelta")
        _check_unknown(
            data,
            {
                "schema_version", "memory_version", "source_observation_id",
                "generated_at", "attempted", "observed_success", "uncertain",
                "revisions",
            },
            "MemoryDelta",
        )
        return cls(
            memory_version=data["memory_version"],
            source_observation_id=data["source_observation_id"],
            generated_at=ClockStamp.from_dict(data["generated_at"]),
            attempted=tuple(data.get("attempted", ())),
            observed_success=tuple(data.get("observed_success", ())),
            uncertain=tuple(data.get("uncertain", ())),
            revisions=tuple(data.get("revisions", ())),
        )
