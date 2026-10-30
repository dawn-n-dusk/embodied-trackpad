"""Runtime validation of protocol messages against episode context.

Construction-time checks in ``contracts.py`` catch malformed fields; this
module catches messages that are well-formed but wrong for *this* episode:
mismatched embodiment or actuator layout, out-of-range or nonfinite
references, expired or wrong-goal intents and chunks, and clock-domain
abuse. Rejected messages must not execute.

Normalized actuator references live in [-1, 1]. A row of exact zeros is a
legitimate reference (e.g. "hold centered position") and is never treated
as an invalid action; executability comes only from ``valid_mask``.

Execution-time semantics (when ``now`` is supplied): a chunk is admissible
on the half-open interval ``[start, start + horizon*dt)``, intersected
with ``[issued_at, valid_until)`` for intents and ``(-inf, valid_until)``
when a chunk carries ``valid_until``. A chunk whose whole execution
window has passed is rejected even when ``valid_until`` is absent or lies
further ahead. Passing ``now=None`` performs structural/historical
validation only (for example when re-checking recorded data); live
admission must pass ``now``.
"""

from __future__ import annotations

from typing import Optional

from .clock import ClockStamp, elapsed_ns, is_expired, require_same_domain
from .contracts import (
    ACTION_REPRESENTATIONS,
    ActionChunk,
    AttemptOutcome,
    EpisodeConfig,
    FramePacket,
    MemoryDelta,
    MotorIntent,
)
from .errors import ValidationError

NORMALIZED_ACTION_LIMIT = 1.0
NS_PER_SECOND = 1_000_000_000


class Validator:
    """Validates messages for one episode configuration."""

    def __init__(self, config: EpisodeConfig) -> None:
        self.config = config

    def validate_frame(self, frame: FramePacket, now: Optional[ClockStamp] = None) -> None:
        """Check frame timing identity; never compares across clock domains."""

        if frame.capture.clock_id == frame.receive.clock_id:
            # Same declared domain: receive must not precede capture.
            elapsed_ns(frame.capture, frame.receive)
        if now is not None and now.clock_id == frame.capture.clock_id:
            if frame.capture.time_ns > now.time_ns:
                raise ValidationError(
                    "non_monotonic",
                    f"frame {frame.frame_id!r} capture time is in the future of now",
                )

    def validate_intent(
        self,
        intent: MotorIntent,
        *,
        expected_goal_id: str,
        expected_goal_version: int,
        now: ClockStamp,
    ) -> None:
        """Check an intent against calibration, goal identity, and expiry.

        ``now`` is the live admission time; an intent issued in the future
        of ``now`` is rejected rather than silently accepted.
        """

        if intent.calibration_id != self.config.calibration_id:
            raise ValidationError(
                "calibration_mismatch",
                f"intent calibration {intent.calibration_id!r} does not match episode "
                f"calibration {self.config.calibration_id!r}",
            )
        if intent.goal_id != expected_goal_id or intent.goal_version != expected_goal_version:
            raise ValidationError(
                "goal_mismatch",
                f"intent references goal {intent.goal_id!r} v{intent.goal_version}, "
                f"expected {expected_goal_id!r} v{expected_goal_version}",
            )
        if is_expired(intent.valid_until, now):
            raise ValidationError(
                "expired",
                f"intent {intent.intent_id!r} expired at "
                f"{intent.valid_until.time_ns} ns ({intent.valid_until.clock_id}), "
                f"now is {now.time_ns} ns",
            )
        # issued_at shares valid_until's domain (contract), and is_expired
        # above already proved now is in that same domain.
        if intent.issued_at.time_ns > now.time_ns:
            raise ValidationError(
                "not_yet_valid",
                f"intent {intent.intent_id!r} was issued at "
                f"{intent.issued_at.time_ns} ns, which is in the future of now "
                f"{now.time_ns} ns",
            )

    def validate_action_chunk(
        self,
        chunk: ActionChunk,
        *,
        expected_goal_id: str,
        expected_goal_version: int,
        expected_observation_id: Optional[str] = None,
        now: Optional[ClockStamp] = None,
    ) -> None:
        """Check an action chunk against embodiment, layout, ranges, goal, expiry.

        Zero-valued rows are valid actions; only ``valid_mask`` marks which
        steps may execute. With ``now`` given (live admission) the chunk
        must lie inside its half-open execution window
        ``[start, start + horizon*dt)``; without ``now`` only structural
        and historical checks run.
        """

        if chunk.embodiment_id != self.config.embodiment_id:
            raise ValidationError(
                "embodiment_mismatch",
                f"chunk embodiment {chunk.embodiment_id!r} does not match episode "
                f"embodiment {self.config.embodiment_id!r}",
            )
        if chunk.actuator_layout_id != self.config.actuator_layout_id:
            raise ValidationError(
                "layout_mismatch",
                f"chunk actuator_layout_id {chunk.actuator_layout_id!r} does not "
                f"match episode layout {self.config.actuator_layout_id!r}; the "
                f"chunk's value ordering is bound to a different actuator_order",
            )
        if self.config.action_representation not in ACTION_REPRESENTATIONS:
            raise ValidationError(
                "bad_field",
                f"unsupported action_representation {self.config.action_representation!r}",
            )
        expected_width = self.config.action_dim
        if chunk.action_width != expected_width:
            raise ValidationError(
                "actuator_order_mismatch",
                f"chunk action width {chunk.action_width} does not match "
                f"actuator_order of length {expected_width} for embodiment "
                f"{self.config.embodiment_id!r}; values must follow actuator_order",
            )
        if len(chunk.values) != chunk.horizon:
            raise ValidationError(
                "shape",
                f"values rows {len(chunk.values)} != horizon {chunk.horizon}",
            )
        if len(chunk.valid_mask) != chunk.horizon:
            raise ValidationError(
                "shape",
                f"valid_mask length {len(chunk.valid_mask)} != horizon {chunk.horizon}",
            )
        for r, row in enumerate(chunk.values):
            for c, value in enumerate(row):
                if abs(value) > NORMALIZED_ACTION_LIMIT:
                    raise ValidationError(
                        "out_of_range",
                        f"values[{r}][{c}]={value} exceeds normalized actuator "
                        f"range [-{NORMALIZED_ACTION_LIMIT}, {NORMALIZED_ACTION_LIMIT}]",
                    )
        if chunk.goal_id != expected_goal_id or chunk.goal_version != expected_goal_version:
            raise ValidationError(
                "goal_mismatch",
                f"chunk references goal {chunk.goal_id!r} v{chunk.goal_version}, "
                f"expected {expected_goal_id!r} v{expected_goal_version}",
            )
        if expected_observation_id is not None and chunk.observation_id != expected_observation_id:
            raise ValidationError(
                "observation_mismatch",
                f"chunk was computed from observation {chunk.observation_id!r}, "
                f"expected {expected_observation_id!r}",
            )
        if now is not None:
            # Live admission requires a shared clock; a foreign clock makes
            # the timing unauditable and is an error, not a skipped check.
            require_same_domain(chunk.start, now)
            if chunk.start.time_ns > now.time_ns:
                raise ValidationError(
                    "not_yet_valid",
                    f"chunk start {chunk.start.time_ns} ns is in the future of now "
                    f"{now.time_ns} ns",
                )
            window_ns = int(round(chunk.horizon * chunk.dt_seconds * NS_PER_SECOND))
            if now.time_ns >= chunk.start.time_ns + window_ns:
                raise ValidationError(
                    "expired",
                    f"chunk execution window [start, start + horizon*dt) = "
                    f"[{chunk.start.time_ns}, {chunk.start.time_ns + window_ns}) ns "
                    f"has passed at now {now.time_ns} ns",
                )
            if chunk.valid_until is not None and is_expired(chunk.valid_until, now):
                raise ValidationError(
                    "expired",
                    f"action chunk for goal {chunk.goal_id!r} expired at "
                    f"{chunk.valid_until.time_ns} ns, now is {now.time_ns} ns",
                )

    def validate_outcome(self, outcome: AttemptOutcome) -> None:
        """Outcome contracts already enforce evidence; re-assert the boundary."""

        if outcome.visible_outcome == "observed_success" and not outcome.evidence_frame_ids:
            raise ValidationError(
                "missing_evidence",
                "observed_success requires evidence frame ids; success is never "
                "inferred from attempt emission",
            )

    def validate_memory_delta(self, delta: MemoryDelta, *, min_version: int = 0) -> None:
        """Check memory versioning and evidence separation."""

        if delta.memory_version < min_version:
            raise ValidationError(
                "non_monotonic",
                f"memory_version {delta.memory_version} is below {min_version}",
            )
        for fact in delta.observed_success:
            if not fact.evidence_frame_ids:
                raise ValidationError(
                    "missing_evidence",
                    "observed_success memory facts require evidence frame ids",
                )
