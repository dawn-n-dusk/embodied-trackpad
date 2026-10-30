"""Clock-domain handling.

Monotonic timestamps are only comparable inside one declared clock domain
(``clock_id``). Values from different hosts or clocks are never subtracted,
and no cross-host frame age is invented: callers must either share one
domain or carry an externally estimated ``age_estimate_ms`` with an
explicit uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ClockDomainError, ValidationError

NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class ClockStamp:
    """An integer nanosecond reading tied to a named monotonic clock domain."""

    clock_id: str
    time_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.clock_id, str) or not self.clock_id:
            raise ValidationError("bad_field", "clock_id must be a non-empty string")
        if isinstance(self.time_ns, bool) or not isinstance(self.time_ns, int):
            raise ValidationError("bad_field", "time_ns must be an integer")
        if self.time_ns < 0:
            raise ValidationError("bad_field", "time_ns must be non-negative")

    def to_dict(self) -> dict:
        return {"clock_id": self.clock_id, "time_ns": self.time_ns}

    @classmethod
    def from_dict(cls, data: dict) -> "ClockStamp":
        if not isinstance(data, dict):
            raise ValidationError("bad_field", "clock stamp must be an object")
        unknown = set(data) - {"clock_id", "time_ns"}
        if unknown:
            raise ValidationError("bad_field", f"unknown clock stamp fields: {sorted(unknown)}")
        return cls(clock_id=data["clock_id"], time_ns=data["time_ns"])


def require_same_domain(a: ClockStamp, b: ClockStamp) -> None:
    """Raise unless both stamps come from the same declared clock domain."""

    if a.clock_id != b.clock_id:
        raise ClockDomainError(
            f"cannot compare monotonic times across clock domains "
            f"{a.clock_id!r} and {b.clock_id!r}; provide an explicit "
            f"synchronization estimate instead"
        )


def elapsed_ns(start: ClockStamp, end: ClockStamp) -> int:
    """Nanoseconds between two stamps in one domain; negative is an error."""

    require_same_domain(start, end)
    delta = end.time_ns - start.time_ns
    if delta < 0:
        raise ValidationError(
            "non_monotonic",
            f"clock {start.clock_id!r} moved backwards: "
            f"{start.time_ns} -> {end.time_ns}",
        )
    return delta


def is_expired(valid_until: ClockStamp, now: ClockStamp) -> bool:
    """True when ``now`` is past ``valid_until`` in the same domain.

    Raises ClockDomainError for cross-domain comparisons instead of
    guessing whether the message is still valid.
    """

    require_same_domain(valid_until, now)
    return now.time_ns > valid_until.time_ns
