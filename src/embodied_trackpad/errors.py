"""Structured validation and recording errors for embodied_trackpad.

Every rejection carries a stable machine-readable ``code`` so callers and
fixtures can assert on the reason without matching message text.
"""

from __future__ import annotations


class TrackpadError(Exception):
    """Base class for all package errors."""


class ValidationError(TrackpadError):
    """A message failed contract or runtime validation and must not execute."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")


class ClockDomainError(ValidationError):
    """Monotonic values from incompatible clock domains were compared."""

    def __init__(self, message: str) -> None:
        super().__init__("clock_domain", message)


class RecorderError(TrackpadError):
    """Append-only recorder misuse (closed, immutable rewrite, whitelist)."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"[{code}] {message}")
