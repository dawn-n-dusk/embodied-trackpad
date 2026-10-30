"""Embodied Trackpad foundation package (protocol v2 draft).

Typed contracts, runtime validation, an append-only episode recorder, and
a deterministic mock dry-run CLI. This package contains no physics, no
learned policy, no screen capture, and no input injection.
"""

from .clock import ClockStamp
from .contracts import (
    SCHEMA_VERSION,
    ActionChunk,
    AttemptOutcome,
    EpisodeConfig,
    EvidencedFact,
    FramePacket,
    MemoryDelta,
    MemoryRevision,
    MotorIntent,
)
from .errors import ClockDomainError, RecorderError, TrackpadError, ValidationError
from .recorder import EpisodeRecorder
from .validation import Validator

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "ActionChunk",
    "AttemptOutcome",
    "ClockDomainError",
    "ClockStamp",
    "EpisodeConfig",
    "EpisodeRecorder",
    "EvidencedFact",
    "FramePacket",
    "MemoryDelta",
    "MemoryRevision",
    "MotorIntent",
    "RecorderError",
    "TrackpadError",
    "ValidationError",
    "Validator",
    "__version__",
]
