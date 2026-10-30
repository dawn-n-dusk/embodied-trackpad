"""Append-only episode recorder.

One recorder owns one episode directory. The config and provenance are
written once at creation and are immutable afterwards; events are only
appended to per-stream JSONL files, and every accepted event is flushed
immediately so an interrupted process still leaves readable partial logs.
Teacher and evaluator records live in separate streams from actor-visible
events. Actor events are validated against the strict contract structure
of their kind (round-tripped through ``from_dict``); unknown fields at any
nesting level are rejected instead of name-substring filtering.

Referential integrity is enforced: intents and chunks must reference a
recorded observation, outcomes must reference a recorded intent, and
``observed_success`` evidence ids must be recorded observations. Duplicate
observation ids are rejected (conflicting or identical). Recording an
event never upgrades its semantics: an emitted attempt stays ``attempted``
unless a separate, evidence-backed success record exists. Failed and
interrupted attempts remain in the log by design.

On clean ``close()`` the manifest records ``status: "completed"``. If the
``with`` body raises, the recorder closes with ``status: "interrupted"``
and the exception note; an episode is never silently marked completed
after an exception. Every recorded artifact is checksummed into
``manifest.json``; the manifest does not hash itself.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Type

from .clock import ClockStamp
from .contracts import (
    ActionChunk,
    AttemptOutcome,
    EpisodeConfig,
    FramePacket,
    MemoryDelta,
    MotorIntent,
    require_safe_path_component,
)
from .errors import RecorderError, ValidationError
from .validation import Validator

EVIDENCE_CLASSES = ("mock_software_fixture", "physics_simulation", "real_device")

ACTOR_STREAM = "actor_events.jsonl"
TEACHER_STREAM = "teacher_events.jsonl"
EVALUATOR_STREAM = "evaluator_events.jsonl"
CONFIG_FILE = "config.json"
PROVENANCE_FILE = "provenance.json"
MANIFEST_FILE = "manifest.json"

#: Strict contract structure accepted for each actor event kind. Payloads
#: are re-validated through from_dict, so unknown or extra fields (such as
#: oracle cursor coordinates or reward signals smuggled into nested
#: objects) are rejected rather than filtered by name.
_ACTOR_CONTRACTS: dict[str, Type] = {
    "observation": FramePacket,
    "intent": MotorIntent,
    "action_chunk": ActionChunk,
    "attempt_outcome": AttemptOutcome,
    "memory_delta": MemoryDelta,
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_actor_payload(kind: str, payload: dict) -> None:
    contract = _ACTOR_CONTRACTS.get(kind)
    if contract is None:
        raise RecorderError("whitelist", f"unknown actor event kind {kind!r}")
    if not isinstance(payload, dict):
        raise RecorderError("whitelist", "actor payload must be an object")
    try:
        contract.from_dict(payload)
    except (ValidationError, KeyError, TypeError) as exc:
        code = getattr(exc, "code", "bad_field")
        raise RecorderError(
            "whitelist",
            f"actor {kind} payload failed strict contract validation "
            f"({code}): {exc}",
        ) from exc


class EpisodeRecorder:
    """Append-only recorder bound to one immutable EpisodeConfig."""

    def __init__(
        self,
        root_dir: str | Path,
        config: EpisodeConfig,
        *,
        source_provenance: dict,
        evidence_class: str,
        training_eligible: bool,
        created_utc: Optional[str] = None,
        context_goal: Optional[Tuple[str, int]] = None,
    ) -> None:
        if evidence_class not in EVIDENCE_CLASSES:
            raise RecorderError(
                "bad_field",
                f"evidence_class must be one of {EVIDENCE_CLASSES}, got {evidence_class!r}",
            )
        if not isinstance(training_eligible, bool):
            raise RecorderError(
                "bad_field",
                f"training_eligible must be a boolean, got {type(training_eligible).__name__}",
            )
        if evidence_class == "mock_software_fixture" and training_eligible:
            raise RecorderError(
                "bad_field",
                "mock_software_fixture episodes are never training eligible; "
                "a mock fixture must not masquerade as training data",
            )
        if not isinstance(source_provenance, dict) or not source_provenance.get("source"):
            raise RecorderError(
                "bad_field", "source_provenance must name its 'source' (e.g. package version)"
            )
        # Defense in depth: EpisodeConfig already enforces this; re-check
        # before any filesystem mutation.
        require_safe_path_component(config.episode_id, "episode_id")

        self.config = config
        self.validator = Validator(config)
        self.evidence_class = evidence_class
        self.training_eligible = training_eligible
        self._goal = context_goal
        self._closed = False
        self._counts = {ACTOR_STREAM: 0, TEACHER_STREAM: 0, EVALUATOR_STREAM: 0}
        # Referential integrity state for this episode.
        self._observations: dict[str, str] = {}  # frame_id -> canonical payload json
        self._intent_ids: set[str] = set()

        self.episode_dir = Path(root_dir) / config.episode_id
        try:
            self.episode_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RecorderError(
                "immutable",
                f"episode directory {self.episode_dir.name!r} already exists; "
                f"episodes are append-only and cannot be reopened for rewrite",
            ) from exc

        self._write_once(
            CONFIG_FILE,
            {"kind": "episode_config", "config": config.to_dict()},
        )
        self._write_once(
            PROVENANCE_FILE,
            {
                "kind": "provenance",
                "created_utc": created_utc or _utc_now_iso(),
                "source": dict(source_provenance),
                "evidence_class": evidence_class,
                "training_eligible": self.training_eligible,
            },
        )
        self._streams = {}
        for name in (ACTOR_STREAM, TEACHER_STREAM, EVALUATOR_STREAM):
            self._streams[name] = (self.episode_dir / name).open("a", encoding="utf-8")

    # -- context ---------------------------------------------------------

    def set_goal_context(self, goal_id: str, goal_version: int) -> None:
        """Declare the goal identity that new intents/chunks must match."""

        self._require_open()
        self._goal = (goal_id, goal_version)

    # -- actor-visible records -------------------------------------------

    def record_observation(self, frame: FramePacket, now: Optional[ClockStamp] = None) -> int:
        self._require_open()
        self.validator.validate_frame(frame, now=now)
        canonical = json.dumps(frame.to_dict(), sort_keys=True)
        existing = self._observations.get(frame.frame_id)
        if existing is not None:
            code = "observation_conflict" if existing != canonical else "duplicate_observation"
            raise RecorderError(
                code,
                f"observation id {frame.frame_id!r} was already recorded"
                + (" with different content" if code == "observation_conflict" else ""),
            )
        seq = self._append_actor("observation", frame.to_dict())
        self._observations[frame.frame_id] = canonical
        return seq

    def record_intent(self, intent: MotorIntent, now: ClockStamp) -> int:
        self._require_open()
        goal_id, goal_version = self._require_goal()
        self.validator.validate_intent(
            intent, expected_goal_id=goal_id, expected_goal_version=goal_version, now=now
        )
        self._require_observation(intent.observation_id, f"intent {intent.intent_id!r}")
        seq = self._append_actor("intent", intent.to_dict())
        self._intent_ids.add(intent.intent_id)
        return seq

    def record_action_chunk(self, chunk: ActionChunk, now: Optional[ClockStamp] = None) -> int:
        self._require_open()
        goal_id, goal_version = self._require_goal()
        self.validator.validate_action_chunk(
            chunk, expected_goal_id=goal_id, expected_goal_version=goal_version, now=now
        )
        self._require_observation(chunk.observation_id, "action chunk")
        return self._append_actor("action_chunk", chunk.to_dict())

    def record_outcome(self, outcome: AttemptOutcome) -> int:
        self._require_open()
        self.validator.validate_outcome(outcome)
        if outcome.intent_id not in self._intent_ids:
            raise RecorderError(
                "unknown_intent",
                f"outcome references intent {outcome.intent_id!r}, which was not "
                f"recorded in this episode",
            )
        if outcome.visible_outcome == "observed_success":
            self._require_evidence(outcome.evidence_frame_ids, "attempt outcome")
        return self._append_actor("attempt_outcome", outcome.to_dict())

    def record_memory_delta(self, delta: MemoryDelta, *, min_version: int = 0) -> int:
        self._require_open()
        self.validator.validate_memory_delta(delta, min_version=min_version)
        self._require_observation(
            delta.source_observation_id, f"memory delta v{delta.memory_version}"
        )
        for fact in delta.observed_success:
            self._require_evidence(fact.evidence_frame_ids, f"memory fact {fact.fact!r}")
        return self._append_actor("memory_delta", delta.to_dict())

    # -- privileged records ----------------------------------------------

    def record_teacher(self, kind: str, payload: dict) -> int:
        """Record privileged teacher data; never mixed into actor events."""

        self._require_open()
        return self._append_privileged(TEACHER_STREAM, "teacher", kind, payload)

    def record_evaluator(self, kind: str, payload: dict) -> int:
        """Record privileged evaluator data; never mixed into actor events."""

        self._require_open()
        return self._append_privileged(EVALUATOR_STREAM, "evaluator", kind, payload)

    # -- lifecycle ---------------------------------------------------------

    def close(
        self,
        closed_utc: Optional[str] = None,
        *,
        status: str = "completed",
        note: Optional[str] = None,
    ) -> dict:
        """Flush streams and write the checksum manifest. Idempotent.

        ``status`` is "completed" only for a clean close; an
        exception-terminated recording must pass "interrupted" with a note
        (the context manager does this automatically). ``closed_utc`` may
        pin the close timestamp for deterministic fixtures.
        """

        if self._closed:
            return self._manifest
        if status not in ("completed", "interrupted"):
            raise RecorderError("bad_field", f"status must be completed|interrupted, got {status!r}")
        for handle in self._streams.values():
            handle.flush()
            handle.close()
        files = {}
        for name in (
            CONFIG_FILE,
            PROVENANCE_FILE,
            ACTOR_STREAM,
            TEACHER_STREAM,
            EVALUATOR_STREAM,
        ):
            path = self.episode_dir / name
            files[name] = {"sha256": _sha256(path), "bytes": path.stat().st_size}
        self._manifest = {
            "kind": "episode_manifest",
            "episode_id": self.config.episode_id,
            "evidence_class": self.evidence_class,
            "training_eligible": self.training_eligible,
            "status": status,
            "closed_utc": closed_utc or _utc_now_iso(),
            "event_counts": dict(self._counts),
            "files": files,
            "note": "manifest does not include a checksum of itself",
        }
        if note:
            self._manifest["interruption"] = note
        manifest_path = self.episode_dir / MANIFEST_FILE
        manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._closed = True
        return self._manifest

    def __enter__(self) -> "EpisodeRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.close(status="interrupted", note=f"{exc_type.__name__}: {exc}")
        return False  # never suppress the exception

    # -- internals ---------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise RecorderError("recorder_closed", "recorder is closed; appends are refused")

    def _require_goal(self) -> Tuple[str, int]:
        if self._goal is None:
            raise RecorderError(
                "bad_field", "no goal context set; call set_goal_context first"
            )
        return self._goal

    def _require_observation(self, observation_id: str, owner: str) -> None:
        if observation_id not in self._observations:
            raise RecorderError(
                "unknown_observation",
                f"{owner} references observation {observation_id!r}, which was not "
                f"recorded in this episode",
            )

    def _require_evidence(self, evidence_ids, owner: str) -> None:
        missing = [fid for fid in evidence_ids if fid not in self._observations]
        if missing:
            raise RecorderError(
                "unknown_evidence",
                f"{owner} cites evidence frames {missing}, which were not recorded "
                f"as observations in this episode; success cannot be inferred",
            )

    def _write_once(self, name: str, payload: dict) -> None:
        path = self.episode_dir / name
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except FileExistsError as exc:
            raise RecorderError(
                "immutable", f"{name} already exists and is immutable"
            ) from exc

    def _append_actor(self, kind: str, payload: dict) -> int:
        _check_actor_payload(kind, payload)
        seq = self._counts[ACTOR_STREAM]
        record = {"seq": seq, "kind": kind, "payload": payload}
        handle = self._streams[ACTOR_STREAM]
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()  # an interrupted process still leaves readable logs
        self._counts[ACTOR_STREAM] += 1
        return seq

    def _append_privileged(self, stream: str, role: str, kind: str, payload: dict) -> int:
        if not isinstance(kind, str) or not kind:
            raise RecorderError("bad_field", "privileged record kind must be a string")
        if not isinstance(payload, dict):
            raise RecorderError("bad_field", "privileged payload must be an object")
        seq = self._counts[stream]
        record = {"seq": seq, "role": role, "kind": kind, "payload": payload}
        handle = self._streams[stream]
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        self._counts[stream] += 1
        return seq
