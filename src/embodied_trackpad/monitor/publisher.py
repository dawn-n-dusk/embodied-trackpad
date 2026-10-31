"""Atomic, bounded publisher for the training-monitor protocol.

A trainer/renderer (or a marked test fixture) owns one ``MonitorPublisher``
per run directory. Each publish writes the frame bytes and the versioned
JSON record with tmp-file + fsync + ``os.replace`` so observers never see a
partial file, and every record is validated against
``monitor.protocol.validate_record`` before it lands. Storage is bounded by
construction: one latest frame, one latest record, one static metadata file
and one capped line log. No history is accumulated.

The publisher never invents training signals: ``train_step`` /
``gradient_updates`` default to ``None`` (JSON null), ``source_kind`` is an
explicit constructor argument with no default, and PNG metadata (size, hash)
is derived from the actual bytes handed over, never from caller claims.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .protocol import (
    FRAME_NAME,
    METADATA_NAME,
    RECORD_NAME,
    MonitorProtocolError,
    _atomic_write_bytes,
    png_dimensions,
    validate_record,
)

LOG_NAME = "publisher.log"
LOG_MAX_LINES = 200


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["MonitorPublisher", "png_dimensions", "utc_now_iso"]


class MonitorPublisher:
    """Publishes latest-frame + versioned-record pairs for one run.

    ``source_kind`` is explicit (no default): the producer must declare what
    it is. ``checkpoint`` is required up front when ``policy_kind='learned'``
    and may be updated per publish as training advances.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        run_id: str,
        source_kind: str,
        world_id: str,
        environment_count: int,
        policy_kind: str = "unknown",
        checkpoint: dict | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.source_kind = source_kind
        self.world_id = world_id
        self.environment_count = environment_count
        self.policy_kind = policy_kind
        self.checkpoint = checkpoint
        self._last_frame_seq = -1
        self._last_frame_block: dict | None = None
        self._log_lines: list[str] = []
        # Validate the static identity eagerly (fail before creating anything).
        validate_record(self._build_record(lifecycle="waiting", frame_seq=0,
                                           physics_step=0, sim_time_s=0.0,
                                           frame=None))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "training-monitor/1",
            "run_id": run_id,
            "source_kind": source_kind,
            "world_id": world_id,
            "environment_count": environment_count,
            "policy_kind": policy_kind,
            "checkpoint": checkpoint,
            "pid": os.getpid(),
            "created_utc": utc_now_iso(),
        }
        _atomic_write_bytes(
            self.run_dir / METADATA_NAME,
            (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        self._log("publisher_open")

    def _build_record(self, *, lifecycle, frame_seq, physics_step, sim_time_s,
                      frame, train_step=None, gradient_updates=None,
                      metrics=None, progress=None, checkpoint="__keep__",
                      source_utc=None, heartbeat_unix=None) -> dict:
        import time

        return {
            "schema_version": "training-monitor/1",
            "run_id": self.run_id,
            "source_kind": self.source_kind,
            "lifecycle": lifecycle,
            "frame_seq": frame_seq,
            "physics_step": physics_step,
            "sim_time_s": sim_time_s,
            "source_utc": source_utc if source_utc is not None else utc_now_iso(),
            "producer": {
                "heartbeat_unix": time.time() if heartbeat_unix is None else heartbeat_unix,
                "progress": progress,
            },
            "train_step": train_step,
            "gradient_updates": gradient_updates,
            "world_id": self.world_id,
            "environment_count": self.environment_count,
            "policy_kind": self.policy_kind,
            "checkpoint": self.checkpoint if checkpoint == "__keep__" else checkpoint,
            "metrics": metrics,
            "frame": frame,
        }

    def publish(
        self,
        *,
        lifecycle: str,
        frame_seq: int,
        physics_step: int,
        sim_time_s: float,
        png: bytes | None = None,
        retain_frame: bool = False,
        train_step: int | None = None,
        gradient_updates: int | None = None,
        metrics: dict | None = None,
        progress: float | None = None,
        checkpoint: dict | None = None,
        source_utc: str | None = None,
        rendered_unix: float | None = None,
    ) -> dict:
        """Validate then atomically publish one record (+ optional new frame).

        ``frame_seq`` must strictly increase per run. When ``png`` is None the
        record carries ``frame: null`` (e.g. waiting with no source); pass
        ``retain_frame=True`` on a terminal publish to keep the previously
        published frame block (with its original render time) as the final
        image instead of re-sending the bytes. ``rendered_unix`` should be the
        actual render instant of the supplied PNG; it defaults to now.
        """
        import time

        if frame_seq <= self._last_frame_seq:
            raise MonitorProtocolError(
                f"frame_seq must strictly increase (last {self._last_frame_seq}, "
                f"got {frame_seq})"
            )
        frame_block: dict | None = None
        if png is not None:
            width, height = png_dimensions(png)
            frame_block = {
                "path": FRAME_NAME,
                "sha256": hashlib.sha256(bytes(png)).hexdigest(),
                "width": width,
                "height": height,
                "rendered_unix": time.time() if rendered_unix is None else rendered_unix,
            }
        elif retain_frame:
            frame_block = self._last_frame_block
        record = self._build_record(
            lifecycle=lifecycle,
            frame_seq=frame_seq,
            physics_step=physics_step,
            sim_time_s=sim_time_s,
            frame=frame_block,
            train_step=train_step,
            gradient_updates=gradient_updates,
            metrics=metrics,
            progress=progress,
            checkpoint="__keep__" if checkpoint is None else checkpoint,
            source_utc=source_utc,
        )
        validate_record(record)
        if png is not None:
            _atomic_write_bytes(self.run_dir / FRAME_NAME, bytes(png))
        _atomic_write_bytes(
            self.run_dir / RECORD_NAME,
            (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"),
        )
        self._last_frame_seq = frame_seq
        if frame_block is not None:
            self._last_frame_block = frame_block
        self._log(
            f"publish lifecycle={lifecycle} frame_seq={frame_seq} "
            f"physics_step={physics_step} train_step={train_step} "
            f"gradient_updates={gradient_updates} frame={'yes' if frame_block else 'no'}"
        )
        return record

    def _log(self, message: str) -> None:
        line = json.dumps({"utc": utc_now_iso(), "event": message}, sort_keys=True)
        self._log_lines.append(line)
        if len(self._log_lines) > LOG_MAX_LINES:
            self._log_lines = self._log_lines[-LOG_MAX_LINES:]
        try:
            _atomic_write_bytes(
                self.run_dir / LOG_NAME,
                ("\n".join(self._log_lines) + "\n").encode("utf-8"),
            )
        except OSError:
            pass  # the log is diagnostic; publishing itself must not fail on it
