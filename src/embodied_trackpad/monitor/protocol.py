"""Training-monitor wire protocol: versioned record schema, strict validation,
truth-preserving display derivation, and a verified run-directory reader.

A trainer/renderer publishes one latest frame PNG plus one versioned JSON
record into a run directory (see ``publisher.py``). This module is the exact
contract both sides share. It is stdlib-only and performs no simulation, no
motor output, and no training; it only describes and checks producer-reported
metadata. Unknown or malformed data fails closed: validation raises
``MonitorProtocolError`` and display derivation never upgrades doubtful
sources into a training claim.

Record (``latest.json``) minimum fields::

    schema_version      == SCHEMA_VERSION
    run_id              non-empty id string
    source_kind         training_environment | checkpoint_evaluation |
                        scripted_diagnostic | test_fixture
    lifecycle           waiting | training | evaluating | stalled | failed | completed
    frame_seq           int >= 0, strictly increasing per run
    physics_step        int >= 0
    sim_time_s          finite float >= 0
    source_utc          ISO-8601 UTC timestamp of record production
    producer            {"heartbeat_unix": finite > 0, "progress": null | [0,1]}
    train_step          null | int >= 0 (missing means null, never invented 0)
    gradient_updates    null | int >= 0 (missing means null, never invented 0)
    world_id            selected world id string
    environment_count   int >= 1
    policy_kind         learned | scripted | unknown
    checkpoint          null | {"sha256": 64-hex, "version": str};
                        REQUIRED when policy_kind == "learned"
    metrics             optional {"loss" | "reward" | "eval_return": finite | null}
    frame               null | {"path": "frame.png", "sha256": 64-hex,
                                "width": int, "height": int, "rendered_unix": float}

``frame.path`` is one fixed relative name (``frame.png``); absolute paths,
parent traversal and nested paths are rejected.

Reader-side boundaries (``read_verified_record``): producer heartbeat and
frame render timestamps may not lie more than ``CLOCK_TOLERANCE_S`` (5 s) in
the future of the reader's clock — a bounded skew tolerance, not a license
for future-dated freshness; record and frame reads are size-capped
(``MAX_RECORD_BYTES`` / ``MAX_FRAME_BYTES``); the PNG IHDR width/height must
equal the JSON metadata; and symlinks resolving outside the run directory
are rejected.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "training-monitor/1"

SOURCE_KINDS = (
    "training_environment",
    "checkpoint_evaluation",
    "scripted_diagnostic",
    "test_fixture",
)
LIFECYCLES = ("waiting", "training", "evaluating", "stalled", "failed", "completed")
POLICY_KINDS = ("learned", "scripted", "unknown")
METRIC_KEYS = ("loss", "reward", "eval_return")

RECORD_NAME = "latest.json"
FRAME_NAME = "frame.png"
METADATA_NAME = "metadata.json"

MAX_ID_LEN = 128
MAX_DIM = 4096
MAX_ENVIRONMENTS = 1_000_000

# Reader-side resource bounds and clock policy.
MAX_RECORD_BYTES = 1 << 20  # 1 MiB: the record is a small JSON document
MAX_FRAME_BYTES = 64 << 20  # 64 MiB: bounds a 4096x4096 PNG decode input
CLOCK_TOLERANCE_S = 5.0  # producer timestamps may lead the reader clock by <= 5 s

LABEL_BASIS = "producer_reported_pending_independent_acceptance"

# Display statuses; LIVE is only ever transport freshness, never a training claim.
STATUS_LIVE = "LIVE"
STATUS_STALLED = "STALLED"
STATUS_WAITING = "WAITING"
STATUS_REPLAY = "REPLAY"
STATUS_ENDED = "ENDED"
STATUS_FAILED = "FAILED"
STATUS_ERROR = "ERROR"

_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class MonitorProtocolError(ValueError):
    """Raised when a record, frame or path violates the monitor contract."""


def _fail(msg: str) -> None:
    raise MonitorProtocolError(msg)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (
        float("inf"),
        float("-inf"),
    )


def _check_id(value, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_ID_LEN:
        _fail(f"{field} must be a non-empty string of at most {MAX_ID_LEN} chars")
    if any(ch not in _ID_CHARS for ch in value):
        _fail(f"{field} contains characters outside [A-Za-z0-9._-]")
    return value


def _check_sha256(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        _fail(f"{field} must be 64 lowercase hex characters")
    return value


def _check_counter(value, field: str, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        _fail(f"{field} must be an int >= {minimum}")
    return value


def _check_optional_counter(value, field: str) -> int | None:
    if value is None:
        return None
    return _check_counter(value, field)


def _check_finite(value, field: str, minimum: float | None = None) -> float:
    if not _is_finite_num(value):
        _fail(f"{field} must be a finite number")
    v = float(value)
    if minimum is not None and v < minimum:
        _fail(f"{field} must be >= {minimum}")
    return v


def _check_utc(value, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{field} must be a non-empty ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        _fail(f"{field} is not parseable ISO-8601")
    if parsed.tzinfo is None:
        _fail(f"{field} must carry a timezone")
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        _fail(f"{field} must be UTC")
    return value


def _validate_frame_path(value) -> str:
    if not isinstance(value, str) or not value:
        _fail("frame.path must be a non-empty relative filename")
    if value != FRAME_NAME:
        # One fixed relative name: rejects absolute paths, "..", separators.
        _fail(f"frame.path must be exactly the fixed relative name {FRAME_NAME!r}")
    return value


def _validate_frame(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail("frame must be null or an object")
    unknown = set(value) - {"path", "sha256", "width", "height", "rendered_unix"}
    if unknown:
        _fail(f"frame has unknown keys: {sorted(unknown)}")
    path = _validate_frame_path(value.get("path"))
    sha = _check_sha256(value.get("sha256"), "frame.sha256")
    width = _check_counter(value.get("width"), "frame.width", 1)
    height = _check_counter(value.get("height"), "frame.height", 1)
    if width > MAX_DIM or height > MAX_DIM:
        _fail(f"frame dimensions must be <= {MAX_DIM}")
    rendered = _check_finite(value.get("rendered_unix"), "frame.rendered_unix", 0.0)
    return {
        "path": path,
        "sha256": sha,
        "width": width,
        "height": height,
        "rendered_unix": rendered,
    }


def _validate_producer(value) -> dict:
    if not isinstance(value, dict):
        _fail("producer must be an object")
    unknown = set(value) - {"heartbeat_unix", "progress"}
    if unknown:
        _fail(f"producer has unknown keys: {sorted(unknown)}")
    heartbeat = _check_finite(value.get("heartbeat_unix"), "producer.heartbeat_unix", 0.0)
    progress = value.get("progress")
    if progress is not None:
        progress = _check_finite(progress, "producer.progress")
        if not 0.0 <= progress <= 1.0:
            _fail("producer.progress must be in [0, 1]")
    return {"heartbeat_unix": heartbeat, "progress": progress}


def _validate_checkpoint(value, policy_kind: str):
    if value is None:
        if policy_kind == "learned":
            _fail("policy_kind 'learned' requires a checkpoint (sha256 + version)")
        return None
    if not isinstance(value, dict):
        _fail("checkpoint must be null or an object")
    unknown = set(value) - {"sha256", "version"}
    if unknown:
        _fail(f"checkpoint has unknown keys: {sorted(unknown)}")
    sha = _check_sha256(value.get("sha256"), "checkpoint.sha256")
    version = value.get("version")
    if not isinstance(version, str) or not version or len(version) > MAX_ID_LEN:
        _fail("checkpoint.version must be a non-empty string")
    return {"sha256": sha, "version": version}


def _validate_metrics(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        _fail("metrics must be null or an object")
    unknown = set(value) - set(METRIC_KEYS)
    if unknown:
        _fail(f"metrics has unknown keys: {sorted(unknown)}")
    out = {}
    for key, v in value.items():
        if v is None:
            out[key] = None
        else:
            out[key] = _check_finite(v, f"metrics.{key}")
    return out


def png_dimensions(png: bytes) -> tuple[int, int]:
    """(width, height) from actual PNG bytes; rejects non-PNG input."""
    if not isinstance(png, (bytes, bytearray)) or not bytes(png).startswith(
        b"\x89PNG\r\n\x1a\n"
    ):
        raise MonitorProtocolError("frame bytes are not a PNG")
    data = bytes(png)
    if len(data) < 26 or data[12:16] != b"IHDR":
        raise MonitorProtocolError("PNG is missing its IHDR chunk")
    width, height = struct.unpack(">II", data[16:24])
    if width < 1 or height < 1:
        raise MonitorProtocolError("PNG dimensions must be positive")
    return width, height


def validate_record(data) -> dict:
    """Validate one decoded JSON record; return a normalized copy.

    Raises MonitorProtocolError on any violation (fail closed). No defaults
    are invented: absent optional counters stay ``None``.
    """
    if not isinstance(data, dict):
        _fail("record must be a JSON object")
    required = {
        "schema_version",
        "run_id",
        "source_kind",
        "lifecycle",
        "frame_seq",
        "physics_step",
        "sim_time_s",
        "source_utc",
        "producer",
        "train_step",
        "gradient_updates",
        "world_id",
        "environment_count",
        "policy_kind",
        "checkpoint",
        "frame",
    }
    missing = required - set(data)
    if missing:
        _fail(f"record is missing required fields: {sorted(missing)}")
    unknown = set(data) - required - {"metrics"}
    if unknown:
        _fail(f"record has unknown fields: {sorted(unknown)}")
    if data["schema_version"] != SCHEMA_VERSION:
        _fail(f"schema_version must be {SCHEMA_VERSION!r}")
    source_kind = data["source_kind"]
    if source_kind not in SOURCE_KINDS:
        _fail(f"source_kind must be one of {SOURCE_KINDS}")
    lifecycle = data["lifecycle"]
    if lifecycle not in LIFECYCLES:
        _fail(f"lifecycle must be one of {LIFECYCLES}")
    policy_kind = data["policy_kind"]
    if policy_kind not in POLICY_KINDS:
        _fail(f"policy_kind must be one of {POLICY_KINDS}")
    environment_count = _check_counter(data["environment_count"], "environment_count", 1)
    if environment_count > MAX_ENVIRONMENTS:
        _fail(f"environment_count must be <= {MAX_ENVIRONMENTS}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _check_id(data["run_id"], "run_id"),
        "source_kind": source_kind,
        "lifecycle": lifecycle,
        "frame_seq": _check_counter(data["frame_seq"], "frame_seq"),
        "physics_step": _check_counter(data["physics_step"], "physics_step"),
        "sim_time_s": _check_finite(data["sim_time_s"], "sim_time_s", 0.0),
        "source_utc": _check_utc(data["source_utc"], "source_utc"),
        "producer": _validate_producer(data["producer"]),
        "train_step": _check_optional_counter(data["train_step"], "train_step"),
        "gradient_updates": _check_optional_counter(
            data["gradient_updates"], "gradient_updates"
        ),
        "world_id": _check_id(data["world_id"], "world_id"),
        "environment_count": environment_count,
        "policy_kind": policy_kind,
        "checkpoint": _validate_checkpoint(data["checkpoint"], policy_kind),
        "metrics": _validate_metrics(data.get("metrics")),
        "frame": _validate_frame(data["frame"]),
    }
    return record


def is_training_active(record: dict) -> bool:
    """Strict conjunction: real training source + declared training phase +
    nonzero measured gradient updates. A counter alone never suffices."""
    return (
        record["source_kind"] == "training_environment"
        and record["lifecycle"] == "training"
        and record["gradient_updates"] is not None
        and record["gradient_updates"] > 0
    )


def derive_display(
    record: dict | None,
    *,
    heartbeat_age_s: float | None = None,
    frame_age_s: float | None = None,
    stale_after_s: float = 10.0,
    historical: bool = False,
) -> dict:
    """Truth-preserving display state for one validated record (or None).

    Status precedence: terminal lifecycles (FAILED/ENDED) first; then
    ``historical`` (a record that already existed when the viewer started) as
    REPLAY, never LIVE; then a missing frame or declared ``waiting`` as
    WAITING; then a declared ``stalled``; then test_fixture /
    scripted_diagnostic sources as TEST / DEMO when fresh and STALLED when
    not — fixtures are never LIVE. Only a fresh, framed, real source may be
    LIVE. ``training_declared`` is the producer-side conjunction; the
    returned ``training_active`` additionally requires the current display to
    be a fresh LIVE training view (not stale, historical, or terminal).
    Fixture/scripted sources always carry a TEST/DEMO badge.
    """
    if record is None:
        return {
            "status": STATUS_WAITING,
            "activity": "no_source",
            "training_active": False,
            "training_declared": False,
            "live": False,
            "fresh": False,
            "source_badge": None,
            "notice_zh": "尚无有效来源记录：等待训练器/渲染器发布首条记录",
            "label_basis": LABEL_BASIS,
        }

    badge = {"test_fixture": "TEST", "scripted_diagnostic": "DEMO"}.get(
        record["source_kind"]
    )
    lifecycle = record["lifecycle"]
    training_declared = is_training_active(record)

    if lifecycle == "completed":
        activity = "ended"
    elif lifecycle == "failed":
        activity = "failed"
    elif record["source_kind"] == "test_fixture":
        activity = "test"
    elif record["source_kind"] == "scripted_diagnostic":
        activity = "demo"
    elif lifecycle == "waiting":
        activity = "waiting"
    elif lifecycle == "stalled":
        activity = "stalled"
    elif lifecycle == "evaluating":
        activity = "evaluating"
    elif training_declared:
        activity = "training_active"
    else:
        activity = "unverified_training"

    heartbeat_fresh = heartbeat_age_s is not None and heartbeat_age_s <= stale_after_s
    frame_fresh = (
        record["frame"] is not None
        and frame_age_s is not None
        and frame_age_s <= stale_after_s
    )
    fresh = heartbeat_fresh and frame_fresh

    if lifecycle == "failed":
        status = STATUS_FAILED
    elif lifecycle == "completed":
        status = STATUS_ENDED
    elif historical:
        status = STATUS_REPLAY
    elif record["frame"] is None or lifecycle == "waiting":
        status = STATUS_WAITING  # never LIVE without an actual frame
    elif lifecycle == "stalled":
        status = STATUS_STALLED
    elif badge is not None:
        status = badge if fresh else STATUS_STALLED  # TEST / DEMO, never LIVE
    else:
        status = STATUS_LIVE if fresh else STATUS_STALLED

    training_active = training_declared and status == STATUS_LIVE

    notices = {
        "no_source": "尚无有效来源记录：等待训练器/渲染器发布首条记录",
        "waiting": "等待训练环境/数据：暂无来源画面",
        "training_active": "训练进行中（生产者上报，未经独立验收）",
        "unverified_training": "训练来源已声明但尚无实测梯度更新：不能确认为训练中",
        "evaluating": "检查点评估中（生产者上报，未经独立验收）",
        "test": "测试影像：来源为测试夹具，非真实训练",
        "demo": "脚本演示：非学习策略，非训练",
        "stalled": "来源已声明停滞",
        "ended": "已结束：保留最终画面，非实时",
        "failed": "已失败：保留最终画面，非实时",
    }
    notice = notices[activity]
    if status == STATUS_REPLAY:
        notice = "历史画面： viewer 启动前已存在，非当前训练"
    elif status == STATUS_WAITING and activity not in ("waiting", "no_source"):
        notice = notices[activity] + "；等待来源画面，暂无帧发布"
    elif status == STATUS_STALLED and activity not in ("stalled",):
        notice = notices[activity] + "；心跳/画面超时未更新"
    return {
        "status": status,
        "activity": activity,
        "training_active": training_active,
        "training_declared": training_declared,
        "live": status == STATUS_LIVE,
        "fresh": fresh,
        "source_badge": badge,
        "notice_zh": notice,
        "label_basis": LABEL_BASIS,
    }


@dataclass(frozen=True)
class Snapshot:
    """One verified run-directory observation (record + exact frame bytes)."""

    record: dict
    frame_bytes: bytes | None
    read_unix: float

    @property
    def identity(self) -> tuple:
        frame = self.record["frame"]
        return (
            self.record["run_id"],
            self.record["frame_seq"],
            None if frame is None else frame["sha256"],
        )


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass  # directory fsync is best-effort durability, not correctness


def _bounded_child(root: Path, name: str, field: str) -> Path:
    """Resolve ``root/name``; reject anything escaping the run directory.

    A symlink whose target lives outside the (resolved) run directory fails
    closed; plain files and in-directory links pass through.
    """
    root_real = root.resolve()
    resolved = (root / name).resolve(strict=False)
    if resolved.parent != root_real:
        _fail(f"{field} resolves outside the run directory")
    return resolved


def _read_capped(path: Path, cap: int, field: str) -> bytes:
    """Read at most ``cap`` bytes; fail closed on larger files."""
    with open(path, "rb") as fh:  # FileNotFoundError propagates
        data = fh.read(cap + 1)
    if len(data) > cap:
        _fail(f"{field} exceeds the {cap}-byte read cap")
    return data


def read_verified_record(run_dir: str | Path, *, now_unix: float | None = None) -> Snapshot:
    """Read and verify the latest record + frame pair from ``run_dir``.

    Fails closed with MonitorProtocolError on malformed JSON, invalid fields,
    a missing frame file, non-PNG bytes, a content-hash mismatch, a PNG IHDR
    size that disagrees with the JSON metadata, oversized files, symlinks
    escaping the run directory, or producer timestamps more than
    ``CLOCK_TOLERANCE_S`` ahead of the reader clock (future timestamps are
    rejected, never clipped to age 0). Absent record file raises
    FileNotFoundError so callers can distinguish "no source yet" from
    "present but invalid".
    """
    import time

    now = time.time() if now_unix is None else now_unix
    run_path = Path(run_dir)
    record_path = _bounded_child(run_path, RECORD_NAME, "record path")
    raw = _read_capped(record_path, MAX_RECORD_BYTES, "record")  # may raise FileNotFoundError
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"record is not valid JSON: {exc}")
    record = validate_record(data)
    heartbeat = record["producer"]["heartbeat_unix"]
    if heartbeat > now + CLOCK_TOLERANCE_S:
        _fail(
            f"producer.heartbeat_unix is {heartbeat - now:.1f}s in the future "
            f"(beyond the {CLOCK_TOLERANCE_S:g}s clock tolerance)"
        )
    frame_bytes: bytes | None = None
    frame = record["frame"]
    if frame is not None:
        if frame["rendered_unix"] > now + CLOCK_TOLERANCE_S:
            _fail(
                f"frame.rendered_unix is {frame['rendered_unix'] - now:.1f}s in the "
                f"future (beyond the {CLOCK_TOLERANCE_S:g}s clock tolerance)"
            )
        frame_path = _bounded_child(run_path, frame["path"], "frame path")
        try:
            frame_bytes = _read_capped(frame_path, MAX_FRAME_BYTES, "frame")
        except FileNotFoundError:
            _fail(f"frame file {frame['path']!r} is missing for a non-null frame record")
        if not frame_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            _fail("frame bytes are not a PNG")
        digest = hashlib.sha256(frame_bytes).hexdigest()
        if digest != frame["sha256"]:
            _fail(
                f"frame content hash mismatch: record {frame['sha256'][:16]}… "
                f"vs file {digest[:16]}…"
            )
        width, height = png_dimensions(frame_bytes)
        if width != frame["width"] or height != frame["height"]:
            _fail(
                f"PNG IHDR {width}x{height} disagrees with record metadata "
                f"{frame['width']}x{frame['height']}"
            )
    return Snapshot(
        record=record,
        frame_bytes=frame_bytes,
        read_unix=now,
    )
