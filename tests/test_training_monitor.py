"""Protocol tests for the training-monitor interface (stdlib only, no server).

Covers the contract boundaries: fixtures/scripted sources can never appear as
real training, the training-active conjunction, checkpoint/missing-field
rules, monotonic sequence enforcement, stale-producer staleness while the
HTTP layer stays alive, terminal completion/failure, atomic/hash-mismatch
fail-closed reads, fixed-path rejection, startup-floor replay protection and
metadata-only polls never resetting freshness. Live socket behavior is
validated separately and recorded under .local/training-monitor/.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import time
import zlib
from pathlib import Path

import pytest

from embodied_trackpad.monitor import (
    MonitorProtocolError,
    MonitorPublisher,
    derive_display,
    is_training_active,
    read_verified_record,
    validate_record,
)

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "training_live_view.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("training_live_view", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def make_png(width: int = 4, height: int = 3, seed: int = 0) -> bytes:
    """Minimal stdlib RGB PNG; seed varies pixels so hashes differ."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(
        b"\x00" + bytes(((seed + i * width + j) % 251 for j in range(width * 3)))
        for i in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


PNG_A = make_png(seed=1)
PNG_B = make_png(seed=7)
NOW = 1_760_000_000.0


def base_record(**over):
    rec = {
        "schema_version": "training-monitor/1",
        "run_id": "run-test-1",
        "source_kind": "training_environment",
        "lifecycle": "training",
        "frame_seq": 3,
        "physics_step": 1200,
        "sim_time_s": 2.4,
        "source_utc": "2026-10-31T04:00:00Z",
        "producer": {"heartbeat_unix": NOW, "progress": 0.5},
        "train_step": 40,
        "gradient_updates": 40,
        "world_id": "world_0",
        "environment_count": 32,
        "policy_kind": "learned",
        "checkpoint": {"sha256": hashlib.sha256(b"ckpt").hexdigest(), "version": "ckpt-40"},
        "metrics": {"loss": 0.25, "reward": None},
        "frame": None,
    }
    rec.update(over)
    return rec


def make_publisher(run_dir, **kw):
    args = dict(
        run_id="run-test-1",
        source_kind="test_fixture",
        world_id="world_0",
        environment_count=1,
        policy_kind="scripted",
    )
    args.update(kw)
    return MonitorPublisher(run_dir, **args)


# ---------------------------------------------------------------------------
# validation: fields, checkpoints, missing counters, paths, finiteness
# ---------------------------------------------------------------------------


def test_valid_record_roundtrip():
    rec = validate_record(base_record())
    assert rec["gradient_updates"] == 40
    assert rec["metrics"]["loss"] == 0.25
    assert rec["metrics"]["reward"] is None


def test_missing_gradient_updates_stays_null_never_zero():
    rec = validate_record(base_record(gradient_updates=None, train_step=None))
    assert rec["gradient_updates"] is None
    assert rec["train_step"] is None
    assert "gradient_updates" in rec  # key present, value null


@pytest.mark.parametrize("field,value", [
    ("sim_time_s", float("nan")),
    ("sim_time_s", float("inf")),
    ("frame_seq", -1),
    ("frame_seq", 1.5),
    ("frame_seq", True),
    ("physics_step", -2),
    ("environment_count", 0),
    ("environment_count", 2.5),
    ("train_step", -1),
    ("gradient_updates", -3),
    ("run_id", ""),
    ("run_id", "bad id/with slash"),
    ("source_kind", "trainig"),
    ("lifecycle", "TRAINING"),
    ("policy_kind", "learned-ish"),
    ("schema_version", "training-monitor/0"),
])
def test_field_rejection(field, value):
    with pytest.raises(MonitorProtocolError):
        validate_record(base_record(**{field: value}))


def test_partial_record_rejected():
    rec = base_record()
    del rec["gradient_updates"]
    with pytest.raises(MonitorProtocolError):
        validate_record(rec)
    rec = base_record()
    del rec["producer"]
    with pytest.raises(MonitorProtocolError):
        validate_record(rec)
    with pytest.raises(MonitorProtocolError):
        validate_record({"schema_version": "training-monitor/1"})
    with pytest.raises(MonitorProtocolError):
        validate_record([1, 2, 3])


def test_unknown_fields_rejected():
    with pytest.raises(MonitorProtocolError):
        validate_record(base_record(extra_field=1))
    rec = base_record()
    rec["metrics"]["mystery"] = 1.0
    with pytest.raises(MonitorProtocolError):
        validate_record(rec)


def test_nonfinite_metrics_and_progress_rejected():
    rec = base_record()
    rec["metrics"]["loss"] = float("inf")
    with pytest.raises(MonitorProtocolError):
        validate_record(rec)
    rec = base_record()
    rec["producer"]["progress"] = float("nan")
    with pytest.raises(MonitorProtocolError):
        validate_record(rec)
    rec = base_record()
    rec["producer"]["progress"] = 1.2
    with pytest.raises(MonitorProtocolError):
        validate_record(rec)
    rec = base_record()
    rec["producer"]["heartbeat_unix"] = float("-inf")
    with pytest.raises(MonitorProtocolError):
        validate_record(rec)


def test_learned_policy_requires_checkpoint():
    with pytest.raises(MonitorProtocolError):
        validate_record(base_record(checkpoint=None))  # learned without checkpoint
    with pytest.raises(MonitorProtocolError):
        validate_record(base_record(checkpoint={"sha256": "xyz", "version": "v"}))
    with pytest.raises(MonitorProtocolError):
        validate_record(
            base_record(checkpoint={"sha256": hashlib.sha256(b"c").hexdigest()})
        )  # missing version
    ok = validate_record(base_record(policy_kind="scripted", checkpoint=None))
    assert ok["checkpoint"] is None


@pytest.mark.parametrize("path", [
    "/etc/passwd",
    "../frame.png",
    "sub/frame.png",
    "./frame.png",
    "frame.png/..",
    "",
    "FRAME.PNG",
])
def test_frame_path_rejection(path):
    frame = {
        "path": path,
        "sha256": hashlib.sha256(PNG_A).hexdigest(),
        "width": 4,
        "height": 3,
        "rendered_unix": NOW,
    }
    with pytest.raises(MonitorProtocolError):
        validate_record(base_record(frame=frame))


def test_frame_block_validation():
    good = {
        "path": "frame.png",
        "sha256": hashlib.sha256(PNG_A).hexdigest(),
        "width": 4,
        "height": 3,
        "rendered_unix": NOW,
    }
    assert validate_record(base_record(frame=good))["frame"]["width"] == 4
    for bad in (dict(good, width=0), dict(good, height=5000),
                dict(good, rendered_unix=float("nan")), dict(good, sha256="0" * 63)):
        with pytest.raises(MonitorProtocolError):
            validate_record(base_record(frame=bad))


# ---------------------------------------------------------------------------
# training-active conjunction and fixture/demo override
# ---------------------------------------------------------------------------


def test_training_active_requires_full_conjunction():
    assert is_training_active(validate_record(base_record())) is True
    # counter alone is not enough: each leg of the conjunction matters
    assert is_training_active(validate_record(base_record(gradient_updates=0))) is False
    assert is_training_active(validate_record(base_record(gradient_updates=None))) is False
    assert is_training_active(validate_record(base_record(lifecycle="evaluating"))) is False
    assert is_training_active(
        validate_record(base_record(source_kind="checkpoint_evaluation"))
    ) is False
    assert is_training_active(
        validate_record(base_record(source_kind="test_fixture"))
    ) is False


def test_fixture_cannot_appear_as_training_even_if_lifecycle_lies():
    # Inconsistent fixture: lifecycle claims training with real-looking numbers.
    rec = validate_record(base_record(source_kind="test_fixture"))
    display = derive_display(rec, heartbeat_age_s=0.1, frame_age_s=0.1, stale_after_s=10.0)
    assert display["training_active"] is False
    assert display["activity"] == "test"
    assert display["source_badge"] == "TEST"
    assert "测试" in display["notice_zh"]
    demo = derive_display(
        validate_record(base_record(source_kind="scripted_diagnostic")),
        heartbeat_age_s=0.1,
        frame_age_s=0.1,
        stale_after_s=10.0,
    )
    assert demo["training_active"] is False
    assert demo["activity"] == "demo"
    assert demo["source_badge"] == "DEMO"
    assert "演示" in demo["notice_zh"]


def test_unverified_training_claim_is_not_training_active():
    rec = validate_record(base_record(gradient_updates=None))
    display = derive_display(rec, heartbeat_age_s=0.1, stale_after_s=10.0)
    assert display["activity"] == "unverified_training"
    assert display["training_active"] is False
    assert "不能确认为训练中" in display["notice_zh"]
    assert display["label_basis"] == "producer_reported_pending_independent_acceptance"


def test_waiting_no_source_and_no_record_labels():
    rec = validate_record(base_record(lifecycle="waiting", frame=None))
    display = derive_display(rec, heartbeat_age_s=0.1, stale_after_s=10.0)
    assert display["status"] == "WAITING"
    assert display["live"] is False
    assert "暂无来源画面" in display["notice_zh"]
    none_display = derive_display(None)
    assert none_display["status"] == "WAITING"
    assert none_display["activity"] == "no_source"


def test_terminal_lifecycle_labels():
    for lifecycle, status, activity in (
        ("completed", "ENDED", "ended"),
        ("failed", "FAILED", "failed"),
    ):
        rec = validate_record(base_record(lifecycle=lifecycle))
        display = derive_display(rec, heartbeat_age_s=0.1, stale_after_s=10.0)
        assert display["status"] == status
        assert display["activity"] == activity
        assert display["live"] is False
        assert display["training_active"] is False
        assert "非实时" in display["notice_zh"]


def test_staleness_boundaries_and_historical_replay():
    rec = validate_record(
        base_record(
            frame={
                "path": "frame.png",
                "sha256": hashlib.sha256(PNG_A).hexdigest(),
                "width": 4,
                "height": 3,
                "rendered_unix": NOW,
            }
        )
    )
    fresh = derive_display(rec, heartbeat_age_s=9.9, frame_age_s=9.9, stale_after_s=10.0)
    assert fresh["status"] == "LIVE"
    stale_hb = derive_display(rec, heartbeat_age_s=10.1, frame_age_s=0.1, stale_after_s=10.0)
    assert stale_hb["status"] == "STALLED"
    stale_fr = derive_display(rec, heartbeat_age_s=0.1, frame_age_s=10.1, stale_after_s=10.0)
    assert stale_fr["status"] == "STALLED"
    replay = derive_display(rec, heartbeat_age_s=0.1, frame_age_s=0.1,
                            stale_after_s=10.0, historical=True)
    assert replay["status"] == "REPLAY"
    assert replay["live"] is False
    assert replay["training_active"] is False  # a historical training record is still not current
    assert "历史画面" in replay["notice_zh"]


def _framed_record(**over):
    over.setdefault("frame", {
        "path": "frame.png",
        "sha256": hashlib.sha256(PNG_A).hexdigest(),
        "width": 4,
        "height": 3,
        "rendered_unix": NOW,
    })
    return validate_record(base_record(**over))


def test_fixture_demo_statuses_are_never_live():
    fx = _framed_record(source_kind="test_fixture")
    fresh = derive_display(fx, heartbeat_age_s=0.1, frame_age_s=0.1, stale_after_s=10.0)
    assert fresh["status"] == "TEST"  # fresh fixture: TEST, never LIVE
    assert fresh["live"] is False
    assert fresh["training_active"] is False
    assert fresh["source_badge"] == "TEST"
    stale = derive_display(fx, heartbeat_age_s=99.0, frame_age_s=0.1, stale_after_s=10.0)
    assert stale["status"] == "STALLED"
    assert stale["source_badge"] == "TEST"  # badge kept while stale
    demo = derive_display(
        _framed_record(source_kind="scripted_diagnostic"),
        heartbeat_age_s=0.1, frame_age_s=0.1, stale_after_s=10.0,
    )
    assert demo["status"] == "DEMO"
    assert demo["live"] is False
    assert demo["source_badge"] == "DEMO"


def test_terminal_beats_historical_and_missing_frame_is_waiting():
    term = _framed_record(lifecycle="completed")
    done = derive_display(term, heartbeat_age_s=0.1, frame_age_s=0.1,
                          stale_after_s=10.0, historical=True)
    assert done["status"] == "ENDED"  # terminal first, not REPLAY
    noframe = validate_record(base_record())  # claims training, has no frame
    d2 = derive_display(noframe, heartbeat_age_s=0.1, stale_after_s=10.0)
    assert d2["status"] == "WAITING"  # never LIVE without an actual frame
    assert d2["training_active"] is False
    assert d2["training_declared"] is True  # declaration kept separate


def test_training_active_requires_current_fresh_live_display():
    rec = _framed_record()
    fresh = derive_display(rec, heartbeat_age_s=0.1, frame_age_s=0.1, stale_after_s=10.0)
    assert fresh["status"] == "LIVE"
    assert fresh["training_active"] is True
    assert fresh["training_declared"] is True
    stale = derive_display(rec, heartbeat_age_s=30.0, frame_age_s=0.1, stale_after_s=10.0)
    assert stale["status"] == "STALLED"
    assert stale["training_active"] is False  # stale display cannot be active training
    assert stale["training_declared"] is True
    hist = derive_display(rec, heartbeat_age_s=0.1, frame_age_s=0.1,
                          stale_after_s=10.0, historical=True)
    assert hist["status"] == "REPLAY"
    assert hist["training_active"] is False


# ---------------------------------------------------------------------------
# publisher: atomic write, monotonic sequence, bounded log, verified reads
# ---------------------------------------------------------------------------


def test_publisher_roundtrip_and_verified_read(tmp_path):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="waiting", frame_seq=0, physics_step=0, sim_time_s=0.0)
    snap = read_verified_record(tmp_path)
    assert snap.record["lifecycle"] == "waiting"
    assert snap.frame_bytes is None
    assert snap.record["gradient_updates"] is None  # never invented zero
    pub.publish(lifecycle="training", frame_seq=1, physics_step=50, sim_time_s=0.1,
                png=PNG_A, gradient_updates=0, metrics={"loss": 1.0})
    snap = read_verified_record(tmp_path)
    assert snap.frame_bytes == PNG_A
    assert snap.record["frame"]["width"] == 4
    assert snap.record["frame"]["sha256"] == hashlib.sha256(PNG_A).hexdigest()
    assert (tmp_path / "metadata.json").exists()
    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert meta["source_kind"] == "test_fixture"


def test_publisher_monotonic_sequence(tmp_path):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="waiting", frame_seq=0, physics_step=0, sim_time_s=0.0)
    with pytest.raises(MonitorProtocolError):
        pub.publish(lifecycle="waiting", frame_seq=0, physics_step=0, sim_time_s=0.0)
    with pytest.raises(MonitorProtocolError):
        pub.publish(lifecycle="waiting", frame_seq=0, physics_step=1, sim_time_s=0.0)
    pub.publish(lifecycle="waiting", frame_seq=1, physics_step=1, sim_time_s=0.0)


def test_publisher_rejects_invalid_before_write(tmp_path):
    pub = make_publisher(tmp_path)
    with pytest.raises(MonitorProtocolError):
        pub.publish(lifecycle="training", frame_seq=1, physics_step=0, sim_time_s=0.0,
                    png=b"not a png")
    with pytest.raises(MonitorProtocolError):
        pub.publish(lifecycle="bogus", frame_seq=1, physics_step=0, sim_time_s=0.0)
    with pytest.raises(MonitorProtocolError):
        pub.publish(lifecycle="training", frame_seq=1, physics_step=0,
                    sim_time_s=float("nan"))
    assert not (tmp_path / "latest.json").exists()  # nothing landed


def test_hash_mismatch_and_missing_frame_fail_closed(tmp_path):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    (tmp_path / "frame.png").write_bytes(PNG_B)  # corrupt/replace after publish
    with pytest.raises(MonitorProtocolError):
        read_verified_record(tmp_path)
    (tmp_path / "frame.png").unlink()
    with pytest.raises(MonitorProtocolError):
        read_verified_record(tmp_path)
    (tmp_path / "latest.json").write_text("{truncated")
    with pytest.raises(MonitorProtocolError):
        read_verified_record(tmp_path)
    (tmp_path / "latest.json").unlink()
    with pytest.raises(FileNotFoundError):
        read_verified_record(tmp_path)  # absent is distinguishable from invalid


def test_reader_rejects_future_producer_timestamps(tmp_path):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    rec = json.loads((tmp_path / "latest.json").read_text())
    rec["producer"]["heartbeat_unix"] = time.time() + 3600.0  # far future: reject
    (tmp_path / "latest.json").write_text(json.dumps(rec) + "\n")
    with pytest.raises(MonitorProtocolError):
        read_verified_record(tmp_path)
    rec["producer"]["heartbeat_unix"] = time.time() + 4.0  # within 5 s tolerance
    (tmp_path / "latest.json").write_text(json.dumps(rec) + "\n")
    assert read_verified_record(tmp_path).record["run_id"] == "run-test-1"
    # a future render instant is rejected too, never clipped to age 0
    pub.publish(lifecycle="training", frame_seq=2, physics_step=2, sim_time_s=0.2,
                png=PNG_A, rendered_unix=time.time() + 3600.0)
    with pytest.raises(MonitorProtocolError):
        read_verified_record(tmp_path)


def test_reader_rejects_ihdr_metadata_mismatch(tmp_path):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    rec = json.loads((tmp_path / "latest.json").read_text())
    rec["frame"]["width"] = 5  # PNG_A is really 4x3
    (tmp_path / "latest.json").write_text(json.dumps(rec) + "\n")
    with pytest.raises(MonitorProtocolError):
        read_verified_record(tmp_path)


def test_reader_rejects_symlink_escaping_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    pub = make_publisher(run_dir)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    outside_png = tmp_path / "outside.png"
    outside_png.write_bytes(PNG_A)  # correct bytes, wrong location
    (run_dir / "frame.png").unlink()
    (run_dir / "frame.png").symlink_to(outside_png)
    with pytest.raises(MonitorProtocolError):
        read_verified_record(run_dir)
    outside_json = tmp_path / "outside.json"
    outside_json.write_text(json.dumps(base_record()) + "\n")
    (run_dir / "latest.json").unlink()
    (run_dir / "latest.json").symlink_to(outside_json)
    with pytest.raises(MonitorProtocolError):
        read_verified_record(run_dir)


def test_reader_caps_oversized_record(tmp_path):
    (tmp_path / "latest.json").write_bytes(b" " * (1 << 20) + b"{}")
    with pytest.raises(MonitorProtocolError):
        read_verified_record(tmp_path)


def test_terminal_publish_retains_final_frame(tmp_path):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    rec = pub.publish(lifecycle="completed", frame_seq=2, physics_step=2, sim_time_s=0.2,
                      retain_frame=True)
    assert rec["frame"]["sha256"] == hashlib.sha256(PNG_A).hexdigest()
    snap = read_verified_record(tmp_path)
    assert snap.frame_bytes == PNG_A
    assert snap.record["lifecycle"] == "completed"


def test_publisher_log_is_bounded(tmp_path):
    pub = make_publisher(tmp_path)
    for i in range(1, 260):
        pub.publish(lifecycle="training", frame_seq=i, physics_step=i, sim_time_s=i * 0.1)
    lines = (tmp_path / "publisher.log").read_text().strip().splitlines()
    assert len(lines) <= 201  # capped (open + last 200 publishes)
    # bounded storage: no png was ever published, so exactly these files exist
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "latest.json", "metadata.json", "publisher.log"
    ]


# ---------------------------------------------------------------------------
# viewer monitor: freshness, startup floor, terminal states, GET-only tool
# ---------------------------------------------------------------------------


def test_monitor_stale_producer_while_http_alive(tmp_path, tool):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    server_info = {"pid": 1, "remaining_wall_s": 100.0}
    fresh = mon.build_state(now_unix=time.time(), server_running=True,
                            server_info=server_info)
    assert fresh["status"] == "REPLAY"  # pre-existing at startup: historical
    future = time.time() + 30.0
    stale = mon.build_state(now_unix=future, server_running=True,
                            server_info=server_info)
    assert stale["status"] == "REPLAY"  # still historical, never upgraded to LIVE
    stopped = mon.build_state(now_unix=time.time(), server_running=False,
                              server_info=server_info)
    assert stopped["status"] == "STOPPED"


def test_monitor_startup_floor_then_new_fixture_frames_show_test(tool, tmp_path):
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)  # floors at frame_seq 1
    pub.publish(lifecycle="training", frame_seq=2, physics_step=2, sim_time_s=0.2, png=PNG_B)
    assert mon.poll_once() == "accepted"
    state = mon.build_state(now_unix=time.time(), server_running=True,
                            server_info={"pid": 1})
    assert state["status"] == "TEST"  # fixture source: fresh but never LIVE
    assert state["historical"] is False
    assert state["activity"] == "test"  # fixture source: never training_active
    assert state["training_active"] is False
    assert state["training_declared"] is False
    assert state["source_badge"] == "TEST"


def test_metadata_only_poll_never_resets_freshness(tmp_path, tool):
    pub = make_publisher(tmp_path)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    assert mon.poll_once() == "accepted"
    accepted = mon.records_accepted
    for _ in range(3):
        assert mon.poll_once() == "unchanged"
    assert mon.records_accepted == accepted  # no new identity accepted
    later = time.time() + 20.0  # unchanged files: ages keep growing
    state = mon.build_state(now_unix=later, server_running=True, server_info={"pid": 1})
    assert state["status"] == "STALLED"


def test_monitor_completion_and_failure_labels(tmp_path, tool):
    pub = make_publisher(tmp_path)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    mon.poll_once()
    pub.publish(lifecycle="completed", frame_seq=2, physics_step=2, sim_time_s=0.2,
                retain_frame=True)
    mon.poll_once()
    state = mon.build_state(now_unix=time.time(), server_running=True,
                            server_info={"pid": 1})
    assert state["status"] == "ENDED"
    assert state["live"] is False
    frame = mon.get_frame()  # final image retained
    assert frame is not None and frame[0] == PNG_A
    pub2 = make_publisher(tmp_path / "other", run_id="run-test-2")
    pub2.publish(lifecycle="failed", frame_seq=1, physics_step=5, sim_time_s=0.3, png=PNG_B)
    mon2 = tool.RunMonitor(tmp_path / "other", stale_after_s=10.0)
    state2 = mon2.build_state(now_unix=time.time(), server_running=True,
                              server_info={"pid": 1})
    assert state2["status"] == "FAILED"  # terminal lifecycle wins over historical REPLAY
    pub2.publish(lifecycle="failed", frame_seq=2, physics_step=6, sim_time_s=0.4,
                 retain_frame=True)
    mon2.poll_once()
    state2 = mon2.build_state(now_unix=time.time(), server_running=True,
                              server_info={"pid": 1})
    assert state2["status"] == "FAILED"


def test_monitor_corrupt_frame_fails_closed(tmp_path, tool):
    pub = make_publisher(tmp_path)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    assert mon.poll_once() == "accepted"
    (tmp_path / "frame.png").write_bytes(PNG_B)  # hash mismatch on next poll
    assert mon.poll_once() == "invalid"
    state = mon.build_state(now_unix=time.time() + 30.0, server_running=True,
                            server_info={"pid": 1})
    assert state["counters"]["reads_rejected"] == 1
    assert state["counters"]["last_error"] is not None
    assert state["status"] == "STALLED"  # last verified frame ages out; never LIVE


def test_monitor_waiting_no_video(tmp_path, tool):
    pub = make_publisher(tmp_path)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub.publish(lifecycle="waiting", frame_seq=1, physics_step=0, sim_time_s=0.0)
    assert mon.poll_once() == "accepted"
    assert mon.get_frame() is None  # no video while waiting
    state = mon.build_state(now_unix=time.time(), server_running=True,
                            server_info={"pid": 1})
    assert state["status"] == "WAITING"
    assert state["frame"] is None
    assert "等待" in state["notice_zh"]


def test_monitor_invalid_first_record_fails_closed(tmp_path, tool):
    (tmp_path / "latest.json").write_text('{"schema_version": "wrong"}')
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    assert mon.poll_once() == "invalid"
    state = mon.build_state(now_unix=time.time(), server_running=True,
                            server_info={"pid": 1})
    assert state["status"] == "ERROR"
    assert mon.get_frame() is None


def test_monitor_sequence_boundaries(tmp_path, tool):
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub = make_publisher(tmp_path)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    assert mon.poll_once() == "accepted"
    assert mon.poll_once() == "unchanged"  # identical record: freshness untouched
    rec = json.loads((tmp_path / "latest.json").read_text())
    rec["metrics"] = {"loss": 9.9}  # same frame_seq, mutated metadata
    (tmp_path / "latest.json").write_text(json.dumps(rec) + "\n")
    assert mon.poll_once() == "invalid"  # same-sequence mutation rejected
    assert mon.records_accepted == 1
    assert mon.current.record["frame_seq"] == 1  # last verified state kept
    rec = json.loads((tmp_path / "latest.json").read_text())
    rec["frame_seq"] = 0  # lower sequence for the same run
    (tmp_path / "latest.json").write_text(json.dumps(rec) + "\n")
    assert mon.poll_once() == "invalid"  # regression rejected
    assert mon.current.record["frame_seq"] == 1
    pub2 = make_publisher(tmp_path, run_id="run-test-2")
    pub2.publish(lifecycle="waiting", frame_seq=0, physics_step=0, sim_time_s=0.0)
    assert mon.poll_once() == "accepted"  # a new run_id resets sequence identity
    assert mon.current.record["run_id"] == "run-test-2"


def test_monitor_frame_headers_carry_run_identity(tmp_path, tool):
    pub = make_publisher(tmp_path)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    mon.poll_once()
    payload, headers = mon.get_frame()
    assert payload == PNG_A
    assert headers["X-Run-Id"] == "run-test-1"
    assert headers["X-Frame-Seq"] == "1"


def test_declared_lifetime_stop_decision(tool):
    # no terminal producer: only the declared deadline stops the server
    assert tool.stop_reason(100.0, 700.0, None) is None
    assert tool.stop_reason(700.0, 700.0, None) == "deadline_expired"
    # terminal producer state stops after the final-frame grace period,
    # long before a training-scale declared deadline
    assert tool.stop_reason(80.0, 86_400.0, 60.0, grace_s=30.0) is None  # within grace
    assert tool.stop_reason(90.0, 86_400.0, 60.0, grace_s=30.0) == "terminal_state"
    assert tool.stop_reason(700.0, 700.0, 60.0) == "deadline_expired"  # deadline wins ties


def test_monitor_marks_terminal_time_for_grace(tmp_path, tool):
    pub = make_publisher(tmp_path)
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1, png=PNG_A)
    mon.poll_once()
    assert mon.terminal_since_unix is None
    pub.publish(lifecycle="completed", frame_seq=2, physics_step=2, sim_time_s=0.2,
                retain_frame=True)
    mon.poll_once()
    assert mon.terminal_since_unix is not None
    # a terminal record already on disk at viewer start starts the grace clock
    mon2 = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    assert mon2.terminal_since_unix is not None


def test_stopped_state_never_reports_training_active(tmp_path, tool):
    pub = make_publisher(
        tmp_path, source_kind="training_environment", policy_kind="learned",
        checkpoint={"sha256": hashlib.sha256(b"ckpt").hexdigest(), "version": "ckpt-1"},
    )
    mon = tool.RunMonitor(tmp_path, stale_after_s=10.0)
    pub.publish(lifecycle="training", frame_seq=1, physics_step=1, sim_time_s=0.1,
                png=PNG_A, gradient_updates=5)
    assert mon.poll_once() == "accepted"
    running = mon.build_state(now_unix=time.time(), server_running=True,
                              server_info={"pid": 1})
    assert running["status"] == "LIVE" and running["training_active"] is True
    stopped = mon.build_state(now_unix=time.time(), server_running=False,
                              server_info={"pid": 1})
    assert stopped["status"] == "STOPPED"
    assert stopped["live"] is False
    assert stopped["training_active"] is False  # a stopped viewer shows no active training


def test_select_stop_reason_preserves_error(tool):
    assert tool.select_stop_reason("error", True, "deadline_expired") == "error"
    assert tool.select_stop_reason("error", False, "terminal_state") == "error"
    assert tool.select_stop_reason("deadline_expired", True, "deadline_expired") == "signal"
    assert tool.select_stop_reason("deadline_expired", False, "terminal_state") == "terminal_state"


def test_tool_constants_and_arg_bounds(tool, capsys):
    import argparse

    assert tool.BIND_HOST == "127.0.0.1"
    assert tool.DEFAULT_PORT == 8770
    assert tool.DEFAULT_MAX_WALL_S == 600.0  # bounded default
    assert tool.MAX_WALL_S_CAP == 86400.0  # explicit declared-lifetime ceiling

    def check(*argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8770)
        parser.add_argument("--max-wall-s", type=float, default=600.0)
        parser.add_argument("--stale-after-s", type=float, default=10.0)
        parser.add_argument("--poll-interval-s", type=float, default=0.5)
        args = parser.parse_args(list(argv))
        tool.validate_args(parser, args)

    check()
    check("--max-wall-s", "600", "--stale-after-s", "120")
    check("--max-wall-s", "600.01")  # explicit declared lifetime beyond the default
    check("--max-wall-s", "86400")
    for bad in ("0", "-1", "86400.01", "nan", "inf"):
        with pytest.raises(SystemExit):
            check("--max-wall-s", bad)
    for bad in ("0", "120.01", "nan", "inf"):
        with pytest.raises(SystemExit):
            check("--stale-after-s", bad)
    for bad in ("0.05", "10.5", "nan"):
        with pytest.raises(SystemExit):
            check("--poll-interval-s", bad)
    with pytest.raises(SystemExit):
        check("--port", "80")
    capsys.readouterr()


def test_tool_get_only_and_html_contract(tool):
    handler = tool.Handler
    for method in ("do_POST", "do_PUT", "do_DELETE", "do_PATCH", "do_HEAD", "do_OPTIONS"):
        assert hasattr(handler, method), method  # explicit 405, not a silent 501
    html = tool.render_index().decode()
    for needle in (
        "/state.json",
        "/frame.png",
        "AbortController",
        "createObjectURL",
        "probe.decode()",
        "lastDisplayedWallMs",
        "displayedIdentity",
        "view.style.display",
        "serverStatus === 'TEST'",
        "X-Run-Id",
        "inFlight",
        "stateInFlight",
        "watchdog",
        "REPLAY",
        "ENDED",
        "FAILED",
        "WAITING",
        "STALLED",
        "TEST",
        "DEMO",
        "测试影像",
        "历史画面",
        "640px",
        "max-height:480px",
    ):
        assert needle in html, needle
    assert "__STALE_MS__" not in html and "__SCOPE_0__" not in html
    assert 'src="/stream"' not in html  # no passive MJPEG display
