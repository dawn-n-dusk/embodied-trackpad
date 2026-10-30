"""Recorder behavior: append-only streams, immutable config, stream
separation, strict actor-payload validation, referential integrity,
flush/interrupted lifecycle, and no inferred success."""

import hashlib
import json
import subprocess
import sys

import pytest

from embodied_trackpad import (
    ActionChunk,
    AttemptOutcome,
    EpisodeRecorder,
    EvidencedFact,
    FramePacket,
    MemoryDelta,
    MotorIntent,
    RecorderError,
    ValidationError,
)


def _recorder(tmp_path, config, **kwargs):
    kwargs.setdefault("source_provenance", {"source": {"package": "embodied_trackpad"}})
    kwargs.setdefault("evidence_class", "mock_software_fixture")
    kwargs.setdefault("training_eligible", False)
    recorder = EpisodeRecorder(tmp_path, config, **kwargs)
    recorder.set_goal_context("goal_a", 1)
    return recorder


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _seed_observation(recorder, frame_kwargs, now):
    recorder.record_observation(FramePacket(**frame_kwargs), now=now)


def test_full_episode_flow(tmp_path, config, frame_kwargs, intent_kwargs, chunk_kwargs, now):
    recorder = _recorder(tmp_path, config)
    _seed_observation(recorder, frame_kwargs, now)
    recorder.record_intent(MotorIntent(**intent_kwargs), now=now)
    recorder.record_action_chunk(ActionChunk(**chunk_kwargs), now=now)
    recorder.record_outcome(
        AttemptOutcome(
            intent_id="intent_0001",
            motion_status="interrupted",
            visible_outcome="uncertain",
            uncertainty=0.6,
            request_replan=True,
        )
    )
    recorder.record_memory_delta(
        MemoryDelta(memory_version=1, source_observation_id="frame_0001",
                    generated_at=now, attempted=("tried",)),
        min_version=1,
    )
    recorder.record_teacher("label", {"ground_truth_target": [0.5, 0.5]})
    recorder.record_evaluator("final", {"score_hint": "privileged"})
    manifest = recorder.close()

    episode_dir = tmp_path / config.episode_id
    actor = _read_jsonl(episode_dir / "actor_events.jsonl")
    teacher = _read_jsonl(episode_dir / "teacher_events.jsonl")
    evaluator = _read_jsonl(episode_dir / "evaluator_events.jsonl")

    assert [event["seq"] for event in actor] == list(range(5))
    assert [event["kind"] for event in actor] == [
        "observation", "intent", "action_chunk", "attempt_outcome", "memory_delta",
    ]
    assert len(teacher) == 1 and len(evaluator) == 1
    # privileged payloads never enter the actor stream
    blob = json.dumps(actor)
    assert "ground_truth" not in blob and "score_hint" not in blob
    # failed/interrupted attempt remains recorded, without success
    outcome = actor[3]["payload"]
    assert outcome["motion_status"] == "interrupted"
    assert outcome["visible_outcome"] != "observed_success"

    assert manifest["event_counts"]["actor_events.jsonl"] == 5
    assert manifest["status"] == "completed"
    for name, info in manifest["files"].items():
        digest = hashlib.sha256((episode_dir / name).read_bytes()).hexdigest()
        assert digest == info["sha256"]
    assert "manifest.json" not in manifest["files"]  # no self-hash


def test_config_is_immutable(tmp_path, config):
    recorder = _recorder(tmp_path, config)
    recorder.close()
    with pytest.raises(RecorderError) as excinfo:
        _recorder(tmp_path, config)
    assert excinfo.value.code == "immutable"


def test_record_after_close_refused(tmp_path, config, frame_kwargs, now):
    recorder = _recorder(tmp_path, config)
    recorder.close()
    with pytest.raises(RecorderError) as excinfo:
        recorder.record_observation(FramePacket(**frame_kwargs), now=now)
    assert excinfo.value.code == "recorder_closed"


def test_invalid_message_not_recorded(tmp_path, config, intent_kwargs, now):
    recorder = _recorder(tmp_path, config)
    intent_kwargs["goal_version"] = 99
    with pytest.raises(ValidationError):
        recorder.record_intent(MotorIntent(**intent_kwargs), now=now)
    manifest = recorder.close()
    assert manifest["event_counts"]["actor_events.jsonl"] == 0


def test_goal_context_required(tmp_path, config, frame_kwargs, intent_kwargs, now):
    recorder = EpisodeRecorder(
        tmp_path,
        config,
        source_provenance={"source": {"package": "embodied_trackpad"}},
        evidence_class="mock_software_fixture",
        training_eligible=False,
    )
    recorder.record_observation(FramePacket(**frame_kwargs), now=now)
    with pytest.raises(RecorderError):
        recorder.record_intent(MotorIntent(**intent_kwargs), now=now)
    recorder.close()


def test_episode_id_cannot_escape_root(tmp_path, config):
    object.__setattr__(config, "episode_id", "../escaped")  # bypass frozen guard
    with pytest.raises(ValidationError) as excinfo:
        _recorder(tmp_path, config)
    assert excinfo.value.code == "unsafe_path"
    assert not (tmp_path.parent / "escaped").exists()


def test_mock_fixture_never_training_eligible(tmp_path, config):
    with pytest.raises(RecorderError) as excinfo:
        _recorder(tmp_path, config, training_eligible=True)
    assert excinfo.value.code == "bad_field"


def test_training_eligible_must_be_boolean(tmp_path, config):
    for bad in (1, "yes", None):
        with pytest.raises(RecorderError) as excinfo:
            _recorder(tmp_path / f"b{len(str(bad))}{type(bad).__name__}", config,
                      training_eligible=bad)
        assert excinfo.value.code == "bad_field"


# -- referential integrity -------------------------------------------------


def test_intent_requires_recorded_observation(tmp_path, config, intent_kwargs, now):
    recorder = _recorder(tmp_path, config)
    with pytest.raises(RecorderError) as excinfo:
        recorder.record_intent(MotorIntent(**intent_kwargs), now=now)
    assert excinfo.value.code == "unknown_observation"
    recorder.close()


def test_chunk_requires_recorded_observation(tmp_path, config, chunk_kwargs, now):
    recorder = _recorder(tmp_path, config)
    with pytest.raises(RecorderError) as excinfo:
        recorder.record_action_chunk(ActionChunk(**chunk_kwargs), now=now)
    assert excinfo.value.code == "unknown_observation"
    recorder.close()


def test_outcome_requires_recorded_intent(tmp_path, config, now):
    recorder = _recorder(tmp_path, config)
    with pytest.raises(RecorderError) as excinfo:
        recorder.record_outcome(
            AttemptOutcome(intent_id="ghost", motion_status="completed",
                           visible_outcome="unknown")
        )
    assert excinfo.value.code == "unknown_intent"
    recorder.close()


def test_success_evidence_must_be_recorded_observations(
    tmp_path, config, frame_kwargs, intent_kwargs, now
):
    recorder = _recorder(tmp_path, config)
    _seed_observation(recorder, frame_kwargs, now)
    recorder.record_intent(MotorIntent(**intent_kwargs), now=now)
    with pytest.raises(RecorderError) as excinfo:
        recorder.record_outcome(
            AttemptOutcome(
                intent_id="intent_0001",
                motion_status="completed",
                visible_outcome="observed_success",
                evidence_frame_ids=("frame_9999",),  # never recorded
            )
        )
    assert excinfo.value.code == "unknown_evidence"
    # with the real frame as evidence the same outcome is accepted
    recorder.record_outcome(
        AttemptOutcome(
            intent_id="intent_0001",
            motion_status="completed",
            visible_outcome="observed_success",
            evidence_frame_ids=("frame_0001",),
        )
    )
    recorder.close()


def test_memory_success_evidence_must_be_recorded(tmp_path, config, frame_kwargs, now):
    recorder = _recorder(tmp_path, config)
    _seed_observation(recorder, frame_kwargs, now)
    with pytest.raises(RecorderError) as excinfo:
        recorder.record_memory_delta(
            MemoryDelta(
                memory_version=1,
                source_observation_id="frame_0001",
                generated_at=now,
                observed_success=(EvidencedFact("done", ("frame_9999",)),),
            ),
            min_version=1,
        )
    assert excinfo.value.code == "unknown_evidence"
    recorder.close()


def test_memory_source_observation_must_exist(tmp_path, config, now):
    recorder = _recorder(tmp_path, config)
    with pytest.raises(RecorderError) as excinfo:
        recorder.record_memory_delta(
            MemoryDelta(memory_version=1, source_observation_id="ghost",
                        generated_at=now),
            min_version=1,
        )
    assert excinfo.value.code == "unknown_observation"
    recorder.close()


def test_duplicate_observation_rejected(tmp_path, config, frame_kwargs, now):
    recorder = _recorder(tmp_path, config)
    _seed_observation(recorder, frame_kwargs, now)
    with pytest.raises(RecorderError) as excinfo:
        _seed_observation(recorder, frame_kwargs, now)  # identical duplicate
    assert excinfo.value.code == "duplicate_observation"
    conflicting = dict(frame_kwargs)
    conflicting["receive"] = type(frame_kwargs["receive"])(
        frame_kwargs["receive"].clock_id, frame_kwargs["receive"].time_ns + 5
    )
    with pytest.raises(RecorderError) as excinfo:
        _seed_observation(recorder, conflicting, now)
    assert excinfo.value.code == "observation_conflict"
    recorder.close()


def test_frame_reuse_for_multiple_decisions_allowed(
    tmp_path, config, frame_kwargs, intent_kwargs, chunk_kwargs, now
):
    """One recorded observation may ground several intents/chunks."""
    recorder = _recorder(tmp_path, config)
    _seed_observation(recorder, frame_kwargs, now)
    recorder.record_intent(MotorIntent(**intent_kwargs), now=now)
    recorder.record_action_chunk(ActionChunk(**chunk_kwargs), now=now)
    second = dict(intent_kwargs, intent_id="intent_0002")
    recorder.record_intent(MotorIntent(**second), now=now)
    recorder.record_action_chunk(ActionChunk(**chunk_kwargs), now=now)
    manifest = recorder.close()
    assert manifest["event_counts"]["actor_events.jsonl"] == 5


# -- strict actor payload validation ----------------------------------------


def test_strict_whitelist_rejects_unknown_nested_payload(tmp_path, config):
    recorder = _recorder(tmp_path, config)
    with pytest.raises(RecorderError) as excinfo:
        recorder._append_actor("observation", {"unrecognized_field": {"reward": 1000}})
    assert excinfo.value.code == "whitelist"
    recorder.close()


def test_strict_whitelist_rejects_nested_extra_field(tmp_path, config, frame_kwargs):
    recorder = _recorder(tmp_path, config)
    payload = FramePacket(**frame_kwargs).to_dict()
    payload["capture"]["oracle_cursor_position"] = [10, 20]  # nested leak
    with pytest.raises(RecorderError) as excinfo:
        recorder._append_actor("observation", payload)
    assert excinfo.value.code == "whitelist"
    recorder.close()


def test_whitelist_rejects_unknown_actor_kind(tmp_path, config):
    recorder = _recorder(tmp_path, config)
    with pytest.raises(RecorderError) as excinfo:
        recorder._append_actor("grader_result", {"x": 1})
    assert excinfo.value.code == "whitelist"
    recorder.close()


# -- lifecycle: flush + interrupted status -----------------------------------


def test_exception_in_context_marks_interrupted(tmp_path, config, frame_kwargs, now):
    with pytest.raises(ValueError):
        with _recorder(tmp_path, config) as recorder:
            _seed_observation(recorder, frame_kwargs, now)
            raise ValueError("boom")
    manifest = json.loads((tmp_path / config.episode_id / "manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert "ValueError" in manifest["interruption"]
    assert manifest["event_counts"]["actor_events.jsonl"] == 1


def test_clean_context_marks_completed(tmp_path, config, frame_kwargs, now):
    with _recorder(tmp_path, config) as recorder:
        _seed_observation(recorder, frame_kwargs, now)
    manifest = json.loads((tmp_path / config.episode_id / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert "interruption" not in manifest


def test_unclosed_process_leaves_readable_partial_log(tmp_path):
    """Events are flushed per append; a killed process keeps readable rows."""
    script = (
        "import os, sys\n"
        "from embodied_trackpad import (\n"
        "    ClockStamp, EpisodeConfig, EpisodeRecorder, FramePacket)\n"
        "cfg = EpisodeConfig(\n"
        "    episode_id='proc_ep', embodiment_id='synthetic_mock_embodiment_v1',\n"
        "    actuator_order=('a', 'b'), joint_order=('a', 'b'),\n"
        "    action_representation='normalized_actuator_position_reference',\n"
        "    units={'length': 'm', 'angle': 'rad', 'time': 's'},\n"
        "    calibration_id='mock_calibration_v1', bridge_version='none-mock',\n"
        "    cursor_profile={})\n"
        "r = EpisodeRecorder(sys.argv[1], cfg,\n"
        "    source_provenance={'source': {'package': 'embodied_trackpad'}},\n"
        "    evidence_class='mock_software_fixture', training_eligible=False)\n"
        "now = ClockStamp('sim_main', 1000)\n"
        "r.record_observation(FramePacket(frame_id='f1', stream_id='s', capture=now,\n"
        "    receive=ClockStamp('sim_main', 2000), image_shape_px=(8, 8),\n"
        "    image_ref='synthetic://f1'), now=now)\n"
        "os._exit(0)\n"  # hard exit: no close(), no manifest; log must still read
    )
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=60)
    lines = (tmp_path / "proc_ep" / "actor_events.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["kind"] == "observation" and record["payload"]["frame_id"] == "f1"


def test_evidence_class_enforced(tmp_path, config):
    with pytest.raises(RecorderError):
        EpisodeRecorder(
            tmp_path,
            config,
            source_provenance={"source": {"package": "embodied_trackpad"}},
            evidence_class="not_a_class",
            training_eligible=False,
        )


def test_provenance_recorded(tmp_path, config):
    recorder = _recorder(
        tmp_path, config, created_utc="2026-10-30T00:00:00Z",
        source_provenance={"source": {"package": "embodied_trackpad",
                                      "source_commit": "abc123"}},
    )
    recorder.close()
    provenance = json.loads((tmp_path / config.episode_id / "provenance.json").read_text())
    assert provenance["created_utc"] == "2026-10-30T00:00:00Z"
    assert provenance["source"]["source"]["source_commit"] == "abc123"
    assert provenance["evidence_class"] == "mock_software_fixture"
    assert provenance["training_eligible"] is False
