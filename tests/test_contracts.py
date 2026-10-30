"""Contract construction, serialization round-trips, and strictness."""

import pytest

from embodied_trackpad import (
    SCHEMA_VERSION,
    ActionChunk,
    AttemptOutcome,
    EpisodeConfig,
    EvidencedFact,
    FramePacket,
    MemoryDelta,
    MemoryRevision,
    MotorIntent,
    ValidationError,
)


def test_episode_config_round_trip(config):
    restored = EpisodeConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.schema_version == SCHEMA_VERSION
    assert restored.action_dim == 4


def test_frame_packet_round_trip(frame_kwargs):
    frame = FramePacket(**frame_kwargs)
    assert FramePacket.from_dict(frame.to_dict()) == frame


def test_motor_intent_round_trip(intent_kwargs):
    intent = MotorIntent(**intent_kwargs)
    assert MotorIntent.from_dict(intent.to_dict()) == intent


def test_action_chunk_round_trip(chunk_kwargs):
    chunk = ActionChunk(**chunk_kwargs)
    restored = ActionChunk.from_dict(chunk.to_dict())
    assert restored == chunk
    assert restored.action_width == 4


def test_attempt_outcome_round_trip():
    outcome = AttemptOutcome(
        intent_id="intent_1",
        motion_status="completed",
        visible_outcome="observed_success",
        evidence_frame_ids=("frame_9",),
        uncertainty=0.2,
    )
    assert AttemptOutcome.from_dict(outcome.to_dict()) == outcome


def test_memory_delta_round_trip(now):
    delta = MemoryDelta(
        memory_version=3,
        source_observation_id="frame_9",
        generated_at=now,
        attempted=("tried to open candidate 3",),
        observed_success=(EvidencedFact("preview visible", ("frame_9",)),),
        uncertain=("whether scroll registered",),
        revisions=(MemoryRevision("fact_12", "corrected count: 2", ("frame_8",)),),
    )
    assert MemoryDelta.from_dict(delta.to_dict()) == delta


def test_unknown_field_rejected(config):
    data = config.to_dict()
    data["surprise_field"] = 1
    with pytest.raises(ValidationError) as excinfo:
        EpisodeConfig.from_dict(data)
    assert excinfo.value.code == "bad_field"


def test_wrong_schema_version_rejected(config):
    data = config.to_dict()
    data["schema_version"] = "trackpad.v1"
    with pytest.raises(ValidationError) as excinfo:
        EpisodeConfig.from_dict(data)
    assert excinfo.value.code == "schema_version"


def test_duplicate_actuator_names_rejected(config):
    data = config.to_dict()
    data["actuator_order"] = ["a", "a"]
    with pytest.raises(ValidationError):
        EpisodeConfig.from_dict(data)


def test_bbox_must_be_ordered(intent_kwargs):
    intent_kwargs["target_bbox_norm"] = (0.4, 0.1, 0.1, 0.4)
    with pytest.raises(ValidationError) as excinfo:
        MotorIntent(**intent_kwargs)
    assert excinfo.value.code == "out_of_range"


def test_confidence_range_enforced(intent_kwargs):
    intent_kwargs["target_confidence"] = 1.5
    with pytest.raises(ValidationError) as excinfo:
        MotorIntent(**intent_kwargs)
    assert excinfo.value.code == "out_of_range"


def test_contracts_are_immutable(config):
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.episode_id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_id",
    [
        "../escaped",
        "..",
        ".",
        "a/b",
        "a\\b",
        "/absolute",
        "C:\\\\temp",
        "C:rel",
        "",
    ],
)
def test_episode_id_must_be_single_safe_path_component(config, bad_id):
    data = config.to_dict()
    data["episode_id"] = bad_id
    with pytest.raises(ValidationError) as excinfo:
        EpisodeConfig.from_dict(data)
    assert excinfo.value.code in ("unsafe_path", "bad_field")


def test_actuator_layout_id_matches_order(config):
    from embodied_trackpad.contracts import compute_actuator_layout_id

    assert config.actuator_layout_id == compute_actuator_layout_id(config.actuator_order)
    restored = EpisodeConfig.from_dict(config.to_dict())
    assert restored.actuator_layout_id == config.actuator_layout_id


def test_tampered_actuator_layout_id_rejected(config):
    data = config.to_dict()
    data["actuator_layout_id"] = "actlayout-" + "0" * 16
    with pytest.raises(ValidationError) as excinfo:
        EpisodeConfig.from_dict(data)
    assert excinfo.value.code == "layout_mismatch"
