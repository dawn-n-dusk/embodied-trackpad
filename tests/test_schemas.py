"""Published JSON Schemas: one consistent draft, valid against fixtures."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from embodied_trackpad import (
    ActionChunk,
    AttemptOutcome,
    EpisodeConfig,
    EvidencedFact,
    FramePacket,
    MemoryDelta,
    MotorIntent,
)
from embodied_trackpad.contracts import JSON_SCHEMA_DRAFT

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"

SCHEMA_FILES = {
    "episode-config.schema.json": EpisodeConfig,
    "frame-packet.schema.json": FramePacket,
    "motor-intent.schema.json": MotorIntent,
    "action-chunk.schema.json": ActionChunk,
    "attempt-outcome.schema.json": AttemptOutcome,
    "memory-delta.schema.json": MemoryDelta,
}


def _load(name):
    return json.loads((SCHEMA_DIR / name).read_text())


def test_all_schemas_share_one_draft():
    for name in SCHEMA_FILES:
        schema = _load(name)
        assert schema["$schema"] == JSON_SCHEMA_DRAFT
        Draft202012Validator.check_schema(schema)


def test_schema_version_constant_matches():
    for name in SCHEMA_FILES:
        schema = _load(name)
        assert schema["properties"]["schema_version"]["const"] == "trackpad.v2-draft"


def test_valid_instances_pass(config, frame_kwargs, intent_kwargs, chunk_kwargs, now):
    instances = {
        "episode-config.schema.json": config.to_dict(),
        "frame-packet.schema.json": FramePacket(**frame_kwargs).to_dict(),
        "motor-intent.schema.json": MotorIntent(**intent_kwargs).to_dict(),
        "action-chunk.schema.json": ActionChunk(**chunk_kwargs).to_dict(),
        "attempt-outcome.schema.json": AttemptOutcome(
            intent_id="i1",
            motion_status="completed",
            visible_outcome="observed_success",
            evidence_frame_ids=("frame_1",),
        ).to_dict(),
        "memory-delta.schema.json": MemoryDelta(
            memory_version=1,
            source_observation_id="frame_1",
            generated_at=now,
            observed_success=(EvidencedFact("f", ("frame_1",)),),
        ).to_dict(),
    }
    for name, instance in instances.items():
        Draft202012Validator(_load(name)).validate(instance)


def test_extra_fields_rejected_by_schema(config):
    schema = _load("episode-config.schema.json")
    instance = config.to_dict()
    instance["bogus"] = 1
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(instance)


def test_action_chunk_schema_bounds(config, chunk_kwargs):
    schema = _load("action-chunk.schema.json")
    chunk = ActionChunk(**chunk_kwargs).to_dict()
    chunk["values"][0][0] = 1.5
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(chunk)


def test_click_command_rejected_by_schema(chunk_kwargs):
    schema = _load("action-chunk.schema.json")
    chunk = ActionChunk(**chunk_kwargs).to_dict()
    chunk["command_type"] = "coordinate_click"
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(chunk)


def test_chunk_without_layout_id_rejected_by_schema(chunk_kwargs):
    schema = _load("action-chunk.schema.json")
    chunk = ActionChunk(**chunk_kwargs).to_dict()
    del chunk["actuator_layout_id"]
    with pytest.raises(Exception):
        Draft202012Validator(schema).validate(chunk)
