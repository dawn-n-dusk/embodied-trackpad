"""Runtime validation boundaries: embodiment, shape, range, goal, expiry,
clock domains, and the rule that zero actions can be valid."""

import pytest

from embodied_trackpad import (
    ActionChunk,
    AttemptOutcome,
    ClockDomainError,
    ClockStamp,
    EvidencedFact,
    FramePacket,
    MemoryDelta,
    MotorIntent,
    ValidationError,
    Validator,
)
from embodied_trackpad.contracts import compute_actuator_layout_id

SIM = "sim_main"


def test_wrong_embodiment_rejected(config, chunk_kwargs, now):
    chunk_kwargs["embodiment_id"] = "other_bot"
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "embodiment_mismatch"


def test_actuator_width_mismatch_rejected(config, chunk_kwargs, now):
    chunk_kwargs["values"] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "actuator_order_mismatch"


def test_reordered_actuator_layout_rejected(config, chunk_kwargs, now):
    """Same width and embodiment, reversed actuator_order: must not pass."""
    import dataclasses

    # keeping the old layout id while reordering fails at construction
    with pytest.raises(ValidationError) as excinfo:
        dataclasses.replace(config, actuator_order=tuple(reversed(config.actuator_order)))
    assert excinfo.value.code == "layout_mismatch"
    # a cleanly reordered config recomputes its layout id, and the old
    # chunk is rejected against it
    reordered = dataclasses.replace(
        config,
        actuator_order=tuple(reversed(config.actuator_order)),
        actuator_layout_id="",
    )
    assert reordered.actuator_layout_id != config.actuator_layout_id
    chunk = ActionChunk(**chunk_kwargs)  # bound to the original layout
    with pytest.raises(ValidationError) as excinfo:
        Validator(reordered).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "layout_mismatch"


def test_foreign_layout_id_rejected(config, chunk_kwargs, now):
    chunk_kwargs["actuator_layout_id"] = compute_actuator_layout_id(("a", "b"))
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "layout_mismatch"


def test_stale_chunk_rejected_without_valid_until(config, chunk_kwargs, now):
    """A chunk past start + horizon*dt is stale even with no valid_until."""
    chunk_kwargs["start"] = ClockStamp(SIM, 1)  # ~1s before now, window is 100ms
    chunk_kwargs["valid_until"] = None
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "expired"


def test_chunk_inside_half_open_window_accepted(config, chunk_kwargs, now):
    """Boundary: start <= now < start + horizon*dt is admissible."""
    chunk_kwargs["valid_until"] = None
    chunk = ActionChunk(**chunk_kwargs)  # horizon 2 * dt 0.05s -> 100 ms
    validator = Validator(config)
    validator.validate_action_chunk(
        chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
    )
    edge = ClockStamp(SIM, now.time_ns + 99_999_999)
    validator.validate_action_chunk(
        chunk, expected_goal_id="goal_a", expected_goal_version=1, now=edge
    )
    past = ClockStamp(SIM, now.time_ns + 100_000_000)
    with pytest.raises(ValidationError) as excinfo:
        validator.validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=past
        )
    assert excinfo.value.code == "expired"


def test_future_chunk_start_rejected_on_admission(config, chunk_kwargs, now):
    chunk_kwargs["start"] = ClockStamp(SIM, now.time_ns + 1_000_000)
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "not_yet_valid"


def test_cross_domain_now_rejected_for_chunk(config, chunk_kwargs, now):
    chunk = ActionChunk(**chunk_kwargs)
    foreign_now = ClockStamp("other_host_clock", now.time_ns)
    with pytest.raises(ClockDomainError):
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=foreign_now
        )


def test_structural_chunk_check_without_now(config, chunk_kwargs):
    """Historical/structural validation runs without a live clock."""
    chunk_kwargs["start"] = ClockStamp(SIM, 0)
    chunk_kwargs["valid_until"] = None
    chunk = ActionChunk(**chunk_kwargs)
    Validator(config).validate_action_chunk(
        chunk, expected_goal_id="goal_a", expected_goal_version=1
    )


def test_future_intent_issued_at_rejected(config, intent_kwargs, now):
    intent_kwargs["issued_at"] = ClockStamp(SIM, now.time_ns + 5_000_000)
    intent_kwargs["valid_until"] = ClockStamp(SIM, now.time_ns + 1_000_000_000)
    intent = MotorIntent(**intent_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_intent(
            intent, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "not_yet_valid"


def test_horizon_shape_mismatch_rejected(chunk_kwargs):
    chunk_kwargs["values"] = ((0.0, 0.0, 0.0, 0.0),)  # 1 row, horizon says 2
    with pytest.raises(ValidationError) as excinfo:
        ActionChunk(**chunk_kwargs)
    assert excinfo.value.code == "shape"


def test_valid_mask_length_mismatch_rejected(chunk_kwargs):
    chunk_kwargs["valid_mask"] = (True,)
    with pytest.raises(ValidationError) as excinfo:
        ActionChunk(**chunk_kwargs)
    assert excinfo.value.code == "shape"


def test_nonfinite_value_rejected(chunk_kwargs):
    chunk_kwargs["values"] = ((0.0, float("nan"), 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValidationError) as excinfo:
        ActionChunk(**chunk_kwargs)
    assert excinfo.value.code == "nonfinite"


def test_out_of_range_value_rejected(config, chunk_kwargs, now):
    chunk_kwargs["values"] = ((1.5, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "out_of_range"


def test_zero_action_is_valid(config, chunk_kwargs, now):
    chunk_kwargs["values"] = ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    chunk = ActionChunk(**chunk_kwargs)
    Validator(config).validate_action_chunk(
        chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
    )


def test_partial_valid_mask_accepted(config, chunk_kwargs, now):
    chunk_kwargs["valid_mask"] = (True, False)
    chunk = ActionChunk(**chunk_kwargs)
    Validator(config).validate_action_chunk(
        chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
    )


def test_coordinate_click_command_rejected(chunk_kwargs):
    chunk_kwargs["command_type"] = "coordinate_click"
    with pytest.raises(ValidationError) as excinfo:
        ActionChunk(**chunk_kwargs)
    assert excinfo.value.code == "command_type"


def test_dt_must_be_positive(chunk_kwargs):
    chunk_kwargs["dt_seconds"] = 0.0
    with pytest.raises(ValidationError) as excinfo:
        ActionChunk(**chunk_kwargs)
    assert excinfo.value.code == "out_of_range"


def test_expired_intent_rejected(config, intent_kwargs, now):
    intent_kwargs["issued_at"] = ClockStamp(SIM, 0)
    intent_kwargs["valid_until"] = ClockStamp(SIM, now.time_ns - 1)
    intent = MotorIntent(**intent_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_intent(
            intent, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "expired"


def test_expired_chunk_rejected(config, chunk_kwargs, now):
    chunk_kwargs["start"] = ClockStamp(SIM, 0)
    chunk_kwargs["valid_until"] = ClockStamp(SIM, now.time_ns - 1)
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "expired"


def test_wrong_goal_version_rejected(config, intent_kwargs, chunk_kwargs, now):
    bad_intent = MotorIntent(**{**intent_kwargs, "goal_version": 2})
    bad_chunk = ActionChunk(**{**chunk_kwargs, "goal_id": "goal_b"})
    validator = Validator(config)
    with pytest.raises(ValidationError) as excinfo:
        validator.validate_intent(
            bad_intent, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "goal_mismatch"
    with pytest.raises(ValidationError) as excinfo:
        validator.validate_action_chunk(
            bad_chunk, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "goal_mismatch"


def test_observation_mismatch_rejected(config, chunk_kwargs, now):
    chunk = ActionChunk(**chunk_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_action_chunk(
            chunk,
            expected_goal_id="goal_a",
            expected_goal_version=1,
            expected_observation_id="frame_9999",
            now=now,
        )
    assert excinfo.value.code == "observation_mismatch"


def test_calibration_mismatch_rejected(config, intent_kwargs, now):
    intent_kwargs["calibration_id"] = "other_calibration"
    intent = MotorIntent(**intent_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_intent(
            intent, expected_goal_id="goal_a", expected_goal_version=1, now=now
        )
    assert excinfo.value.code == "calibration_mismatch"


def test_same_domain_latency_computed(frame_kwargs):
    frame = FramePacket(**frame_kwargs)
    assert frame.measured_latency_ms() == pytest.approx(2.0)


def test_cross_domain_frame_age_never_invented(frame_kwargs, now):
    frame_kwargs["receive"] = ClockStamp("other_host_clock", now.time_ns + 2_000_000)
    frame = FramePacket(**frame_kwargs)
    with pytest.raises(ClockDomainError):
        frame.measured_latency_ms()


def test_cross_domain_expiry_check_rejected(config, intent_kwargs):
    intent = MotorIntent(**intent_kwargs)
    foreign_now = ClockStamp("other_host_clock", intent_kwargs["issued_at"].time_ns)
    with pytest.raises(ClockDomainError):
        Validator(config).validate_intent(
            intent, expected_goal_id="goal_a", expected_goal_version=1, now=foreign_now
        )


def test_non_monotonic_frame_rejected(config, frame_kwargs):
    frame_kwargs["receive"] = ClockStamp(SIM, frame_kwargs["capture"].time_ns - 1)
    frame = FramePacket(**frame_kwargs)
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_frame(frame)
    assert excinfo.value.code == "non_monotonic"


def test_observed_success_requires_evidence():
    with pytest.raises(ValidationError) as excinfo:
        AttemptOutcome(
            intent_id="i1", motion_status="completed", visible_outcome="observed_success"
        )
    assert excinfo.value.code == "missing_evidence"


def test_failed_attempt_records_without_success():
    outcome = AttemptOutcome(
        intent_id="i1",
        motion_status="interrupted",
        visible_outcome="uncertain",
        uncertainty=0.7,
        request_replan=True,
    )
    assert outcome.visible_outcome != "observed_success"


def test_memory_success_fact_requires_evidence(now):
    # an evidenced success fact is valid
    MemoryDelta(
        memory_version=1,
        source_observation_id="frame_1",
        generated_at=now,
        observed_success=(EvidencedFact("done", ("frame_1",)),),
    )
    # a success fact without evidence is not
    with pytest.raises(ValidationError) as excinfo:
        EvidencedFact("done", ())
    assert excinfo.value.code == "missing_evidence"


def test_memory_version_monotonicity(config, now):
    delta = MemoryDelta(
        memory_version=0, source_observation_id="frame_1", generated_at=now
    )
    with pytest.raises(ValidationError) as excinfo:
        Validator(config).validate_memory_delta(delta, min_version=1)
    assert excinfo.value.code == "non_monotonic"
