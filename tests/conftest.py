import pytest

from embodied_trackpad import ClockStamp, EpisodeConfig

SIM_CLOCK = "sim_main"
ACTUATORS = ("arm_x", "arm_y", "finger_a_flex", "finger_b_flex")


@pytest.fixture
def config() -> EpisodeConfig:
    return EpisodeConfig(
        episode_id="test_episode",
        embodiment_id="synthetic_mock_embodiment_v1",
        actuator_order=ACTUATORS,
        joint_order=ACTUATORS,
        action_representation="normalized_actuator_position_reference",
        units={"length": "m", "angle": "rad", "time": "s"},
        calibration_id="mock_calibration_v1",
        bridge_version="none-mock",
        cursor_profile={"size_px": 32, "style": "mock"},
    )


@pytest.fixture
def now() -> ClockStamp:
    return ClockStamp(SIM_CLOCK, 1_000_000_000)


@pytest.fixture
def frame_kwargs(now):
    return dict(
        frame_id="frame_0001",
        stream_id="mock_stream",
        capture=now,
        receive=ClockStamp(SIM_CLOCK, now.time_ns + 2_000_000),
        image_shape_px=(64, 64),
        image_ref="synthetic://test/frame_0001",
    )


@pytest.fixture
def intent_kwargs(config, now):
    return dict(
        intent_id="intent_0001",
        goal_id="goal_a",
        goal_version=1,
        observation_id="frame_0001",
        interaction_kind="move",
        target_bbox_norm=(0.1, 0.1, 0.4, 0.4),
        target_confidence=0.9,
        calibration_id=config.calibration_id,
        issued_at=now,
        valid_until=ClockStamp(SIM_CLOCK, now.time_ns + 1_000_000_000),
    )


@pytest.fixture
def chunk_kwargs(config, now):
    return dict(
        policy_version="mock_policy_v1",
        embodiment_id=config.embodiment_id,
        actuator_layout_id=config.actuator_layout_id,
        goal_id="goal_a",
        goal_version=1,
        observation_id="frame_0001",
        command_type="actuator_position_reference",
        start=now,
        dt_seconds=0.05,
        horizon=2,
        values=((0.0, 0.1, -0.1, 0.0), (0.2, 0.0, 0.0, -0.2)),
        valid_mask=(True, True),
        valid_until=ClockStamp(SIM_CLOCK, now.time_ns + 1_000_000_000),
    )
