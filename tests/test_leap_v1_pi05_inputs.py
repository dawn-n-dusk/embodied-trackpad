"""Boundary tests for the pure-NumPy native pi0.5 input repacker (pi05_inputs).

Synthetic ``VisualExample`` fixtures only (no corpus/index machinery): the
repacker consumes the already-accepted dataclass. Covers exact pixel/state/
action mapping with real missing-image masks, oracle-leak whitelist, copy and
mutation isolation, rejection of invalid shapes/dtypes/non-finite values,
fabricated sensor masks and non-native timing, and the 32-mask/16-native
contract separation.
"""
from __future__ import annotations

import numpy as np
import pytest

from embodied_trackpad.leap_v1.pi05_inputs import (
    ACTION_DIM,
    DEFAULT_PROMPT,
    Pi05NativeSample,
    prepare_pi05_native_sample,
)
from embodied_trackpad.leap_v1.visual_dataset import (
    ACTION_SEMANTICS,
    DT_S,
    HORIZON,
    HORIZON_S,
    NUM_CHANNELS,
    PNG_HEIGHT,
    PNG_WIDTH,
    VisualCorpusError,
    VisualExample,
)


def _rgb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(PNG_HEIGHT, PNG_WIDTH, 3), dtype=np.uint8)


def _timing() -> dict:
    return {
        "dt_s": DT_S,
        "horizon": HORIZON,
        "horizon_s": HORIZON_S,
        "first_step_index": 1451,
        "end_exclusive_step_index": 1476,
        "consumed_prev_sim_time_s": 2.9,
        "first_sim_time_s": 2.902,
        "end_sim_time_s": 2.95,
        "action_semantics": ACTION_SEMANTICS,
    }


def _example(*, seed: int = 0, abs_scale: float = 1e-4) -> VisualExample:
    q = np.full(NUM_CHANNELS, 0.25, dtype=np.float32)
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    executed[:, 3] = abs_scale * np.sin(np.linspace(0, np.pi, HORIZON)).astype(np.float32)
    observation = {
        "screen_rgb": _rgb(seed),
        "q_rad16": q,
        "qvel_rad_s16": np.zeros(NUM_CHANNELS, dtype=np.float32),
        "previous_ctrl_rad16": np.zeros(NUM_CHANNELS, dtype=np.float32),
        "fsr_N4": None,
        "motor_current_A16": None,
        "wrist": None,
        "fsr_valid4": [False, False, False, False],
        "motor_current_valid16": [False] * NUM_CHANNELS,
        "wrist_valid": False,
    }
    provenance = {
        "sample_id": f"syn:ep-000:frame-{seed}",
        "action_window": _timing(),
    }
    return VisualExample(
        observation=observation,
        target={"target_ctrl_rad": executed},
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# 1. exact pixel/state/action mapping and real missing-image masks
# --------------------------------------------------------------------------- #


def test_exact_mapping_pixels_state_actions_and_missing_image_masks():
    ex = _example(seed=5)
    out = prepare_pi05_native_sample(ex)

    assert isinstance(out, Pi05NativeSample)
    data = out.raw_model_data
    assert set(data.keys()) == {"state", "image", "image_mask", "actions", "prompt"}

    assert data["state"].shape == (NUM_CHANNELS,)
    assert data["state"].dtype == np.float32
    np.testing.assert_array_equal(data["state"], ex.observation["q_rad16"])

    image = data["image"]
    assert set(image.keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert image["base_0_rgb"].shape == (PNG_HEIGHT, PNG_WIDTH, 3)
    assert image["base_0_rgb"].dtype == np.uint8
    np.testing.assert_array_equal(image["base_0_rgb"], ex.observation["screen_rgb"])
    # Real missing wrist views: zero-filled, but honestly flagged absent.
    assert image["left_wrist_0_rgb"].shape == (PNG_HEIGHT, PNG_WIDTH, 3)
    assert image["left_wrist_0_rgb"].dtype == np.uint8
    assert image["right_wrist_0_rgb"].dtype == np.uint8
    assert not image["left_wrist_0_rgb"].any()
    assert not image["right_wrist_0_rgb"].any()
    assert data["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    assert all(type(v) is bool for v in data["image_mask"].values())

    actions = data["actions"]
    assert actions.shape == (HORIZON, NUM_CHANNELS)
    assert actions.dtype == np.float32
    # Absolute executed references unchanged (no delta/renormalization).
    np.testing.assert_array_equal(actions, ex.target["target_ctrl_rad"])
    assert actions[12, 3] != 0.0

    assert data["prompt"] == DEFAULT_PROMPT
    custom = prepare_pi05_native_sample(ex, prompt="drag slowly")
    assert custom.raw_model_data["prompt"] == "drag slowly"


# --------------------------------------------------------------------------- #
# 2. no oracle leak: whitelist model dict even with extras in metadata
# --------------------------------------------------------------------------- #


def test_no_oracle_leak_extras_in_provenance_or_target_stay_out():
    ex = _example(seed=1)
    ex.provenance["error_xy_px"] = [8.0, 4.0]
    ex.provenance["teacher_state"] = {"privileged": np.ones(3)}
    ex.provenance["force_N"] = [0.1, 0.2, 0.3]
    ex.provenance["task_profile"] = {"oracle_gain": 9.9}
    ex.target["error_profile_rad"] = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    out = prepare_pi05_native_sample(ex)

    data = out.raw_model_data
    assert set(data.keys()) == {"state", "image", "image_mask", "actions", "prompt"}
    assert set(data["image"].keys()) == {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    flat = [data["state"], *data["image"].values(), data["actions"]]
    assert all(9.9 not in np.unique(arr) for arr in flat)
    assert "teacher_state" not in data and "error_xy_px" not in data
    # Privileged metadata remains visible only in the copied provenance.
    assert out.provenance["teacher_state"]["privileged"].sum() == 3.0


# --------------------------------------------------------------------------- #
# 3. copy / mutation isolation both directions
# --------------------------------------------------------------------------- #


def test_result_arrays_and_provenance_are_copies():
    ex = _example(seed=2)
    pristine_rgb = ex.observation["screen_rgb"].copy()
    pristine_q = ex.observation["q_rad16"].copy()
    pristine_actions = ex.target["target_ctrl_rad"].copy()
    pristine_dt = ex.provenance["action_window"]["dt_s"]

    out = prepare_pi05_native_sample(ex)

    # Mutating the result must not touch the example...
    out.raw_model_data["state"][:] = -1.0
    out.raw_model_data["actions"][:] = 0.0
    out.raw_model_data["image"]["base_0_rgb"][:] = 0
    out.raw_model_data["image"]["left_wrist_0_rgb"][:] = 7
    out.provenance["action_window"]["dt_s"] = 0.5
    out.provenance["unused_observation_keys"].append("forged")
    np.testing.assert_array_equal(ex.observation["screen_rgb"], pristine_rgb)
    np.testing.assert_array_equal(ex.observation["q_rad16"], pristine_q)
    np.testing.assert_array_equal(ex.target["target_ctrl_rad"], pristine_actions)
    assert ex.provenance["action_window"]["dt_s"] == pristine_dt
    assert "forged" not in ex.provenance.get("unused_observation_keys", [])

    # ...and a second, independent repack of the untouched example is pristine.
    out2 = prepare_pi05_native_sample(ex)
    np.testing.assert_array_equal(out2.raw_model_data["image"]["base_0_rgb"], pristine_rgb)
    assert not out2.raw_model_data["image"]["left_wrist_0_rgb"].any()
    np.testing.assert_array_equal(out2.raw_model_data["actions"], pristine_actions)
    assert out2.provenance["action_window"]["dt_s"] == pristine_dt


# --------------------------------------------------------------------------- #
# 4. rejection of invalid shapes/dtypes/non-finite/sensor masks/timing
# --------------------------------------------------------------------------- #


def test_empty_or_nonstring_prompt_rejected():
    ex = _example()
    for bad in ("", "   ", None, 0, ["x"]):
        with pytest.raises(VisualCorpusError, match="prompt"):
            prepare_pi05_native_sample(ex, prompt=bad)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda ex: ex.observation.update(screen_rgb=np.zeros((600, 800, 3), dtype=np.float32)), "screen_rgb"),
        (lambda ex: ex.observation.update(screen_rgb=np.zeros((100, 100, 3), dtype=np.uint8)), "screen_rgb"),
        (lambda ex: ex.observation.update(q_rad16=np.zeros(NUM_CHANNELS, dtype=np.float64)), "q_rad16"),
        (lambda ex: ex.observation.update(q_rad16=np.zeros((NUM_CHANNELS, 1), dtype=np.float32)), "q_rad16"),
        (lambda ex: ex.observation["q_rad16"].__setitem__(4, np.nan), "non-finite"),
        (lambda ex: ex.target.update(target_ctrl_rad=np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float64)), "target_ctrl_rad"),
        (lambda ex: ex.target.update(target_ctrl_rad=np.zeros((HORIZON - 1, NUM_CHANNELS), dtype=np.float32)), "target_ctrl_rad"),
        (lambda ex: ex.target["target_ctrl_rad"].__setitem__((0, 0), float("inf")), "non-finite"),
        (lambda ex: ex.observation.update(fsr_N4=[0.0] * 4), "fsr"),
        (lambda ex: ex.observation.update(wrist=np.zeros((4, 4), dtype=np.float32)), "wrist"),
        (lambda ex: ex.observation.update(fsr_valid4=[False, False, False, True]), "fsr_valid4"),
        (lambda ex: ex.observation.update(fsr_valid4=[0, 0, 0, 0]), "fsr_valid4"),
        (lambda ex: ex.observation.update(motor_current_valid16=[False] * (NUM_CHANNELS - 1)), "motor_current_valid16"),
        (lambda ex: ex.observation.update(wrist_valid=0), "wrist_valid"),
        (lambda ex: ex.observation.update(wrist_valid=True), "wrist_valid"),
        (lambda ex: ex.provenance.update(action_window=None), "action_window"),
        (lambda ex: ex.provenance["action_window"].update(dt_s=0.004), "native"),
        (lambda ex: ex.provenance["action_window"].update(horizon=32), "native"),
        (lambda ex: ex.provenance["action_window"].update(horizon_s=0.064), "native"),
        (lambda ex: ex.provenance["action_window"].update(action_semantics="delta_nominal"), "action_semantics"),
        (lambda ex: ex.provenance["action_window"].update(first_sim_time_s=float("nan")), "finite"),
        (lambda ex: ex.provenance["action_window"].update(end_sim_time_s=3.5), "inconsistent"),
    ],
)
def test_invalid_example_rejected(mutate, match):
    ex = _example()
    mutate(ex)
    with pytest.raises(VisualCorpusError, match=match):
        prepare_pi05_native_sample(ex)


# --------------------------------------------------------------------------- #
# 5. loss_mask32 contract vs. untouched native 16-dim state/actions
# --------------------------------------------------------------------------- #


def test_loss_mask32_outside_raw_data_and_native16_time_contract():
    ex = _example()
    out = prepare_pi05_native_sample(ex)

    assert out.loss_mask32.shape == (ACTION_DIM,)
    assert out.loss_mask32.dtype == np.bool_
    assert out.loss_mask32[:NUM_CHANNELS].all()
    assert not out.loss_mask32[NUM_CHANNELS:].any()
    assert "loss_mask32" not in out.raw_model_data

    # Raw state/actions remain native 16 BEFORE official normalization/tokenizer
    # and PadStatesAndActions; this repacker does not pad, resize, fit stats or
    # apply any loss.
    assert out.raw_model_data["state"].shape == (NUM_CHANNELS,)
    assert out.raw_model_data["actions"].shape == (HORIZON, NUM_CHANNELS)
    assert out.raw_model_data["image"]["base_0_rgb"].shape == (PNG_HEIGHT, PNG_WIDTH, 3)

    prov = out.provenance
    assert prov["model_contract"] == {"pi05": True, "action_dim": 32, "action_horizon": 25}
    assert prov["native_dt_s"] == DT_S
    assert prov["action_window"]["dt_s"] == DT_S
    assert prov["action_window"]["horizon"] == HORIZON
    assert prov["action_window"]["horizon_s"] == HORIZON_S
    assert prov["action_window"]["action_semantics"] == ACTION_SEMANTICS
    assert prov["pending_transforms"] == {
        "normalization": "PENDING",
        "padding": "PENDING",
        "resize": "PENDING",
        "tokenization": "PENDING",
        "loss_application": "PENDING",
    }
    # qvel/previous_ctrl are explicitly unused, not silently consumed.
    assert prov["unused_observation_keys"] == ["qvel_rad_s16", "previous_ctrl_rad16"]
    assert "qvel_rad_s16" not in out.raw_model_data
    assert "previous_ctrl_rad16" not in out.raw_model_data


# --------------------------------------------------------------------------- #
# absent-sensor flags: copied into provenance only, isolated from the example
# --------------------------------------------------------------------------- #


def test_absent_sensor_flags_copied_to_provenance_and_isolated():
    ex = _example()
    out = prepare_pi05_native_sample(ex)

    assert out.provenance["absent_sensors"] == {
        "fsr_N4": None,
        "motor_current_A16": None,
        "wrist": None,
        "fsr_valid4": [False, False, False, False],
        "motor_current_valid16": [False] * NUM_CHANNELS,
        "wrist_valid": False,
    }
    # Flags stay out of the model input entirely.
    assert "absent_sensors" not in out.raw_model_data
    for key in out.provenance["absent_sensors"]:
        assert key not in out.raw_model_data

    # Mutating the output flags must not change the example observation.
    out.provenance["absent_sensors"]["fsr_valid4"][0] = True
    out.provenance["absent_sensors"]["motor_current_valid16"][0] = True
    out.provenance["absent_sensors"]["wrist_valid"] = True
    out.provenance["absent_sensors"]["fsr_N4"] = [1.0] * 4
    assert ex.observation["fsr_valid4"] == [False, False, False, False]
    assert ex.observation["motor_current_valid16"] == [False] * NUM_CHANNELS
    assert ex.observation["wrist_valid"] is False
    assert ex.observation["fsr_N4"] is None
