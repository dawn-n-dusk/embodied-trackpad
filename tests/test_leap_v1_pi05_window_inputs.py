"""Focused tests for pi05_window_inputs (raw 25x20 ms window sample repacker).

Synthetic in-memory WindowObservation only: no corpus images, no GPU, no
OpenPI. Covers accepted raw layout, exact retention, rounding record, masks,
class distinctness, mutation isolation, semantics mapping, and the rejection
boundary (wrong time/units/semantics/types, malformed provenance, invented
sensors, old native input types).
"""
from __future__ import annotations

import py_compile

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import pi05_window_inputs as wi
from embodied_trackpad.leap_v1.pi05_inputs import (
    DEFAULT_PROMPT,
    Pi05NativeSample,
)
from embodied_trackpad.leap_v1.temporal_windows import (
    TemporalWindowError,
    WindowObservation,
)
from embodied_trackpad.leap_v1.visual_dataset import VisualExample

PY_FILE = wi.__file__

SOURCE_SEMANTICS = "absolute_executed_actuator_command_rad"
CANONICAL_SEMANTICS = "absolute_executed_joint_reference_rad"


def _observation():
    screen = (
        np.arange(600 * 800 * 3, dtype=np.int64).reshape(600, 800, 3) % 251
    ).astype(np.uint8)
    q = (np.arange(16, dtype=np.float64) * 0.01 + 0.1).astype(np.float32)
    obs = {
        "screen_rgb": screen,
        "q_rad16": q,
        "qvel_rad_s16": np.full(16, 7.0, dtype=np.float32),
        "previous_ctrl_rad16": np.full(16, 0.2, dtype=np.float32),
        "fsr_N4": None,
        "motor_current_A16": None,
        "wrist": None,
        "fsr_valid4": [False] * 4,
        "motor_current_valid16": [False] * 16,
        "wrist_valid": False,
    }
    return obs, screen, q


def _target():
    base = np.linspace(-0.3, 0.4, 25, dtype=np.float64)[:, None]
    actions = np.repeat(base, 16, axis=1) + np.arange(16, dtype=np.float64)[None, :] * 1e-4
    tgt = {
        "target_ctrl_rad": actions,
        "dt_s": 0.02,
        "horizon": 25,
        "horizon_s": 0.5,
        "coarse_endpoint_sim_time_s": np.arange(1, 26, dtype=np.float64) * 0.02,
        "action_semantics": SOURCE_SEMANTICS,
    }
    return tgt, actions


def _provenance():
    return {
        "format_version": "leap_v1_temporal_window/v1",
        "sample_id": "s-0",
        "corpus_id": "corpus-a_1",
        "index_sha256": "ab" * 32,
        "row_indices": tuple(range(10)),
        "episode_id": 0,
        "frame_id": 3,
        "split": "train",
        "frame_monotonic_time_s": 12.5,
        "capture_timing": {"frame_age_ms": [1.0, 2.0]},
        "clock_meta": {"simulation_clock_id": "sim0"},
        "action_window": {
            "dt_s": 0.002,
            "horizon": 25,
            "horizon_s": 0.05,
            "action_semantics": CANONICAL_SEMANTICS,
            "first_step_index": 1,
            "end_exclusive_step_index": 26,
            "consumed_prev_step_index": 0,
            "first_sim_time_s": 0.002,
            "end_sim_time_s": 0.05,
            "consumed_prev_sim_time_s": 0.0,
        },
        "prev_action_semantics": CANONICAL_SEMANTICS,
    }


@pytest.fixture
def parts():
    observation, screen, q = _observation()
    target, actions = _target()
    provenance = _provenance()
    window = WindowObservation(
        observation=observation, target=target, provenance=provenance
    )
    return {
        "window": window,
        "observation": observation,
        "target": target,
        "provenance": provenance,
        "screen": screen,
        "q": q,
        "actions": actions,
    }


def _rewindow(p, *, observation=None, target=None, provenance=None):
    return WindowObservation(
        observation=p["observation"] if observation is None else observation,
        target=p["target"] if target is None else target,
        provenance=p["provenance"] if provenance is None else provenance,
    )


def test_module_compiles():
    py_compile.compile(PY_FILE, doraise=True)


def test_accepted_raw_layout(parts):
    sample = wi.prepare_pi05_window_sample(parts["window"])
    raw = sample.raw_model_data
    assert set(raw) == {"state", "image", "image_mask", "actions", "prompt"}
    assert raw["state"].dtype == np.float32 and raw["state"].shape == (16,)
    np.testing.assert_array_equal(raw["state"], parts["q"])
    assert raw["image"]["base_0_rgb"].dtype == np.uint8
    assert raw["image"]["base_0_rgb"].shape == (600, 800, 3)
    np.testing.assert_array_equal(raw["image"]["base_0_rgb"], parts["screen"])
    assert raw["image"]["base_0_rgb"] is not parts["screen"]
    np.testing.assert_array_equal(raw["image"]["left_wrist_0_rgb"], 0)
    np.testing.assert_array_equal(raw["image"]["right_wrist_0_rgb"], 0)
    assert raw["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    assert raw["actions"].dtype == np.float32 and raw["actions"].shape == (25, 16)
    np.testing.assert_array_equal(raw["actions"], parts["actions"].astype(np.float32))
    assert raw["prompt"] == DEFAULT_PROMPT
    assert sample.coarse_actions_rad64.dtype == np.float64
    np.testing.assert_array_equal(sample.coarse_actions_rad64, parts["actions"])
    assert sample.coarse_actions_rad64 is not parts["actions"]


def test_custom_prompt_retained(parts):
    sample = wi.prepare_pi05_window_sample(parts["window"], prompt="Tap the target.")
    assert sample.raw_model_data["prompt"] == "Tap the target."


def test_explicit_f32_rounding_error(parts):
    sample = wi.prepare_pi05_window_sample(parts["window"])
    expected = float(
        np.max(np.abs(parts["actions"].astype(np.float32).astype(np.float64) - parts["actions"]))
    )
    got = sample.provenance["actions_f64_to_f32_max_abs_error_rad"]
    assert isinstance(got, float) and got == expected
    assert 0.0 < got < 1e-5


def test_masks_pending_and_contracts(parts):
    sample = wi.prepare_pi05_window_sample(parts["window"])
    assert sample.loss_mask32.dtype == bool and sample.loss_mask32.shape == (32,)
    assert sample.loss_mask32[:16].all() and not sample.loss_mask32[16:].any()
    pending = sample.provenance["pending_transforms"]
    assert set(pending) == {"normalization", "padding", "resize", "tokenization", "loss_application"}
    assert all(v == "PENDING" for v in pending.values())
    assert sample.provenance["sample_format_version"] == wi.SAMPLE_FORMAT_VERSION
    assert sample.provenance["frame_monotonic_time_s"] == 12.5
    assert sample.provenance["unused_observation_keys"] == ["qvel_rad_s16", "previous_ctrl_rad16"]
    assert sample.provenance["absent_sensors"]["fsr_N4"] is None
    assert sample.provenance["absent_sensors"]["wrist_valid"] is False


def test_distinct_class_not_native_subclass(parts):
    sample = wi.prepare_pi05_window_sample(parts["window"])
    assert isinstance(sample, wi.Pi05WindowSample)
    assert not isinstance(sample, Pi05NativeSample)
    assert type(sample) is wi.Pi05WindowSample


def test_semantics_mapping_recorded(parts):
    sample = wi.prepare_pi05_window_sample(parts["window"])
    wc = sample.provenance["window_contract"]
    assert wc["source_action_semantics"] == SOURCE_SEMANTICS
    assert wc["action_semantics"] == CANONICAL_SEMANTICS
    assert wc["native_dt_s"] == 0.002 and wc["target_dt_s"] == 0.02
    assert wc["horizon"] == 25 and wc["horizon_s"] == 0.5
    np.testing.assert_allclose(
        wc["coarse_endpoint_sim_time_s"],
        np.arange(1, 26, dtype=np.float64) * 0.02,
        atol=0.0,
        rtol=0.0,
    )
    # Original native action_window record stays untouched and canonical.
    assert (
        sample.provenance["action_window"]["action_semantics"] == CANONICAL_SEMANTICS
    )


def test_raw_model_data_has_no_oracle_or_teacher_fields(parts):
    parts["provenance"]["target_class"] = "secret"
    parts["provenance"]["error_xy"] = [1.0, 2.0]
    sample = wi.prepare_pi05_window_sample(parts["window"])
    raw = sample.raw_model_data
    for key in raw:
        assert "target_class" not in key and "error_xy" not in key and "teacher" not in key
    assert isinstance(raw["prompt"], str)
    assert "target_class" not in sample.raw_model_data
    assert "error_xy" not in sample.raw_model_data


def test_mutation_isolation(parts):
    sample = wi.prepare_pi05_window_sample(parts["window"])
    # Source mutations after prepare must not leak into the sample.
    parts["screen"][0, 0, 0] = 255
    parts["q"][0] = 9.0
    parts["actions"][0, 0] = 9.0
    parts["provenance"]["capture_timing"]["frame_age_ms"][0] = 99.0
    parts["provenance"]["action_window"]["end_sim_time_s"] = 9.0
    assert sample.raw_model_data["image"]["base_0_rgb"][0, 0, 0] != 255
    assert sample.raw_model_data["state"][0] != 9.0
    assert sample.raw_model_data["actions"][0, 0] != 9.0
    assert sample.provenance["capture_timing"]["frame_age_ms"][0] == 1.0
    assert sample.provenance["action_window"]["end_sim_time_s"] == 0.05
    # Returned-sample mutations must not leak back into the source or the f64 copy.
    sample.raw_model_data["image"]["base_0_rgb"][1, 1, 1] = 77
    sample.raw_model_data["state"][1] = 5.0
    sample.raw_model_data["actions"][1, 1] = 5.0
    sample.coarse_actions_rad64[2, 2] = 8.0
    sample.provenance["capture_timing"]["frame_age_ms"][1] = 55.0
    sample.provenance["window_contract"]["coarse_endpoint_sim_time_s"][0] = -1.0
    assert parts["screen"][1, 1, 1] != 77
    assert parts["q"][1] != 5.0
    assert parts["actions"][1, 1] != 5.0
    assert parts["actions"][2, 2] != 8.0
    assert parts["provenance"]["capture_timing"]["frame_age_ms"][1] == 2.0


def test_identity_format_strict(parts):
    # Uppercase hex and dash/underscore corpus names are accepted.
    prov = {**parts["provenance"], "index_sha256": "AB" * 32, "corpus_id": "my-corpus_2"}
    wi.prepare_pi05_window_sample(_rewindow(parts, provenance=prov))
    # 63-hex or 65-hex hashes are rejected.
    for bad_sha in ("ab" * 31 + "a", "ab" * 32 + "a"):
        with pytest.raises(TemporalWindowError):
            wi.prepare_pi05_window_sample(
                _rewindow(parts, provenance={**parts["provenance"], "index_sha256": bad_sha})
            )
    # Non-hex 64-char strings are rejected.
    with pytest.raises(TemporalWindowError):
        wi.prepare_pi05_window_sample(
            _rewindow(parts, provenance={**parts["provenance"], "index_sha256": "z" * 64})
        )


_REJECTIONS = [
    ("coarse_dt_old", lambda p: {"target": {**p["target"], "dt_s": 0.002}}),
    ("coarse_horizon_24", lambda p: {"target": {**p["target"], "horizon": 24}}),
    ("coarse_horizon_s_wrong", lambda p: {"target": {**p["target"], "horizon_s": 0.05}}),
    (
        "endpoint_shifted",
        lambda p: {
            "target": {
                **p["target"],
                "coarse_endpoint_sim_time_s": p["target"]["coarse_endpoint_sim_time_s"] + 1e-6,
            }
        },
    ),
    (
        "endpoint_nonfinite",
        lambda p: {
            "target": {
                **p["target"],
                "coarse_endpoint_sim_time_s": _with(p["target"]["coarse_endpoint_sim_time_s"], 0, np.nan),
            }
        },
    ),
    (
        "endpoint_wrong_dtype",
        lambda p: {
            "target": {
                **p["target"],
                "coarse_endpoint_sim_time_s": p["target"]["coarse_endpoint_sim_time_s"].astype(np.float32),
            }
        },
    ),
    ("semantics_delta", lambda p: {"target": {**p["target"], "action_semantics": "delta_joint_delta_rad"}}),
    ("semantics_canonical_rejected", lambda p: {"target": {**p["target"], "action_semantics": CANONICAL_SEMANTICS}}),
    (
        "actions_nonfinite",
        lambda p: {"target": {**p["target"], "target_ctrl_rad": _with(p["actions"], (0, 0), np.inf)}},
    ),
    (
        "actions_wrong_dtype",
        lambda p: {"target": {**p["target"], "target_ctrl_rad": p["actions"].astype(np.float32)}},
    ),
    (
        "actions_overflow_f32",
        lambda p: {"target": {**p["target"], "target_ctrl_rad": np.full((25, 16), 1e300)}},
    ),
    ("image_wrong_dtype", lambda p: {"observation": {**p["observation"], "screen_rgb": p["screen"].astype(np.float32)}}),
    ("image_wrong_shape", lambda p: {"observation": {**p["observation"], "screen_rgb": p["screen"][:, :799, :]}}),
    ("q_wrong_dtype", lambda p: {"observation": {**p["observation"], "q_rad16": p["q"].astype(np.float64)}}),
    ("q_nonfinite", lambda p: {"observation": {**p["observation"], "q_rad16": _with(p["q"], 0, np.nan)}}),
    ("q_wrong_len", lambda p: {"observation": {**p["observation"], "q_rad16": p["q"][:15]}}),
    (
        "native_dt_wrong",
        lambda p: {"provenance": {**p["provenance"], "action_window": {**p["provenance"]["action_window"], "dt_s": 0.02}}},
    ),
    (
        "native_horizon_wrong",
        lambda p: {"provenance": {**p["provenance"], "action_window": {**p["provenance"]["action_window"], "horizon": 24}}},
    ),
    (
        "native_end_time_wrong",
        lambda p: {"provenance": {**p["provenance"], "action_window": {**p["provenance"]["action_window"], "end_sim_time_s": 0.06}}},
    ),
    (
        "native_first_time_wrong",
        lambda p: {"provenance": {**p["provenance"], "action_window": {**p["provenance"]["action_window"], "first_sim_time_s": 0.004}}},
    ),
    (
        "native_semantics_source_rejected",
        lambda p: {"provenance": {**p["provenance"], "action_window": {**p["provenance"]["action_window"], "action_semantics": SOURCE_SEMANTICS}}},
    ),
    (
        "native_window_missing",
        lambda p: {"provenance": {k: v for k, v in p["provenance"].items() if k != "action_window"}},
    ),
    ("format_version_wrong", lambda p: {"provenance": {**p["provenance"], "format_version": "leap_v1_temporal_window/v2"}}),
    ("split_invalid", lambda p: {"provenance": {**p["provenance"], "split": "heldout"}}),
    ("row_indices_too_few", lambda p: {"provenance": {**p["provenance"], "row_indices": tuple(range(9))}}),
    ("row_indices_unsorted", lambda p: {"provenance": {**p["provenance"], "row_indices": (0, 2, 1, 3, 4, 5, 6, 7, 8, 9)}}),
    ("row_indices_negative", lambda p: {"provenance": {**p["provenance"], "row_indices": tuple(range(-1, 9))}}),
    ("row_indices_bool", lambda p: {"provenance": {**p["provenance"], "row_indices": (True,) + tuple(range(1, 10))}}),
    ("episode_id_str", lambda p: {"provenance": {**p["provenance"], "episode_id": "0"}}),
    ("frame_id_negative", lambda p: {"provenance": {**p["provenance"], "frame_id": -1}}),
    ("corpus_id_empty", lambda p: {"provenance": {**p["provenance"], "corpus_id": ""}}),
    ("corpus_id_space", lambda p: {"provenance": {**p["provenance"], "corpus_id": "cor pus"}}),
    ("corpus_id_slash", lambda p: {"provenance": {**p["provenance"], "corpus_id": "a/b"}}),
    ("corpus_id_dot", lambda p: {"provenance": {**p["provenance"], "corpus_id": "a.b"}}),
    ("sha_short", lambda p: {"provenance": {**p["provenance"], "index_sha256": "ab" * 16}}),
    ("sha_nonhex", lambda p: {"provenance": {**p["provenance"], "index_sha256": "zz" * 32}}),
    ("fsr_invented", lambda p: {"observation": {**p["observation"], "fsr_N4": np.zeros(4)}}),
    ("motor_current_invented", lambda p: {"observation": {**p["observation"], "motor_current_A16": np.zeros(16)}}),
    ("wrist_invented", lambda p: {"observation": {**p["observation"], "wrist": {"left": None}}}),
    ("fsr_valid_true", lambda p: {"observation": {**p["observation"], "fsr_valid4": [True] * 4}}),
    ("motor_valid_numeric", lambda p: {"observation": {**p["observation"], "motor_current_valid16": [0] * 16}}),
    ("wrist_valid_true", lambda p: {"observation": {**p["observation"], "wrist_valid": True}}),
]


def _with(arr, index, value):
    out = np.array(arr, copy=True)
    out[index] = value
    return out


@pytest.mark.parametrize("case", [c for c in _REJECTIONS], ids=[c[0] for c in _REJECTIONS])
def test_rejections(parts, case):
    _, make_overrides = case
    overrides = make_overrides(parts)
    window = _rewindow(
        parts,
        observation=overrides.get("observation"),
        target=overrides.get("target"),
        provenance=overrides.get("provenance"),
    )
    with pytest.raises(TemporalWindowError):
        wi.prepare_pi05_window_sample(window)


def test_rejects_empty_prompt(parts):
    with pytest.raises(TemporalWindowError):
        wi.prepare_pi05_window_sample(parts["window"], prompt="   ")


def test_rejects_old_input_types(parts):
    visual = VisualExample(
        observation=parts["observation"],
        target=parts["target"],
        provenance=parts["provenance"],
    )
    with pytest.raises(TemporalWindowError):
        wi.prepare_pi05_window_sample(visual)
    native = Pi05NativeSample(
        raw_model_data={}, loss_mask32=np.zeros(32, dtype=bool), provenance={}
    )
    with pytest.raises(TemporalWindowError):
        wi.prepare_pi05_window_sample(native)

    class Duck:
        observation = parts["observation"]
        target = parts["target"]
        provenance = parts["provenance"]

    with pytest.raises(TemporalWindowError):
        wi.prepare_pi05_window_sample(Duck())
