"""Microfix guard tests for the LEAP v1 learned evaluator helper.

These tests are NumPy/stub only: no GPU, native app, physics, network, or
optimizer. They verify the three microfixes:
  1. build_feature_vector rejects manually mutated valid=False and non-None
     absent-sensor fields (FSR/current/wrist).
  2. stop_and_reap_client follows the native SandboxClient contract
     (_send stop 1s, quit 1s, no second terminate/kill sequence).
  3. REQUIRED_SOURCE_FILES includes normal_priority.py and leap_sim/pad_env.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import small_policy as sp
from embodied_trackpad.leap_v1.learned_evaluator import (
    REQUIRED_SOURCE_FILES,
    EvaluatorError,
    EvaluationContract,
    PolicyInference,
    build_actor_input,
    build_feature_vector,
    stop_and_reap_client,
)


def _synthetic_npz(rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    rng = rng or np.random.default_rng(42)
    return {
        "W1": rng.normal(size=(sp.HIDDEN_DIM, sp.INPUT_DIM)).astype(np.float32) * 0.01,
        "b1": np.zeros(sp.HIDDEN_DIM, dtype=np.float32),
        "W2": rng.normal(size=(sp.HIDDEN_DIM, sp.HIDDEN_DIM)).astype(np.float32) * 0.01,
        "b2": np.zeros(sp.HIDDEN_DIM, dtype=np.float32),
        "W3": rng.normal(size=(sp.OUTPUT_DIM, sp.HIDDEN_DIM)).astype(np.float32) * 0.01,
        "b3": np.zeros(sp.OUTPUT_DIM, dtype=np.float32),
        "feature_mean": np.zeros(sp.INPUT_DIM, dtype=np.float32),
        "feature_std": np.ones(sp.INPUT_DIM, dtype=np.float32),
    }


def _write_npz(path: Path, npz: dict[str, np.ndarray]) -> None:
    np.savez(path, **npz)


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.npz"
    _write_npz(path, _synthetic_npz())
    return path


def _good_actor_input(previous_ctrl_rad16=None):
    return build_actor_input(
        error_xy_px=(1.0, -2.0),
        q_rad16=np.linspace(0.0, 0.1, sp.NUM_CHANNELS, dtype=np.float32),
        qvel_rad_s16=np.zeros(sp.NUM_CHANNELS, dtype=np.float32),
        previous_ctrl_rad16=(
            previous_ctrl_rad16
            if previous_ctrl_rad16 is not None
            else np.zeros(sp.NUM_CHANNELS, dtype=np.float64)
        ),
    )


def test_build_feature_vector_rejects_valid_false():
    actor_input = _good_actor_input()
    actor_input["valid"] = False
    with pytest.raises(EvaluatorError, match="valid"):
        build_feature_vector(actor_input)


def test_build_feature_vector_rejects_non_none_fsr():
    actor_input = _good_actor_input()
    actor_input["fsr_N4"] = np.zeros(4, dtype=np.float32)
    with pytest.raises(EvaluatorError, match="fsr"):
        build_feature_vector(actor_input)


def test_build_feature_vector_rejects_non_none_motor_current():
    actor_input = _good_actor_input()
    actor_input["motor_current_A16"] = np.zeros(sp.NUM_CHANNELS, dtype=np.float32)
    with pytest.raises(EvaluatorError, match="motor_current"):
        build_feature_vector(actor_input)


def test_build_feature_vector_rejects_non_none_wrist():
    actor_input = _good_actor_input()
    actor_input["wrist"] = np.zeros((64, 64, 3), dtype=np.float32)
    with pytest.raises(EvaluatorError, match="wrist"):
        build_feature_vector(actor_input)


def test_policy_inference_roundtrip_preserves_float64_anchor(synthetic_checkpoint):
    contract = EvaluationContract(
        status="accepted_for_leap_v1_small_policy_evaluation",
        allow_optimizer=False,
        sample_index_sha256="0" * 64,
        source_commit="deadbeef",
        dataset_schema="v1-small-policy-smoke.v1",
        checkpoint_sha256=sp.sha256_file(synthetic_checkpoint),
        checkpoint_identity="synthetic-mock-checkpoint",
        target_class=4,
        approved_actor_fields=sp.APPROVED_ACTOR_FIELDS,
        source_file_hashes={},
        limits={},
        purpose="microfix_test",
        suite_id="heldout-axis-v1",
        task_profile="near_pointing_v1",
        native_binary_sha256=None,
    )
    inference = PolicyInference(synthetic_checkpoint, contract)
    anchor = np.linspace(0.05, 0.55, sp.NUM_CHANNELS, dtype=np.float64)
    actor_input = _good_actor_input(previous_ctrl_rad16=anchor)
    result = inference.predict(actor_input)
    assert result["features"].dtype == np.float32
    assert result["features"].shape == (sp.INPUT_DIM,)
    assert result["previous_ctrl_rad16"].dtype == np.float64
    assert np.allclose(result["previous_ctrl_rad16"], anchor)
    assert result["predicted_deltas_rad"].dtype == np.float32
    assert result["requested_controls_rad"].dtype == np.float64


class _StubProc:
    def __init__(self, returncode=None):
        self.pid = 1234
        self.returncode = returncode


class _StubClient:
    def __init__(self, send_return=None, quit_return=0, send_raise=None, quit_raise=None):
        self.proc = _StubProc(returncode=None)
        self._send_return = send_return if send_return is not None else {"type": "stopped"}
        self._quit_return = quit_return
        self._send_raise = send_raise
        self._quit_raise = quit_raise
        self._send_log = []
        self._quit_log = []

    def _send(self, obj, timeout):
        self._send_log.append((obj, timeout))
        if self._send_raise is not None:
            raise self._send_raise
        return self._send_return

    def quit(self, timeout=1.0):
        self._quit_log.append(timeout)
        if self._quit_raise is not None:
            raise self._quit_raise
        self.proc.returncode = self._quit_return
        return self._quit_return


def test_stop_and_reap_uses_send_stop_and_quit_with_one_second_caps():
    client = _StubClient()
    info = stop_and_reap_client(client)
    assert client._send_log == [({"cmd": "stop"}, 1.0)]
    assert client._quit_log == [1.0]
    assert info["child_reaped"] is True
    assert info["child_returncode"] == 0
    assert info["teardown"]["stop_error"] is None
    assert info["teardown"]["quit_error"] is None


def test_stop_and_reap_caps_overly_large_bounds():
    client = _StubClient()
    info = stop_and_reap_client(client, bounds={"stop": 10.0, "quit": 10.0})
    assert client._send_log == [({"cmd": "stop"}, 1.0)]
    assert client._quit_log == [1.0]
    assert info["child_reaped"] is True


def test_stop_and_reap_attempts_quit_after_stop_error():
    client = _StubClient(send_raise=RuntimeError("stop timeout"))
    info = stop_and_reap_client(client)
    assert info["teardown"]["stop_error"] is not None
    assert len(client._quit_log) == 1
    assert info["child_reaped"] is True
    assert info["child_returncode"] == 0


def test_stop_and_reap_counts_quit_error_as_failure_even_if_reaped():
    client = _StubClient(quit_raise=RuntimeError("app did not exit"), quit_return=-9)
    client.proc.returncode = -9
    info = stop_and_reap_client(client)
    assert info["teardown"]["quit_error"] is not None
    assert info["child_reaped"] is False
    assert info["child_returncode"] == -9


def test_stop_and_reap_does_not_unconditionally_poll_proc():
    class NoPollProc:
        pid = 1234
        returncode = None

    class NoPollClient(_StubClient):
        def __init__(self):
            super().__init__()
            self.proc = NoPollProc()

    client = NoPollClient()
    info = stop_and_reap_client(client)
    assert info["child_reaped"] is True
    assert info["child_returncode"] == 0


def test_required_source_files_includes_action_chain_guards():
    assert "src/embodied_trackpad/leap_v1/normal_priority.py" in REQUIRED_SOURCE_FILES
    assert "src/embodied_trackpad/leap_sim/pad_env.py" in REQUIRED_SOURCE_FILES
    # Committed common files must not have been removed.
    assert "src/embodied_trackpad/leap_v1/small_policy.py" in REQUIRED_SOURCE_FILES
    assert "src/embodied_trackpad/leap_v1/learned_evaluator.py" in REQUIRED_SOURCE_FILES
    assert "tools/evaluate_leap_v1_small_policy.py" in REQUIRED_SOURCE_FILES
    # Profile/device behavior is source-bound through the native app sources.
    assert "macos/TrackpadSandbox/Sources/main.swift" in REQUIRED_SOURCE_FILES
    assert "macos/TrackpadSandbox/Sources/DeviceModel.swift" in REQUIRED_SOURCE_FILES
    assert "macos/TrackpadSandbox/Info.plist" in REQUIRED_SOURCE_FILES
    assert "macos/TrackpadSandbox/TrackpadSandbox.entitlements" in REQUIRED_SOURCE_FILES
