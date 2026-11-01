"""Unit/stub tests for the frozen LEAP v1 small-policy evaluator.

These tests use synthetic checkpoints and do not require a real GPU checkpoint,
native app, or physics execution.  They verify permit binding, input guards,
deployment arithmetic, action validation, counterfactual recording, and runner
lifecycle helpers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import small_policy as sp
from embodied_trackpad.leap_v1.learned_evaluator import (
    ALLOWED_TARGET_CLASSES,
    MAX_ACTION_IMAGES,
    EvaluatorError,
    PolicyInference,
    build_actor_input,
    build_feature_vector,
    compute_screen_dependence,
    compute_source_file_hashes,
    decode_controls_deployment,
    load_and_validate_checkpoint,
    stop_and_reap_client,
    validate_controls_against_ranges,
    validate_evaluation_permit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_npz(rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Return a valid small-policy weight dictionary (NumPy only)."""
    rng = rng or np.random.default_rng(12345)
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


def _make_good_permit(tmp_path: Path, checkpoint_path: Path, target_class: int = 4,
                      mutate_source_hash: str | None = None,
                      mutate_limit: dict | None = None,
                      source_commit: str = "5344a3a",
                      dataset_schema: str = "v1-small-policy-smoke.v1") -> tuple[Path, dict]:
    sample_index = tmp_path / "samples.jsonl"
    sample_index.write_text('{"episode_id":0,"frame_id":0}\n')
    index_sha = sp.sha256_file(sample_index)
    source_hashes = compute_source_file_hashes(REPO_ROOT)
    if mutate_source_hash:
        source_hashes[mutate_source_hash] = "0" * 64
    permit = {
        "status": "accepted_for_leap_v1_small_policy_evaluation",
        "allow_optimizer": False,
        "sample_index_sha256": index_sha,
        "source_commit": source_commit,
        "dataset_schema": dataset_schema,
        "checkpoint_sha256": sp.sha256_file(checkpoint_path),
        "checkpoint_identity": "synthetic-mock-checkpoint",
        "target_class": target_class,
        "approved_actor_fields": list(sp.APPROVED_ACTOR_FIELDS),
        "source_file_hashes": source_hashes,
        "limits": {"max_action_images": 120, "max_wall_s": 120,
                   "max_sim_s": 20, "dir_cap_bytes": 60 * 1024 * 1024,
                   "batch_cap_bytes": 160 * 1024 * 1024},
    }
    if mutate_limit:
        permit["limits"].update(mutate_limit)
    permit_path = tmp_path / "permit.json"
    permit_path.write_text(json.dumps(permit))
    return permit_path, permit


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.npz"
    _write_npz(path, _synthetic_npz())
    return path


def test_load_and_validate_checkpoint_ok(synthetic_checkpoint):
    npz, sha = load_and_validate_checkpoint(synthetic_checkpoint)
    assert sha == sp.sha256_file(synthetic_checkpoint)
    assert set(npz.keys()) == {"W1", "b1", "W2", "b2", "W3", "b3", "feature_mean", "feature_std"}
    assert npz["W1"].shape == (sp.HIDDEN_DIM, sp.INPUT_DIM)


def test_load_and_validate_checkpoint_rejects_nan_weight(tmp_path):
    npz = _synthetic_npz()
    npz["W1"][0, 0] = np.nan
    path = tmp_path / "bad.npz"
    _write_npz(path, npz)
    with pytest.raises(EvaluatorError, match="finite|checkpoint validation"):
        load_and_validate_checkpoint(path)


def test_validate_permit_accepts_valid(tmp_path, synthetic_checkpoint):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint, target_class=4)
    contract = validate_evaluation_permit(
        permit,
        checkpoint_path=synthetic_checkpoint,
        sample_index_path=tmp_path / "samples.jsonl",
        source_commit="5344a3a",
        dataset_schema="v1-small-policy-smoke.v1",
        target_class=4,
        source_file_hashes=compute_source_file_hashes(REPO_ROOT),
    )
    assert contract.target_class == 4
    assert contract.allow_optimizer is False
    assert contract.checkpoint_sha256 == sp.sha256_file(synthetic_checkpoint)


def test_validate_permit_rejects_bad_status(synthetic_checkpoint):
    with pytest.raises(EvaluatorError, match="status"):
        validate_evaluation_permit({"status": "draft", "allow_optimizer": False,
                                    "sample_index_sha256": "0" * 64,
                                    "source_commit": "a", "dataset_schema": "x",
                                    "checkpoint_sha256": "0" * 64,
                                    "checkpoint_identity": "x", "target_class": 4,
                                    "approved_actor_fields": list(sp.APPROVED_ACTOR_FIELDS),
                                    "source_file_hashes": {},
                                    "limits": {"max_action_images": 1, "max_wall_s": 1,
                                               "max_sim_s": 1, "dir_cap_bytes": 1,
                                               "batch_cap_bytes": 1}})


def test_validate_permit_rejects_optimizer_true(tmp_path, synthetic_checkpoint):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint)
    permit["allow_optimizer"] = True
    with pytest.raises(EvaluatorError, match="allow_optimizer"):
        validate_evaluation_permit(permit)


def test_validate_permit_rejects_wrong_target_class(tmp_path, synthetic_checkpoint):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint, target_class=4)
    with pytest.raises(EvaluatorError, match="target_class"):
        validate_evaluation_permit(permit, target_class=10)


def test_validate_permit_rejects_class_outside_allowed(tmp_path, synthetic_checkpoint):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint, target_class=4)
    permit["target_class"] = 7
    with pytest.raises(EvaluatorError, match="target_class"):
        validate_evaluation_permit(permit)


def test_validate_permit_rejects_changed_checkpoint(tmp_path, synthetic_checkpoint):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint)
    other = tmp_path / "other.npz"
    _write_npz(other, _synthetic_npz(np.random.default_rng(999)))
    with pytest.raises(EvaluatorError, match="checkpoint_sha256 mismatch"):
        validate_evaluation_permit(permit, checkpoint_path=other)


def test_validate_permit_rejects_changed_source_file(tmp_path, synthetic_checkpoint):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint,
                                            mutate_source_hash="src/embodied_trackpad/leap_v1/small_policy.py")
    with pytest.raises(EvaluatorError, match="source_file_hashes mismatch"):
        validate_evaluation_permit(permit, source_file_hashes=compute_source_file_hashes(REPO_ROOT))


def test_validate_permit_rejects_exceeded_budget(tmp_path, synthetic_checkpoint):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint,
                                            mutate_limit={"max_action_images": 121})
    with pytest.raises(EvaluatorError, match="max_action_images exceeds cap"):
        validate_evaluation_permit(permit)


def test_build_actor_input_exact_feature_shape():
    actor_input = build_actor_input(
        error_xy_px=(10.0, -5.0),
        q_rad16=np.zeros(sp.NUM_CHANNELS),
        qvel_rad_s16=np.zeros(sp.NUM_CHANNELS),
        previous_ctrl_rad16=np.zeros(sp.NUM_CHANNELS),
    )
    features = build_feature_vector(actor_input)
    assert features.shape == (sp.INPUT_DIM,)
    assert features.dtype == np.float32
    assert np.allclose(features[:2], [10.0, -5.0])


def test_build_actor_input_rejects_rejected_fields():
    actor_input = build_actor_input(
        error_xy_px=(0.0, 0.0),
        q_rad16=np.zeros(sp.NUM_CHANNELS),
        qvel_rad_s16=np.zeros(sp.NUM_CHANNELS),
        previous_ctrl_rad16=np.zeros(sp.NUM_CHANNELS),
    )
    bad = dict(actor_input)
    bad["force"] = [1.0]
    with pytest.raises(EvaluatorError, match="rejected"):
        build_feature_vector(bad)


def test_build_actor_input_rejects_wrist_image():
    actor_input = build_actor_input(
        error_xy_px=(0.0, 0.0),
        q_rad16=np.zeros(sp.NUM_CHANNELS),
        qvel_rad_s16=np.zeros(sp.NUM_CHANNELS),
        previous_ctrl_rad16=np.zeros(sp.NUM_CHANNELS),
    )
    bad = dict(actor_input)
    bad["wrist"] = np.zeros((64, 64, 3))
    with pytest.raises(EvaluatorError, match="wrist"):
        build_feature_vector(bad)


def test_build_actor_input_rejects_nan():
    q = np.zeros(sp.NUM_CHANNELS)
    q[3] = np.nan
    with pytest.raises(EvaluatorError, match="non-finite"):
        build_actor_input(
            error_xy_px=(1.0, 1.0),
            q_rad16=q,
            qvel_rad_s16=np.zeros(sp.NUM_CHANNELS),
            previous_ctrl_rad16=np.zeros(sp.NUM_CHANNELS),
        )


def test_policy_inference_predict_full16(synthetic_checkpoint, tmp_path):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint, target_class=4)
    contract = validate_evaluation_permit(permit, checkpoint_path=synthetic_checkpoint,
                                          target_class=4)
    inference = PolicyInference(synthetic_checkpoint, contract)
    actor_input = build_actor_input(
        error_xy_px=(12.0, 0.0),
        q_rad16=np.zeros(sp.NUM_CHANNELS),
        qvel_rad_s16=np.zeros(sp.NUM_CHANNELS),
        previous_ctrl_rad16=np.zeros(sp.NUM_CHANNELS),
    )
    result = inference.predict(actor_input)
    assert result["predicted_deltas_rad"].shape == (sp.HORIZON, sp.NUM_CHANNELS)
    assert result["requested_controls_rad"].shape == (sp.HORIZON, sp.NUM_CHANNELS)
    assert result["requested_controls_rad"].dtype == np.float64
    assert result["predicted_deltas_rad"].dtype == np.float32


def test_decode_controls_deployment_preserves_caps():
    """Float64 accumulation keeps block/substep caps even near large anchors."""
    previous_ctrl = np.full(sp.NUM_CHANNELS, 0.6, dtype=np.float64)
    deltas = np.full((sp.HORIZON, sp.NUM_CHANNELS), sp.ACTION_CAP_RAD, dtype=np.float32)
    controls = decode_controls_deployment(deltas, previous_ctrl)
    assert controls.dtype == np.float64
    # First step and each subsequent step are within the per-substep cap.
    assert np.max(np.abs(controls[0] - previous_ctrl)) <= sp.ACTION_CAP_RAD + 1e-12
    assert np.max(np.abs(controls[1:] - controls[:-1])) <= sp.ACTION_CAP_RAD + 1e-12
    # Block cap from the anchor.
    assert np.max(np.abs(controls - previous_ctrl)) <= sp.BLOCK_CAP_RAD + 1e-12


def test_decode_controls_deployment_rejects_excess_block():
    previous_ctrl = np.zeros(sp.NUM_CHANNELS, dtype=np.float64)
    deltas = np.full((sp.HORIZON, sp.NUM_CHANNELS), sp.ACTION_CAP_RAD * 1.1, dtype=np.float32)
    with pytest.raises(EvaluatorError, match="substep delta"):
        decode_controls_deployment(deltas, previous_ctrl)


def test_validate_controls_against_ranges_rejects_out_of_range():
    ranges = [(-0.5, 0.5)] * sp.NUM_CHANNELS
    controls = np.zeros((sp.HORIZON, sp.NUM_CHANNELS), dtype=np.float64)
    controls[10, 5] = 1.0
    with pytest.raises(EvaluatorError, match="outside range"):
        validate_controls_against_ranges(controls, ranges)


def test_compute_screen_dependence_records_no_physics(synthetic_checkpoint, tmp_path):
    permit_path, permit = _make_good_permit(tmp_path, synthetic_checkpoint, target_class=4)
    contract = validate_evaluation_permit(permit, checkpoint_path=synthetic_checkpoint, target_class=4)
    inference = PolicyInference(synthetic_checkpoint, contract)
    actor_input = build_actor_input(
        error_xy_px=(12.0, -8.0),
        q_rad16=np.zeros(sp.NUM_CHANNELS),
        qvel_rad_s16=np.zeros(sp.NUM_CHANNELS),
        previous_ctrl_rad16=np.zeros(sp.NUM_CHANNELS),
    )
    cf = compute_screen_dependence(actor_input, inference.npz)
    assert "base_predicted_deltas_rad" in cf
    assert "perturbations" in cf
    for name, info in cf["perturbations"].items():
        assert "max_abs_delta_rad" in info
        assert "perturbed_error_xy_px" in info
    # Counterfactual uses the same q/qvel/prevctrl; only error changed.
    assert cf["base_error_xy_px"] == [12.0, -8.0]


def test_stop_and_reap_client_uses_send_stop_and_quit():
    class FakeProc:
        pid = 1234
        returncode = None

    class FakeClient:
        proc = FakeProc()

        def __init__(self):
            self.sent = []

        def _send(self, obj, timeout=30.0):
            self.sent.append((obj, timeout))
            return {"type": "stopped"}

        def quit(self, timeout=1.0):
            self.proc.returncode = 0
            return 0

    client = FakeClient()
    info = stop_and_reap_client(client, bounds={"stop": 1.0, "quit": 1.0})
    assert any(obj.get("cmd") == "stop" for obj, _ in client.sent)
    assert info["child_reaped"] is True
    assert info["child_returncode"] == 0
    assert info["teardown"]["requested_bounds_s"]["stop"] <= 1.0


def test_stop_and_reap_client_continues_after_send_stop_error():
    class FakeProc:
        pid = 1234
        returncode = None

    class FailingStopClient:
        proc = FakeProc()
        quit_called = False

        def _send(self, obj, timeout=30.0):
            raise RuntimeError("stop failed")

        def quit(self, timeout=1.0):
            self.quit_called = True
            self.proc.returncode = 0
            return 0

    client = FailingStopClient()
    info = stop_and_reap_client(client)
    assert client.quit_called
    assert info["child_reaped"] is True
    assert info["teardown"]["stop_error"] is not None


def test_runner_helpers_terminal_no_action():
    """A terminal success frame records action_requested=false and no next_ctrl."""
    pytest.importorskip("mujoco")
    from tools import evaluate_leap_v1_small_policy as runner
    import io

    R = {"actor": io.StringIO(), "cycle": 5}
    frame = {"path": "cycle-005.png", "png": b"pngbytes", "frame_id": 5,
             "monotonic_time_s": 1.0}
    est = type("E", (), {"error_px": (2.0, 1.0)})()
    runner.log_capture_reject(R, frame, est, 12.0, ["success_terminal_no_action"])
    line = json.loads(R["actor"].getvalue())
    assert line["phase"] == "gate_rejected"
    assert line["action_requested"] is False


def test_runner_helpers_success_and_release():
    """Import runner helpers and exercise success/release logic without physics/app."""
    pytest.importorskip("mujoco")
    from tools import evaluate_leap_v1_small_policy as runner

    assert runner.ALLOWED_CLASSES == {4, 10}
    # Success requires 3 frames <=3px and >=50% reduction.
    assert runner.evaluate_success(20.0, [15.0, 8.0, 4.0]) is False
    assert runner.evaluate_success(20.0, [3.0, 3.0, 2.0]) is True
    assert runner.evaluate_success(5.0, [3.0, 3.0, 3.0]) is False  # reduction < 50%

    # Release requires contiguous trailing records with no pad contacts.
    good = [
        {"sim_time_s": 0.0, "contacts": []},
        {"sim_time_s": 0.002, "contacts": []},
    ]
    assert runner.release_from_records(good) is False  # too short


def test_allowed_target_classes():
    assert ALLOWED_TARGET_CLASSES == {4, 10}


def _import_runner():
    pytest.importorskip("mujoco")
    from tools import evaluate_leap_v1_small_policy as runner
    return runner


def _minimal_R(actor: "io.StringIO") -> dict:
    """Return a minimal runner state dict for stub visual-phase tests."""
    import io
    import tempfile
    import time
    from pathlib import Path

    q = np.zeros(sp.NUM_CHANNELS, dtype=float)
    qvel = np.zeros(sp.NUM_CHANNELS, dtype=float)
    out = Path(tempfile.mkdtemp())
    return {
        "actor": actor,
        "teacher": io.StringIO(),
        "touch": io.StringIO(),
        "planner": io.StringIO(),
        "cycle": 0,
        "step_index": 0,
        "seq": 0,
        "ctrl": np.zeros(sp.NUM_CHANNELS),
        "errors": [],
        "initial_error": None,
        "final_error": None,
        "success": False,
        "cleanup_records": [],
        "last_record": None,
        "full_map": type("M", (), {"ranges": [(-0.9, 0.9)] * sp.NUM_CHANNELS})(),
        "contract": type("C", (), {"limits": {
            "max_action_images": 120,
            "max_wall_s": 120,
            "max_sim_s": 20,
            "dir_cap_bytes": 64 * 1024 * 1024,
            "batch_cap_bytes": 160 * 1024 * 1024,
        }})(),
        "env": type("E", (), {"data": type("D", (), {
            "time": 0.0,
            "qpos": q,
            "qvel": qvel,
        })()})(),
        "client": None,
        "reconciler": type("Rec", (), {
            "offset_s": 0.0,
            "uncertainty_s": 0.0,
            "uncertainty_ms": 0.0,
            "frame_age_ms": lambda self, f: 0.0,
        })(),
        "t0": time.monotonic(),
        "operation_wall_s": 120.0,
        "out": out,
    }


def test_main_validates_binding_and_frozen_seed_before_run(monkeypatch, tmp_path, synthetic_checkpoint):
    """main validates git HEAD, source hashes, index/schema and seed=class before run."""
    runner = _import_runner()
    git_head = runner._git_head(REPO_ROOT)
    permit_path, permit = _make_good_permit(
        tmp_path, synthetic_checkpoint, target_class=4,
        source_commit=git_head or "unknown",
        dataset_schema=runner.DATASET_SCHEMA,
    )
    sample_index = tmp_path / "samples.jsonl"
    sample_index.write_text('{"episode_id":0,"frame_id":0}\n')
    out_dir = tmp_path / "run-out"
    run_calls = []

    def fake_run(seed, out_dir_arg, contract, inference, target_class):
        run_calls.append({"seed": seed, "out_dir": out_dir_arg,
                          "target_class": target_class, "contract": contract})
        return {"learning_success": False}

    monkeypatch.setattr(runner, "run", fake_run)
    argv = [
        "--seed", "4", "--class", "4",
        "--checkpoint", str(synthetic_checkpoint),
        "--permit", str(permit_path),
        "--sample-index", str(sample_index),
        "--out", str(out_dir),
    ]
    rc = runner.main(argv)
    assert len(run_calls) == 1
    assert run_calls[0]["seed"] == 4
    assert run_calls[0]["target_class"] == 4
    assert run_calls[0]["out_dir"] == str(out_dir)
    assert run_calls[0]["contract"].dataset_schema == runner.DATASET_SCHEMA
    assert rc == 1  # fake run reports learning_success=False


@pytest.mark.parametrize("fail_step", [1, 2])
def test_learned_phase_step_error_preserves_intent_and_count(fail_step, monkeypatch, tmp_path):
    """A PhysicsStepError records the failed substep and preserves intent/count."""
    runner = _import_runner()
    import io
    R = _minimal_R(io.StringIO())
    R["gate"] = type("Gate", (), {
        "check": lambda self, f, e, a: type("D", (), {"ok": True, "reasons": []})()
    })()
    frame = {"path": "cycle-001.png", "png": b"pngbytes", "frame_id": 1,
             "monotonic_time_s": 1.0}
    est = type("E", (), {"error_px": (20.0, 0.0)})()

    class FakeInf:
        npz = _synthetic_npz()

        def predict(self, actor_input):
            return {
                "requested_controls_rad": np.zeros((sp.HORIZON, sp.NUM_CHANNELS), dtype=np.float64),
                "predicted_deltas_rad": np.zeros((sp.HORIZON, sp.NUM_CHANNELS), dtype=np.float32),
                "features": np.zeros(sp.INPUT_DIM, dtype=np.float32),
            }

    physics_calls = []

    def _step_record(step_index, frame_id, ok, reasons):
        return runner.vp.StepRecord(
            step_index=step_index, sim_time_s=0.0,
            frame_id=frame_id, ctrl=[], qpos=[], qvel=[], contacts=[],
            aggregate={}, warning_delta=[], checks={}, ok=ok,
            failure_reasons=reasons, ctrl_shape=[],
        )

    def fake_physics_step(env, ctrl, *, step_index, frame_id, **kwargs):
        physics_calls.append(step_index)
        if len(physics_calls) == fail_step:
            raise runner.vp.PhysicsStepError(
                _step_record(step_index, frame_id, ok=False, reasons=["stub"])
            )
        return _step_record(step_index, frame_id, ok=True, reasons=[])

    monkeypatch.setattr(runner, "capture_estimate", lambda R_local: (frame, est, 0.0))
    monkeypatch.setattr(runner, "send_touch", lambda R_local, agg: None)
    monkeypatch.setattr(runner.vp, "physics_step", fake_physics_step)

    with pytest.raises(runner.vp.PhysicsStepError):
        runner.visual_phase_learned(R, FakeInf())

    actor_lines = [json.loads(l) for l in R["actor"].getvalue().splitlines()]
    intent = [l for l in actor_lines if l["phase"] == "learned_action_intent"][0]
    executed = [l for l in actor_lines if l["phase"] == "learned_action_executed"][0]
    assert intent["planned_step_index_start"] == 1
    assert executed["outcome"] == "physics_unsafe"
    assert executed["executed_step_count"] == fail_step

    teacher_lines = [json.loads(l) for l in R["teacher"].getvalue().splitlines()]
    failing = [l for l in teacher_lines if l["phase"] == "learned" and l.get("failure_reasons") == ["stub"]][0]
    assert failing["step_index"] == fail_step
    assert failing["frame_id"] == 1
    assert len(physics_calls) == fail_step


def test_run_lifecycle_skips_cleanup_on_unsafe_invokes_teardown(monkeypatch, tmp_path, synthetic_checkpoint):
    """On physics unsafe, run skips physical cleanup but still runs software teardown."""
    runner = _import_runner()
    git_head = runner._git_head(REPO_ROOT)
    permit_path, permit = _make_good_permit(
        tmp_path, synthetic_checkpoint, target_class=4,
        source_commit=git_head or "unknown",
        dataset_schema=runner.DATASET_SCHEMA,
    )
    sample_index = tmp_path / "samples.jsonl"
    sample_index.write_text('{"episode_id":0,"frame_id":0}\n')
    contract = validate_evaluation_permit(
        permit, checkpoint_path=synthetic_checkpoint,
        sample_index_path=sample_index, source_commit=git_head,
        dataset_schema=runner.DATASET_SCHEMA, target_class=4,
        source_file_hashes=compute_source_file_hashes(REPO_ROOT),
    )
    inference = PolicyInference(synthetic_checkpoint, contract)

    class FakeEnv:
        model = type("M", (), {})()

    class FakeMap:
        ranges = [(-0.9, 0.9)] * runner.vp.NU
        act_index = list(range(runner.vp.NU))
        joint_names = list(sp.MOTOR_JOINTS)

    class FakeClient:
        def __init__(self, out_dir):
            self.out_dir = out_dir
            self.proc = type("P", (), {"pid": 1234, "returncode": None})()

        def reset(self, seed, task_profile=None):
            pass

        def capture(self, save_as=None):
            return {"frame_id": 1, "png": b"x", "path": "x.png",
                    "monotonic_time_s": 1.0}

        def _send(self, obj, timeout=30.0):
            if obj.get("cmd") == "stop":
                return {"type": "stopped"}
            return {"type": "touch_ok", "sequence": obj.get("sequence", 0)}

        def raw(self, obj, timeout=30.0):
            return self._send(obj, timeout)

        def quit(self, timeout=1.0):
            self.proc.returncode = 0
            return 0

    class FakeReconciler:
        offset_s = 0.0
        uncertainty_s = 0.0
        uncertainty_ms = 0.0

        @classmethod
        def sample(cls, capture):
            return cls()

    class FakeGate:
        def __init__(self, cfg):
            pass

        def check(self, frame, est, age_ms):
            return type("D", (), {"ok": True, "reasons": []})()

    cleanup_called = []
    teardown_calls = []

    monkeypatch.setattr(runner.vp, "make_env", lambda: FakeEnv())
    monkeypatch.setattr(runner.vp, "build_named_mapping", lambda model, joints: FakeMap())
    monkeypatch.setattr(runner.vp, "reset_open", lambda env: None)
    def fake_prep(R):
        R["last_record"] = {"aggregate": {}}

    monkeypatch.setattr(runner, "prep_phase", fake_prep)
    monkeypatch.setattr(runner, "send_touch", lambda R, agg: None)
    monkeypatch.setattr(runner, "ClockReconciler", FakeReconciler)
    monkeypatch.setattr(runner, "FrameGate", FakeGate)
    monkeypatch.setattr(runner, "SandboxClient", FakeClient)
    monkeypatch.setattr(
        runner, "visual_phase_learned",
        lambda R, inf: (_ for _ in ()).throw(runner.vp.PhysicsStepError(
            runner.vp.StepRecord(
                step_index=1, sim_time_s=0.0, frame_id=1, ctrl=[],
                qpos=[], qvel=[], contacts=[], aggregate={}, warning_delta=[],
                checks={}, ok=False, failure_reasons=["stub"], ctrl_shape=[],
            )
        )),
    )
    monkeypatch.setattr(runner, "cleanup_phase", lambda R: cleanup_called.append(True) or True)

    def fake_teardown(client, bounds=None):
        teardown_calls.append((client, bounds))
        return {"child_pid": getattr(getattr(client, "proc", None), "pid", None),
                "child_returncode": 0, "child_reaped": True, "teardown": {}}

    monkeypatch.setattr(runner, "stop_and_reap_client", fake_teardown)

    out_dir = tmp_path / "unsafe-out"
    summary = runner.run(4, out_dir, contract, inference, target_class=4)
    assert not cleanup_called
    assert len(teardown_calls) == 1
    assert summary["physics_unsafe"] is True


def test_learned_phase_success_terminal_frame_no_action(monkeypatch, tmp_path):
    """Three fresh success frames terminate without requesting a physical action."""
    runner = _import_runner()
    import io
    R = _minimal_R(io.StringIO())
    R["initial_error"] = 20.0
    R["errors"] = [20.0, 2.0, 2.0]
    R["gate"] = type("Gate", (), {
        "check": lambda self, f, e, a: type("D", (), {"ok": True, "reasons": []})()
    })()
    frames = [
        {"path": "c1.png", "png": b"a", "frame_id": 1, "monotonic_time_s": 1.0},
        {"path": "c2.png", "png": b"b", "frame_id": 2, "monotonic_time_s": 1.1},
        {"path": "c3.png", "png": b"c", "frame_id": 3, "monotonic_time_s": 1.2},
    ]
    ests = [
        type("E", (), {"error_px": (2.0, 0.0)})(),
        type("E", (), {"error_px": (1.0, 0.0)})(),
        type("E", (), {"error_px": (1.0, 0.0)})(),
    ]
    it = iter(zip(frames, ests))
    monkeypatch.setattr(runner, "capture_estimate", lambda R_local: (*(next(it)), 0.0))

    class SentinelInf:
        @staticmethod
        def predict(actor_input):
            raise AssertionError("terminal success frame must not call inference.predict")

    monkeypatch.setattr(runner, "step", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("terminal success frame must not call step")))
    monkeypatch.setattr(runner, "send_touch", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("terminal success frame must not call send_touch")))

    stop = runner.visual_phase_learned(R, SentinelInf())
    assert stop == "success"
    assert R["success"] is True
    lines = [json.loads(l) for l in R["actor"].getvalue().splitlines()]
    assert all(l["phase"] == "gate_rejected" for l in lines)
    assert lines[-1]["reasons"] == ["success_terminal_no_action"]
    assert lines[-1]["action_requested"] is False
    assert not any(l["phase"] == "learned_action_intent" for l in lines)


def test_check_budgets_honors_smaller_wall_budget(tmp_path):
    """The operation wall budget is capped by the permit and enforced by check_budgets."""
    runner = _import_runner()
    import time
    out = tmp_path / "budget-out"
    out.mkdir()
    R = {
        "t0": time.monotonic() - 50.0,
        "operation_wall_s": 20.0,
        "env": type("E", (), {"data": type("D", (), {"time": 0.0})()})(),
        "out": out,
        "contract": type("C", (), {"limits": {
            "max_sim_s": 20,
            "dir_cap_bytes": 64 * 1024 * 1024,
        }})(),
    }
    with pytest.raises(runner.ResourceStop, match="operation_wall_budget_exceeded"):
        runner.check_budgets(R, R["contract"].limits)


def test_learned_phase_preserves_rejected_and_initial_below_min_frames(monkeypatch, tmp_path):
    """Gate-rejected and initial-below-min frames are preserved in the actor log."""
    runner = _import_runner()
    import io

    # Gate-rejected frame.
    R = _minimal_R(io.StringIO())
    frame_rej = {"path": "rej.png", "png": b"rej", "frame_id": 7,
                 "monotonic_time_s": 2.0}
    est_rej = type("E", (), {"error_px": (5.0, 0.0)})()
    R["gate"] = type("Gate", (), {
        "check": lambda self, f, e, a: type("D", (), {"ok": False, "reasons": ["bad_age"]})()
    })()
    monkeypatch.setattr(runner, "capture_estimate", lambda R_local: (frame_rej, est_rej, 33.0))
    stop = runner.visual_phase_learned(R, None)
    assert stop.startswith("frame_gate_rejected")
    lines = [json.loads(l) for l in R["actor"].getvalue().splitlines()]
    assert lines[0]["frame_id"] == 7
    assert lines[0]["png_sha256"] == hashlib.sha256(b"rej").hexdigest()
    assert lines[0]["frame_age_ms"] == 33.0
    assert lines[0]["action_requested"] is False

    # Initial-error-below-minimum frame.
    R2 = _minimal_R(io.StringIO())
    frame_low = {"path": "low.png", "png": b"low", "frame_id": 8,
                 "monotonic_time_s": 3.0}
    est_low = type("E", (), {"error_px": (2.0, 0.0)})()
    R2["gate"] = type("Gate", (), {
        "check": lambda self, f, e, a: type("D", (), {"ok": True, "reasons": []})()
    })()
    monkeypatch.setattr(runner, "capture_estimate", lambda R_local: (frame_low, est_low, 12.0))
    stop2 = runner.visual_phase_learned(R2, None)
    assert stop2.startswith("initial_error_below_min")
    lines2 = [json.loads(l) for l in R2["actor"].getvalue().splitlines()]
    assert lines2[0]["frame_id"] == 8
    assert lines2[0]["reasons"] == ["initial_error_below_min"]
    assert lines2[0]["action_requested"] is False
