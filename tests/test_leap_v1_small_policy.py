"""Pure-NumPy tests for the LEAP v1 small-policy baseline.

These tests do not import PyTorch or MuJoCo. They exercise the feature/action
contracts, normalization, label extraction, permit validation, and NumPy
forward pass using synthetic samples.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1.small_policy import (
    ACTION_CAP_RAD,
    APPROVED_ACTOR_FIELDS,
    BLOCK_CAP_RAD,
    HIDDEN_DIM,
    HORIZON,
    INPUT_DIM,
    MOTION4_INDICES,
    MOTOR_JOINTS,
    NUM_CHANNELS,
    OUTPUT_DIM,
    SmallPolicyError,
    SmallPolicyNormalizer,
    compute_channel_metrics,
    compute_loss,
    compute_rmse,
    compute_split_counts,
    decode_controls,
    decode_deltas,
    extract_features,
    extract_label_delta,
    forward_numpy,
    load_sample_index,
    normalize_features,
    sha256_file,
    split_by_target_class,
    validate_permit,
)


def _sample_record(target_class: int = 0, split: str = "train", frame_id: int = 1) -> dict:
    """Build a minimal valid sample record (synthetic, not real physics)."""
    np.random.seed(frame_id + target_class * 1000)
    error_xy = np.array([10.0 - frame_id * 0.1, 0.0], dtype=np.float32)
    q = np.zeros(NUM_CHANNELS, dtype=np.float32)
    q[0] = 0.1 * frame_id
    qvel = np.zeros(NUM_CHANNELS, dtype=np.float32)
    qvel[0] = 0.01
    prev_ctrl = np.zeros(NUM_CHANNELS, dtype=np.float32)
    prev_ctrl[0] = 0.05

    # Build executed_ctrl as a small ramp and derive deltas.
    deltas = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    deltas[:, 0] = 1e-5 * np.sin(np.linspace(0, np.pi, HORIZON))
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    executed[0] = prev_ctrl + deltas[0]
    for t in range(1, HORIZON):
        executed[t] = executed[t - 1] + deltas[t]

    # cumulative offset from previous_ctrl (for validation only).
    cumulative_offset = executed - prev_ctrl

    return {
        "episode_id": target_class,
        "frame_id": frame_id,
        "split": split,
        "target_class": target_class,
        "teacher_only": True,
        "training_eligible": False,
        "pending_coordinator": True,
        "actor_input": {
            "png_ref": f".local/synthetic/seed-{target_class:03d}/cycle-{frame_id:03d}.png",
            "png_sha256": "0" * 64,
            "error_xy_px": error_xy.tolist(),
            "q_rad16": q.tolist(),
            "qvel_rad_s16": qvel.tolist(),
            "previous_ctrl_rad16": prev_ctrl.tolist(),
            "frame_monotonic_time_s": 1000.0 + frame_id,
            "valid": {
                "q_valid": True,
                "qvel_valid": True,
                "previous_ctrl_valid": True,
                "png_valid": True,
                "visual_valid": True,
                "fsr_missing_mask": [False] * 4,
                "motor_current_missing_mask": [False] * 16,
            },
            "fsr_N4": None,
            "motor_current_A16": None,
        },
        "action_window": {
            "consumed_prev_step_index": 100 + frame_id,
            "first_step_index": 101 + frame_id,
            "end_exclusive_step_index": 126 + frame_id,
            "first_sim_time_s": 0.2 + frame_id * 0.05,
            "end_sim_time_s": 0.25 + frame_id * 0.05,
            "dt_s": 0.002,
        },
        "labels": {
            "executed_ctrl_rad": executed.tolist(),
            "delta_from_frame_previous_ctrl_rad": cumulative_offset.tolist(),
            "nominal_endpoint_rad_for_reference_only": executed[-1].tolist(),
        },
    }


def test_import_does_not_require_torch_or_mujoco():
    """Importing small_policy must not pull torch or mujoco into sys.modules."""
    before = set(sys.modules.keys())
    # Re-import to be sure.
    import embodied_trackpad.leap_v1.small_policy as sp

    after = set(sys.modules.keys())
    new = after - before
    assert "torch" not in new
    assert "mujoco" not in new
    assert sp.INPUT_DIM == 50


def test_extract_features_shape_and_order():
    rec = _sample_record()
    feat = extract_features(rec)
    assert feat.shape == (INPUT_DIM,)
    assert feat.dtype == np.float32
    assert np.isfinite(feat).all()
    # Order: error_xy (2), q (16), qvel (16), previous_ctrl (16).
    actor = rec["actor_input"]
    assert np.allclose(feat[:2], actor["error_xy_px"])
    assert np.allclose(feat[2:18], actor["q_rad16"])
    assert np.allclose(feat[18:34], actor["qvel_rad_s16"])
    assert np.allclose(feat[34:50], actor["previous_ctrl_rad16"])


def test_extract_features_rejects_non_finite():
    rec = _sample_record()
    rec["actor_input"]["q_rad16"][0] = float("inf")
    with pytest.raises(SmallPolicyError, match="non-finite"):
        extract_features(rec)


def test_extract_features_rejects_actor_leak_fields():
    # Privileged / metadata fields must be rejected.
    for bad_key in ("force", "jacobian", "reward", "outcome", "seed", "class", "oracle"):
        rec = _sample_record()
        rec["actor_input"][bad_key] = [0.0] * 16
        with pytest.raises(SmallPolicyError, match="rejected fields"):
            extract_features(rec)

    # Unknown fields are also rejected.
    rec = _sample_record()
    rec["actor_input"]["extra_sensor_A"] = [100.0, 200.0]
    with pytest.raises(SmallPolicyError, match="unknown fields"):
        extract_features(rec)


def test_features_unchanged_by_allowed_metadata():
    rec1 = _sample_record()
    rec2 = _sample_record()
    rec2["actor_input"]["png_ref"] = ".local/other.png"
    rec2["actor_input"]["png_sha256"] = "a" * 64
    rec2["actor_input"]["frame_monotonic_time_s"] = 1e9
    rec2["actor_input"]["valid"]["png_valid"] = False
    rec2["actor_input"]["fsr_N4"] = None
    rec2["actor_input"]["motor_current_A16"] = None
    assert np.allclose(extract_features(rec1), extract_features(rec2))


def test_extract_features_allows_null_wrist_metadata():
    rec = _sample_record()
    rec["actor_input"]["wrist"] = None
    rec["actor_input"]["valid"]["wrist_valid"] = False
    feat = extract_features(rec)
    assert feat.shape == (INPUT_DIM,)
    assert np.isfinite(feat).all()


def test_extract_features_rejects_nonnull_wrist_image():
    rec = _sample_record()
    rec["actor_input"]["wrist"] = {"image": "not_metadata"}
    with pytest.raises(SmallPolicyError, match="wrist"):
        extract_features(rec)


def test_features_unchanged_by_top_level_metadata():
    rec1 = _sample_record()
    rec2 = _sample_record()
    rec2["training_eligible"] = True
    rec2["teacher_only"] = False
    rec2["pending_coordinator"] = False
    rec2["action_window"]["first_sim_time_s"] = 1e9
    assert np.allclose(extract_features(rec1), extract_features(rec2))


def test_extract_label_delta_shape_and_first_row_relative_to_previous_ctrl():
    rec = _sample_record()
    delta = extract_label_delta(rec)
    assert delta.shape == (HORIZON, NUM_CHANNELS)
    prev = np.asarray(rec["actor_input"]["previous_ctrl_rad16"], dtype=np.float32)
    executed = np.asarray(rec["labels"]["executed_ctrl_rad"], dtype=np.float32)
    assert np.allclose(delta[0], executed[0] - prev)
    assert np.allclose(delta[1:], executed[1:] - executed[:-1])


def test_extract_label_delta_derives_when_key_missing():
    rec = _sample_record()
    del rec["labels"]["delta_from_frame_previous_ctrl_rad"]
    delta = extract_label_delta(rec)
    prev = np.asarray(rec["actor_input"]["previous_ctrl_rad16"], dtype=np.float32)
    executed = np.asarray(rec["labels"]["executed_ctrl_rad"], dtype=np.float32)
    assert np.allclose(delta[0], executed[0] - prev)


def test_extract_label_delta_rejects_inconsistent_delta():
    rec = _sample_record()
    rec["labels"]["delta_from_frame_previous_ctrl_rad"][0][0] += 1.0
    with pytest.raises(SmallPolicyError, match="inconsistent"):
        extract_label_delta(rec)


def test_label_is_per_step_delta_not_nominal_endpoint():
    rec = _sample_record()
    delta = extract_label_delta(rec)
    nominal = np.asarray(rec["labels"]["nominal_endpoint_rad_for_reference_only"], dtype=np.float32)
    # The full delta block is not the endpoint vector.
    assert delta.shape != nominal.shape
    # The cumulative sum of deltas plus previous_ctrl equals the endpoint.
    prev = np.asarray(rec["actor_input"]["previous_ctrl_rad16"], dtype=np.float32)
    assert np.allclose(prev + np.sum(delta, axis=0), nominal)


def test_action_cap_and_cumulative_reconstruction():
    logits = np.linspace(-10, 10, OUTPUT_DIM).astype(np.float32)
    deltas = decode_deltas(logits)
    assert deltas.shape == (HORIZON, NUM_CHANNELS)
    assert np.all(np.abs(deltas) <= ACTION_CAP_RAD + 1e-8)
    assert not np.allclose(deltas, 0.0)

    prev = np.zeros(NUM_CHANNELS, dtype=np.float32)
    controls = decode_controls(deltas, prev)
    assert controls.shape == (HORIZON, NUM_CHANNELS)
    assert np.allclose(controls, np.cumsum(deltas, axis=0))
    # Cumulative displacement over a block is bounded by horizon * action_cap.
    assert np.all(np.abs(controls[-1] - prev) <= HORIZON * ACTION_CAP_RAD + 1e-8)


def test_tanh_saturates_action_cap():
    large_logits = np.full(OUTPUT_DIM, 100.0, dtype=np.float32)
    deltas = decode_deltas(large_logits)
    assert np.allclose(deltas, ACTION_CAP_RAD)


def test_normalizer_fit_on_train_only():
    rng = np.random.RandomState(42)
    train = rng.randn(100, INPUT_DIM).astype(np.float32)
    val = rng.randn(20, INPUT_DIM).astype(np.float32) + 50.0  # very different distribution
    test = rng.randn(20, INPUT_DIM).astype(np.float32) - 50.0

    norm = SmallPolicyNormalizer().fit(train)
    train_norm = norm.transform(train)
    val_norm = norm.transform(val)
    test_norm = norm.transform(test)

    # Train should be roughly standard normal.
    assert np.allclose(train_norm.mean(axis=0), 0.0, atol=1e-1)
    assert np.allclose(train_norm.std(axis=0), 1.0, atol=1e-1)

    # Fitting stats must come only from train; val/test std should not be 1.
    assert not np.allclose(val_norm.std(axis=0), 1.0, atol=1e-1)
    assert not np.allclose(test_norm.std(axis=0), 1.0, atol=1e-1)

    # Std floor is enforced.
    assert (norm.feature_std >= 1e-3 - 1e-9).all()


def test_normalizer_state_save_load():
    rng = np.random.RandomState(7)
    train = rng.randn(50, INPUT_DIM).astype(np.float32)
    norm = SmallPolicyNormalizer().fit(train)
    state = norm.state_dict()

    norm2 = SmallPolicyNormalizer()
    norm2.load_state_dict(state)
    assert np.allclose(norm2.feature_mean, norm.feature_mean)
    assert np.allclose(norm2.feature_std, norm.feature_std)


def test_target_class_grouping():
    split_for_tc = {
        0: "train", 1: "validation", 2: "train", 3: "train", 4: "test",
        5: "train", 6: "train", 7: "validation", 8: "train", 9: "train",
        10: "test", 11: "train",
    }
    records = []
    for tc in range(12):
        records.extend([_sample_record(target_class=tc, split=split_for_tc[tc], frame_id=i) for i in range(5)])
    splits = split_by_target_class(records)
    counts = compute_split_counts(records)
    assert counts["train"] == 8 * 5
    assert counts["validation"] == 2 * 5
    assert counts["test"] == 2 * 5
    # No target class leakage across splits.
    train_classes = {r["target_class"] for r in splits["train"]}
    val_classes = {r["target_class"] for r in splits["validation"]}
    test_classes = {r["target_class"] for r in splits["test"]}
    assert train_classes == {0, 2, 3, 5, 6, 8, 9, 11}
    assert val_classes == {1, 7}
    assert test_classes == {4, 10}
    assert not (train_classes & val_classes)
    assert not (train_classes & test_classes)
    assert not (val_classes & test_classes)
    # Every declared split matches its target class.
    for rec in records:
        assert rec["split"] == split_for_tc[rec["target_class"]]


def test_target_class_outside_range_rejected():
    records = [_sample_record(target_class=12)]
    with pytest.raises(SmallPolicyError, match="target_class 12"):
        split_by_target_class(records)


def test_compute_loss_normalized():
    pred = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    target = np.full((HORIZON, NUM_CHANNELS), ACTION_CAP_RAD, dtype=np.float32)
    prev = np.zeros(NUM_CHANNELS, dtype=np.float32)
    metrics = compute_loss(pred, target, prev)
    # Increment MSE normalized by (2e-4)^2 with perfect error 2e-4 -> 1.0.
    assert np.isclose(metrics["increment_mse_normalized"], 1.0, rtol=1e-5)
    # Cumulative error at step t is (t+1)*2e-4; normalized MSE is the mean of
    # those squared errors divided by (0.005)^2 = (25*2e-4)^2.
    expected_cum = np.mean(np.arange(1, HORIZON + 1, dtype=np.float32) ** 2) / (HORIZON**2)
    assert np.isclose(metrics["cumulative_mse_normalized"], expected_cum, rtol=1e-5)
    assert np.isclose(metrics["loss"], 1.0 + 0.25 * expected_cum, rtol=1e-4)


def test_compute_channel_metrics():
    pred = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    target = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    target[:, 0] = 0.1
    metrics = compute_channel_metrics(pred, target)
    assert metrics["rmse_per_channel_rad"].shape == (NUM_CHANNELS,)
    assert np.isclose(metrics["rmse_per_channel_rad"][0], 0.1)
    assert np.isclose(metrics["rmse_all_channels_rad"], 0.1 / np.sqrt(NUM_CHANNELS), rtol=1e-5)
    # Motion4 is the four index joints [0,1,2,3]; only channel 0 differs here.
    assert np.isclose(metrics["rmse_motion4_rad"], 0.1 / 2.0, rtol=1e-5)


def _identity_npz() -> dict[str, np.ndarray]:
    """Build an NPZ that maps feature 0 unchanged through the network."""
    w1 = np.zeros((HIDDEN_DIM, INPUT_DIM), dtype=np.float32)
    w1[:INPUT_DIM, :] = np.eye(INPUT_DIM, dtype=np.float32)
    return {
        "W1": w1,
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": np.eye(HIDDEN_DIM, dtype=np.float32),
        "b2": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W3": np.zeros((OUTPUT_DIM, HIDDEN_DIM), dtype=np.float32),
        "b3": np.zeros(OUTPUT_DIM, dtype=np.float32),
        "feature_mean": np.zeros(INPUT_DIM, dtype=np.float32),
        "feature_std": np.ones(INPUT_DIM, dtype=np.float32),
    }


def test_forward_numpy_with_synthetic_weights():
    """Known synthetic weights produce a deterministic forward pass."""
    npz = _identity_npz()
    features = np.zeros(INPUT_DIM, dtype=np.float32)
    features[0] = 1.0
    deltas = forward_numpy(npz, features)
    assert deltas.shape == (HORIZON, NUM_CHANNELS)
    assert np.allclose(deltas, 0.0)

    # Non-zero bias on final layer maps directly to tanh-scaled delta.
    npz["b3"] = np.full(OUTPUT_DIM, np.arctanh(0.5), dtype=np.float32)
    deltas = forward_numpy(npz, features)
    assert np.allclose(deltas, ACTION_CAP_RAD * 0.5, rtol=1e-5)

    # Normalization is applied exactly once.
    npz["feature_mean"] = np.full(INPUT_DIM, 1.0, dtype=np.float32)
    npz["feature_std"] = np.full(INPUT_DIM, 2.0, dtype=np.float32)
    npz["b3"] = np.zeros(OUTPUT_DIM, dtype=np.float32)
    features = np.full(INPUT_DIM, 1.0, dtype=np.float32)
    # (1-1)/2 = 0 everywhere, so output should still be zero.
    assert np.allclose(forward_numpy(npz, features), 0.0)


def test_forward_numpy_rejects_bad_shapes():
    npz = _identity_npz()
    with pytest.raises(SmallPolicyError, match="features"):
        forward_numpy(npz, np.zeros(INPUT_DIM + 1, dtype=np.float32))

    npz_bad = _identity_npz()
    npz_bad["W2"] = np.zeros((HIDDEN_DIM - 1, HIDDEN_DIM), dtype=np.float32)
    with pytest.raises(SmallPolicyError, match="W2"):
        forward_numpy(npz_bad, np.zeros(INPUT_DIM, dtype=np.float32))


def test_npz_save_load_allow_pickle_false(tmp_path: Path):
    npz = _identity_npz()
    path = tmp_path / "model.npz"
    np.savez(path, **npz)
    loaded = np.load(path, allow_pickle=False)
    for key in npz:
        assert loaded[key].dtype != object
    features = np.zeros(INPUT_DIM, dtype=np.float32)
    features[0] = 1.0
    assert np.allclose(forward_numpy(loaded, features), 0.0)


def test_validate_permit_accepts_valid():
    permit = {
        "status": "accepted_for_small_policy_smoke",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1, "max_wall_s": 1800},
        "approved_actor_fields": list(APPROVED_ACTOR_FIELDS),
    }
    contract = validate_permit(
        permit,
        sample_index_sha256="a" * 64,
        source_commit="abc123",
        dataset_schema="v1-small-policy-smoke-v1",
        split_counts={"train": 453, "validation": 141, "test": 80},
    )
    assert contract.allow_optimizer


def test_validate_permit_rejects_missing_keys():
    permit = {"status": "accepted_for_small_policy_smoke"}
    with pytest.raises(SmallPolicyError, match="missing keys"):
        validate_permit(permit, sample_index_sha256="a" * 64)


def test_validate_permit_rejects_bad_status():
    permit = {
        "status": "pending",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1, "max_wall_s": 1800},
        "approved_actor_fields": list(APPROVED_ACTOR_FIELDS),
    }
    with pytest.raises(SmallPolicyError, match="accepted_for_small_policy_smoke"):
        validate_permit(permit, sample_index_sha256="a" * 64)


def test_validate_permit_rejects_hash_mismatch():
    permit = {
        "status": "accepted_for_small_policy_smoke",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1, "max_wall_s": 1800},
        "approved_actor_fields": list(APPROVED_ACTOR_FIELDS),
    }
    with pytest.raises(SmallPolicyError, match="sample_index_sha256 mismatch"):
        validate_permit(permit, sample_index_sha256="b" * 64)


def test_validate_permit_rejects_split_count_mismatch():
    permit = {
        "status": "accepted_for_small_policy_smoke",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1, "max_wall_s": 1800},
        "approved_actor_fields": list(APPROVED_ACTOR_FIELDS),
    }
    with pytest.raises(SmallPolicyError, match="split_counts"):
        validate_permit(
            permit,
            sample_index_sha256="a" * 64,
            split_counts={"train": 452, "validation": 141, "test": 80},
        )


def test_validate_permit_rejects_missing_limits():
    permit = {
        "status": "accepted_for_small_policy_smoke",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1},
        "approved_actor_fields": list(APPROVED_ACTOR_FIELDS),
    }
    with pytest.raises(SmallPolicyError, match="max_wall_s"):
        validate_permit(permit, sample_index_sha256="a" * 64)


def test_validate_permit_rejects_wrong_actor_fields():
    permit = {
        "status": "accepted_for_small_policy_smoke",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1, "max_wall_s": 1800},
        "approved_actor_fields": ["error_xy_px", "q_rad16", "qvel_rad_s16"],
    }
    with pytest.raises(SmallPolicyError, match="approved_actor_fields"):
        validate_permit(permit, sample_index_sha256="a" * 64)


def test_load_sample_index(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    split_for_tc = {0: "train", 1: "validation", 4: "test"}
    records = [_sample_record(target_class=tc, split=split_for_tc[tc], frame_id=i) for tc in [0, 1, 4] for i in range(2)]
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    loaded = load_sample_index(path)
    assert len(loaded) == 6
    # Duplicate id rejected.
    bad_path = tmp_path / "bad.jsonl"
    with open(bad_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_sample_record(frame_id=1)) + "\n")
        f.write(json.dumps(_sample_record(frame_id=1)) + "\n")
    with pytest.raises(SmallPolicyError, match="duplicate"):
        load_sample_index(bad_path)


def test_motor_joint_order_matches_model_py():
    """Hard-coded order in this module must match model.py::MOTOR_JOINTS."""
    expected = [
        "if_mcp",
        "if_rot",
        "if_pip",
        "if_dip",
        "mf_mcp",
        "mf_rot",
        "mf_pip",
        "mf_dip",
        "rf_mcp",
        "rf_rot",
        "rf_pip",
        "rf_dip",
        "th_cmc",
        "th_axl",
        "th_mcp",
        "th_ipl",
    ]
    assert MOTOR_JOINTS == expected
    assert MOTION4_INDICES == [0, 1, 2, 3]


def test_extract_label_delta_nonuniform_three_step_with_hold():
    """Per-step increments are derived even when only 3 of 25 steps move."""
    prev = np.zeros(NUM_CHANNELS, dtype=np.float64)
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float64)
    # Three non-uniform moves on channel 0, rest hold.
    executed[0] = prev.copy()
    executed[1, 0] = 1e-5
    executed[2, 0] = 1e-5 + 5e-5
    executed[3, 0] = 1e-5 + 5e-5 - 2e-5
    for t in range(4, HORIZON):
        executed[t] = executed[t - 1]

    rec = {
        "episode_id": 0,
        "frame_id": 1,
        "split": "train",
        "target_class": 0,
        "actor_input": {
            "error_xy_px": [0.0, 0.0],
            "q_rad16": [0.0] * NUM_CHANNELS,
            "qvel_rad_s16": [0.0] * NUM_CHANNELS,
            "previous_ctrl_rad16": prev.astype(np.float32).tolist(),
        },
        "action_window": {
            "consumed_prev_step_index": 0,
            "first_step_index": 1,
            "end_exclusive_step_index": 26,
            "first_sim_time_s": 0.0,
            "end_sim_time_s": 0.05,
            "dt_s": 0.002,
        },
        "labels": {
            "executed_ctrl_rad": executed.tolist(),
            "delta_from_frame_previous_ctrl_rad": (executed - prev).tolist(),
        },
    }
    delta = extract_label_delta(rec)
    assert delta.shape == (HORIZON, NUM_CHANNELS)
    assert np.allclose(delta[0], 0.0)
    assert np.isclose(delta[1, 0], 1e-5)
    assert np.isclose(delta[2, 0], 5e-5)
    assert np.isclose(delta[3, 0], -2e-5)
    assert np.allclose(delta[4:, 0], 0.0)
    # All per-step deltas respect the action cap.
    assert np.all(np.abs(delta) <= ACTION_CAP_RAD + 1e-12)


def test_validate_permit_rejects_string_false_and_bad_limits():
    base = {
        "status": "accepted_for_small_policy_smoke",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1, "max_wall_s": 1800},
        "approved_actor_fields": list(APPROVED_ACTOR_FIELDS),
    }
    # String "false" must fail because it is not boolean True.
    bad = dict(base, allow_optimizer="false")
    with pytest.raises(SmallPolicyError, match="boolean True"):
        validate_permit(bad, sample_index_sha256="a" * 64)

    # Limits above baseline caps fail.
    for key, val in [("max_updates", 2001), ("max_gpu", 2), ("max_wall_s", 1801)]:
        bad = dict(base)
        bad["limits"] = dict(base["limits"], **{key: val})
        with pytest.raises(SmallPolicyError, match="exceeds baseline cap"):
            validate_permit(bad, sample_index_sha256="a" * 64)

    # Non-positive limit fails.
    bad = dict(base)
    bad["limits"] = dict(base["limits"], max_updates=0)
    with pytest.raises(SmallPolicyError, match="positive integer"):
        validate_permit(bad, sample_index_sha256="a" * 64)


def test_real_sample_index_compatibility():
    """READ-ONLY parser-compatibility check for an external pilot index.

    This test is independent of index acceptance or training eligibility.
    It only verifies that the configured real-data index parses correctly
    and records structural invariants; separate approval is required for use.
    """
    env_path = os.environ.get("TRACKPAD_PILOT_INDEX")
    if not env_path:
        pytest.skip("TRACKPAD_PILOT_INDEX not set; optional real-data check skipped")
    index_path = Path(env_path)
    if not index_path.exists():
        pytest.skip("TRACKPAD_PILOT_INDEX file does not exist")

    before_hash = sha256_file(index_path)
    records = load_sample_index(index_path)
    after_hash = sha256_file(index_path)
    assert before_hash == after_hash, "index file was modified during read-only parsing"

    counts = compute_split_counts(records)
    assert counts == {"train": 453, "validation": 141, "test": 80}

    features = np.stack([extract_features(r) for r in records])
    labels = np.stack([extract_label_delta(r) for r in records])
    assert features.shape == (674, INPUT_DIM)
    assert labels.shape == (674, HORIZON, NUM_CHANNELS)
    assert np.isfinite(features).all()
    assert np.isfinite(labels).all()
    # Per-step increments must respect the action cap before float32 conversion.
    assert np.all(np.abs(labels) <= ACTION_CAP_RAD + 1e-12)

    # Spot-check that cumulative-offset field is consistent when present.
    for r in records[:10]:
        prev = np.asarray(r["actor_input"]["previous_ctrl_rad16"], dtype=np.float64)
        executed = np.asarray(r["labels"]["executed_ctrl_rad"], dtype=np.float64)
        if "delta_from_frame_previous_ctrl_rad" in r["labels"]:
            cumulative = np.asarray(r["labels"]["delta_from_frame_previous_ctrl_rad"], dtype=np.float64)
            assert np.allclose(cumulative, executed - prev, atol=1e-9, rtol=1e-9)
