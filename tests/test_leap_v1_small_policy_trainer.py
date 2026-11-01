"""Pure-NumPy/AST/mock tests for the LEAP v1 small-policy trainer.

These tests do not import or run PyTorch or MuJoCo. They verify the trainer's
static contracts, GPU guard, source manifest, and NumPy-only helper parity.
The GPU smoke entrypoint is present but skipped by default.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from embodied_trackpad.leap_v1.small_policy import (
    ACTION_CAP_RAD,
    APPROVED_ACTOR_FIELDS,
    HIDDEN_DIM,
    HORIZON,
    INPUT_DIM,
    MOTION4_INDICES,
    NUM_CHANNELS,
    OUTPUT_DIM,
    TEST_TARGET_CLASSES,
    SmallPolicyError,
    SmallPolicyNormalizer,
    decode_controls_batched,
    forward_numpy,
    normalize_features,
)

# The trainer module must not pull torch/mujoco at import time.
import tools.train_leap_v1_small_policy as trainer


def test_train_module_imports_no_torch_or_mujoco():
    """Importing the trainer in a fresh interpreter must not pull torch or mujoco."""
    repo_root = Path(__file__).resolve().parents[1]
    child_script = "import sys; import tools.train_leap_v1_small_policy as _t; forbidden = {'torch', 'mujoco'}; found = [m for m in forbidden if m in sys.modules]; print('ok' if not found else 'found:' + ','.join(found)); sys.exit(1 if found else 0)"
    env = os.environ.copy()
    pythonpath_parts = [str(repo_root), str(repo_root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath_parts = env["PYTHONPATH"].split(os.pathsep) + pythonpath_parts
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    result = subprocess.run(
        [sys.executable, "-c", child_script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "torch" not in result.stdout
    assert "mujoco" not in result.stdout


def test_target_classes_import_available():
    """Held-out target classes are exposed by the small_policy module."""
    assert TEST_TARGET_CLASSES == frozenset({4, 10})


def _identity_npz() -> dict[str, np.ndarray]:
    """Build an NPZ that maps the first feature through the network unchanged.

    W3 is nonzero so that a double-normalization mismatch actually diverges.
    """
    w1 = np.zeros((HIDDEN_DIM, INPUT_DIM), dtype=np.float32)
    w1[:INPUT_DIM, :] = np.eye(INPUT_DIM, dtype=np.float32)
    w3 = np.zeros((OUTPUT_DIM, HIDDEN_DIM), dtype=np.float32)
    m = min(OUTPUT_DIM, HIDDEN_DIM)
    w3[:m, :m] = np.eye(m, dtype=np.float32)
    return {
        "W1": w1,
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": np.eye(HIDDEN_DIM, dtype=np.float32),
        "b2": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W3": w3,
        "b3": np.zeros(OUTPUT_DIM, dtype=np.float32),
        "feature_mean": np.zeros(INPUT_DIM, dtype=np.float32),
        "feature_std": np.ones(INPUT_DIM, dtype=np.float32),
    }


def _torch_forward_from_npz(npz: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """Simulate a Torch model that receives already-normalized features.

    This skips the normalizer so it can be used as ``model_forward_torch_batched``
    when verifying that ``verify_numpy_torch_match`` feeds raw features to the
    NumPy path and normalized features to the Torch path.
    """
    mock_npz = dict(npz)
    mock_npz["feature_mean"] = np.zeros(INPUT_DIM, dtype=np.float32)
    mock_npz["feature_std"] = np.ones(INPUT_DIM, dtype=np.float32)
    return forward_numpy(mock_npz, features)


def test_forward_numpy_batch_matches_single():
    """Batched forward_numpy must equal looping over individual samples."""
    npz = _identity_npz()
    rng = np.random.RandomState(7)
    features = rng.randn(17, INPUT_DIM).astype(np.float32)

    batched = forward_numpy(npz, features)
    single = np.stack([forward_numpy(npz, f) for f in features])

    assert batched.shape == (17, HORIZON, NUM_CHANNELS)
    assert np.allclose(batched, single)


def test_forward_numpy_rejects_negative_std():
    """A malformed negative stored std must fail closed."""
    npz = _identity_npz()
    npz["feature_std"] = np.full(INPUT_DIM, -1.0, dtype=np.float32)
    with pytest.raises(SmallPolicyError, match="feature_std"):
        forward_numpy(npz, np.zeros(INPUT_DIM, dtype=np.float32))


def test_forward_numpy_rejects_sub_floor_std():
    """A stored std below the 1e-3 floor must also be rejected."""
    npz = _identity_npz()
    npz["feature_std"] = np.full(INPUT_DIM, 1e-4, dtype=np.float32)
    with pytest.raises(SmallPolicyError, match="feature_std"):
        forward_numpy(npz, np.zeros(INPUT_DIM, dtype=np.float32))


def test_decode_controls_batched_shape_and_cumsum():
    """Batched decode must reconstruct controls with cumsum on the time axis."""
    rng = np.random.RandomState(3)
    deltas = (ACTION_CAP_RAD * (2 * rng.rand(5, HORIZON, NUM_CHANNELS) - 1)).astype(np.float32)
    prev = rng.randn(5, NUM_CHANNELS).astype(np.float32)
    ctrl = decode_controls_batched(deltas, prev)
    assert ctrl.shape == (5, HORIZON, NUM_CHANNELS)
    expected = prev[:, None, :] + np.cumsum(deltas, axis=1)
    assert np.allclose(ctrl, expected)


def test_stable_silu_no_overflow():
    """SiLU must stay finite for very large positive and negative inputs."""
    from embodied_trackpad.leap_v1 import small_policy as sp

    x = np.array([1000.0, -1000.0, 0.0], dtype=np.float32)
    y = sp._silu(x)
    assert np.isfinite(y).all()
    # x -> +inf should approach x; x -> -inf should approach 0.
    assert np.isclose(y[0], 1000.0, rtol=1e-5)
    assert np.isclose(y[1], 0.0, atol=1e-6)
    assert np.isclose(y[2], 0.0, atol=1e-6)


def test_contract_template_is_proposal():
    """The bundled contract template must be a proposed, optimizer-disabled baseline."""
    path = Path("env/leap-v1-small-policy-contract.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "proposed"
    assert data["allow_optimizer"] is False
    assert data["model"]["motion4_indices"] == [0, 1, 2, 3]
    assert "validation" in data["target_class_split"]
    assert data["target_class_split"]["validation"] == [1, 7]
    expected = data.get("expected_split_counts") or data.get("split_counts")
    assert expected == {"train": 453, "validation": 141, "test": 80}


def test_source_manifest_hashes_expected_files():
    """Source manifest must cover the trainer, helper, and contract."""
    manifest = trainer._build_source_manifest()
    expected = {
        "tools/train_leap_v1_small_policy.py",
        "src/embodied_trackpad/leap_v1/small_policy.py",
        "env/leap-v1-small-policy-contract.json",
    }
    assert set(manifest.keys()) == expected
    for h in manifest.values():
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


def _trainer_ast() -> ast.AST:
    source = Path("tools/train_leap_v1_small_policy.py").read_text(encoding="utf-8")
    return ast.parse(source)


def _function_body(func_name: str) -> list[ast.AST]:
    tree = _trainer_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node.body
    raise AssertionError(f"function {func_name} not found in trainer")


def test_model_architecture_via_ast():
    """Static check that the MLP is 50->128->128->400 with SiLU."""
    body = _function_body("build_model")
    source = ast.unparse(body)
    assert "nn.Linear(INPUT_DIM, HIDDEN_DIM)" in source
    assert "nn.Linear(HIDDEN_DIM, HIDDEN_DIM)" in source
    assert "nn.Linear(HIDDEN_DIM, OUTPUT_DIM)" in source
    assert "torch.nn.functional.silu" in source


def test_compute_loss_torch_is_batched_via_ast():
    """Static check for batched loss: reshape Bx25x16 and cumsum on dim 1."""
    body = _function_body("compute_loss_torch")
    source = ast.unparse(body)
    assert "view(-1, HORIZON, NUM_CHANNELS)" in source
    assert "cumsum(pred_deltas, dim=1)" in source
    assert "previous_ctrl[:, None, :]" in source


def test_train_loop_uses_validation_split_via_ast():
    """Static check that validation uses the canonical 'validation' split key."""
    tree = _trainer_ast()
    source = ast.unparse(tree)
    assert "features['validation']" in source
    assert "labels['validation']" in source
    assert "previous_ctrls['validation']" in source
    assert "features['val']" not in source


def test_no_cpu_fallback_in_train():
    """Static check that train() never constructs a CPU device as a fallback."""
    tree = _trainer_ast()
    source = ast.unparse(tree)
    assert '"cpu"' not in source
    assert "cuda:0" in source


def _mock_torch(cuda_available: bool = True, device_count: int = 1, device_name: str = "NVIDIA H800") -> mock.MagicMock:
    m = mock.MagicMock()
    m.cuda.is_available.return_value = cuda_available
    m.cuda.device_count.return_value = device_count
    m.cuda.get_device_name.return_value = device_name
    return m


def test_gpu_guard_requires_slurm_and_h800():
    """GPU guard must fail closed without exactly one H800 under SLURM."""
    mock_torch = _mock_torch()
    with mock.patch.dict(sys.modules, {"torch": mock_torch}):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345", "CUDA_VISIBLE_DEVICES": "0"}):
            assert trainer._require_gpu_and_slurm() == ("12345", "NVIDIA H800")

        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            with pytest.raises(SmallPolicyError, match="SLURM_JOB_ID"):
                trainer._require_gpu_and_slurm()

        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345", "CUDA_VISIBLE_DEVICES": "0,1"}, clear=True):
            with pytest.raises(SmallPolicyError, match="exactly one GPU"):
                trainer._require_gpu_and_slurm()

        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345", "CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            mock_torch.cuda.device_count.return_value = 2
            with pytest.raises(SmallPolicyError, match="exactly 1 CUDA"):
                trainer._require_gpu_and_slurm()

        mock_torch.cuda.device_count.return_value = 1
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "12345", "CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            mock_torch.cuda.get_device_name.return_value = "NVIDIA A100"
            with pytest.raises(SmallPolicyError, match="H800"):
                trainer._require_gpu_and_slurm()


def test_state_diff_helper_catches_shape_mismatch():
    """The resume-comparison helper must detect shape/key mismatches."""
    a = {"x": np.ones((3, 4)), "y": {"z": np.ones(2)}}
    b_ok = {"x": np.ones((3, 4)), "y": {"z": np.ones(2)}}
    b_bad_shape = {"x": np.ones((3, 5)), "y": {"z": np.ones(2)}}
    b_bad_key = {"x": np.ones((3, 4))}
    assert trainer._max_state_diff(a, b_ok) == 0.0
    assert trainer._max_state_diff(a, b_bad_shape) == float("inf")
    assert trainer._max_state_diff(a, b_bad_key) == float("inf")


@pytest.mark.skipif(
    os.environ.get("SLURM_JOB_ID") is None
    or "H800" not in os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    reason="GPU smoke requires a single H800 Slurm allocation; not run locally",
)
def test_gpu_smoke_entrypoint_placeholder():
    """Placeholder for the coordinator-authorized H800 smoke test.

    This test is skipped unless the local process is already inside the approved
    H800 allocation. It exists only as the entrypoint for the future GPU run.
    """
    pytest.skip("GPU smoke not executed in this local run")


def _make_namespace(**kwargs: Any) -> argparse.Namespace:
    defaults = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "grad_clip": 1.0,
        "seed": 17,
        "max_updates": 2000,
        "val_every": 100,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _valid_manifest_and_permit() -> tuple[dict[str, Any], dict[str, Any]]:
    files = trainer._build_source_manifest()
    permit: dict[str, Any] = {
        "status": "accepted_for_small_policy_smoke",
        "allow_optimizer": True,
        "sample_index_sha256": "a" * 64,
        "source_commit": "abc123def4567890abcd1234567890abcdef1234",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "split_counts": {"train": 453, "validation": 141, "test": 80},
        "limits": {"max_updates": 2000, "max_gpu": 1, "max_wall_s": 1800},
        "approved_actor_fields": list(APPROVED_ACTOR_FIELDS),
        "source_hashes": dict(files),
    }
    manifest = {
        "source_commit": "abc123def4567890abcd1234567890abcdef1234",
        "files": dict(files),
    }
    return manifest, permit


def test_validate_frozen_hyperparams_accepts_defaults():
    permit = {"limits": {"max_updates": 2000}}
    hp = trainer.validate_frozen_hyperparams(_make_namespace(), permit)
    assert hp["lr"] == 1e-3
    assert hp["batch_size"] == 64


@pytest.mark.parametrize(
    "field,bad,msg",
    [
        ("lr", 2e-3, "lr must be 0.001"),
        ("weight_decay", 0.0, "weight_decay must be 0.0001"),
        ("batch_size", 32, "batch_size must be 64"),
        ("grad_clip", 0.5, "grad_clip must be 1.0"),
        ("seed", 42, "seed must be 17"),
        ("val_every", 50, "val_every must be 100"),
        ("max_updates", 2001, "max_updates must be <= 2000"),
    ],
)
def test_validate_frozen_hyperparams_rejects_deviation(field, bad, msg):
    permit = {"limits": {"max_updates": 2000}}
    with pytest.raises(SmallPolicyError, match=msg):
        trainer.validate_frozen_hyperparams(_make_namespace(**{field: bad}), permit)


def test_validate_frozen_hyperparams_respects_permit_max_updates():
    permit = {"limits": {"max_updates": 1000}}
    with pytest.raises(SmallPolicyError, match="exceeds permit limit"):
        trainer.validate_frozen_hyperparams(_make_namespace(max_updates=1500), permit)


def test_validate_source_manifest_accepts_valid():
    manifest, permit = _valid_manifest_and_permit()
    trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT, git_commit=permit["source_commit"])


def test_validate_source_manifest_rejects_empty_source_commit():
    manifest, permit = _valid_manifest_and_permit()
    manifest["source_commit"] = ""
    with pytest.raises(SmallPolicyError, match="source manifest source_commit is empty"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT)


def test_validate_source_manifest_rejects_commit_mismatch():
    manifest, permit = _valid_manifest_and_permit()
    manifest["source_commit"] = "deadbeef" * 5
    with pytest.raises(SmallPolicyError, match="source_commit mismatch"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT)


def test_validate_source_manifest_rejects_git_head_mismatch():
    manifest, permit = _valid_manifest_and_permit()
    with pytest.raises(SmallPolicyError, match="git HEAD"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT, git_commit="badc0ffee" * 4)


def test_validate_source_manifest_rejects_hash_mismatch():
    manifest, permit = _valid_manifest_and_permit()
    manifest["files"]["tools/train_leap_v1_small_policy.py"] = "0" * 64
    with pytest.raises(SmallPolicyError, match="manifest hash mismatch"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT)


def test_validate_source_manifest_rejects_permit_hash_mismatch():
    manifest, permit = _valid_manifest_and_permit()
    permit["source_hashes"]["tools/train_leap_v1_small_policy.py"] = "0" * 64
    with pytest.raises(SmallPolicyError, match="permit source_hashes mismatch"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT)


def test_validate_source_manifest_rejects_absolute_path():
    manifest, permit = _valid_manifest_and_permit()
    h = manifest["files"].pop("tools/train_leap_v1_small_policy.py")
    manifest["files"][str((trainer.REPO_ROOT / "tools/train_leap_v1_small_policy.py").resolve())] = h
    with pytest.raises(SmallPolicyError, match="path is not safe"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT)


def test_validate_source_manifest_rejects_path_escape():
    manifest, permit = _valid_manifest_and_permit()
    h = manifest["files"].pop("tools/train_leap_v1_small_policy.py")
    manifest["files"]["../tools/train_leap_v1_small_policy.py"] = h
    with pytest.raises(SmallPolicyError, match="path is not safe"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT)


def test_validate_source_manifest_rejects_wrong_file_set():
    manifest, permit = _valid_manifest_and_permit()
    manifest["files"]["extra.txt"] = "0" * 64
    del manifest["files"]["env/leap-v1-small-policy-contract.json"]
    with pytest.raises(SmallPolicyError, match="exactly"):
        trainer.validate_source_manifest(manifest, permit, trainer.REPO_ROOT)


def test_verify_numpy_torch_match_parity_with_nonstandard_normalizer(tmp_path, monkeypatch):
    """Torch (normalized once) and NumPy (raw once) must agree within 1e-8 rad.

    Nonzero mean and nonunit std ensure double-normalization would diverge.
    """
    npz = _identity_npz()
    npz["feature_mean"] = np.linspace(-1.0, 1.0, INPUT_DIM, dtype=np.float32)
    npz["feature_std"] = np.linspace(0.5, 2.0, INPUT_DIM, dtype=np.float32)
    npz_path = tmp_path / "parity.npz"
    np.savez(npz_path, **npz)

    rng = np.random.RandomState(11)
    raw_features = rng.randn(7, INPUT_DIM).astype(np.float32)
    normalized_features = normalize_features(raw_features, npz["feature_mean"], npz["feature_std"])
    previous_ctrls = rng.randn(7, NUM_CHANNELS).astype(np.float32)

    def mock_forward(model, features):
        return _torch_forward_from_npz(npz, features)

    monkeypatch.setattr(trainer, "model_forward_torch_batched", mock_forward)

    result = trainer.verify_numpy_torch_match(
        None, npz_path, raw_features, normalized_features, previous_ctrls
    )
    assert result["within_tolerance"]
    assert result["max_abs_delta_rad"] <= 1e-8
    assert result["max_abs_control_rad"] <= 1e-8


def test_verify_numpy_torch_match_detects_double_normalization(tmp_path, monkeypatch):
    """Regression: feeding normalized features to the NumPy path must fail."""
    npz = _identity_npz()
    npz["feature_mean"] = np.linspace(-1.0, 1.0, INPUT_DIM, dtype=np.float32)
    npz["feature_std"] = np.linspace(0.5, 2.0, INPUT_DIM, dtype=np.float32)
    npz_path = tmp_path / "parity.npz"
    np.savez(npz_path, **npz)

    rng = np.random.RandomState(11)
    raw_features = rng.randn(7, INPUT_DIM).astype(np.float32)
    normalized_features = normalize_features(raw_features, npz["feature_mean"], npz["feature_std"])
    previous_ctrls = rng.randn(7, NUM_CHANNELS).astype(np.float32)

    def mock_forward(model, features):
        return _torch_forward_from_npz(npz, features)

    monkeypatch.setattr(trainer, "model_forward_torch_batched", mock_forward)

    # Incorrect old call: both paths receive normalized features.
    result = trainer.verify_numpy_torch_match(
        None, npz_path, normalized_features, normalized_features, previous_ctrls
    )
    assert not result["within_tolerance"]


def _make_sample(target_class: int, frame_id: int, prev_ctrl: np.ndarray | None = None) -> dict[str, Any]:
    """Build a minimal valid sample; wrist is null as required by the actor contract."""
    if prev_ctrl is None:
        prev_ctrl = np.zeros(NUM_CHANNELS, dtype=np.float32)
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    executed[0] = prev_ctrl
    return {
        "episode_id": target_class,
        "frame_id": frame_id,
        "split": "train" if target_class in {0, 2, 3, 5, 6, 8, 9, 11} else "validation" if target_class in {1, 7} else "test",
        "target_class": target_class,
        "actor_input": {
            "error_xy_px": np.zeros(2, dtype=np.float32),
            "q_rad16": np.zeros(NUM_CHANNELS, dtype=np.float32),
            "qvel_rad_s16": np.zeros(NUM_CHANNELS, dtype=np.float32),
            "previous_ctrl_rad16": prev_ctrl,
            "wrist": None,
        },
        "labels": {"executed_ctrl_rad": executed},
        "action_window": {"dt_s": 0.002, "first_step_index": 0, "end_exclusive_step_index": HORIZON},
    }


def test_batch_sampler_basic_iteration():
    """The sampler yields full batches and advances position deterministically."""
    sampler = trainer._BatchSampler(20, 4)
    batches = [sampler.next_batch() for _ in range(5)]
    for b in batches:
        assert b.shape == (4,)
        assert len(set(b.tolist())) == 4
    assert sampler.position == 20  # at end; rollover happens on next call
    assert sampler.epoch == 0


def test_batch_sampler_rollover_matches_identity_shuffle():
    """Rollover uses the same np.random shuffle that the main loop would use."""
    np.random.seed(13)
    sampler = trainer._BatchSampler(10, 4)
    _ = sampler.next_batch()
    _ = sampler.next_batch()
    third = sampler.next_batch()
    assert sampler.epoch == 1
    assert sampler.position == 4  # consumed 4 from the new shuffle

    np.random.seed(13)
    sampler2 = trainer._BatchSampler(10, 4)
    _ = sampler2.next_batch()
    _ = sampler2.next_batch()
    third2 = sampler2.next_batch()
    assert np.array_equal(third, third2)


def test_batch_sampler_state_restore_reproduces_batches():
    """Saving and loading sampler state reproduces the exact batch sequence."""
    np.random.seed(7)
    sampler = trainer._BatchSampler(20, 4)
    _ = sampler.next_batch()
    _ = sampler.next_batch()
    state = sampler.state_dict()

    np.random.seed(7)
    sampler2 = trainer._BatchSampler(20, 4)
    sampler2.load_state_dict(state)
    rest = [sampler2.next_batch() for _ in range(10)]

    np.random.seed(7)
    sampler3 = trainer._BatchSampler(20, 4)
    _ = sampler3.next_batch()
    _ = sampler3.next_batch()
    expected = [sampler3.next_batch() for _ in range(10)]

    for a, b in zip(rest, expected):
        assert np.array_equal(a, b)


def test_batch_sampler_tail_resume_reproduces_rollover():
    """Resuming from the tail of an epoch reproduces the rollover shuffle."""
    n, batch_size = 453, 64
    np.random.seed(42)
    sampler = trainer._BatchSampler(n, batch_size)
    for _ in range(7):
        _ = sampler.next_batch()
    assert sampler.position == 448
    assert sampler.epoch == 0

    rng_state = np.random.get_state()
    state = sampler.state_dict()
    expected = sampler.next_batch()

    np.random.set_state(rng_state)
    sampler2 = trainer._BatchSampler(n, batch_size)
    sampler2.load_state_dict(state)
    restored = sampler2.next_batch()
    assert np.array_equal(restored, expected)
    assert sampler2.position == 64
    assert sampler2.epoch == 1


def test_batch_sampler_rejects_invalid_state():
    """Loading a malformed sampler state must fail closed."""
    sampler = trainer._BatchSampler(20, 4)
    bad_state = {"epoch": 0, "order": np.arange(19), "position": 0}
    with pytest.raises(SmallPolicyError, match="shape mismatch"):
        sampler.load_state_dict(bad_state)

    bad_perm = {"epoch": 0, "order": np.zeros(20, dtype=np.int64), "position": 0}
    with pytest.raises(SmallPolicyError, match="permutation"):
        sampler.load_state_dict(bad_perm)


def test_validate_frozen_hyperparams_rejects_too_few_updates():
    """max_updates must be large enough for the step-100 determinism proof."""
    permit = {"limits": {"max_updates": 2000}}
    with pytest.raises(SmallPolicyError, match="step-100"):
        trainer.validate_frozen_hyperparams(_make_namespace(max_updates=100), permit)


def test_validate_resume_checkpoint_accepts_valid():
    """A matching checkpoint should pass strict resume validation."""
    files = trainer._build_source_manifest()
    contract = {
        "source_commit": "abc123def4567890abcd1234567890abcdef1234",
        "dataset_schema": "v1-small-policy-smoke-v1",
        "dataset_index_sha256": "a" * 64,
        "split_counts_observed": {"train": 10, "validation": 4, "test": 2},
    }
    hyperparams = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "grad_clip": 1.0,
        "seed": 17,
        "val_every": 100,
        "max_updates": 2000,
    }
    raw = np.random.randn(20, INPUT_DIM).astype(np.float32)
    norm = SmallPolicyNormalizer().fit(raw)
    ckpt = {
        "source_manifest": dict(files),
        "contract": contract,
        "hyperparams": hyperparams,
        "normalizer": norm.state_dict(),
        "step": 100,
    }
    trainer._validate_resume_checkpoint(ckpt, files, contract, hyperparams, raw)


def test_validate_resume_checkpoint_rejects_source_manifest_mismatch():
    files = trainer._build_source_manifest()
    bad_files = {k: ("0" * 64 if k == "tools/train_leap_v1_small_policy.py" else v) for k, v in files.items()}
    contract = {"source_commit": "abc", "dataset_schema": "x", "dataset_index_sha256": "a" * 64, "split_counts_observed": {}}
    hyperparams = {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "grad_clip": 1.0, "seed": 17, "val_every": 100, "max_updates": 2000}
    raw = np.random.randn(20, INPUT_DIM).astype(np.float32)
    ckpt = {"source_manifest": bad_files, "contract": contract, "hyperparams": hyperparams, "normalizer": SmallPolicyNormalizer().fit(raw).state_dict(), "step": 50}
    with pytest.raises(SmallPolicyError, match="source manifest mismatch"):
        trainer._validate_resume_checkpoint(ckpt, files, contract, hyperparams, raw)


def test_validate_resume_checkpoint_rejects_normalizer_mismatch():
    files = trainer._build_source_manifest()
    contract = {"source_commit": "abc", "dataset_schema": "x", "dataset_index_sha256": "a" * 64, "split_counts_observed": {}}
    hyperparams = {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64, "grad_clip": 1.0, "seed": 17, "val_every": 100, "max_updates": 2000}
    raw_a = np.random.randn(20, INPUT_DIM).astype(np.float32)
    raw_b = np.random.randn(20, INPUT_DIM).astype(np.float32) + 10.0
    ckpt = {
        "source_manifest": files,
        "contract": contract,
        "hyperparams": hyperparams,
        "normalizer": SmallPolicyNormalizer().fit(raw_a).state_dict(),
        "step": 50,
    }
    with pytest.raises(SmallPolicyError, match="normalizer"):
        trainer._validate_resume_checkpoint(ckpt, files, contract, hyperparams, raw_b)


def test_load_checkpoint_validates_required_keys(tmp_path, monkeypatch):
    """load_checkpoint must reject checkpoints missing required keys."""
    fake_ckpt = {"step": 10, "model_state_dict": {}}
    mock_torch = mock.MagicMock()
    mock_torch.load.return_value = fake_ckpt
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    with pytest.raises(SmallPolicyError, match="missing keys"):
        trainer.load_checkpoint(tmp_path / "x.pt", "cpu")

    assert mock_torch.load.called
    args, kwargs = mock_torch.load.call_args
    assert args[0] == tmp_path / "x.pt"
    assert kwargs.get("map_location") == "cpu"


def test_train_failure_paths_return_nonzero_via_ast():
    """Static check that train() returns 1 on controlled failure paths."""
    body = _function_body("train")
    source = ast.unparse(body)
    assert "failed = True" in source
    assert "return 1" in source
    assert "report['failure_reason']" in source
    assert "failed_progress_checkpoint" in source
