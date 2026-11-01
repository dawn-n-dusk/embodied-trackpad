"""Fine-tuning warm-start tests for the LEAP v1 small-policy trainer.

Source-only gates for explicit whole-network initialization: permit spec
validation, init NPZ loading, frozen-hyperparameter enforcement, strict-resume
binding, and provenance wiring. PyTorch-specific parity tests are skipped on
NumPy-only runtimes and must not be reported as passed there. An optional
real-baseline NPZ check is env-driven (TRACKPAD_BASELINE_NPZ) and stays out of
the public default path.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1.small_policy import (
    HIDDEN_DIM,
    HORIZON,
    INPUT_DIM,
    NUM_CHANNELS,
    OUTPUT_DIM,
    SmallPolicyError,
    SmallPolicyNormalizer,
    forward_numpy,
)

import tools.train_leap_v1_small_policy as trainer

BASELINE_NPZ_ENV = "TRACKPAD_BASELINE_NPZ"
BASELINE_NPZ_SHA256 = "2f38c5152647306a99be474f9ad93d05e890ae497312962911a19938ad4cb2d0"

FINETUNE_SPEC = {
    "mode": "finetune",
    "npz_sha256": "a" * 64,
    "normalizer_policy": "preserve_initial",
}


def _write_npz(path: Path, *, skip=(), mutate=None, contract=None) -> Path:
    rng = np.random.RandomState(3)
    arrays = {
        "W1": (rng.randn(HIDDEN_DIM, INPUT_DIM) * 0.01).astype(np.float32),
        "b1": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W2": (rng.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.01).astype(np.float32),
        "b2": np.zeros(HIDDEN_DIM, dtype=np.float32),
        "W3": (rng.randn(OUTPUT_DIM, HIDDEN_DIM) * 0.01).astype(np.float32),
        "b3": np.zeros(OUTPUT_DIM, dtype=np.float32),
        "feature_mean": np.zeros(INPUT_DIM, dtype=np.float32),
        "feature_std": np.ones(INPUT_DIM, dtype=np.float32),
        "contract_json": np.asarray(
            bytes(json.dumps(contract if contract is not None else {"purpose": "small_policy_smoke"}), "utf-8")
        ),
    }
    for key in skip:
        arrays.pop(key)
    if mutate:
        mutate(arrays)
    np.savez(path, **arrays)
    return path


# --- validate_initialization_spec (pure) -------------------------------------


def test_spec_absent_means_scratch():
    assert trainer.validate_initialization_spec({}, None) is None
    assert trainer.validate_initialization_spec({"purpose": "small_policy_smoke"}, None) is None


def test_init_npz_without_spec_rejected():
    with pytest.raises(SmallPolicyError, match="requires permit initialization"):
        trainer.validate_initialization_spec({}, "some.npz")


def test_finetune_spec_requires_init_npz_unless_resuming():
    with pytest.raises(SmallPolicyError, match="requires --init-npz"):
        trainer.validate_initialization_spec({"initialization": dict(FINETUNE_SPEC)}, None)
    # Strict resume carries the frozen initialization in the checkpoint.
    spec = trainer.validate_initialization_spec({"initialization": dict(FINETUNE_SPEC)}, None, resuming=True)
    assert spec == FINETUNE_SPEC


def test_resume_and_init_npz_conflict_rejected():
    with pytest.raises(SmallPolicyError, match="cannot be combined with --resume"):
        trainer.validate_initialization_spec(
            {"initialization": dict(FINETUNE_SPEC)}, "some.npz", resuming=True
        )


@pytest.mark.parametrize(
    "spec,msg",
    [
        ({"mode": "scratch", "npz_sha256": "a" * 64, "normalizer_policy": "preserve_initial"}, "unsupported initialization.mode"),
        ({"mode": "finetune", "npz_sha256": "a" * 64, "normalizer_policy": "refit"}, "unsupported initialization.normalizer_policy"),
        ({"mode": "finetune", "npz_sha256": "zzz", "normalizer_policy": "preserve_initial"}, "64-character hex"),
    ],
)
def test_bad_spec_fields_rejected(spec, msg):
    with pytest.raises(SmallPolicyError, match=msg):
        trainer.validate_initialization_spec({"initialization": spec}, None, resuming=True)


def test_spec_hash_must_match_file(tmp_path):
    path = _write_npz(tmp_path / "init.npz")
    bad = dict(FINETUNE_SPEC, npz_sha256="b" * 64)
    with pytest.raises(SmallPolicyError, match="sha256 mismatch"):
        trainer.validate_initialization_spec({"initialization": bad}, path)

    good = dict(FINETUNE_SPEC, npz_sha256=trainer.sha256_file(path))
    spec = trainer.validate_initialization_spec({"initialization": good}, path)
    assert spec == good


def test_spec_missing_file_rejected():
    spec = dict(FINETUNE_SPEC)
    with pytest.raises(SmallPolicyError, match="not found"):
        trainer.validate_initialization_spec({"initialization": spec}, "no/such/file.npz")


# --- load_init_npz (pure NumPy) ----------------------------------------------


def test_load_init_npz_valid(tmp_path):
    path = _write_npz(tmp_path / "init.npz")
    arrays = trainer.load_init_npz(path)
    assert set(arrays) == set(trainer.REQUIRED_INIT_NPZ_KEYS)
    assert arrays["W1"].shape == (HIDDEN_DIM, INPUT_DIM)
    assert arrays["feature_std"].shape == (INPUT_DIM,)
    assert all(np.isfinite(a).all() for a in arrays.values())


def test_load_init_npz_missing_key(tmp_path):
    path = _write_npz(tmp_path / "init.npz", skip=("b3",))
    with pytest.raises(SmallPolicyError, match="missing keys"):
        trainer.load_init_npz(path)


def test_load_init_npz_missing_contract(tmp_path):
    path = _write_npz(tmp_path / "init.npz", skip=("contract_json",))
    with pytest.raises(SmallPolicyError, match="contract_json"):
        trainer.load_init_npz(path)


def test_load_init_npz_bad_contract_json(tmp_path):
    path = _write_npz(tmp_path / "init.npz", mutate=lambda a: a.__setitem__("contract_json", np.asarray(b"{")))
    with pytest.raises(SmallPolicyError, match="contract_json"):
        trainer.load_init_npz(path)


def test_load_init_npz_shape_mismatch(tmp_path):
    def mutate(a):
        a["W1"] = np.zeros((HIDDEN_DIM, INPUT_DIM + 1), dtype=np.float32)

    path = _write_npz(tmp_path / "init.npz", mutate=mutate)
    with pytest.raises(SmallPolicyError, match="shape mismatch"):
        trainer.load_init_npz(path)


def test_load_init_npz_non_finite_and_bad_std(tmp_path):
    def bad_value(a):
        a["b1"] = np.full(HIDDEN_DIM, np.nan, dtype=np.float32)

    with pytest.raises(SmallPolicyError, match="non-finite"):
        trainer.load_init_npz(_write_npz(tmp_path / "nan.npz", mutate=bad_value))

    def bad_std(a):
        a["feature_std"] = np.full(INPUT_DIM, 1e-4, dtype=np.float32)

    with pytest.raises(SmallPolicyError, match="feature_std"):
        trainer.load_init_npz(_write_npz(tmp_path / "std.npz", mutate=bad_std))


# --- frozen hyperparameters ---------------------------------------------------


def _ns(**kw):
    import argparse

    base = dict(lr=1e-3, weight_decay=1e-4, batch_size=64, grad_clip=1.0, seed=17, max_updates=2000, val_every=100)
    base.update(kw)
    return argparse.Namespace(**base)


def test_finetune_hyperparams_accepted():
    permit = {"initialization": dict(FINETUNE_SPEC), "limits": {"max_updates": 500}}
    hp = trainer.validate_frozen_hyperparams(_ns(lr=1e-4, max_updates=500), permit)
    assert hp["lr"] == 1e-4 and hp["max_updates"] == 500


def test_finetune_hyperparams_reject_legacy_values():
    permit = {"initialization": dict(FINETUNE_SPEC), "limits": {"max_updates": 500}}
    with pytest.raises(SmallPolicyError, match="finetune initialization"):
        trainer.validate_frozen_hyperparams(_ns(lr=1e-3, max_updates=500), permit)
    with pytest.raises(SmallPolicyError, match="<= 500"):
        trainer.validate_frozen_hyperparams(_ns(lr=1e-4, max_updates=501), permit)


def test_finetune_hyperparams_respect_permit_limit():
    permit = {"initialization": dict(FINETUNE_SPEC), "limits": {"max_updates": 400}}
    with pytest.raises(SmallPolicyError, match="exceeds permit limit"):
        trainer.validate_frozen_hyperparams(_ns(lr=1e-4, max_updates=500), permit)


def test_scratch_defaults_unchanged():
    hp = trainer.validate_frozen_hyperparams(_ns(), {"limits": {"max_updates": 2000}})
    assert hp["lr"] == 1e-3 and hp["max_updates"] == 2000
    with pytest.raises(SmallPolicyError, match="lr must be 0.001"):
        trainer.validate_frozen_hyperparams(_ns(lr=1e-4), {"limits": {"max_updates": 2000}})


# --- strict resume binding (pure NumPy) ---------------------------------------


def _resume_context():
    manifest = {p: "a" * 64 for p in trainer.REQUIRED_SOURCE_FILES}
    contract = {
        "source_commit": "c" * 40,
        "dataset_schema": "v1-small-policy-smoke-v1",
        "dataset_index_sha256": "d" * 64,
        "split_counts_observed": {"train": 4, "validation": 2, "test": 2},
    }
    hyperparams = {
        "lr": 1e-4, "weight_decay": 1e-4, "batch_size": 64, "grad_clip": 1.0,
        "seed": 17, "val_every": 100, "max_updates": 500,
    }
    rng = np.random.RandomState(0)
    raw_train = (rng.randn(32, INPUT_DIM) * 0.1).astype(np.float32)
    return manifest, contract, hyperparams, raw_train


def _ckpt(norm: SmallPolicyNormalizer, initialization=None, step=5):
    manifest, contract, hyperparams, _ = _resume_context()
    return {
        "step": step,
        "source_manifest": manifest,
        "contract": contract,
        "hyperparams": hyperparams,
        "normalizer": norm.state_dict(),
        "initialization": initialization,
    }


def test_finetune_resume_binds_init_and_frozen_normalizer():
    frozen_mean = np.full(INPUT_DIM, 0.25, dtype=np.float32)
    frozen_std = np.full(INPUT_DIM, 2.0, dtype=np.float32)
    norm = SmallPolicyNormalizer()
    norm.load_state_dict({"feature_mean": frozen_mean, "feature_std": frozen_std})
    init = dict(FINETUNE_SPEC)  # npz_sha256 matches the permit spec
    init["frozen_feature_mean"] = frozen_mean.tolist()
    init["frozen_feature_std"] = frozen_std.tolist()

    manifest, contract, hyperparams, raw_train = _resume_context()
    contract = dict(contract, initialization=dict(FINETUNE_SPEC))
    ckpt = _ckpt(norm, initialization=init)
    ckpt["contract"] = dict(ckpt["contract"], initialization=dict(FINETUNE_SPEC))
    trainer._validate_resume_checkpoint(
        ckpt, manifest, contract, hyperparams, raw_train, initialization=FINETUNE_SPEC
    )


def test_finetune_resume_rejects_init_mismatch():
    frozen_mean = np.zeros(INPUT_DIM, dtype=np.float32)
    frozen_std = np.ones(INPUT_DIM, dtype=np.float32)
    norm = SmallPolicyNormalizer()
    norm.load_state_dict({"feature_mean": frozen_mean, "feature_std": frozen_std})
    init = dict(FINETUNE_SPEC, npz_sha256="c" * 64)  # != permit "a"*64
    init["frozen_feature_mean"] = frozen_mean.tolist()
    init["frozen_feature_std"] = frozen_std.tolist()

    manifest, contract, hyperparams, raw_train = _resume_context()
    contract = dict(contract, initialization=dict(FINETUNE_SPEC))
    ckpt = _ckpt(norm, initialization=init)
    ckpt["contract"] = dict(ckpt["contract"], initialization=dict(FINETUNE_SPEC))
    with pytest.raises(SmallPolicyError, match="initialization mismatch"):
        trainer._validate_resume_checkpoint(
            ckpt, manifest, contract, hyperparams, raw_train, initialization=FINETUNE_SPEC
        )

    # Missing initialization metadata entirely (contract carries it, top-level does not).
    ckpt = _ckpt(norm, initialization=None)
    ckpt["contract"] = dict(ckpt["contract"], initialization=dict(FINETUNE_SPEC))
    with pytest.raises(SmallPolicyError, match="missing initialization metadata"):
        trainer._validate_resume_checkpoint(
            ckpt, manifest, contract, hyperparams, raw_train, initialization=FINETUNE_SPEC
        )


@pytest.mark.parametrize(
    "current_init,ckpt_contract_init",
    [(None, dict(FINETUNE_SPEC)), (dict(FINETUNE_SPEC), None)],
)
def test_resume_rejects_contract_provenance_direction_mismatch(current_init, ckpt_contract_init):
    """Duplicated provenance must agree in both directions: a finetune checkpoint
    under a scratch contract (or vice versa) is rejected, never coerced."""
    frozen_mean = np.zeros(INPUT_DIM, dtype=np.float32)
    frozen_std = np.ones(INPUT_DIM, dtype=np.float32)
    norm = SmallPolicyNormalizer()
    norm.load_state_dict({"feature_mean": frozen_mean, "feature_std": frozen_std})
    init = dict(FINETUNE_SPEC)
    init["frozen_feature_mean"] = frozen_mean.tolist()
    init["frozen_feature_std"] = frozen_std.tolist()

    manifest, contract, hyperparams, raw_train = _resume_context()
    contract = dict(contract, initialization=current_init)
    ckpt = _ckpt(norm, initialization=init if current_init is not None else None)
    ckpt["contract"] = dict(ckpt["contract"], initialization=ckpt_contract_init)
    with pytest.raises(SmallPolicyError, match="contract initialization mismatch"):
        trainer._validate_resume_checkpoint(
            ckpt, manifest, contract, hyperparams, raw_train, initialization=current_init
        )


def test_finetune_resume_rejects_nonfinite_or_nonpositive_frozen_std():
    frozen_mean = np.zeros(INPUT_DIM, dtype=np.float32)
    frozen_std = np.ones(INPUT_DIM, dtype=np.float32)
    norm = SmallPolicyNormalizer()
    norm.load_state_dict({"feature_mean": frozen_mean, "feature_std": frozen_std})

    manifest, contract, hyperparams, raw_train = _resume_context()
    contract = dict(contract, initialization=dict(FINETUNE_SPEC))

    bad_std = dict(FINETUNE_SPEC)
    bad_std["frozen_feature_mean"] = frozen_mean.tolist()
    bad_std["frozen_feature_std"] = [0.0] * INPUT_DIM
    ckpt = _ckpt(norm, initialization=bad_std)
    ckpt["contract"] = dict(ckpt["contract"], initialization=dict(FINETUNE_SPEC))
    with pytest.raises(SmallPolicyError, match="std must be positive"):
        trainer._validate_resume_checkpoint(
            ckpt, manifest, contract, hyperparams, raw_train, initialization=FINETUNE_SPEC
        )

    nan_mean = dict(FINETUNE_SPEC)
    nan_mean["frozen_feature_mean"] = [float("nan")] * INPUT_DIM
    nan_mean["frozen_feature_std"] = frozen_std.tolist()
    ckpt = _ckpt(norm, initialization=nan_mean)
    ckpt["contract"] = dict(ckpt["contract"], initialization=dict(FINETUNE_SPEC))
    with pytest.raises(SmallPolicyError, match="not finite"):
        trainer._validate_resume_checkpoint(
            ckpt, manifest, contract, hyperparams, raw_train, initialization=FINETUNE_SPEC
        )


def test_finetune_resume_normalizer_compare_is_exact_float32():
    """An almost-equal frozen normalizer (allclose but not bit-exact) is rejected."""
    frozen_mean = np.full(INPUT_DIM, 0.25, dtype=np.float32)
    frozen_std = np.full(INPUT_DIM, 2.0, dtype=np.float32)
    norm = SmallPolicyNormalizer()
    norm.load_state_dict({"feature_mean": frozen_mean, "feature_std": frozen_std})

    # One ulp-ish drift in float32 space: passes np.allclose, fails array_equal.
    drifted = frozen_mean.copy()
    drifted[0] = np.float32(np.float32(drifted[0]) + np.finfo(np.float32).eps)
    init = dict(FINETUNE_SPEC)
    init["frozen_feature_mean"] = drifted.tolist()
    init["frozen_feature_std"] = frozen_std.tolist()

    manifest, contract, hyperparams, raw_train = _resume_context()
    contract = dict(contract, initialization=dict(FINETUNE_SPEC))
    ckpt = _ckpt(norm, initialization=init)
    ckpt["contract"] = dict(ckpt["contract"], initialization=dict(FINETUNE_SPEC))
    assert np.allclose(np.asarray(init["frozen_feature_mean"], dtype=np.float32), frozen_mean)
    with pytest.raises(SmallPolicyError, match="frozen initialization"):
        trainer._validate_resume_checkpoint(
            ckpt, manifest, contract, hyperparams, raw_train, initialization=FINETUNE_SPEC
        )


def test_finetune_resume_validates_normalizer_against_frozen_not_refit():
    rng = np.random.RandomState(1)
    raw_train = (rng.randn(32, INPUT_DIM) * 0.7 + 3.0).astype(np.float32)
    refit = SmallPolicyNormalizer().fit(raw_train)  # what a new-data refit would give

    frozen_mean = np.zeros(INPUT_DIM, dtype=np.float32)
    frozen_std = np.ones(INPUT_DIM, dtype=np.float32)
    init = dict(FINETUNE_SPEC)
    init["frozen_feature_mean"] = frozen_mean.tolist()
    init["frozen_feature_std"] = frozen_std.tolist()

    manifest, contract, hyperparams, _ = _resume_context()
    # Checkpoint whose normalizer was silently refit on the new data: a legacy
    # check would pass (train-refit == train-refit), the frozen check must not.
    ckpt = _ckpt(refit, initialization=init)
    with pytest.raises(SmallPolicyError, match="frozen initialization"):
        trainer._validate_resume_checkpoint(
            ckpt, manifest, contract, hyperparams, raw_train, initialization=FINETUNE_SPEC
        )


def test_scratch_resume_still_uses_train_refit_check():
    rng = np.random.RandomState(2)
    raw_train = (rng.randn(32, INPUT_DIM) * 0.5).astype(np.float32)
    fit_norm = SmallPolicyNormalizer().fit(raw_train)

    manifest, contract, hyperparams, _ = _resume_context()
    trainer._validate_resume_checkpoint(_ckpt(fit_norm), manifest, contract, hyperparams, raw_train)

    wrong = SmallPolicyNormalizer()
    wrong.load_state_dict({"feature_mean": np.ones(INPUT_DIM, dtype=np.float32), "feature_std": np.ones(INPUT_DIM, dtype=np.float32)})
    with pytest.raises(SmallPolicyError, match="recomputed train-only normalizer"):
        trainer._validate_resume_checkpoint(_ckpt(wrong), manifest, contract, hyperparams, raw_train)


# --- source wiring gates -------------------------------------------------------


def _trainer_source() -> str:
    return Path("tools/train_leap_v1_small_policy.py").read_text(encoding="utf-8")


def test_finetune_wiring_present_via_ast():
    raw = _trainer_source()
    # Raw source keeps the literal argparse spelling (ast.unparse re-quotes).
    assert '"--init-npz"' in raw
    assert "best_step_000000.pt" in raw
    assert "resume_start_determinism_check.pt" in raw
    assert "init parity failed" in raw
    assert "cannot be combined with --resume" in raw
    assert "frozen_feature_mean" in raw
    assert "preserve_initial" in raw
    assert '"initialization"' in raw


def test_resume_start_comparison_wiring_via_ast():
    """Resumed runs must gain a disk-restored first-update comparison."""
    tree = ast.parse(_trainer_source())
    train_fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "train"
    )
    source = ast.unparse(train_fn)
    assert "resume_start_determinism_check" in source
    assert "checked_at_step" in source
    # The comparison still requires an actual following update, never a faked one.
    assert "comparison_updates += 1" in source


# --- init NPZ dtype/finiteness gates (finding: no silent casts) ----------------


def test_load_init_npz_rejects_complex_and_bool(tmp_path):
    def complex_arr(a):
        a["W1"] = np.zeros((HIDDEN_DIM, INPUT_DIM), dtype=np.complex64)

    with pytest.raises(SmallPolicyError, match="unsupported dtype"):
        trainer.load_init_npz(_write_npz(tmp_path / "cplx.npz", mutate=complex_arr))

    def bool_arr(a):
        a["b1"] = np.zeros(HIDDEN_DIM, dtype=np.bool_)

    with pytest.raises(SmallPolicyError, match="unsupported dtype"):
        trainer.load_init_npz(_write_npz(tmp_path / "bool.npz", mutate=bool_arr))


def test_load_init_npz_rejects_overflow_after_float32_conversion(tmp_path):
    def huge(a):
        a["b1"] = np.full(HIDDEN_DIM, 1e39, dtype=np.float64)  # finite in f64, inf in f32

    with pytest.raises(SmallPolicyError, match="float32 conversion"):
        trainer.load_init_npz(_write_npz(tmp_path / "huge.npz", mutate=huge))


# --- optional real baseline NPZ regression (env-driven, read-only) --------------


@pytest.mark.skipif(
    os.environ.get(BASELINE_NPZ_ENV) is None,
    reason=f"set {BASELINE_NPZ_ENV} to the coordinator-provided baseline NPZ to run",
)
def test_real_baseline_npz_sha_and_load():
    path = Path(os.environ[BASELINE_NPZ_ENV])
    if not path.exists():
        pytest.skip(f"{BASELINE_NPZ_ENV} does not exist")
    assert trainer.sha256_file(path) == BASELINE_NPZ_SHA256
    arrays = trainer.load_init_npz(path)
    assert arrays["W3"].shape == (OUTPUT_DIM, HIDDEN_DIM)
    permit = {"initialization": dict(FINETUNE_SPEC, npz_sha256=BASELINE_NPZ_SHA256)}
    assert trainer.validate_initialization_spec(permit, path)["mode"] == "finetune"


# --- torch-only parity (skipped on NumPy-only runtimes) -----------------------


def test_torch_init_parity_and_trainable():
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(5)
    arrays = {
        "W1": (rng.randn(HIDDEN_DIM, INPUT_DIM) * 0.05).astype(np.float32),
        "b1": rng.randn(HIDDEN_DIM).astype(np.float32) * 0.01,
        "W2": (rng.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.05).astype(np.float32),
        "b2": rng.randn(HIDDEN_DIM).astype(np.float32) * 0.01,
        "W3": (rng.randn(OUTPUT_DIM, HIDDEN_DIM) * 0.05).astype(np.float32),
        "b3": rng.randn(OUTPUT_DIM).astype(np.float32) * 0.01,
        "feature_mean": rng.randn(INPUT_DIM).astype(np.float32) * 0.1,
        "feature_std": (np.abs(rng.randn(INPUT_DIM)) + 0.5).astype(np.float32),
    }
    model = trainer.build_model("cpu")
    model.load_state_dict(trainer.init_state_dict_torch(arrays), strict=True)
    assert all(p.requires_grad for p in model.parameters())

    features = rng.randn(6, INPUT_DIM).astype(np.float32)
    norm = SmallPolicyNormalizer()
    norm.load_state_dict({"feature_mean": arrays["feature_mean"], "feature_std": arrays["feature_std"]})
    normalized = norm.transform(features)
    torch_deltas = trainer.model_forward_torch_batched(model, normalized)
    numpy_deltas = forward_numpy(arrays, features)
    assert float(np.max(np.abs(torch_deltas - numpy_deltas))) <= 1e-8

    # Fresh optimizer for a warm start has no stepped state.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    assert opt.state_dict()["state"] == {}
