"""Boundary tests for the native pi0.5 normalization stats/application module.

Synthetic legacy-profile sample indexes are written under tmp_path and loaded
through the real ``VisualCorpus`` + ``small_policy.load_sample_index``
validators, so fit-side tests exercise the same guards as production corpora.
Group 1 checks stats against hand-computed (pure-Python, no NumPy) mean /
population-std / linear-interpolated quantiles; group 2 proves validation and
test rows cannot leak into fitted stats; group 3 covers the corpus namespace
and index-hash guards; group 4 covers the official quantile map on zero-range
and out-of-range channels plus copy/mutation isolation; group 5 rejects
malformed stats and malformed native samples. One host-local test fits the two
authorized runtime corpora and checks the published train/action-step totals.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import small_policy
from embodied_trackpad.leap_v1.pi05_inputs import (
    ACTION_DIM,
    DEFAULT_PROMPT,
    prepare_pi05_native_sample,
)
from embodied_trackpad.leap_v1.pi05_normalization import (
    _EPSILON,
    fit_pi05_norm_stats,
    normalize_pi05_native_sample,
)
from embodied_trackpad.leap_v1.visual_dataset import (
    ACTION_SEMANTICS,
    DT_S,
    HORIZON,
    HORIZON_S,
    NUM_CHANNELS,
    PNG_HEIGHT,
    PNG_WIDTH,
    VisualCorpus,
    VisualCorpusError,
    VisualExample,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# reference implementations: hand-computed mean/std/quantiles (pure Python)
# --------------------------------------------------------------------------- #


def _ref_mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _ref_std(xs: list[float]) -> float:
    m = _ref_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _ref_quantile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    pos = (len(s) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


# --------------------------------------------------------------------------- #
# synthetic corpus fixtures (legacy profile: target_class == episode_id % 12)
# --------------------------------------------------------------------------- #

_PNG_SHA = "ab" * 32
_HAND_CORPUS_ID = "hand-fixture-corpus"
_HAND_INDEX_SHA = "cd" * 32


def _row(
    episode_id: int,
    frame_id: int,
    split: str,
    target_class: int,
    q: np.ndarray,
    executed: np.ndarray,
) -> dict:
    return {
        "episode_id": episode_id,
        "frame_id": frame_id,
        "split": split,
        "target_class": target_class,
        "actor_input": {
            "png_ref": f"frames/ep-{episode_id:03d}-f{frame_id:03d}.png",
            "png_sha256": _PNG_SHA,
            "error_xy_px": [0.0, 0.0],
            "q_rad16": [float(v) for v in q],
            "qvel_rad_s16": [0.0] * NUM_CHANNELS,
            "previous_ctrl_rad16": [0.0] * NUM_CHANNELS,
            "valid": {
                "png_valid": True,
                "visual_valid": True,
                "q_valid": True,
                "qvel_valid": True,
                "previous_ctrl_valid": True,
                "fsr_valid4": [False, False, False, False],
                "motor_current_valid16": [False] * NUM_CHANNELS,
                "wrist_valid": False,
            },
            "fsr_N4": None,
            "motor_current_A16": None,
            "wrist": None,
        },
        "action_window": {
            "dt_s": DT_S,
            "first_step_index": 0,
            "end_exclusive_step_index": HORIZON,
        },
        "labels": {
            "executed_ctrl_rad": [[float(v) for v in step] for step in executed],
        },
    }


def _train_rows() -> tuple[dict, dict]:
    q_a = np.arange(NUM_CHANNELS, dtype=np.float64)
    ex_a = np.zeros((HORIZON, NUM_CHANNELS))
    ex_a[:, 0] = np.arange(HORIZON) * 0.001
    ex_a[:, 1] = [0.0 if t % 2 == 0 else 1.0 for t in range(HORIZON)]
    ex_a[:, 2:] = 0.25
    row_a = _row(0, 0, "train", 0, q_a, ex_a)

    q_b = np.arange(15, -1, -1, dtype=np.float64)
    ex_b = np.zeros((HORIZON, NUM_CHANNELS))
    ex_b[:, 0] = (HORIZON - 1 - np.arange(HORIZON)) * 0.001
    ex_b[:, 1] = 0.5
    ex_b[:, 2:] = 0.25
    row_b = _row(2, 0, "train", 2, q_b, ex_b)
    return row_a, row_b


def _leak_rows() -> tuple[dict, dict]:
    huge = np.full(NUM_CHANNELS, 1e6)
    val_row = _row(1, 0, "validation", 1, huge, np.full((HORIZON, NUM_CHANNELS), 1e6))
    tiny = np.full(NUM_CHANNELS, -1e6)
    test_row = _row(4, 0, "test", 4, tiny, np.full((HORIZON, NUM_CHANNELS), -1e6))
    return val_row, test_row


def _corpus(tmp_path: Path, name: str, corpus_id: str, rows: list[dict]) -> VisualCorpus:
    index_path = tmp_path / name / "sample_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    evidence = tmp_path / name / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    return VisualCorpus(
        index_path=index_path,
        evidence_root=evidence,
        expected_index_sha256=small_policy.sha256_file(index_path),
        corpus_id=corpus_id,
    )


@pytest.fixture
def corpus(tmp_path: Path) -> VisualCorpus:
    """One corpus: 2 train rows + 1 validation + 1 test leak row."""
    row_a, row_b = _train_rows()
    val_row, test_row = _leak_rows()
    return _corpus(tmp_path, "mixed", "mixed-corpus", [row_a, row_b, val_row, test_row])


# --------------------------------------------------------------------------- #
# native sample fixture (same recipe as tests/test_leap_v1_pi05_inputs.py)
# --------------------------------------------------------------------------- #


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


def _native_sample(q: np.ndarray, executed: np.ndarray, provenance_extra: dict | None = None):
    observation = {
        "screen_rgb": _rgb(7),
        "q_rad16": np.asarray(q, dtype=np.float32),
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
        "sample_id": "syn:ep-000:frame-0",
        "action_window": _timing(),
        "corpus_id": _HAND_CORPUS_ID,
        "index_sha256": _HAND_INDEX_SHA,
    }
    if provenance_extra:
        provenance.update(provenance_extra)
    example = VisualExample(
        observation=observation,
        target={"target_ctrl_rad": np.asarray(executed, dtype=np.float32)},
        provenance=provenance,
    )
    return prepare_pi05_native_sample(example)


def _hand_stats() -> dict:
    """Stats with a constant channel (0), a normal channel (1), and a
    sub-epsilon-range channel (2) in both groups.

    Metadata is mandatory for normalization, so even hand-computed stats
    carry fixture corpus provenance that the default synthetic samples bind
    to; this does not weaken the production contract (``fit_pi05_norm_stats``
    is the only real source).
    """
    stats = {}
    for group in ("state", "actions"):
        mean = np.zeros(NUM_CHANNELS)
        std = np.ones(NUM_CHANNELS)
        q01 = np.full(NUM_CHANNELS, -1.0)
        q99 = np.full(NUM_CHANNELS, 1.0)
        mean[0] = 0.5
        std[0] = 0.0
        q01[0] = 0.5
        q99[0] = 0.5
        q01[2] = 0.25
        q99[2] = 0.25 + 1e-9
        std[2] = 1e-10
        mean[2] = 0.25
        stats[group] = {
            "mean": [float(v) for v in mean],
            "std": [float(v) for v in std],
            "q01": [float(v) for v in q01],
            "q99": [float(v) for v in q99],
        }
    stats["metadata"] = {
        "method": "hand-fixture",
        "corpus_ids": [_HAND_CORPUS_ID],
        "index_sha256": [_HAND_INDEX_SHA],
        "train_rows_per_corpus": [0],
        "train_rows_total": 0,
        "state_rows_total": 0,
        "action_steps_total": 0,
    }
    return stats


# --------------------------------------------------------------------------- #
# 1. hand-computed quantiles / mean / std (independent pure-Python reference)
# --------------------------------------------------------------------------- #


def test_fit_matches_hand_computed_stats(corpus: VisualCorpus):
    stats = fit_pi05_norm_stats([corpus])

    assert set(stats["state"].keys()) == {"mean", "std", "q01", "q99"}
    assert set(stats["actions"].keys()) == {"mean", "std", "q01", "q99"}

    # state: two train rows, channel c holds {c, 15-c}.
    for c in range(NUM_CHANNELS):
        xs = [float(c), float(15 - c)]
        assert stats["state"]["mean"][c] == pytest.approx(_ref_mean(xs))
        assert stats["state"]["std"][c] == pytest.approx(_ref_std(xs))
        assert stats["state"]["q01"][c] == pytest.approx(_ref_quantile(xs, 0.01))
        assert stats["state"]["q99"][c] == pytest.approx(_ref_quantile(xs, 0.99))

    # actions channel 0: each k*0.001 for k in 0..24 appears exactly twice.
    ch0 = []
    for k in range(HORIZON):
        ch0.extend([k * 0.001, k * 0.001])
    assert stats["actions"]["mean"][0] == pytest.approx(_ref_mean(ch0))
    assert stats["actions"]["mean"][0] == pytest.approx(0.012)
    assert stats["actions"]["std"][0] == pytest.approx(_ref_std(ch0))
    assert stats["actions"]["q01"][0] == pytest.approx(_ref_quantile(ch0, 0.01))
    assert stats["actions"]["q99"][0] == pytest.approx(_ref_quantile(ch0, 0.99))

    # actions channel 1: thirteen 0.0, twenty-five 0.5, twelve 1.0.
    ch1 = [0.0] * 13 + [0.5] * 25 + [1.0] * 12
    assert stats["actions"]["mean"][1] == pytest.approx(_ref_mean(ch1))
    assert stats["actions"]["mean"][1] == pytest.approx(0.49)
    assert stats["actions"]["std"][1] == pytest.approx(_ref_std(ch1))
    assert stats["actions"]["q01"][1] == pytest.approx(_ref_quantile(ch1, 0.01))
    assert stats["actions"]["q99"][1] == pytest.approx(_ref_quantile(ch1, 0.99))

    meta = stats["metadata"]
    assert meta["train_rows_total"] == 2
    assert meta["state_rows_total"] == 2
    assert meta["action_steps_total"] == 2 * HORIZON
    assert meta["corpus_ids"] == ["mixed-corpus"]
    assert meta["index_sha256"] == [corpus.index_sha256]
    assert meta["train_rows_per_corpus"] == [2]
    assert "Normalize" in meta["method"]
    json.dumps(stats)  # serializable as-is


def test_fit_float64_precision_not_float32(corpus: VisualCorpus):
    stats = fit_pi05_norm_stats([corpus])
    # The two state values differ by exactly 1.0 per channel; a float32 fit
    # would snap q01/q99 onto grid points and lose the interpolation.
    for c in range(NUM_CHANNELS):
        lo, hi = sorted([float(c), float(15 - c)])
        assert stats["state"]["q01"][c] == pytest.approx(lo + 0.01 * (hi - lo))
        assert stats["state"]["q99"][c] == pytest.approx(lo + 0.99 * (hi - lo))


# --------------------------------------------------------------------------- #
# 2. train-only fitting: validation/test rows must not contribute
# --------------------------------------------------------------------------- #


def test_only_train_rows_contribute_no_eval_leakage(corpus: VisualCorpus):
    stats = fit_pi05_norm_stats([corpus])

    assert stats["metadata"]["train_rows_total"] == 2
    # The leak rows carry +-1e6 in every channel; any contribution would blow
    # the fitted values up by orders of magnitude.
    for group in ("state", "actions"):
        for key in ("mean", "std", "q01", "q99"):
            assert all(abs(v) < 100.0 for v in stats[group][key]), (group, key)

    # Fit over a corpus built from train rows only must give identical stats.
    row_a, row_b = _train_rows()
    train_only = _corpus(
        corpus.index_path.parent, "trainonly", "train-only", [row_a, row_b]
    )
    stats2 = fit_pi05_norm_stats([train_only])
    for group in ("state", "actions"):
        for key in ("mean", "std", "q01", "q99"):
            assert stats2[group][key] == pytest.approx(stats[group][key])


# --------------------------------------------------------------------------- #
# 3. corpus namespace + index-hash guards
# --------------------------------------------------------------------------- #


def test_duplicate_corpus_id_rejected(corpus: VisualCorpus):
    with pytest.raises(VisualCorpusError, match="duplicate corpus_id"):
        fit_pi05_norm_stats([corpus, corpus])


def test_same_index_hash_different_corpus_id_rejected(tmp_path: Path, corpus: VisualCorpus):
    alias_evidence = tmp_path / "alias-evidence"
    alias_evidence.mkdir()
    alias = VisualCorpus(
        index_path=corpus.index_path,
        evidence_root=alias_evidence,
        expected_index_sha256=corpus.index_sha256,
        corpus_id="renamed-corpus",
    )
    with pytest.raises(VisualCorpusError, match="duplicate index SHA-256"):
        fit_pi05_norm_stats([corpus, alias])


def test_index_mutated_during_read_rejected(corpus: VisualCorpus, monkeypatch):
    real = small_policy.sha256_file
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 2:  # the post-load re-hash inside fit
            return "ff" * 32
        return real(path)

    monkeypatch.setattr(small_policy, "sha256_file", flaky)
    with pytest.raises(VisualCorpusError, match="changed while fitting"):
        fit_pi05_norm_stats([corpus])


def test_index_rehash_mismatch_with_corpus_identity_rejected(
    corpus: VisualCorpus, monkeypatch
):
    monkeypatch.setattr(small_policy, "sha256_file", lambda path: "ee" * 32)
    with pytest.raises(VisualCorpusError, match="no longer matches"):
        fit_pi05_norm_stats([corpus])


def test_fit_input_guards(tmp_path: Path):
    with pytest.raises(VisualCorpusError, match="corpora is empty"):
        fit_pi05_norm_stats([])
    with pytest.raises(VisualCorpusError, match="must be validated VisualCorpus"):
        fit_pi05_norm_stats([object()])

    row_a, _ = _train_rows()
    val_row, _ = _leak_rows()
    no_train = _corpus(tmp_path, "notrain", "no-train", [val_row])
    with pytest.raises(VisualCorpusError, match="no train rows"):
        fit_pi05_norm_stats([no_train])

    # A two-corpus sequence with distinct ids/hashes is accepted.
    _, row_b = _train_rows()
    second = _corpus(tmp_path, "second", "second-corpus", [row_b])
    first = _corpus(tmp_path, "first", "first-corpus", [row_a])
    stats = fit_pi05_norm_stats([first, second])
    assert stats["metadata"]["corpus_ids"] == ["first-corpus", "second-corpus"]
    assert stats["metadata"]["train_rows_total"] == 2
    assert stats["metadata"]["train_rows_per_corpus"] == [1, 1]


# --------------------------------------------------------------------------- #
# 4. official quantile map: zero-range channel, range report, copy isolation
# --------------------------------------------------------------------------- #


def test_normalize_zero_range_and_range_violation_report():
    stats = _hand_stats()
    q = np.zeros(NUM_CHANNELS, dtype=np.float32)
    q[0] = 0.5  # constant channel, exactly at q01 == q99 -> y = -1.0
    q[1] = 5.0  # far outside [-1, 1] -> must NOT be clipped
    q[2] = 0.25  # sub-epsilon range -> y = -1.0
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    executed[:, 0] = 0.5
    executed[:, 1] = 5.0
    executed[:, 2] = 0.25
    sample = _native_sample(q, executed)

    out = normalize_pi05_native_sample(sample, stats)
    data = out.raw_model_data

    assert data["state"].dtype == np.float32
    assert data["actions"].dtype == np.float32
    assert np.isfinite(data["state"]).all()
    assert np.isfinite(data["actions"]).all()
    # Official formula, no clipping, no invented variance.
    assert data["state"][0] == pytest.approx(-1.0)
    assert data["state"][1] > 1.0
    assert data["state"][2] == pytest.approx(-1.0)
    assert data["actions"][0, 0] == pytest.approx(-1.0)
    assert data["actions"][24, 1] > 1.0

    report = out.provenance["normalization"]
    assert report["epsilon"] == _EPSILON
    assert report["state"]["constant_channels"] == [0]
    assert report["state"]["below_epsilon_channels"] == [2]
    assert report["state"]["out_of_range_counts"][1] == 1
    assert report["state"]["out_of_range_counts"][0] == 0
    assert report["actions"]["constant_channels"] == [0]
    assert report["actions"]["below_epsilon_channels"] == [2]
    assert report["actions"]["out_of_range_counts"][1] == HORIZON
    assert len(report["actions"]["out_of_range_counts"]) == NUM_CHANNELS
    assert len(report["stats"]["stats_sha256"]) == 64
    assert report["stats"]["corpus_ids"] == [_HAND_CORPUS_ID]
    assert report["stats"]["index_sha256"] == [_HAND_INDEX_SHA]

    # Pending flags: only normalization flips; everything else is retained.
    assert out.provenance["pending_transforms"]["normalization"] == "APPLIED"
    for name, flag in out.provenance["pending_transforms"].items():
        if name != "normalization":
            assert flag == "PENDING"


def test_normalize_matches_official_formula_exactly():
    stats = _hand_stats()
    q = np.linspace(-2.0, 3.0, NUM_CHANNELS).astype(np.float32)
    rng = np.random.default_rng(3)
    executed = rng.normal(0.0, 4.0, size=(HORIZON, NUM_CHANNELS)).astype(np.float32)
    sample = _native_sample(q, executed)

    out = normalize_pi05_native_sample(sample, stats)

    def official(x, g):
        q01 = np.asarray(stats[g]["q01"], dtype=np.float64)
        q99 = np.asarray(stats[g]["q99"], dtype=np.float64)
        return ((x.astype(np.float64) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(
            np.float32
        )

    np.testing.assert_array_equal(out.raw_model_data["state"], official(q, "state"))
    np.testing.assert_array_equal(
        out.raw_model_data["actions"], official(executed, "actions")
    )


def test_normalize_preserves_and_isolates_copies():
    stats = _hand_stats()
    q = np.linspace(-0.5, 0.5, NUM_CHANNELS).astype(np.float32)
    executed = np.full((HORIZON, NUM_CHANNELS), 0.1, dtype=np.float32)
    sample = _native_sample(q, executed)
    pristine_rgb = sample.raw_model_data["image"]["base_0_rgb"].copy()
    pristine_prompt = sample.raw_model_data["prompt"]
    pristine_loss = sample.loss_mask32.copy()
    pristine_raw_state = sample.raw_model_data["state"].copy()
    pristine_raw_actions = sample.raw_model_data["actions"].copy()

    out = normalize_pi05_native_sample(sample, stats)

    # Content preserved; containers independent.
    np.testing.assert_array_equal(out.raw_model_data["image"]["base_0_rgb"], pristine_rgb)
    assert out.raw_model_data["image"]["base_0_rgb"] is not sample.raw_model_data["image"]["base_0_rgb"]
    assert out.raw_model_data["prompt"] == pristine_prompt == DEFAULT_PROMPT
    np.testing.assert_array_equal(out.loss_mask32, pristine_loss)
    assert out.loss_mask32 is not sample.loss_mask32
    assert out.provenance is not sample.provenance

    # Mutating the output must not touch the input sample...
    out.raw_model_data["state"][:] = 99.0
    out.raw_model_data["actions"][:] = 99.0
    out.raw_model_data["image"]["base_0_rgb"][:] = 0
    out.raw_model_data["image_mask"]["base_0_rgb"] = False
    out.loss_mask32[:] = False
    out.provenance["pending_transforms"]["padding"] = "FORGED"
    out.provenance["normalization"]["state"]["out_of_range_counts"][0] = 999
    np.testing.assert_array_equal(sample.raw_model_data["state"], pristine_raw_state)
    np.testing.assert_array_equal(sample.raw_model_data["actions"], pristine_raw_actions)
    np.testing.assert_array_equal(sample.raw_model_data["image"]["base_0_rgb"], pristine_rgb)
    assert sample.raw_model_data["image_mask"]["base_0_rgb"] is True
    np.testing.assert_array_equal(sample.loss_mask32, pristine_loss)
    assert sample.provenance["pending_transforms"]["normalization"] == "PENDING"
    assert sample.provenance["pending_transforms"]["padding"] == "PENDING"

    # ...and a second normalization of the untouched sample is deterministic.
    out2 = normalize_pi05_native_sample(sample, stats)
    expected_state = (
        (pristine_raw_state.astype(np.float64) - np.asarray(stats["state"]["q01"]))
        / (np.asarray(stats["state"]["q99"]) - np.asarray(stats["state"]["q01"]) + 1e-6)
        * 2.0
        - 1.0
    ).astype(np.float32)
    np.testing.assert_array_equal(out2.raw_model_data["state"], expected_state)


# --------------------------------------------------------------------------- #
# 5. malformed stats / malformed native input rejection
# --------------------------------------------------------------------------- #


def _good_stats() -> dict:
    stats = _hand_stats()
    # channel 1 range is violated by _native_sample defaults; widen it.
    for group in ("state", "actions"):
        stats[group]["q99"][1] = 10.0
    return stats


def _fresh_sample():
    q = np.linspace(-0.5, 0.5, NUM_CHANNELS).astype(np.float32)
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32)
    return _native_sample(q, executed)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.pop("actions"), "missing group"),
        (lambda s: s["state"].pop("q99"), "missing key"),
        (lambda s: s["state"].update(mean=np.zeros(NUM_CHANNELS - 1).tolist()), "must be float64\\[16\\]"),
        (lambda s: s["state"].update(std=[float("nan")] * NUM_CHANNELS), "non-finite"),
        (lambda s: s["state"].update(q99=[-2.0] * NUM_CHANNELS), "q99 < q01"),
        (lambda s: s["actions"].update(std=[-1.0] * NUM_CHANNELS), "negative std"),
    ],
)
def test_malformed_stats_rejected(mutate, match):
    stats = _good_stats()
    mutate(stats)
    with pytest.raises(VisualCorpusError, match=match):
        normalize_pi05_native_sample(_fresh_sample(), stats)


def test_non_mapping_stats_rejected():
    with pytest.raises(VisualCorpusError, match="must be a mapping"):
        normalize_pi05_native_sample(_fresh_sample(), [("state", {})])


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.raw_model_data.__setitem__("state", s.raw_model_data["state"].astype(np.float64)), "float32"),
        (lambda s: s.raw_model_data["state"].__setitem__(0, np.nan), "non-finite"),
        (lambda s: s.raw_model_data.__setitem__("actions", np.zeros((HORIZON - 1, NUM_CHANNELS), dtype=np.float32)), "float32"),
        (lambda s: s.raw_model_data["actions"].__setitem__((0, 0), float("inf")), "non-finite"),
        (lambda s: s.raw_model_data["image"].__setitem__("base_0_rgb", np.zeros((PNG_HEIGHT, PNG_WIDTH, 3), dtype=np.float32)), "uint8"),
        (lambda s: s.raw_model_data["image"].__setitem__("base_0_rgb", np.zeros((100, 100, 3), dtype=np.uint8)), "uint8"),
        (lambda s: s.raw_model_data["image_mask"].__setitem__("base_0_rgb", 1), "boolean"),
        (lambda s: s.raw_model_data.__setitem__("prompt", ""), "prompt"),
        (lambda s: object.__setattr__(s, "loss_mask32", np.zeros(ACTION_DIM, dtype=np.int64)), "loss_mask32"),
        (lambda s: object.__setattr__(s, "loss_mask32", np.ones(ACTION_DIM, dtype=bool)), "first 16"),
        (lambda s: object.__setattr__(s, "loss_mask32", np.r_[np.ones(NUM_CHANNELS - 1, dtype=bool), np.zeros(ACTION_DIM - NUM_CHANNELS + 1, dtype=bool)]), "first 16"),
        (lambda s: object.__setattr__(s, "loss_mask32", np.r_[np.zeros(NUM_CHANNELS, dtype=bool), np.ones(ACTION_DIM - NUM_CHANNELS, dtype=bool)]), "first 16"),
        (lambda s: s.raw_model_data["image_mask"].__setitem__("base_0_rgb", False), "base_0_rgb"),
        (lambda s: s.raw_model_data["image_mask"].__setitem__("left_wrist_0_rgb", True), "left_wrist_0_rgb"),
        (lambda s: s.raw_model_data["image"]["right_wrist_0_rgb"].__setitem__((0, 0, 0), 1), "all-zero"),
        (lambda s: s.provenance.__delitem__("pending_transforms"), "pending_transforms"),
        (lambda s: s.provenance["pending_transforms"].__setitem__("normalization", "APPLIED"), "twice"),
    ],
)
def test_malformed_native_sample_rejected(mutate, match):
    sample = _fresh_sample()
    mutate(sample)
    with pytest.raises(VisualCorpusError, match=match):
        normalize_pi05_native_sample(sample, _good_stats())


def test_non_sample_input_rejected():
    with pytest.raises(VisualCorpusError, match="Pi05NativeSample"):
        normalize_pi05_native_sample(object(), _good_stats())


def test_repeat_normalization_rejected():
    stats = _good_stats()
    sample = _fresh_sample()
    out = normalize_pi05_native_sample(sample, stats)
    with pytest.raises(VisualCorpusError, match="refusing to normalize twice"):
        normalize_pi05_native_sample(out, stats)


def _bound_pair(tmp_path: Path):
    """Fitted stats over two corpora plus a sample bound to the first."""
    row_a, row_b = _train_rows()
    first = _corpus(tmp_path, "bound1", "bound-corpus", [row_a])
    second = _corpus(tmp_path, "bound2", "second-corpus", [row_b])
    stats = fit_pi05_norm_stats([first, second])
    q = np.linspace(-0.5, 0.5, NUM_CHANNELS).astype(np.float32)
    sample = _native_sample(
        q,
        np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32),
        provenance_extra={"corpus_id": "bound-corpus", "index_sha256": first.index_sha256},
    )
    return stats, sample, first, second


def test_normalize_with_fitted_stats_binds_provenance(tmp_path: Path):
    stats, sample, first, second = _bound_pair(tmp_path)
    out = normalize_pi05_native_sample(sample, stats)

    identity = out.provenance["normalization"]["stats"]
    assert identity["corpus_ids"] == ["bound-corpus", "second-corpus"]
    assert identity["index_sha256"] == [first.index_sha256, second.index_sha256]
    assert len(identity["stats_sha256"]) == 64
    assert out.provenance["normalization"]["bound_to_sample_provenance"] is True

    # Re-labelling the OTHER corpus in the fit metadata changes the identity
    # provenance while the numeric digest stays unchanged.
    relabelled = copy.deepcopy(stats)
    relabelled["metadata"]["corpus_ids"][1] = "renamed-corpus"
    identity2 = normalize_pi05_native_sample(sample, relabelled).provenance["normalization"]["stats"]
    assert identity2["corpus_ids"] == ["bound-corpus", "renamed-corpus"]
    assert identity2["stats_sha256"] == identity["stats_sha256"]


def test_normalize_binding_mismatch_rejected(tmp_path: Path):
    stats, sample, first, _ = _bound_pair(tmp_path)

    other = _native_sample(
        np.zeros(NUM_CHANNELS, dtype=np.float32),
        np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32),
        provenance_extra={"corpus_id": "other-corpus", "index_sha256": first.index_sha256},
    )
    with pytest.raises(VisualCorpusError, match="not among the fitted corpora"):
        normalize_pi05_native_sample(other, stats)

    wrong_sha = _native_sample(
        np.zeros(NUM_CHANNELS, dtype=np.float32),
        np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float32),
        provenance_extra={"corpus_id": "bound-corpus", "index_sha256": "ff" * 32},
    )
    with pytest.raises(VisualCorpusError, match="does not match the SHA"):
        normalize_pi05_native_sample(wrong_sha, stats)

    unbound_sample = _fresh_sample()
    del unbound_sample.provenance["corpus_id"]
    del unbound_sample.provenance["index_sha256"]
    with pytest.raises(VisualCorpusError, match="cannot bind fitted stats"):
        normalize_pi05_native_sample(unbound_sample, stats)

    misaligned = copy.deepcopy(stats)
    misaligned["metadata"]["index_sha256"] = misaligned["metadata"]["index_sha256"][:1]
    with pytest.raises(VisualCorpusError, match="are misaligned"):
        normalize_pi05_native_sample(sample, misaligned)


# --------------------------------------------------------------------------- #
# host-local: the two authorized runtime corpora (skipped when absent)
# --------------------------------------------------------------------------- #

_AUTHORIZED = (
    (
        ".local/v1-visual/pilot-audit-integrated-20261101",
        "1553eb0675469ce89aac110d25d4b344caea7202145433d9f207228ea4c2ece1",
    ),
    (
        ".local/v1-visual/diagonal-ft01-audit-20261101",
        "825ceeb7f8a5f384f8c85c211f11a2e5a6cde1cd2e5f0a50b4fff5dbe5247eb4",
    ),
)


def _authorized_present() -> bool:
    return all(
        (_REPO_ROOT / rel / "sample_index.jsonl").is_file() for rel, _ in _AUTHORIZED
    )


@pytest.mark.skipif(not _authorized_present(), reason="authorized local v1-visual corpora not on this host")
def test_authorized_local_corpora_fit_counts_and_roundtrip():
    corpora = [
        VisualCorpus(
            index_path=_REPO_ROOT / rel / "sample_index.jsonl",
            evidence_root=_REPO_ROOT,
            expected_index_sha256=sha,
            corpus_id=rel.rsplit("/", 1)[-1],
        )
        for rel, sha in _AUTHORIZED
    ]
    stats = fit_pi05_norm_stats(corpora)

    meta = stats["metadata"]
    assert meta["train_rows_total"] == 453 + 254
    assert meta["state_rows_total"] == 707
    assert meta["action_steps_total"] == 17675  # 707 train rows x 25 fine steps
    assert meta["train_rows_per_corpus"] == [453, 254]
    assert meta["index_sha256"] == [sha for _, sha in _AUTHORIZED]
    json.dumps(stats)
    for group in ("state", "actions"):
        assert set(stats[group].keys()) == {"mean", "std", "q01", "q99"}
        for vec in stats[group].values():
            assert len(vec) == NUM_CHANNELS
            assert all(math.isfinite(v) for v in vec)

    corpus = corpora[0]
    train_idx = corpus.split_indices("train")[0]
    try:
        example = corpus.example(train_idx)
    except VisualCorpusError as exc:
        pytest.skip(f"referenced PNGs unavailable for decode roundtrip: {exc}")
    native = prepare_pi05_native_sample(example)
    out = normalize_pi05_native_sample(native, stats)

    assert out.raw_model_data["state"].dtype == np.float32
    assert out.raw_model_data["actions"].dtype == np.float32
    assert np.isfinite(out.raw_model_data["state"]).all()
    assert np.isfinite(out.raw_model_data["actions"]).all()
    assert out.provenance["pending_transforms"]["normalization"] == "APPLIED"
    assert out.provenance["pending_transforms"]["padding"] == "PENDING"

    q01 = np.asarray(stats["state"]["q01"], dtype=np.float64)
    q99 = np.asarray(stats["state"]["q99"], dtype=np.float64)
    expected = (
        (example.observation["q_rad16"].astype(np.float64) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    ).astype(np.float32)
    np.testing.assert_array_equal(out.raw_model_data["state"], expected)
