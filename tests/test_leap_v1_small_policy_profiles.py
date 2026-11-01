"""Profile-aware split/index tests for the LEAP v1 small policy.

Covers the scoped ``near_diagonal_v1`` task profile (8 classes, whole
episodes, shared collection) against ``load_sample_index`` and
``split_by_target_class``. Synthetic fixtures only; legacy
``near_pointing_v1`` behaviour is owned by tests/test_leap_v1_small_policy.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1.small_policy import (
    HORIZON,
    INPUT_DIM,
    NUM_CHANNELS,
    SmallPolicyError,
    compute_split_counts,
    extract_features,
    load_sample_index,
    split_by_target_class,
)

PROFILE = "near_diagonal_v1"
COLLECTION = "diag-pilot-a"
SCOPED_SPLIT = {
    0: "train", 1: "train", 2: "train", 3: "train",
    4: "validation", 5: "test", 6: "validation", 7: "test",
}


def _actor() -> dict:
    return {
        "error_xy_px": [8.0, 4.0],
        "q_rad16": [0.01] * NUM_CHANNELS,
        "qvel_rad_s16": [0.0] * NUM_CHANNELS,
        "previous_ctrl_rad16": [0.0] * NUM_CHANNELS,
    }


def _labels() -> dict:
    prev = np.zeros(NUM_CHANNELS, dtype=np.float64)
    executed = np.zeros((HORIZON, NUM_CHANNELS), dtype=np.float64)
    executed[:, 0] = 1e-5 * np.sin(np.linspace(0, np.pi, HORIZON))
    return {
        "executed_ctrl_rad": executed.tolist(),
        "delta_from_frame_previous_ctrl_rad": (executed - prev).tolist(),
    }


def _window() -> dict:
    return {"first_step_index": 1, "end_exclusive_step_index": 26, "dt_s": 0.002}


def _scoped_record(seed: int = 0, *, split: str | None = None, collection_id=COLLECTION,
                   frame_id: int = 1, **overrides) -> dict:
    rec = {
        "episode_id": seed,
        "frame_id": frame_id,
        "split": split if split is not None else SCOPED_SPLIT[seed],
        "target_class": seed,
        "task_profile": PROFILE,
        "collection_id": collection_id,
        "target_key": f"{PROFILE}:{seed}",
        "episode_key": f"{collection_id}:{PROFILE}:seed-{seed:03d}",
        "actor_input": _actor(),
        "action_window": _window(),
        "labels": _labels(),
    }
    rec.update(overrides)
    return rec


def _legacy_record(target_class: int = 0, *, split: str = "train",
                   episode_id: int = 0, frame_id: int = 1) -> dict:
    return {
        "episode_id": episode_id,
        "frame_id": frame_id,
        "split": split,
        "target_class": target_class,
        "actor_input": _actor(),
        "action_window": _window(),
        "labels": _labels(),
    }


def _write_index(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path


def _assert_both_paths_reject(tmp_path: Path, rec: dict, name: str = "bad") -> None:
    """Direct split_by_target_class and the loader must reject identically."""
    with pytest.raises(SmallPolicyError):
        split_by_target_class([rec])
    path = _write_index(tmp_path / f"{name}.jsonl", [rec])
    with pytest.raises(SmallPolicyError):
        load_sample_index(path)


def test_scoped_index_valid_8_classes(tmp_path: Path):
    recs = [_scoped_record(s, frame_id=i + 1) for i, s in enumerate(range(8))]
    loaded = load_sample_index(_write_index(tmp_path / "index.jsonl", recs))
    assert len(loaded) == 8
    assert compute_split_counts(loaded) == {"train": 4, "validation": 2, "test": 2}
    for rec in loaded:
        assert extract_features(rec).shape == (INPUT_DIM,)


def test_scoped_metadata_not_in_features():
    r1 = _scoped_record(0)
    r2 = _scoped_record(0, collection_id="diag-pilot-b")
    r3 = _scoped_record(0, target_key="near_diagonal_v1:0", episode_key="x:y:seed-000")
    assert np.allclose(extract_features(r1), extract_features(r2))
    assert np.allclose(extract_features(r1), extract_features(r3))


def test_split_by_target_class_scoped_direct():
    splits = split_by_target_class([_scoped_record(s) for s in range(8)])
    assert {r["episode_id"] for r in splits["train"]} == {0, 1, 2, 3}
    assert {r["episode_id"] for r in splits["validation"]} == {4, 6}
    assert {r["episode_id"] for r in splits["test"]} == {5, 7}


def test_wrong_split_for_seed_rejected(tmp_path: Path):
    # Seed 4 is validation; declaring train must fail in both paths.
    _assert_both_paths_reject(tmp_path, _scoped_record(4, split="train"), "wrong_split")


def test_unknown_profile_rejected(tmp_path: Path):
    rec = _scoped_record(0)
    rec["task_profile"] = "side_ways_v9"
    _assert_both_paths_reject(tmp_path, rec, "unknown_profile")


def test_wrong_target_key_rejected(tmp_path: Path):
    _assert_both_paths_reject(
        tmp_path, _scoped_record(0, target_key="near_diagonal_v1:99"), "wrong_target_key")


def test_wrong_episode_key_rejected(tmp_path: Path):
    _assert_both_paths_reject(
        tmp_path, _scoped_record(0, episode_key="diag-pilot-a:near_diagonal_v1:seed-001"),
        "wrong_episode_key")


def test_missing_collection_rejected(tmp_path: Path):
    rec = _scoped_record(0)
    del rec["collection_id"]
    _assert_both_paths_reject(tmp_path, rec, "missing_collection")


def test_target_class_must_equal_seed(tmp_path: Path):
    _assert_both_paths_reject(tmp_path, _scoped_record(4, target_class=5), "tc_ne_seed")


def test_bool_and_string_seed_rejected(tmp_path: Path):
    _assert_both_paths_reject(tmp_path, _scoped_record(0, episode_id=True), "bool_seed")
    _assert_both_paths_reject(tmp_path, _scoped_record(0, episode_id="4"), "string_seed")


def test_duplicate_scoped_frame_rejected(tmp_path: Path):
    recs = [_scoped_record(0, frame_id=1), _scoped_record(0, frame_id=1)]
    path = _write_index(tmp_path / "dup.jsonl", recs)
    with pytest.raises(SmallPolicyError, match="duplicate"):
        load_sample_index(path)
    # Same seed with distinct frame ids is fine.
    recs = [_scoped_record(0, frame_id=1), _scoped_record(0, frame_id=2)]
    assert len(load_sample_index(_write_index(tmp_path / "ok.jsonl", recs))) == 2


def test_mixed_collections_rejected(tmp_path: Path):
    recs = [_scoped_record(0), _scoped_record(1, collection_id="diag-pilot-b")]
    path = _write_index(tmp_path / "mixed_coll.jsonl", recs)
    with pytest.raises(SmallPolicyError, match="mixed"):
        load_sample_index(path)


def test_mixed_profiles_rejected(tmp_path: Path):
    scoped = _scoped_record(0)
    axial_profile = _legacy_record(2, split="train")
    axial_profile["task_profile"] = "near_pointing_v1"
    path = _write_index(tmp_path / "mixed_prof.jsonl", [scoped, axial_profile])
    with pytest.raises(SmallPolicyError, match="mixed"):
        load_sample_index(path)


def test_mixed_scoped_and_legacy_rejected(tmp_path: Path):
    recs = [_scoped_record(0), _legacy_record(0, split="train")]
    path = _write_index(tmp_path / "mixed_legacy.jsonl", recs)
    with pytest.raises(SmallPolicyError, match="mixed"):
        load_sample_index(path)


def test_legacy_record_with_scoped_keys_rejected(tmp_path: Path):
    rec = _legacy_record(0, split="train")
    rec["collection_id"] = COLLECTION
    _assert_both_paths_reject(tmp_path, rec, "legacy_scoped_keys")


def test_legacy_records_still_load_unchanged(tmp_path: Path):
    recs = [
        _legacy_record(0, split="train", episode_id=0),
        _legacy_record(1, split="validation", episode_id=1),
        _legacy_record(4, split="test", episode_id=4),
    ]
    loaded = load_sample_index(_write_index(tmp_path / "legacy.jsonl", recs))
    assert compute_split_counts(loaded) == {"train": 1, "validation": 1, "test": 1}
    path = _write_index(
        tmp_path / "legacy_dup.jsonl",
        [_legacy_record(0, split="train"), _legacy_record(0, split="train")],
    )
    with pytest.raises(SmallPolicyError, match="duplicate"):
        load_sample_index(path)
