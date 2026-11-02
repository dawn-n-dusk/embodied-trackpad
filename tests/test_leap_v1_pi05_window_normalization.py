"""Focused fit tests for the 20 ms window normalization wrapper (synthetic corpora).

Group 1 fits a real ``VisualCorpus`` built from 11 contiguous legacy-profile
train rows (two overlapping 0.5 s windows) and checks state/actions stats
against an independent pure-Python reference, the explicit 20 ms metadata and
the ordered provenance digest. Group 2 proves complete extreme
validation/test window chains cannot change fitted train statistics.
``VisualCorpus.example`` is monkeypatched to fail in every test: fitting must
never decode images. Negative guard tests live in a later task.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1 import small_policy
from embodied_trackpad.leap_v1.pi05_window_normalization import (
    fit_pi05_window_norm_stats,
)
from embodied_trackpad.leap_v1.visual_dataset import (
    NUM_CHANNELS,
    VisualCorpus,
)

_DT = 0.002
_PNG_SHA = "ab" * 32
# Verified legacy axial split mapping (target_class == episode_id % 12).
_SPLIT_OF = {0: "train", 1: "validation", 2: "train", 3: "train", 4: "test",
             5: "train", 6: "train", 7: "validation", 8: "train", 9: "train",
             10: "test", 11: "train"}


def _ref_mean(xs):
    return sum(xs) / len(xs)


def _ref_std(xs):
    m = _ref_mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _ref_quantile(xs, p):
    s = sorted(xs)
    pos = (len(s) - 1) * p
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (pos - lo) * (s[hi] - s[lo])


def _train_value(step, c):
    return 0.5 if c == 7 else step * 1e-3 + c * 1e-4  # channel 7 constant


def _train_q(k):
    return [0.75 if c == 5 else k + 0.25 * c for c in range(NUM_CHANNELS)]  # ch 5 constant


def _row(episode_id, k, split, q, executed, prev):
    first = 25 * k
    return {
        "episode_id": episode_id,
        "frame_id": k,
        "split": split,
        "target_class": episode_id % 12,
        "actor_input": {
            "png_ref": f"frames/ep-{episode_id:03d}-f{k:03d}.png",
            "png_sha256": _PNG_SHA,
            "error_xy_px": [0.0, 0.0],
            "q_rad16": [float(v) for v in q],
            "qvel_rad_s16": [0.0] * NUM_CHANNELS,
            "previous_ctrl_rad16": [float(v) for v in prev],
            "valid": {
                "png_valid": True, "visual_valid": True, "q_valid": True,
                "qvel_valid": True, "previous_ctrl_valid": True,
                "fsr_valid4": [False, False, False, False],
                "motor_current_valid16": [False] * NUM_CHANNELS,
                "wrist_valid": False,
            },
            "fsr_N4": None,
            "motor_current_A16": None,
            "wrist": None,
        },
        "action_window": {
            "dt_s": _DT,
            "first_step_index": first,
            "end_exclusive_step_index": first + 25,
            "consumed_prev_step_index": first - 1,
            "consumed_prev_sim_time_s": (first - 1) * _DT,
            "first_sim_time_s": first * _DT,
            "end_sim_time_s": (first + 24) * _DT,
        },
        "labels": {"executed_ctrl_rad": [[float(v) for v in step] for step in executed]},
    }


def _chain(episode_id, n, value_fn, q_fn):
    """n contiguous rows in one episode; previous ctrl = prior executed last."""
    split = _SPLIT_OF[episode_id % 12]
    rows = []
    for k in range(n):
        first = 25 * k
        executed = [[value_fn(first + t, c) for c in range(NUM_CHANNELS)]
                    for t in range(25)]
        prev = [value_fn(first - 1, c) for c in range(NUM_CHANNELS)]
        rows.append(_row(episode_id, k, split, q_fn(k), executed, prev))
    return rows


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
def no_decode(monkeypatch):
    def _boom(self, i):
        raise AssertionError("VisualCorpus.example must not be called while fitting")
    monkeypatch.setattr(VisualCorpus, "example", _boom)


def _assert_no_private_paths(stats, *corpora):
    blob = json.dumps(stats)
    for corpus in corpora:
        assert str(corpus.index_path) not in blob


def _check_group(stats, group, ref_channels):
    for c in range(NUM_CHANNELS):
        xs = ref_channels(c)
        assert stats[group]["mean"][c] == pytest.approx(_ref_mean(xs)), (group, c)
        assert stats[group]["std"][c] == pytest.approx(_ref_std(xs)), (group, c)
        assert stats[group]["q01"][c] == pytest.approx(_ref_quantile(xs, 0.01)), (group, c)
        assert stats[group]["q99"][c] == pytest.approx(_ref_quantile(xs, 0.99)), (group, c)


def test_fit_two_overlapping_train_windows_hand_computed(tmp_path, no_decode):
    corpus = _corpus(tmp_path, "train", "train-corpus",
                     _chain(0, 11, _train_value, _train_q))
    stats = fit_pi05_window_norm_stats([corpus])

    meta = stats["metadata"]
    assert meta["train_windows_total"] == 2
    assert meta["state_rows_total"] == 2  # one state per window start row
    assert meta["action_refs_total"] == 50  # 2 windows x 25 coarse refs
    assert meta["train_windows_per_corpus"] == [2]

    # state: window starts are rows 0 and 1 (episode 0).
    _check_group(stats, "state", lambda c: [_train_q(0)[c], _train_q(1)[c]])
    # actions: coarse refs are fine[9::10] over steps 0..249 and 25..274.
    def action_ref(c):
        return [_train_value(s0 + 10 * j + 9, c)
                for s0 in (0, 25) for j in range(25)]
    _check_group(stats, "actions", action_ref)

    assert meta["constant_channel_indices"]["state"] == [5]
    assert meta["constant_channel_indices"]["actions"] == [7]
    assert meta["format_version"] == "leap-v1-pi05-window-norm-v1"
    assert meta["fit_split"] == "train"
    assert meta["native_dt_s"] == 0.002
    assert meta["target_dt_s"] == 0.02
    assert meta["horizon"] == 25
    assert meta["horizon_s"] == 0.5
    assert meta["action_semantics"] == "absolute_executed_joint_reference_rad"
    assert meta["corpus_ids"] == ["train-corpus"]
    assert meta["index_sha256"] == [corpus.index_sha256]

    # Ordered provenance digest, recomputed independently from the contract.
    records = [
        {"corpus_id": "train-corpus", "index_sha256": corpus.index_sha256,
         "episode_id": 0, "frame_id": 0, "first_step_index": 0,
         "row_indices": list(range(10))},
        {"corpus_id": "train-corpus", "index_sha256": corpus.index_sha256,
         "episode_id": 0, "frame_id": 1, "first_step_index": 25,
         "row_indices": list(range(1, 11))},
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    assert meta["train_window_provenance_sha256"] == hashlib.sha256(
        payload.encode("utf-8")).hexdigest()

    json.dumps(stats)  # serializable as-is
    _assert_no_private_paths(stats, corpus)


def test_extreme_eval_windows_do_not_change_train_stats(tmp_path, no_decode):
    train_only = _corpus(tmp_path, "trainonly", "mixed-corpus",
                         _chain(0, 11, _train_value, _train_q))
    stats_ref = fit_pi05_window_norm_stats([train_only])

    val_rows = _chain(1, 10, lambda s, c: 1e6, lambda k: [1e6] * NUM_CHANNELS)
    test_rows = _chain(4, 10, lambda s, c: -1e6, lambda k: [-1e6] * NUM_CHANNELS)
    mixed = _corpus(tmp_path, "mixed", "mixed-corpus",
                    _chain(0, 11, _train_value, _train_q) + val_rows + test_rows)
    stats = fit_pi05_window_norm_stats([mixed])

    assert stats["metadata"]["train_windows_total"] == 2  # eval windows excluded
    for group in ("state", "actions"):
        for key in ("mean", "std", "q01", "q99"):
            assert stats[group][key] == pytest.approx(stats_ref[group][key]), (group, key)
    assert stats["metadata"]["index_sha256"] != stats_ref["metadata"]["index_sha256"]
    assert (stats["metadata"]["train_window_provenance_sha256"]
            != stats_ref["metadata"]["train_window_provenance_sha256"])
    json.dumps(stats)
    _assert_no_private_paths(stats, mixed)


# --------------------------------------------------------------------------- #
# 3. negative guards: fail closed on bad input or corrupted index identity
# --------------------------------------------------------------------------- #


def test_input_guards(tmp_path, no_decode):
    corpus = _corpus(tmp_path, "g1", "guard-corpus", _chain(0, 11, _train_value, _train_q))
    with pytest.raises(ValueError):
        fit_pi05_window_norm_stats([])
    with pytest.raises(ValueError):
        fit_pi05_window_norm_stats([object()])
    with pytest.raises(ValueError):
        fit_pi05_window_norm_stats([corpus, corpus])  # duplicate corpus_id

    alias_evidence = tmp_path / "g1-alias-evidence"
    alias_evidence.mkdir()
    alias = VisualCorpus(  # same index bytes, different valid corpus id
        index_path=corpus.index_path,
        evidence_root=alias_evidence,
        expected_index_sha256=corpus.index_sha256,
        corpus_id="renamed-corpus",
    )
    with pytest.raises(ValueError):
        fit_pi05_window_norm_stats([corpus, alias])  # duplicate index SHA

    # Index bytes modified after corpus construction: fit-side re-hash fails closed.
    with open(corpus.index_path, "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError):
        fit_pi05_window_norm_stats([corpus])


def test_no_complete_train_window_rejected(tmp_path, no_decode):
    nine = _corpus(tmp_path, "g2", "nine-corpus", _chain(0, 9, _train_value, _train_q))
    with pytest.raises(ValueError):
        fit_pi05_window_norm_stats([nine])  # 9 rows < 10: no full window
    val_only = _corpus(
        tmp_path, "g3", "val-corpus",
        _chain(1, 10, lambda s, c: 1e6, lambda k: [1e6] * NUM_CHANNELS),
    )
    with pytest.raises(ValueError):
        fit_pi05_window_norm_stats([val_only])  # complete windows, all validation


@pytest.mark.parametrize("mutate", ["nan", "string"])
def test_nonfinite_or_nonnumeric_train_q_rejected(tmp_path, no_decode, mutate):
    rows = _chain(0, 11, _train_value, _train_q)
    bad = []
    for i, r in enumerate(rows):
        r = dict(r)
        r["actor_input"] = dict(r["actor_input"])
        if i == 0:
            r["actor_input"]["q_rad16"] = (
                [float("nan")] * NUM_CHANNELS
                if mutate == "nan" else ["x"] * NUM_CHANNELS
            )
        bad.append(r)
    with pytest.raises(ValueError):  # constructor validators or the fitter may catch it
        corpus = _corpus(tmp_path, f"g4-{mutate}", f"bad-q-{mutate}", bad)
        fit_pi05_window_norm_stats([corpus])


@pytest.mark.parametrize(
    "states,actions",
    [
        (np.zeros((1, 16)), np.full((1, 25, 16), np.nan)),  # nonfinite actions
        (np.zeros((0, 16)), np.zeros((0, 25, 16))),  # zero windows
        (np.zeros((2, 16)), np.zeros((3, 25, 16))),  # N mismatch
        (np.zeros((1, 15)), np.zeros((1, 25, 16))),  # wrong state shape
        (np.zeros((1, 16)), np.zeros((1, 24, 16))),  # wrong action shape
    ],
)
def test_fit_window_arrays_guards(states, actions):
    from embodied_trackpad.leap_v1.pi05_window_normalization import _fit_window_arrays

    with pytest.raises(ValueError):
        _fit_window_arrays(states, actions)
