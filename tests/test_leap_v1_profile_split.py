"""Focused tests for the profile-scoped split_of helper.

No episode files, no auditor runs: pure mapping semantics only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC = str(REPO_ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from embodied_trackpad.leap_v1 import pilot_dataset as pd  # noqa: E402


def test_default_profile_matches_legacy_axial_table():
    expected = {}
    for s in (0, 2, 3, 5, 6, 8, 9, 11):
        expected[s] = "train"
    for s in (1, 7):
        expected[s] = "validation"
    for s in (4, 10):
        expected[s] = "test"
    for seed in range(12):
        assert pd.split_of(seed) == expected[seed]  # implicit default profile
        assert pd.split_of(seed, task_profile="near_pointing_v1") == expected[seed]


def test_legacy_sets_preserved_unchanged():
    assert pd.TRAIN_SEEDS == {0, 2, 3, 5, 6, 8, 9, 11}
    assert pd.VAL_SEEDS == {1, 7}
    assert pd.TEST_SEEDS == {4, 10}
    assert pd.DEFAULT_TASK_PROFILE == "near_pointing_v1"


def test_axial_invalid_seeds_rejected_both_call_forms():
    for bad in (-1, 12, 100):
        assert pd.split_of(bad) == "rejected"
        assert pd.split_of(bad, task_profile="near_pointing_v1") == "rejected"


def test_diagonal_groups_all_eight_seeds():
    expected = {}
    for s in (0, 1, 2, 3):
        expected[s] = "train"
    for s in (4, 6):
        expected[s] = "validation"
    for s in (5, 7):
        expected[s] = "test"
    for seed in range(8):
        assert pd.split_of(seed, task_profile="near_diagonal_v1") == expected[seed]


def test_diagonal_invalid_seeds_rejected():
    for bad in (-1, 8, 9, 100):
        assert pd.split_of(bad, task_profile="near_diagonal_v1") == "rejected"


def test_unknown_profile_rejected_not_silently_axial():
    # Seed 0 is axial-train; under an unknown profile it must NOT fall back.
    assert pd.split_of(0, task_profile="near_diagonal_v1_EXTRA") == "rejected"
    assert pd.split_of(4, task_profile="bogus") == "rejected"
    assert pd.split_of(0, task_profile="") == "rejected"


def test_seed_collision_across_profiles_is_scoped():
    # Same seed, different profile -> different split; no silent merge.
    assert pd.split_of(4) == "test"                      # axial
    assert pd.split_of(4, task_profile="near_diagonal_v1") == "validation"
    assert pd.split_of(5) == "train"                     # axial
    assert pd.split_of(5, task_profile="near_diagonal_v1") == "test"
    assert pd.split_of(7) == "validation"                # axial
    assert pd.split_of(7, task_profile="near_diagonal_v1") == "test"
