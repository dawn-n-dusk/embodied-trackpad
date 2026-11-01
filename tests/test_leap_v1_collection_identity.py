"""Pure-helper tests for near_diagonal_v1 collection identity validation.

No PNG fixtures, no filesystem, no real episodes. Documents are plain dicts.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from embodied_trackpad.leap_v1.pilot_collection import (  # noqa: E402
    validate_diagonal_collection_identity,
)
from embodied_trackpad.leap_v1.pilot_dataset import split_of  # noqa: E402

DIAG = "near_diagonal_v1"
COLL = "diagdev-A_1"
GOOD_COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
OTHER_COMMIT = "f" * 40


def _index():
    return [{"seed": s, "group": split_of(s, task_profile=DIAG)}
            for s in range(8)]


def _docs():
    manifest = {
        "task_profile": DIAG,
        "collection_id": COLL,
        "source_commit": GOOD_COMMIT,
        "seeds_attempt_order": list(range(8)),
        "prospective_split": {
            "train": [0, 1, 2, 3],
            "validation": [4, 6],
            "test": [5, 7],
            "prospective_index": _index(),
        },
        "prospective_index": _index(),
        "extra_metadata_ignored": {"anything": True},
    }
    batch = {
        "task_profile": DIAG,
        "collection_id": COLL,
        "source_commit": GOOD_COMMIT,
        "attempted_seeds": list(range(8)),
        "successful_seeds": list(range(8)),
        "prospective_index": _index(),
    }
    acceptance = {
        "collection_id": COLL,
        "source_commit": GOOD_COMMIT,
        "config": {"task_profile": DIAG, "seeds": list(range(8))},
        "prospective_split": {"train": [0, 1, 2, 3],
                              "validation": [4, 6], "test": [5, 7]},
    }
    return manifest, batch, acceptance


def _validate(manifest, batch, acceptance, coll=COLL):
    return validate_diagonal_collection_identity(manifest, batch, acceptance,
                                                 coll)


# --- valid shapes ------------------------------------------------------------

def test_valid_complete_collection_no_issues():
    assert _validate(*_docs()) == []


@pytest.mark.parametrize("successful", [[], [0], [0, 3, 5, 7]])
def test_valid_partial_successes_allowed(successful):
    manifest, batch, acceptance = _docs()
    batch["successful_seeds"] = successful
    assert _validate(manifest, batch, acceptance) == []


def test_valid_split_order_may_differ():
    manifest, batch, acceptance = _docs()
    manifest["prospective_split"]["train"] = [3, 1, 0, 2]
    acceptance["prospective_split"]["validation"] = [6, 4]
    assert _validate(manifest, batch, acceptance) == []


def test_valid_single_prospective_index_location():
    manifest, batch, acceptance = _docs()
    # only the nested manifest.prospective_split.prospective_index remains
    del manifest["prospective_index"]
    del batch["prospective_index"]
    assert _validate(manifest, batch, acceptance) == []


# --- identity mismatches ------------------------------------------------------

@pytest.mark.parametrize("which", ["manifest", "batch", "acceptance"])
def test_wrong_task_profile_rejected(which):
    manifest, batch, acceptance = _docs()
    if which == "manifest":
        manifest["task_profile"] = "near_pointing_v1"
    elif which == "batch":
        batch["task_profile"] = "near_pointing_v1"
    else:
        acceptance["config"]["task_profile"] = "near_pointing_v1"
    issues = _validate(manifest, batch, acceptance)
    assert any("task_profile" in i for i in issues)


@pytest.mark.parametrize("which", ["manifest", "batch", "acceptance"])
def test_wrong_collection_id_rejected(which):
    manifest, batch, acceptance = _docs()
    if which == "manifest":
        manifest["collection_id"] = "other"
    elif which == "batch":
        batch["collection_id"] = "other"
    else:
        acceptance["collection_id"] = "other"
    issues = _validate(manifest, batch, acceptance)
    assert any("collection_id" in i for i in issues)


@pytest.mark.parametrize("bad", [None, "", "bad!id", "has space", 42])
def test_unsafe_requested_collection_id_rejected(bad):
    issues = _validate(*_docs(), coll=bad)
    assert any("not a safe identifier" in i for i in issues)


@pytest.mark.parametrize("which", ["manifest", "batch", "acceptance"])
def test_source_commit_shape_rejected(which):
    manifest, batch, acceptance = _docs()
    target = {"manifest": manifest, "batch": batch,
              "acceptance": acceptance}[which]
    target["source_commit"] = "abc123"
    issues = _validate(manifest, batch, acceptance)
    assert any("source_commit" in i and "40-hex" in i for i in issues)


def test_source_commit_mismatch_rejected():
    manifest, batch, acceptance = _docs()
    batch["source_commit"] = OTHER_COMMIT
    issues = _validate(manifest, batch, acceptance)
    assert any("source_commit mismatch" in i for i in issues)


# --- seed ordering ------------------------------------------------------------

@pytest.mark.parametrize("key,doc", [
    ("seeds_attempt_order", "manifest"),
    ("seeds", "acceptance"),
    ("attempted_seeds", "batch"),
])
def test_wrong_seed_order_rejected(key, doc):
    manifest, batch, acceptance = _docs()
    target = {"manifest": manifest, "batch": batch,
              "acceptance": acceptance["config"]}[doc]
    target[key] = [0, 0, 1, 2, 3, 4, 5, 6]
    issues = _validate(manifest, batch, acceptance)
    assert any(key in i for i in issues)


def test_bool_seed_rejected():
    manifest, batch, acceptance = _docs()
    acceptance["config"]["seeds"][3] = True
    issues = _validate(manifest, batch, acceptance)
    assert any("seeds" in i and "integer" in i for i in issues)


@pytest.mark.parametrize("bad", [[0, 9], [0, 0], [True]])
def test_bad_successful_seeds_rejected(bad):
    manifest, batch, acceptance = _docs()
    batch["successful_seeds"] = bad
    issues = _validate(manifest, batch, acceptance)
    assert any("successful_seeds" in i for i in issues)


# --- prospective split / index -------------------------------------------------

def test_mutually_swapped_manifest_acceptance_splits_rejected():
    manifest, batch, acceptance = _docs()
    manifest["prospective_split"]["train"] = [4, 6]
    manifest["prospective_split"]["validation"] = [0, 1, 2, 3]
    acceptance["prospective_split"]["train"] = [0, 1, 2, 3, 4]
    acceptance["prospective_split"]["test"] = [5, 6, 7]
    issues = _validate(manifest, batch, acceptance)
    assert any("manifest.prospective_split.train" in i for i in issues)
    assert any("acceptance.prospective_split.test" in i for i in issues)


@pytest.mark.parametrize("dup", [True, "bool-seed", 99])
def test_split_unknown_or_duplicate_seed_rejected(dup):
    manifest, batch, acceptance = _docs()
    manifest["prospective_split"]["train"] = [0, 1, 2, dup]
    issues = _validate(manifest, batch, acceptance)
    assert any("manifest.prospective_split.train" in i for i in issues)


def test_missing_all_prospective_index_rejected():
    manifest, batch, acceptance = _docs()
    del manifest["prospective_split"]["prospective_index"]
    del manifest["prospective_index"]
    del batch["prospective_index"]
    issues = _validate(manifest, batch, acceptance)
    assert any("no prospective_index" in i for i in issues)


def test_contradictory_prospective_index_rejected():
    manifest, batch, acceptance = _docs()
    manifest["prospective_index"][0] = {"seed": 0, "group": "test"}
    issues = _validate(manifest, batch, acceptance)
    assert any("manifest.prospective_index" in i for i in issues)
    # the other two locations stay valid: no blanket failure
    assert not any("batch_summary.prospective_index" in i for i in issues)
    assert not any("prospective_split.prospective_index" in i for i in issues)


def test_duplicate_seed_in_one_index_rejected():
    manifest, batch, acceptance = _docs()
    batch["prospective_index"][1] = {"seed": 0, "group": "train"}
    issues = _validate(manifest, batch, acceptance)
    assert any("batch_summary.prospective_index" in i for i in issues)


def test_incomplete_index_rejected():
    manifest, batch, acceptance = _docs()
    manifest["prospective_index"] = manifest["prospective_index"][:5]
    issues = _validate(manifest, batch, acceptance)
    assert any("cover seeds 0..7 exactly once" in i for i in issues)


# --- malformed containers / mutation ------------------------------------------

def test_malformed_containers_rejected_without_traceback():
    manifest, batch, acceptance = _docs()
    assert any("manifest" in i for i in _validate(None, batch, acceptance))
    assert any("batch_summary" in i for i in _validate(manifest, None, acceptance))
    assert any("acceptance" in i for i in _validate(manifest, batch, "x"))
    assert any("acceptance.config" in i for i in _validate(
        manifest, batch, {"source_commit": GOOD_COMMIT}))
    manifest2, batch2, _ = _docs()
    manifest2["prospective_split"] = [1, 2]
    assert any("prospective_split must be an object" in i
               for i in _validate(manifest2, batch2, _docs()[2]))
    manifest3, batch3, acc3 = _docs()
    manifest3["prospective_index"] = ["nope"]
    assert any("must be an object" in i
               for i in _validate(manifest3, batch3, acc3))


def test_inputs_not_mutated():
    manifest, batch, acceptance = _docs()
    snapshot = copy.deepcopy((manifest, batch, acceptance))
    _validate(manifest, batch, acceptance)
    assert (manifest, batch, acceptance) == snapshot
