"""Focused tests for M2a profile/collection identity in pilot_dataset.

Synthetic fixtures only: reuses _make_fixture from the pilot-data audit
tests with shallow metadata/summary profile tagging. No real episodes.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from embodied_trackpad.leap_v1 import pilot_dataset as pd  # noqa: E402
from embodied_trackpad.leap_v1.pilot_dataset import (  # noqa: E402
    EpisodeAudit, audit_episode, build_dataset_manifest, split_of,
)

_spec = importlib.util.spec_from_file_location(
    "pilot_audit_fixtures",
    REPO_ROOT / "tests" / "test_leap_v1_pilot_data_audit.py",
)
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)

DIAG = "near_diagonal_v1"
AXIAL = "near_pointing_v1"
COLL = "diagdev-A_1"
ACC = {"controller_kind": "privileged_visual_teacher",
       "config": {"planner_kind": "force-governed"}}
LEGACY_SAMPLE_KEYS = {"episode_id", "frame_id", "split", "target_class",
                      "actor_input", "action_window", "labels", "metadata",
                      "training_eligible", "pending_coordinator"}
ACTOR_INPUT_KEYS = {"png_ref", "png_sha256", "error_xy_px", "q_rad16",
                    "qvel_rad_s16", "previous_ctrl_rad16", "valid",
                    "fsr_N4", "motor_current_A16", "wrist"}


def _tagged(tmp_path, seed, profile=DIAG, collection=COLL):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ep, _ = fx._make_fixture(tmp_path, seed=seed)
    meta = json.loads((ep / "metadata.json").read_text())
    meta["task_profile"] = profile
    if collection is not None:
        meta["collection_id"] = collection
    (ep / "metadata.json").write_text(json.dumps(meta))
    summ = json.loads((ep / "summary.json").read_text())
    summ["task_profile"] = profile
    if collection is not None:
        summ["collection_id"] = collection
    (ep / "summary.json").write_text(json.dumps(summ))
    return ep


def _mk_audit(seed, collection=COLL, accepted=True):
    return EpisodeAudit(seed=seed, episode_dir=f"/fake/seed-{seed:03d}",
                        summary={"steps": 0, "wall_s": 0.0},
                        task_profile=DIAG, collection_id=collection,
                        accepted=accepted)


def _mk_sample(seed):
    return {"episode_id": seed, "frame_id": 2,
            "split": split_of(seed, task_profile=DIAG),
            "target_class": seed, "task_profile": DIAG, "collection_id": COLL,
            "target_key": f"{DIAG}:{seed}",
            "episode_key": f"{COLL}:{DIAG}:seed-{seed:03d}",
            "actor_input": {}, "action_window": {}, "labels": {}}


def _manifest(audits, samples, **kw):
    opts = {"task_profile": DIAG, "collection_id": COLL}
    opts.update(kw)
    return build_dataset_manifest(audits, samples, ACC, "m", "b", {}, **opts)


# --- valid diagonal audits: splits + scoped keys + no feature leak -----------

def test_diagonal_seeds_4_5_7_splits_and_scoped_keys(tmp_path):
    expected_split = {4: "validation", 5: "test", 7: "test"}
    for seed in (4, 5, 7):
        ep = _tagged(tmp_path / f"s{seed}", seed)
        audit, samples = audit_episode(ep, tmp_path / f"s{seed}",
                                       task_profile=DIAG, collection_id=COLL)
        assert audit.accepted and not audit.rejection_reasons
        assert audit.task_profile == DIAG and audit.collection_id == COLL
        assert samples
        for s in samples:
            assert s["split"] == expected_split[seed]
            assert s["episode_id"] == seed and s["target_class"] == seed
            assert s["task_profile"] == DIAG and s["collection_id"] == COLL
            assert s["target_key"] == f"{DIAG}:{seed}"
            assert s["episode_key"] == f"{COLL}:{DIAG}:seed-{seed:03d}"
            assert set(s["actor_input"]) == ACTOR_INPUT_KEYS  # no identity leak
            assert len(s["labels"]["executed_ctrl_rad"]) == pd.SUBSTEPS_PER_FRAME
            assert all(len(row) == pd.N_JOINTS
                       for row in s["labels"]["executed_ctrl_rad"])


def test_all_eight_class_keys_in_manifest(tmp_path):
    audits = [_mk_audit(s) for s in range(8)]
    samples = [_mk_sample(s) for s in range(8)]
    m = _manifest(audits, samples)
    assert m["task_profile"] == DIAG and m["collection_id"] == COLL
    assert m["target_keys"] == [f"{DIAG}:{s}" for s in range(8)]
    assert m["target_classes"] == list(range(8))
    assert m["split_counts"]["train"] == 4
    assert m["split_counts"]["validation"] == 2
    assert m["split_counts"]["test"] == 2
    for entry, s in zip(m["per_episode"], range(8)):
        assert entry["target_key"] == f"{DIAG}:{s}"
        assert entry["episode_key"] == f"{COLL}:{DIAG}:seed-{s:03d}"


# --- request gating -----------------------------------------------------------

def test_missing_or_unsafe_collection_id_rejected(tmp_path):
    ep = _tagged(tmp_path, 0)
    for bad in (None, "", "bad!id", "has space"):
        with pytest.raises(ValueError):
            audit_episode(ep, tmp_path, task_profile=DIAG, collection_id=bad)
    with pytest.raises(ValueError):
        _manifest([_mk_audit(0)], [_mk_sample(0)], collection_id="bad!id")


def test_profile_conflicts_rejected(tmp_path):
    # metadata/summary disagree with each other
    ep = _tagged(tmp_path / "a", 0)
    summ = json.loads((ep / "summary.json").read_text())
    summ["task_profile"] = AXIAL
    (ep / "summary.json").write_text(json.dumps(summ))
    audit, samples = audit_episode(ep, tmp_path / "a",
                                   task_profile=DIAG, collection_id=COLL)
    assert not audit.accepted and samples == []
    assert any("task_profile" in r for r in audit.rejection_reasons)
    # episode declares diagonal but request is axial -> never silently axial
    ep2 = _tagged(tmp_path / "b", 0)
    audit2, _ = audit_episode(ep2, tmp_path / "b")
    assert not audit2.accepted
    assert any("task_profile" in r for r in audit2.rejection_reasons)
    # metadata collection_id conflict
    ep3 = _tagged(tmp_path / "c", 0, collection="other-coll")
    audit3, _ = audit_episode(ep3, tmp_path / "c",
                              task_profile=DIAG, collection_id=COLL)
    assert not audit3.accepted
    assert any("collection_id" in r for r in audit3.rejection_reasons)


def test_acceptance_profile_gate(tmp_path):
    ep = _tagged(tmp_path / "ok", 0)
    acc_bad = {"controller_kind": "privileged_visual_teacher",
               "config": {"planner_kind": "force-governed", "task_profile": AXIAL}}
    audit, _ = audit_episode(ep, tmp_path / "ok", acceptance=acc_bad,
                             task_profile=DIAG, collection_id=COLL)
    assert not audit.accepted
    acc_missing = {"controller_kind": "privileged_visual_teacher",
                   "config": {"planner_kind": "force-governed"}}
    audit, _ = audit_episode(ep, tmp_path / "ok", acceptance=acc_missing,
                             task_profile=DIAG, collection_id=COLL)
    assert not audit.accepted  # absence no longer slips through for diagonal
    acc_ok = {"controller_kind": "privileged_visual_teacher",
              "config": {"planner_kind": "force-governed", "task_profile": DIAG}}
    audit, samples = audit_episode(ep, tmp_path / "ok", acceptance=acc_ok,
                                   task_profile=DIAG, collection_id=COLL)
    assert audit.accepted and samples
    # legacy: absent acceptance profile still allowed
    (tmp_path / "ax").mkdir(parents=True, exist_ok=True)
    ep_ax, _ = fx._make_fixture(tmp_path / "ax", seed=7)
    audit_ax, samples_ax = audit_episode(ep_ax, tmp_path / "ax", acceptance=acc_missing)
    assert audit_ax.accepted and samples_ax


def test_invalid_profile_and_out_of_range_seed(tmp_path):
    ep = _tagged(tmp_path, 0)
    with pytest.raises(ValueError):
        audit_episode(ep, tmp_path, task_profile="bogus", collection_id=COLL)
    with pytest.raises(ValueError):
        _manifest([_mk_audit(0)], [_mk_sample(0)], task_profile="bogus")
    ep8 = _tagged(tmp_path / "s8", 8)
    audit, samples = audit_episode(ep8, tmp_path / "s8",
                                   task_profile=DIAG, collection_id=COLL)
    assert not audit.accepted and samples == []
    assert any("outside near_diagonal_v1 split" in r for r in audit.rejection_reasons)
    (tmp_path / "s12").mkdir(parents=True, exist_ok=True)
    ep12, _ = fx._make_fixture(tmp_path / "s12", seed=12)
    audit12, samples12 = audit_episode(ep12, tmp_path / "s12")
    assert not audit12.accepted and samples12 == []  # axial out-of-range too


# --- legacy default unchanged -------------------------------------------------

def test_default_axial_sample_and_manifest_keys_unchanged(tmp_path):
    ep, _ = fx._make_fixture(tmp_path, seed=7)
    audit, samples = audit_episode(ep, tmp_path)
    assert audit.accepted and audit.task_profile == AXIAL
    assert samples and all(set(s) == LEGACY_SAMPLE_KEYS for s in samples)
    assert all("task_profile" not in s and "episode_key" not in s for s in samples)
    assert all(s["split"] == "validation" and s["target_class"] == 7 for s in samples)
    m = build_dataset_manifest([audit], samples, ACC, "m", "b", {})
    assert "task_profile" not in m and "collection_id" not in m
    assert "target_keys" not in m
    assert m["split_counts"]["validation"] == len(samples)
    # legacy identity in a legacy request is valid (scoped-in-legacy
    # rejection is covered in test_manifest_rejects_mixed_collection_*)


# --- truthful partial manifests + manifest rejections --------------------------

def test_partial_and_zero_sample_manifests_truthful():
    audits = [_mk_audit(s) for s in (0, 4)]
    samples = [_mk_sample(0)]
    m = _manifest(audits, samples)
    assert m["total_action_frames"] == 1
    assert m["target_keys"] == [f"{DIAG}:0"]
    assert m["split_counts"]["train"] == 1
    m0 = _manifest(audits, [])
    assert m0["total_action_frames"] == 0 and m0["target_keys"] == []


def test_manifest_rejects_wrong_class_key_split_and_unaccepted(tmp_path):
    audits = [_mk_audit(s) for s in (0, 1)]
    base = [_mk_sample(0), _mk_sample(1)]
    with pytest.raises(ValueError):
        _manifest(audits, [{**base[0], "target_class": 99}, base[1]])
    with pytest.raises(ValueError):
        _manifest(audits, [{**base[0], "target_key": f"{DIAG}:99"}, base[1]])
    with pytest.raises(ValueError):
        _manifest(audits, [{**base[0], "episode_key": "wrong"}, base[1]])
    with pytest.raises(ValueError):
        _manifest(audits, [{**base[0], "split": "test"}, base[1]])
    with pytest.raises(ValueError):
        _manifest(audits, base + [_mk_sample(2)])  # no audit for seed 2
    unaccepted = [_mk_audit(2), _mk_audit(3)]
    unaccepted[1].accepted = False
    with pytest.raises(ValueError):
        _manifest(unaccepted, [_mk_sample(2), _mk_sample(3)])


def test_manifest_rejects_mixed_collection_and_duplicate_audits():
    with pytest.raises(ValueError):
        _manifest([_mk_audit(0), _mk_audit(1, collection="other")],
                  [_mk_sample(0), _mk_sample(1)])
    with pytest.raises(ValueError):
        _manifest([_mk_audit(0), _mk_audit(0)],
                  [_mk_sample(0), _mk_sample(0)])
    # scoped samples cannot enter a legacy manifest
    with pytest.raises(ValueError):
        build_dataset_manifest([_mk_audit(0)], [_mk_sample(0)], ACC, "m", "b", {})
