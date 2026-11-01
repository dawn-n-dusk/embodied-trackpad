"""Integration tests for the teacher-config gate in audit_episode.

Reuses the enriched ``_tagged`` diagonal fixture (coherent clock timing +
force-governed teacher config) and the original legacy fixture. Synthetic
only; no real episodes or raw evidence changes.
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

from embodied_trackpad.leap_v1.pilot_dataset import audit_episode  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "leap_v1_profile_export_fixtures",
    REPO_ROOT / "tests" / "test_leap_v1_profile_export.py",
)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)

DIAG = pe.DIAG
COLL = pe.COLL
GOOD_HASH = "b" * 40


def _acceptance(**cfg_over):
    cfg = {"planner_kind": "force-governed", "task_profile": DIAG,
           **pe.ACC_TEACHER_CONFIG}
    source = cfg_over.pop("source_commit", None)
    cfg.update(cfg_over)
    acc = {"controller_kind": "privileged_visual_teacher", "config": cfg}
    if source is not None:
        acc["source_commit"] = source
    return acc


def _audit(tmp_path, seed=0, acceptance=None, mutate_summary=None):
    ep = pe._tagged(tmp_path / f"s{seed}", seed)
    if mutate_summary is not None:
        summ = json.loads((ep / "summary.json").read_text())
        mutate_summary(summ)
        (ep / "summary.json").write_text(json.dumps(summ))
    return audit_episode(ep, tmp_path / f"s{seed}", acceptance=acceptance,
                         task_profile=DIAG, collection_id=COLL)


def test_valid_diagonal_config_and_source_stamped_in_metadata(tmp_path):
    audit, samples = _audit(tmp_path, acceptance=_acceptance(
        source_commit=GOOD_HASH))
    assert audit.accepted and len(samples) == 4
    for s in samples:
        md = s["metadata"]
        tc = md["teacher_configuration"]
        assert tc["contact_mode"] == pe.TEACHER_CONTACT
        assert tc["planner_kind"] == "force-governed"
        assert tc["force_config"] == pe.FORCE_CONFIG
        assert tc["force_governor_config"] == pe.GOVERNOR_CONFIG
        assert md["collection_source_commit"] == GOOD_HASH


def test_mismatched_target_force_rejected_before_image_work(tmp_path):
    audit, samples = _audit(
        tmp_path,
        mutate_summary=lambda s: s["force_config"].update(target_force_n=0.8))
    assert not audit.accepted and samples == []
    assert audit.png_verified == 0 and audit.png_failed == 0
    assert any("teacher configuration" in r for r in audit.rejection_reasons)


def test_missing_acceptance_force_config_rejected(tmp_path):
    audit, _ = _audit(tmp_path, acceptance={"controller_kind": "x",
                                            "config": {"planner_kind":
                                                       "force-governed",
                                                       "task_profile": DIAG}})
    assert not audit.accepted
    assert any("teacher configuration" in r for r in audit.rejection_reasons)


def test_missing_governor_config_rejected(tmp_path):
    audit, _ = _audit(
        tmp_path,
        mutate_summary=lambda s: s.pop("force_governor_config"))
    assert not audit.accepted
    assert any("teacher configuration" in r for r in audit.rejection_reasons)


def test_no_source_fabrication_when_acceptance_none(tmp_path):
    audit, samples = _audit(tmp_path)
    assert audit.accepted
    for s in samples:
        assert s["metadata"]["collection_source_commit"] is None
        assert s["metadata"]["teacher_configuration"]["contact_mode"] == \
            pe.TEACHER_CONTACT


def test_legacy_axial_output_retains_no_teacher_metadata(tmp_path):
    ep, _ = pe.fx._make_fixture(tmp_path, seed=7)
    audit, samples = audit_episode(ep, tmp_path)
    assert audit.accepted
    for s in samples:
        assert "teacher_configuration" not in s["metadata"]
        assert "collection_source_commit" not in s["metadata"]
