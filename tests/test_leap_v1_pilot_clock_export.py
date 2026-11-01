"""M2b2 clock-integration tests for pilot_dataset.

Synthetic fixtures only: reuses the enriched ``_tagged`` diagonal fixture
(with coherent new capture timing) and the original legacy fixture from
the pilot-data audit tests. No real episodes, no raw evidence changes.
"""
from __future__ import annotations

import copy
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
    audit_episode, build_dataset_manifest,
)

_spec = importlib.util.spec_from_file_location(
    "leap_v1_profile_export_fixtures",
    REPO_ROOT / "tests" / "test_leap_v1_profile_export.py",
)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)

DIAG = pe.DIAG
COLL = pe.COLL
ACC = pe.ACC
OFFSET = pe.CLOCK_OFFSET
UNC = pe.CLOCK_UNC
OLD_CLOCK_NOTE = "clock offset and uncertainty are unavailable for this offline export"


def _audit_diagonal(tmp_path, seed=0):
    ep = pe._tagged(tmp_path / f"s{seed}", seed)
    return audit_episode(ep, tmp_path / f"s{seed}",
                         task_profile=DIAG, collection_id=COLL)


def _manifest(audit, samples):
    return build_dataset_manifest([audit], samples, ACC, "m", "b", {},
                                  task_profile=DIAG, collection_id=COLL)


def _rewrite_actor(ep, mutate):
    path = ep / "actor.jsonl"
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            mutate(rec)
            out.append(json.dumps(rec))
    path.write_text("\n".join(out) + "\n")


def _stale(rec):
    """Coherently age one record by +300 ms (t3 up, durations/age follow)."""
    ct = rec["capture_timing"]
    ct["t3_monotonic_s"] += 0.3
    ct["extract_ms"] += 300.0
    ct["total_ms"] += 300.0
    rec["frame_age_ms"] += 300.0


def test_valid_diagonal_timing_metadata_preserved_and_actor_input_unchanged(tmp_path):
    audit, samples = _audit_diagonal(tmp_path)
    assert audit.accepted and audit.timing_validated_frames == 5
    assert audit.timing_legacy_frames == 0
    assert len(samples) == 4
    for s in samples:
        md = s["metadata"]
        assert md["clock_offset_s"] == OFFSET
        assert md["clock_uncertainty_s"] == UNC
        assert md["png_backend"] == "pillow"
        assert md["png_backend_version"] == "fixture-v1"
        ct = md["capture_timing"]
        assert ct["clock_offset_s"] == OFFSET and ct["png_backend"] == "pillow"
        expected_age = (ct["t3_monotonic_s"] + OFFSET
                        - md["frame_monotonic_time_s"]) * 1000.0
        assert md["frame_age_ms"] == pytest.approx(expected_age, abs=1e-9)
        assert md["worst_frame_age_ms"] == pytest.approx(
            expected_age + UNC * 1000.0, abs=1e-9)
        assert set(s["actor_input"]) == pe.ACTOR_INPUT_KEYS


def test_scoped_manifest_reports_validated_availability(tmp_path):
    audit, samples = _audit_diagonal(tmp_path)
    m = _manifest(audit, samples)
    assert m["capture_timing"] == {
        "validated_frames": 5, "legacy_frames": 0, "invalid_frames": 0,
        "frame_availability": "validated"}
    entry = m["per_episode"][0]
    assert entry["timing_validated_frames"] == 5
    assert entry["timing_legacy_frames"] == 0
    assert entry["timing_invalid_frames"] == 0
    assert OLD_CLOCK_NOTE not in m["limitations"]
    assert any("measured values" in lim for lim in m["limitations"])


@pytest.mark.parametrize("phase,expected_samples", [
    ("visual_step", 3), ("success_terminal", 4),
])
def test_coherently_stale_action_and_terminal_rejected(tmp_path, phase, expected_samples):
    ep = pe._tagged(tmp_path / "stale", 0)

    def mutate(rec):
        if rec["phase"] == phase and rec["frame_id"] == (
                2 if phase == "visual_step" else 6):
            _stale(rec)

    _rewrite_actor(ep, mutate)
    audit, samples = audit_episode(ep, tmp_path / "stale",
                                   task_profile=DIAG, collection_id=COLL)
    assert not audit.accepted
    assert any("capture timing" in i["reason"] for i in audit.validation_issues)
    assert len(samples) == expected_samples
    stale_id = 2 if phase == "visual_step" else 6
    assert stale_id not in [s["frame_id"] for s in samples]
    total = audit.actor_entries
    assert audit.timing_validated_frames == total - 1
    assert audit.timing_legacy_frames == 0


def test_missing_required_timing_rejected_and_counted_invalid_not_legacy(tmp_path):
    ep = pe._tagged(tmp_path / "missing", 0)
    _rewrite_actor(ep, lambda rec: rec.pop("capture_timing", None))
    audit, samples = audit_episode(ep, tmp_path / "missing",
                                   task_profile=DIAG, collection_id=COLL)
    assert not audit.accepted and samples == []
    assert audit.timing_validated_frames == 0
    assert audit.timing_legacy_frames == 0
    m = _manifest(audit, [])
    assert m["capture_timing"]["frame_availability"] == "incomplete"
    assert m["capture_timing"] == {
        "validated_frames": 0, "legacy_frames": 0, "invalid_frames": 5,
        "frame_availability": "incomplete"}
    assert entry_invalid(m) == 5
    assert any("never approved" in lim for lim in m["limitations"])


def entry_invalid(m):
    return m["per_episode"][0]["timing_invalid_frames"]


def test_newtiming_axial_manifest_keeps_timing_metadata(tmp_path):
    (tmp_path / "ax").mkdir(parents=True, exist_ok=True)
    ep, _ = pe.fx._make_fixture(tmp_path / "ax", seed=7)
    pe._enrich_clock(ep)
    audit, samples = audit_episode(ep, tmp_path / "ax")
    assert audit.accepted and audit.timing_validated_frames == 5
    assert samples and samples[0]["metadata"]["clock_offset_s"] == OFFSET
    m = build_dataset_manifest([audit], samples, ACC, "m", "b", {})
    assert m["capture_timing"]["frame_availability"] == "validated"
    assert m["capture_timing"]["validated_frames"] == 5
    assert OLD_CLOCK_NOTE not in m["limitations"]
    assert "capture_timing" not in build_dataset_manifest([], [], ACC, "m", "b", {})


def test_legacy_axial_fixture_output_unchanged(tmp_path):
    ep, _ = pe.fx._make_fixture(tmp_path, seed=7)
    audit, samples = audit_episode(ep, tmp_path)
    assert audit.accepted and audit.timing_legacy_frames == 5
    assert audit.timing_validated_frames == 0
    for s in samples:
        assert s["metadata"]["clock_offset_s"] is None
        assert s["metadata"]["clock_uncertainty_s"] is None
        assert "capture_timing" not in s["metadata"]
    m = build_dataset_manifest([audit], samples, ACC, "m", "b", {})
    assert "capture_timing" not in m
    assert OLD_CLOCK_NOTE in m["limitations"]


@pytest.mark.parametrize("ref_kind", ["repo_relative", "wrong_episode", "traversal"])
def test_png_repo_relative_refs_and_rejections(tmp_path, ref_kind):
    ep = pe._tagged(tmp_path / "png", 0)
    ep_name = ep.name
    sibling = tmp_path / "png" / "sibling-episode"
    sibling.mkdir()
    for png in sorted(ep.glob("cycle-*.png")):
        (sibling / png.name).write_bytes(png.read_bytes())

    def mutate(rec):
        base = Path(rec["png_ref"]).name
        if ref_kind == "repo_relative":
            rec["png_ref"] = f"{ep_name}/{base}"
        elif ref_kind == "wrong_episode":
            rec["png_ref"] = f"sibling-episode/{base}"
        else:
            rec["png_ref"] = f"sibling-episode/../sibling-episode/{base}"

    _rewrite_actor(ep, mutate)
    audit, _ = audit_episode(ep, tmp_path / "png",
                             task_profile=DIAG, collection_id=COLL)
    if ref_kind == "repo_relative":
        assert audit.accepted and audit.png_verified == 5 and audit.png_failed == 0
    else:
        assert not audit.accepted and audit.png_failed == 5
        assert any("not in episode dir" in i["reason"]
                   for i in audit.validation_issues)


def test_episode_fixture_not_mutated_by_audit(tmp_path):
    ep = pe._tagged(tmp_path / "immut", 0)
    actor_before = (ep / "actor.jsonl").read_text()
    summary_before = (ep / "summary.json").read_text()
    audit_episode(ep, tmp_path / "immut", task_profile=DIAG, collection_id=COLL)
    assert (ep / "actor.jsonl").read_text() == actor_before
    assert (ep / "summary.json").read_text() == summary_before


def test_symlinked_png_resolved_to_other_episode_rejected(tmp_path):
    ep = pe._tagged(tmp_path / "link", 0)
    sibling = tmp_path / "link" / "real-episode"
    sibling.mkdir()
    target = sibling / "cycle-001.png"
    target.write_bytes((ep / "cycle-001.png").read_bytes())
    link = ep / "cycle-001.png"
    link.unlink()
    link.symlink_to(target)
    audit, _ = audit_episode(ep, tmp_path / "link",
                             task_profile=DIAG, collection_id=COLL)
    assert not audit.accepted
    assert audit.png_failed == 1
    assert audit.png_verified == 4
    assert any("not in episode dir" in i["reason"]
               for i in audit.validation_issues)
