"""Tests for validate_teacher_configuration (M2b teacher-config helper).

Compact in-memory fixtures only; no PNG fixtures, no real episodes.
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

from embodied_trackpad.leap_v1.pilot_provenance import (  # noqa: E402
    validate_teacher_configuration,
)

CONTACT = "pointing-light-lowtarget"
FORCE = {"target_force_n": 0.6, "gain_m_per_ns": 0.004,
         "max_normal_speed_m_s": 0.002, "max_joint_step_rad": 0.0002,
         "max_cycle_delta_rad": 0.005}
GOV = {"substeps_per_frame": 25, "target_force_n": 0.6,
       "tangential_stop_force_n": 0.9,
       "per_substep_total_joint_delta_cap_rad": 0.0002,
       "max_cycle_delta_rad": 0.005}
GOOD_HASH = "a" * 40


def _summary(**over):
    s = {"contact_mode": CONTACT, "planner_kind": "force-governed",
         "force_config": copy.deepcopy(FORCE),
         "force_governor_config": copy.deepcopy(GOV)}
    s.update(over)
    return s


def _metadata(**over):
    m = {"contact_mode": CONTACT,
         "planner": {"planner_kind": "force-governed",
                     "force_config": copy.deepcopy(FORCE),
                     "force_governor_config": copy.deepcopy(GOV),
                     "damping": 0.05}}
    m.update(over)
    return m


def _acceptance(**cfg_over):
    cfg = {"contact_mode": CONTACT, "planner_kind": "force-governed",
           "force_config": copy.deepcopy(FORCE),
           "force_governor_config": copy.deepcopy(GOV)}
    source = cfg_over.pop("source_commit", None)
    cfg.update(cfg_over)
    acc = {"controller_kind": "privileged_visual_teacher", "config": cfg}
    if source is not None:
        acc["source_commit"] = source
    return acc


def _both(target=0.6):
    force = {**FORCE, "target_force_n": target}
    gov = {**GOV, "target_force_n": target}
    return (_summary(force_config=force, force_governor_config=gov),
            _metadata(planner={"planner_kind": "force-governed",
                               "force_config": copy.deepcopy(force),
                               "force_governor_config": copy.deepcopy(gov),
                               "damping": 0.05}))


def test_valid_config_returns_copies_and_none_source():
    out = validate_teacher_configuration(_summary(), _metadata(),
                                         _acceptance(), required=True)
    assert out["collection_source_commit"] is None
    tc = out["teacher_configuration"]
    assert tc["contact_mode"] == CONTACT and tc["planner_kind"] == "force-governed"
    assert tc["force_config"] == FORCE and tc["force_governor_config"] == GOV
    tc["force_config"]["target_force_n"] = 99.0
    assert FORCE["target_force_n"] == 0.6


@pytest.mark.parametrize("target", [0.6, 0.8])
def test_exact_matched_physical_policy_values_allowed(target):
    summary, metadata = _both(target=target)
    acc = _acceptance(force_config={**FORCE, "target_force_n": target},
                      force_governor_config={**GOV, "target_force_n": target})
    out = validate_teacher_configuration(summary, metadata, acc, required=True)
    assert out["teacher_configuration"]["force_config"]["target_force_n"] == target


def test_mismatched_target_force_rejected():
    summary, metadata = _both(target=0.6)
    metadata["planner"]["force_config"] = {**FORCE, "target_force_n": 0.8}
    with pytest.raises(ValueError, match="force_config"):
        validate_teacher_configuration(summary, metadata, required=True)


@pytest.mark.parametrize("build", [
    lambda: (_summary(force_governor_config=None), _metadata()),
    lambda: (_summary(), _metadata(contact_mode=None)),
    lambda: (_summary(), {"planner": "not-a-dict"}),
    lambda: (_summary(), _metadata(planner={"planner_kind": "other"})),
])
def test_missing_governor_contact_mode_or_planner_rejected(build):
    summary, metadata = build()
    with pytest.raises(ValueError):
        validate_teacher_configuration(summary, metadata, required=True)


@pytest.mark.parametrize("build", [
    lambda: (_summary(), _metadata(), {"controller_kind": "x", "config": {"planner_kind": "force-governed"}}),
    lambda: (_summary(), _metadata(),
             _acceptance(force_config={**FORCE, "target_force_n": 0.8})),
    lambda: (_summary(), _metadata(), _acceptance(contact_mode="other-mode")),
])
def test_acceptance_missing_fields_or_mismatch_rejected(build):
    summary, metadata, acc = build()
    with pytest.raises(ValueError):
        validate_teacher_configuration(summary, metadata, acc, required=True)


@pytest.mark.parametrize("mutate", [
    lambda s, m: s["force_config"].update(target_force_n=float("nan")),
    lambda s, m: m["planner"]["force_config"].update(target_force_n=True),
    lambda s, m: m["planner"]["force_config"].update(extra_key=1.0),
    lambda s, m: s["force_governor_config"].pop("tangential_stop_force_n"),
    lambda s, m: s["force_governor_config"].update(substeps_per_frame=24),
    lambda s, m: s["force_governor_config"].update(substeps_per_frame=25.0),
    lambda s, m: s["force_governor_config"].update(substeps_per_frame=True),
    lambda s, m: s["force_config"].update(gain_m_per_ns=0.0),
])
def test_invalid_numeric_keyset_or_substeps_rejected(mutate):
    summary, metadata = _summary(), _metadata()
    mutate(summary, metadata)
    with pytest.raises(ValueError):
        validate_teacher_configuration(summary, metadata, required=True)


def test_source_commit_none_valid_and_invalid():
    summary, metadata = _summary(), _metadata()
    out = validate_teacher_configuration(summary, metadata, required=True)
    assert out["collection_source_commit"] is None
    acc = _acceptance(source_commit=GOOD_HASH)
    out = validate_teacher_configuration(summary, metadata, acc, required=True)
    assert out["collection_source_commit"] == GOOD_HASH
    acc = _acceptance(source_commit="abc123")
    with pytest.raises(ValueError, match="source_commit"):
        validate_teacher_configuration(summary, metadata, acc, required=True)


@pytest.mark.parametrize("mutate", [
    lambda acc: acc["config"]["force_governor_config"].update(substeps_per_frame=25.0),
    lambda acc: acc["config"]["force_governor_config"].update(substeps_per_frame=True),
    lambda acc: acc["config"]["force_config"].update(target_force_n=float("nan")),
    lambda acc: acc["config"]["force_config"].update(gain_m_per_ns=True),
    lambda acc: acc["config"].pop("force_governor_config"),
])
def test_acceptance_schema_validated_before_equality(mutate):
    # dict equality alone would pass 25.0 == 25 or True == 1; the helper
    # must schema-check the acceptance configs first.
    summary, metadata = _summary(), _metadata()
    acc = _acceptance()
    mutate(acc)
    with pytest.raises(ValueError):
        validate_teacher_configuration(summary, metadata, acc, required=True)


def test_output_isolation_retains_local_inputs():
    summary, metadata = _summary(), _metadata()
    snap = copy.deepcopy((summary, metadata))
    out = validate_teacher_configuration(summary, metadata, _acceptance(),
                                         required=True)
    out["teacher_configuration"]["force_config"]["target_force_n"] = 99.0
    out["teacher_configuration"]["force_governor_config"]["substeps_per_frame"] = 1
    assert (summary, metadata) == snap


def test_legacy_not_required_returns_empty_and_inputs_untouched():
    snapshot = copy.deepcopy((FORCE, GOV))
    assert validate_teacher_configuration(_summary(), _metadata()) == {}
    assert (FORCE, GOV) == snapshot
