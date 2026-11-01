"""Pure regression tests for LEAP v1 pilot contact-boundary validation.

No physics stepping, no app launch, no live data collection.
"""
from __future__ import annotations

from typing import Any

import pytest

from embodied_trackpad.leap_v1.pilot_dataset import (
    MAX_PAD_FORCE_N,
    PENETRATION_TOL_M,
    EpisodeAudit,
    _validate_teacher_steps,
    _validate_touch,
)


def _base_step(
    contacts: list[Any] | None = None,
    aggregate: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_index": 1,
        "sim_time_s": 0.002,
        "ctrl": [0.0] * 16,
        "qpos": [0.0] * 16,
        "qvel": [0.0] * 16,
        "ctrl_shape": [16],
        "checks": {k: True for k in [
            "pre_step_state_finite", "finite_state", "raw_contacts_valid",
            "pad_force<=5N", "penetration<=tol", "no_unintended_contact",
            "joint_excursion<=tol", "no_new_warnings",
        ]},
        "ok": True,
        "failure_reasons": [],
        "warning_delta": [0] * 7,
        "contacts": contacts if contacts is not None else [],
        "aggregate": aggregate if aggregate is not None else {},
    }
    if overrides:
        step.update(overrides)
    return step


def _contact(**overrides: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "kind": "tip_pad_top",
        "normal_force_n": 0.8,
        "force6": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
        "frame": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
        "dist_m": -1e-05,
        "pos_world": [0.07, 0.006, 0.1],
        "pad_local": [0.07, 0.006, 0.005],
    }
    c.update(overrides)
    return c


def _aggregate(finger_id: str = "index", **overrides: Any) -> dict[str, Any]:
    a: dict[str, Any] = {
        "finger_id": finger_id,
        "x_m": 0.07,
        "y_m": 0.006,
        "normal_force_n": 0.8,
        "contact_points": 1,
    }
    a.update(overrides)
    return a


class TestValidateTeacherContacts:
    def test_two_raw_contacts_one_finger_pass(self):
        step = _base_step(contacts=[
            _contact(normal_force_n=2.5, force6=[2.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _contact(normal_force_n=2.5, force6=[2.5, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ], aggregate={"index": _aggregate(normal_force_n=5.0, contact_points=2)})
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert not audit.validation_issues, audit.validation_issues

    def test_total_pad_force_6n_rejected(self):
        step = _base_step(contacts=[
            _contact(normal_force_n=3.0, force6=[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _contact(normal_force_n=3.0, force6=[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ], aggregate={"index": _aggregate(normal_force_n=6.0, contact_points=2)})
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("total pad force" in i["reason"] for i in audit.validation_issues)

    def test_contact_force_negative_rejected(self):
        step = _base_step(contacts=[_contact(normal_force_n=-0.1)])
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("contact force negative" in i["reason"] for i in audit.validation_issues)

    def test_empty_force6_rejected(self):
        step = _base_step(contacts=[_contact(force6=[])])
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("force6 not 6 finite vector" in i["reason"] for i in audit.validation_issues)

    def test_missing_vector_key_rejected(self):
        step = _base_step(contacts=[_contact()])
        del step["contacts"][0]["pos_world"]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("pos_world not 3 finite vector" in i["reason"] for i in audit.validation_issues)

    def test_penetration_over_tol_rejected(self):
        step = _base_step(contacts=[_contact(dist_m=-0.002)])
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("penetration" in i["reason"] for i in audit.validation_issues)

    def test_malformed_contact_not_dict_rejected(self):
        step = _base_step(contacts=["not-a-dict"])
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("contact not a dict" in i["reason"] for i in audit.validation_issues)

    def test_aggregate_force_negative_rejected(self):
        step = _base_step(contacts=[_contact()], aggregate={"index": _aggregate(normal_force_n=-0.1)})
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("aggregate force negative" in i["reason"] for i in audit.validation_issues)

    def test_aggregate_sum_mismatch_rejected(self):
        step = _base_step(contacts=[_contact(normal_force_n=1.0)], aggregate={"index": _aggregate(normal_force_n=2.0)})
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("aggregate force sum" in i["reason"] for i in audit.validation_issues)

    def test_aggregate_x_nan_rejected(self):
        step = _base_step(contacts=[_contact()], aggregate={"index": _aggregate(x_m=float("nan"))})
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_teacher_steps([step], audit)
        assert any("aggregate x/y non-finite" in i["reason"] for i in audit.validation_issues)


class TestValidateTouch:
    def test_two_raw_one_finger_touch_passes(self):
        step = _base_step(contacts=[
            _contact(normal_force_n=2.5),
            _contact(normal_force_n=2.5),
        ], aggregate={"index": _aggregate(normal_force_n=5.0, contact_points=2)})
        teacher_by_sim_time = {0.002: step}
        touch = [{
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": 1,
                "simulation_time_s": 0.002,
                "contacts": [{"finger_id": "index", "x_m": 0.07, "y_m": 0.006, "normal_force_n": 5.0}],
            },
            "ack": {"type": "touch_ok", "sequence": 1},
        }]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_touch(touch, teacher_by_sim_time, audit)
        assert not audit.validation_issues, audit.validation_issues

    def test_touch_count_matches_aggregate_not_raw_count(self):
        # One aggregate finger with two raw contact points; payload count must be 1.
        step = _base_step(contacts=[_contact(), _contact()], aggregate={"index": _aggregate()})
        teacher_by_sim_time = {0.002: step}
        touch = [{
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": 1,
                "simulation_time_s": 0.002,
                "contacts": [{"finger_id": "index", "x_m": 0.07, "y_m": 0.006, "normal_force_n": 0.8}],
            },
            "ack": {"type": "touch_ok", "sequence": 1},
        }]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_touch(touch, teacher_by_sim_time, audit)
        assert not audit.validation_issues, audit.validation_issues

    def test_touch_nan_x_rejected(self):
        step = _base_step(contacts=[_contact()], aggregate={"index": _aggregate()})
        teacher_by_sim_time = {0.002: step}
        touch = [{
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": 1,
                "simulation_time_s": 0.002,
                "contacts": [{"finger_id": "index", "x_m": float("nan"), "y_m": 0.006, "normal_force_n": 0.8}],
            },
            "ack": {"type": "touch_ok", "sequence": 1},
        }]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_touch(touch, teacher_by_sim_time, audit)
        assert any("x_m non-finite" in i["reason"] for i in audit.validation_issues)

    def test_touch_missing_finger_rejected(self):
        step = _base_step(contacts=[_contact()], aggregate={
            "index": _aggregate(),
            "middle": _aggregate(finger_id="middle", x_m=0.08),
        })
        teacher_by_sim_time = {0.002: step}
        touch = [{
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": 1,
                "simulation_time_s": 0.002,
                "contacts": [{"finger_id": "index", "x_m": 0.07, "y_m": 0.006, "normal_force_n": 0.8}],
            },
            "ack": {"type": "touch_ok", "sequence": 1},
        }]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_touch(touch, teacher_by_sim_time, audit)
        assert any("missing fingers" in i["reason"] for i in audit.validation_issues)

    def test_touch_duplicate_finger_rejected(self):
        step = _base_step(contacts=[_contact(), _contact()], aggregate={"index": _aggregate()})
        teacher_by_sim_time = {0.002: step}
        touch = [{
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": 1,
                "simulation_time_s": 0.002,
                "contacts": [
                    {"finger_id": "index", "x_m": 0.07, "y_m": 0.006, "normal_force_n": 0.4},
                    {"finger_id": "index", "x_m": 0.07, "y_m": 0.006, "normal_force_n": 0.4},
                ],
            },
            "ack": {"type": "touch_ok", "sequence": 1},
        }]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_touch(touch, teacher_by_sim_time, audit)
        assert any("duplicate finger_id" in i["reason"] for i in audit.validation_issues)

    def test_empty_touch_hides_aggregate_rejected(self):
        step = _base_step(contacts=[_contact()], aggregate={"index": _aggregate()})
        teacher_by_sim_time = {0.002: step}
        touch = [{
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": 1,
                "simulation_time_s": 0.002,
                "contacts": [],
            },
            "ack": {"type": "touch_ok", "sequence": 1},
        }]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_touch(touch, teacher_by_sim_time, audit)
        assert any("empty touch payload hides" in i["reason"] for i in audit.validation_issues)

    def test_touch_force_mismatch_rejected(self):
        step = _base_step(contacts=[_contact()], aggregate={"index": _aggregate()})
        teacher_by_sim_time = {0.002: step}
        touch = [{
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": 1,
                "simulation_time_s": 0.002,
                "contacts": [{"finger_id": "index", "x_m": 0.07, "y_m": 0.006, "normal_force_n": 0.81}],
            },
            "ack": {"type": "touch_ok", "sequence": 1},
        }]
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        _validate_touch(touch, teacher_by_sim_time, audit)
        assert any("normal_force_n mismatch" in i["reason"] for i in audit.validation_issues)
