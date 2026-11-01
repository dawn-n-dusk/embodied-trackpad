"""Focused regression tests for LEAP v1 pilot terminal/action-window/clock fixes.

No physics stepping, no app launch, no live data collection.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from embodied_trackpad.leap_v1.pilot_dataset import (
    DT_S,
    N_JOINTS,
    SUBSTEPS_PER_FRAME,
    EpisodeAudit,
    _final_release_empty_span_s,
    _validate_force_intents,
    audit_episode,
)
from embodied_trackpad.leap_v1.visual_servo import decode_png_bytes, extract_visual_estimate
from tests.test_leap_v1_pilot_data_audit import (
    _base_summary,
    _make_png_bytes,
    _prep_step,
    _visual_step,
    sha256_file,
)


def _recompute_error(png_path: Path) -> float:
    rgb = decode_png_bytes(png_path.read_bytes())
    est = extract_visual_estimate(rgb)
    err = est.error_px
    assert err is not None, f"visual estimate failed for {png_path}"
    return math.hypot(*err)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_teacher(
    seed_frames: dict[int, dict],
    total_steps: int,
    contact_end_step: int,
) -> list[dict]:
    records: list[dict] = []
    sorted_frames = sorted(seed_frames.items())
    first_frame_id, first_info = sorted_frames[0]
    for i in range(1, first_info["first_step_index"] - 1):
        records.append(_prep_step(i))
    prev_last_step = 0
    for idx, (fid, info) in enumerate(sorted_frames):
        first = info["first_step_index"]
        if idx == 0:
            prior = _prep_step(first - 1)
            prior["qpos"] = [0.0] * N_JOINTS
            records.append(prior)
        else:
            # the step immediately before this frame is the last visual step of
            # the previous frame; fill any genuine gap with prep steps
            for i in range(prev_last_step + 1, first - 1):
                records.append(_prep_step(i, has_contact=i <= contact_end_step))
        for k in range(SUBSTEPS_PER_FRAME):
            idx_step = first + k
            ctrl = info["ctrl"]
            has_contact = idx_step <= contact_end_step
            records.append(_visual_step(idx_step, fid, ctrl, has_contact=has_contact))
        prev_last_step = first + SUBSTEPS_PER_FRAME - 1
    last_visual = max(info["first_step_index"] + SUBSTEPS_PER_FRAME - 1 for info in seed_frames.values())
    for i in range(last_visual + 1, total_steps + 1):
        records.append(_prep_step(i, has_contact=i <= contact_end_step))
    return records


def _make_planner(frames: list[dict], teacher_by_index: dict[int, dict]) -> list[dict]:
    planner: list[dict] = []
    nominal_delta = [0.001] + [0.0] * (N_JOINTS - 1)
    for info in frames:
        fid = info["frame_id"]
        first = info["first_step_index"]
        planner.append({
            "kind": "plan",
            "planner_kind": "force-governed",
            "cycle": fid - 1,
            "frame_id": fid,
            "nominal_frame_delta": nominal_delta,
        })
        for k in range(SUBSTEPS_PER_FRAME):
            consumed = first - 1 + k
            intended = consumed + 1
            consumed_rec = teacher_by_index.get(consumed, {})
            intended_rec = teacher_by_index.get(intended, {})
            corr = intended_rec.get("ctrl", [0.0] * N_JOINTS)
            agg_force = sum(
                float(a.get("normal_force_n", 0.0))
                for a in (consumed_rec.get("aggregate") or {}).values()
                if isinstance(a, dict)
            )
            planner.append({
                "kind": "force_substep_intent",
                "planner_kind": "force-governed",
                "cycle": fid - 1,
                "frame_id": fid,
                "consumed_record_step_index": consumed,
                "consumed_record_sim_time_s": consumed * DT_S,
                "intended_next_step_index": intended,
                "measured_force_n": agg_force,
                "contact_point_world_m": [0.0, 0.0, 0.0],
                "normal_row": [1.0, 0.0, 0.0, 0.0],
                "prior_ctrl": consumed_rec.get("ctrl", [0.0] * N_JOINTS),
                "normal_ctrl": corr,
                "corrected_ctrl": corr,
                "helper_info": {},
                "governor_info": {"gate": 0.5, "common_scale": 1.0},
                "nominal_frame_delta": nominal_delta,
            })
    return planner


def _make_minimal_episode(
    tmp_path: Path,
    seed: int = 7,
    target_offsets: list[int] = (30, 2, 2),
    terminal_offset: int = 0,
    total_steps: int = 400,
    contact_end_step: int = 110,
) -> tuple[Path, list[dict], list[dict], list[dict]]:
    ep = tmp_path / f"seed-{seed:03d}"
    ep.mkdir()
    cursor = (400, 300)
    n_visual = len(target_offsets)

    frames: list[dict] = []
    for i, off in enumerate(target_offsets):
        tx = cursor[0] + off
        png_path = ep / f"cycle-{i + 1:03d}.png"
        png_path.write_bytes(_make_png_bytes(
            cursor_center=cursor,
            target_center=(tx, cursor[1]),
            cursor_radius=15,
            target_radius=5,
        ))
        frames.append({
            "frame_id": i + 2,
            "first_step_index": 11 + i * SUBSTEPS_PER_FRAME,
            "png_path": png_path,
            "png_sha256": sha256_file(png_path),
            "error_px": _recompute_error(png_path),
            "ctrl": [0.001 * (i + 1)] + [0.0] * (N_JOINTS - 1),
        })

    terminal_png_path = ep / f"cycle-{n_visual + 1:03d}.png"
    terminal_png_path.write_bytes(_make_png_bytes(
        cursor_center=cursor,
        target_center=(cursor[0] + terminal_offset, cursor[1]),
        cursor_radius=15,
        target_radius=5,
    ))
    terminal_error = _recompute_error(terminal_png_path)
    terminal_frame_id = frames[-1]["frame_id"] + 1
    terminal_cycle = n_visual + 1

    seed_frames = {
        f["frame_id"]: {"first_step_index": f["first_step_index"], "ctrl": f["ctrl"]}
        for f in frames
    }
    teacher = _make_teacher(seed_frames, total_steps, contact_end_step)
    teacher_by_index = {r["step_index"]: r for r in teacher}
    last_visual_step = teacher_by_index[frames[-1]["first_step_index"] + SUBSTEPS_PER_FRAME - 1]

    actor: list[dict] = []
    base_time = 1000.0
    for i, f in enumerate(frames):
        prior = teacher_by_index.get(f["first_step_index"] - 1, {})
        actor.append({
            "phase": "visual_step",
            "cycle": i + 1,
            "png_ref": f["png_path"].name,
            "png_sha256": f["png_sha256"],
            "frame_id": f["frame_id"],
            "frame_monotonic_time_s": base_time + i * 0.05,
            "frame_age_ms": 10.0 + i,
            "error_px": f["error_px"],
            "q": prior.get("qpos", [0.0] * N_JOINTS),
            "qvel": prior.get("qvel", [0.0] * N_JOINTS),
            "prev_ctrl": prior.get("ctrl", [0.0] * N_JOINTS),
            "next_ctrl": f["ctrl"],
            "next_ctrl_is_nominal": True,
            "action_requested": True,
        })
    actor.append({
        "phase": "success_terminal",
        "cycle": terminal_cycle,
        "png_ref": terminal_png_path.name,
        "png_sha256": sha256_file(terminal_png_path),
        "frame_id": terminal_frame_id,
        "frame_monotonic_time_s": base_time + n_visual * 0.05,
        "frame_age_ms": 5.0,
        "error_px": terminal_error,
        "q": last_visual_step["qpos"],
        "qvel": last_visual_step["qvel"],
        "prev_ctrl": last_visual_step["ctrl"],
        "next_ctrl": None,
        "next_ctrl_is_nominal": False,
        "action_requested": False,
    })

    summary = _base_summary(seed, action_frames=len(actor), steps=total_steps)
    summary["initial_error_px"] = frames[0]["error_px"]
    summary["final_error_px"] = terminal_error
    (ep / "summary.json").write_text(json.dumps(summary))
    (ep / "metadata.json").write_text(json.dumps({
        "controller_kind": "privileged_visual_teacher",
        "training_eligible": False,
        "seed": seed,
    }))

    _write_jsonl(ep / "teacher.jsonl", teacher)
    _write_jsonl(ep / "actor.jsonl", actor)
    _write_jsonl(ep / "planner.jsonl", _make_planner(frames, teacher_by_index))
    _write_jsonl(ep / "touch.jsonl", [])
    return ep, frames, actor, teacher


class TestForceIntentFrameId:
    def test_intent_frame_id_matches_intended_step(self):
        teacher_by_index = {
            10: {"step_index": 10, "frame_id": 1, "sim_time_s": 0.02, "ctrl": [0.0] * N_JOINTS, "aggregate": {}},
            11: {"step_index": 11, "frame_id": 2, "sim_time_s": 0.022, "ctrl": [0.1] + [0.0] * (N_JOINTS - 1), "aggregate": {}},
        }
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        planner = [{
            "kind": "force_substep_intent",
            "intended_next_step_index": 11,
            "consumed_record_step_index": 10,
            "frame_id": 2,
            "measured_force_n": 0.0,
            "prior_ctrl": [0.0] * N_JOINTS,
            "corrected_ctrl": [0.1] + [0.0] * (N_JOINTS - 1),
            "consumed_record_sim_time_s": 0.02,
        }]
        intents = _validate_force_intents(planner, teacher_by_index, audit)
        assert 11 in intents
        assert audit.force_intent_mismatches == 0
        assert not any("frame_id" in i["reason"] for i in audit.validation_issues)

    def test_intent_frame_id_from_consumed_step_rejected(self):
        teacher_by_index = {
            10: {"step_index": 10, "frame_id": 1, "sim_time_s": 0.02, "ctrl": [0.0] * N_JOINTS, "aggregate": {}},
            11: {"step_index": 11, "frame_id": 2, "sim_time_s": 0.022, "ctrl": [0.1] + [0.0] * (N_JOINTS - 1), "aggregate": {}},
        }
        audit = EpisodeAudit(seed=0, episode_dir="ep", summary={}, metadata={})
        planner = [{
            "kind": "force_substep_intent",
            "intended_next_step_index": 11,
            "consumed_record_step_index": 10,
            "frame_id": 1,  # wrong: matches consumed step, not intended step
            "measured_force_n": 0.0,
            "prior_ctrl": [0.0] * N_JOINTS,
            "corrected_ctrl": [0.1] + [0.0] * (N_JOINTS - 1),
            "consumed_record_sim_time_s": 0.02,
        }]
        _validate_force_intents(planner, teacher_by_index, audit)
        assert audit.force_intent_mismatches > 0
        assert any("frame_id" in i["reason"] for i in audit.validation_issues)


class TestTerminalValidation:
    def test_terminal_accepted_with_error_state_clock_order(self, tmp_path: Path):
        ep, frames, actor, teacher = _make_minimal_episode(tmp_path)
        audit, samples = audit_episode(ep, tmp_path)
        assert audit.terminal_frame is not None
        assert audit.accepted, (audit.rejection_reasons, audit.validation_issues)
        assert len(samples) == len(frames)
        # Terminal error is included as the third final success frame:
        # without it the last visual errors [30, 2, 2] would include 30 > 3 px.
        assert audit.terminal_frame["error_px"] == 0.0

    def test_terminal_state_mismatch_rejected(self, tmp_path: Path):
        ep, frames, actor, teacher = _make_minimal_episode(tmp_path)
        actor[-1]["q"] = [99.0] * N_JOINTS
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("terminal q does not match" in i["reason"] for i in audit.validation_issues)

    def test_terminal_not_last_rejected(self, tmp_path: Path):
        ep, frames, actor, teacher = _make_minimal_episode(tmp_path)
        actor.insert(0, actor.pop())
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("terminal is not the last" in i["reason"] for i in audit.validation_issues)


class TestFinalReleaseSpan:
    def test_trailing_empty_span_boundary(self):
        teacher: list[dict] = []
        for i in range(1, 501):
            teacher.append({
                "sim_time_s": i * DT_S,
                "contacts": [{"kind": "tip_pad_top"}],
                "aggregate": {"index": {"normal_force_n": 0.8}},
            })
        for i in range(501, 750):
            teacher.append({"sim_time_s": i * DT_S, "contacts": [], "aggregate": {}})
        first_empty, last_sim, span = _final_release_empty_span_s(teacher)
        assert span == pytest.approx(0.496, abs=1e-9)
        teacher.append({"sim_time_s": 750 * DT_S, "contacts": [], "aggregate": {}})
        first_empty, last_sim, span = _final_release_empty_span_s(teacher)
        assert span == pytest.approx(0.498, abs=1e-9)
        teacher.append({"sim_time_s": 751 * DT_S, "contacts": [], "aggregate": {}})
        first_empty, last_sim, span = _final_release_empty_span_s(teacher)
        assert span == pytest.approx(0.500, abs=1e-9)

    def test_short_final_release_rejected(self, tmp_path: Path):
        ep, frames, actor, teacher = _make_minimal_episode(tmp_path, total_steps=160, contact_end_step=110)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("empty-contact span" in i["reason"] for i in audit.validation_issues)


class TestActionWindowAndClockDomain:
    def test_action_window_end_is_last_executed_step(self, tmp_path: Path):
        ep, frames, actor, teacher = _make_minimal_episode(tmp_path)
        audit, samples = audit_episode(ep, tmp_path)
        assert audit.accepted
        for s in samples:
            w = s["action_window"]
            last_step_index = w["end_exclusive_step_index"] - 1
            assert w["end_sim_time_s"] == last_step_index * DT_S
            assert w["end_sim_time_s"] != (last_step_index + 1) * DT_S
            assert w["consumed_prev_sim_time_s"] == (w["first_step_index"] - 1) * DT_S
            assert w["end_exclusive_step_index"] - w["first_step_index"] == SUBSTEPS_PER_FRAME

    def test_clock_domain_in_metadata_not_actor_input(self, tmp_path: Path):
        ep, frames, actor, teacher = _make_minimal_episode(tmp_path)
        audit, samples = audit_episode(ep, tmp_path)
        assert audit.accepted
        for i, s in enumerate(samples):
            ai = s["actor_input"]
            assert "native_capture_clock_id" not in ai
            assert "clock_offset_uncertainty_s" not in ai
            assert "frame_monotonic_time_s" not in ai
            assert ai["error_xy_px"] is not None
            assert len(ai["q_rad16"]) == N_JOINTS
            assert len(ai["qvel_rad_s16"]) == N_JOINTS
            assert len(ai["previous_ctrl_rad16"]) == N_JOINTS
            labels = s["labels"]
            assert "executed_ctrl_rad" in labels
            assert "delta_from_frame_previous_ctrl_rad" in labels
            assert "nominal_endpoint_rad_for_reference_only" in labels
            md = s["metadata"]
            assert md["native_capture_clock_id"] == "native_app_monotonic"
            assert md["simulation_clock_id"] == "episode_simulation_relative"
            assert md["frame_monotonic_time_s"] == actor[i]["frame_monotonic_time_s"]
            assert md["clock_offset_s"] is None
            assert md["clock_uncertainty_s"] is None
