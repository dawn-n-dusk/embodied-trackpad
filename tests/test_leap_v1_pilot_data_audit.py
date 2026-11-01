"""Offline audit/index tests for LEAP v1 pilot episodes.

No physics stepping, no app launch, no live data collection.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from embodied_trackpad.leap_v1.pilot_dataset import (
    N_JOINTS,
    SUBSTEPS_PER_FRAME,
    DT_S,
    audit_episode,
    build_dataset_manifest,
    load_jsonl,
    sha256_file,
    split_of,
)
from embodied_trackpad.leap_v1.visual_servo import decode_png_bytes, extract_visual_estimate


REPO_ROOT = Path(__file__).resolve().parents[1]


def _recompute_error(png_path: Path) -> float:
    rgb = decode_png_bytes(png_path.read_bytes())
    est = extract_visual_estimate(rgb)
    err = est.error_px
    assert err is not None, f"visual estimate failed for {png_path}"
    return math.hypot(*err)


def _make_png_bytes(
    width: int = 800,
    height: int = 600,
    cursor_center: tuple[int, int] = (400, 300),
    target_center: tuple[int, int] = (430, 300),
    cursor_radius: int = 8,
    target_radius: int = 15,
) -> bytes:
    """Create a minimal valid 8-bit RGB PNG with red cursor and blue target patches."""
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = max(0, cursor_center[1] - cursor_radius), max(0, cursor_center[0] - cursor_radius)
    y1, x1 = min(height, cursor_center[1] + cursor_radius), min(width, cursor_center[0] + cursor_radius)
    rgb[y0:y1, x0:x1] = [255, 0, 0]
    y0, x0 = max(0, target_center[1] - target_radius), max(0, target_center[0] - target_radius)
    y1, x1 = min(height, target_center[1] + target_radius), min(width, target_center[0] + target_radius)
    rgb[y0:y1, x0:x1] = [0, 100, 255]

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y].tobytes())
    compressed = zlib.compress(bytes(raw), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    out = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    return out


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _base_summary(seed: int = 7, action_frames: int = 4, steps: int = 600) -> dict:
    return {
        "seed": seed,
        "near_pointing_success": True,
        "physics_unsafe": False,
        "child_reaped": True,
        "release_measured": True,
        "steps": steps,
        "accepted_frames": action_frames,
        "initial_error_px": 30.0,
        "final_error_px": 0.0,
        "wall_s": 1.0,
        "bytes": 1024,
        "stop_reason": "success",
        "controller_kind": "privileged_visual_teacher",
        "planner_kind": "force-governed",
        "training_eligible": False,
    }


def _make_teacher(
    seed_frames: dict[int, dict],
    total_steps: int = 600,
    contacts_start: int = 100,
    contacts_end: int = 300,
) -> list[dict]:
    records: list[dict] = []
    sorted_frames = sorted(seed_frames.items())
    first_frame_id, first_info = sorted_frames[0]
    # prep steps before the first visual frame's prior step
    for i in range(1, first_info["first_step_index"] - 1):
        records.append(_prep_step(i))
    prev_last_step = 0
    for idx, (fid, info) in enumerate(sorted_frames):
        first = info["first_step_index"]
        if idx == 0:
            # explicit prior prep step for the first visual frame
            prior = _prep_step(first - 1)
            prior["qpos"] = [float(fid) * 0.001 + c * 0.001 for c in info["ctrl"]]
            records.append(prior)
        else:
            # fill any gap between the previous visual frame and this one;
            # the step just before this frame is the last step of the prior frame
            for i in range(prev_last_step + 1, first - 1):
                has_contact = contacts_start <= i <= contacts_end
                records.append(_prep_step(i, has_contact=has_contact))
        for k in range(SUBSTEPS_PER_FRAME):
            idx_step = first + k
            ctrl = info["ctrl"]
            has_contact = contacts_start <= idx_step <= contacts_end
            records.append(_visual_step(idx_step, fid, ctrl, has_contact=has_contact))
        prev_last_step = first + SUBSTEPS_PER_FRAME - 1
    # tail steps: continue physics stepping until total_steps, with contact
    # according to the configured contact window (empty tail provides release span)
    last_visual = max(info["first_step_index"] + SUBSTEPS_PER_FRAME - 1 for info in seed_frames.values())
    for i in range(last_visual + 1, total_steps + 1):
        has_contact = contacts_start <= i <= contacts_end
        records.append(_prep_step(i, has_contact=has_contact))
    return records


CONTACT_TEMPLATE = {
    "geom1": 0, "geom2": 31, "body1": "world", "body2": "if_ds",
    "kind": "tip_pad_top",
    "normal_force_n": 0.8,
    "force6": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    "frame": [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
    "dist_m": -1e-05,
    "pos_world": [0.07, 0.006, 0.1],
    "pad_local": [0.07, 0.006, 0.005],
    "top_face_cos": 1.0,
}


def _aggregate_template() -> dict:
    return {
        "index": {
            "finger_id": "index",
            "x_m": 0.07,
            "y_m": 0.006,
            "normal_force_n": 0.8,
            "contact_points": 1,
        }
    }


def _prep_step(i: int, has_contact: bool = False) -> dict:
    contacts = [dict(CONTACT_TEMPLATE)] if has_contact else []
    aggregate = _aggregate_template() if has_contact else {}
    return {
        "step_index": i,
        "sim_time_s": i * DT_S,
        "frame_id": None,
        "phase": "prep_open",
        "ctrl": [0.0] * N_JOINTS,
        "qpos": [0.0] * N_JOINTS,
        "qvel": [0.0] * N_JOINTS,
        "contacts": contacts,
        "aggregate": aggregate,
        "warning_delta": [0] * 7,
        "checks": {k: True for k in [
            "pre_step_state_finite", "finite_state", "raw_contacts_valid",
            "pad_force<=5N", "penetration<=tol", "no_unintended_contact",
            "joint_excursion<=tol", "no_new_warnings",
        ]},
        "ok": True,
        "failure_reasons": [],
        "ctrl_shape": [N_JOINTS],
    }


def _visual_step(idx: int, fid: int, base_ctrl: list[float], has_contact: bool = True) -> dict:
    ctrl = [base_ctrl[i] + 0.0001 * (idx % SUBSTEPS_PER_FRAME + 1) for i in range(N_JOINTS)]
    qpos = [0.001 * float(fid) + c * 0.001 for c in ctrl]
    contacts = [dict(CONTACT_TEMPLATE)] if has_contact else []
    aggregate = _aggregate_template() if has_contact else {}
    return {
        "step_index": idx,
        "sim_time_s": idx * DT_S,
        "frame_id": fid,
        "phase": "visual",
        "ctrl": ctrl,
        "qpos": qpos,
        "qvel": [0.0] * N_JOINTS,
        "contacts": contacts,
        "aggregate": aggregate,
        "warning_delta": [0] * 7,
        "checks": {k: True for k in [
            "pre_step_state_finite", "finite_state", "raw_contacts_valid",
            "pad_force<=5N", "penetration<=tol", "no_unintended_contact",
            "joint_excursion<=tol", "no_new_warnings",
        ]},
        "ok": True,
        "failure_reasons": [],
        "ctrl_shape": [N_JOINTS],
    }


def _make_actor(
    frames: list[dict],
    teacher_by_index: dict[int, dict],
    ep: Path,
    seed: int = 7,
) -> list[dict]:
    actor: list[dict] = []
    base_time = 1000.0
    for i, f in enumerate(frames):
        fid = f["frame_id"]
        first_step_index = f["first_step_index"]
        prior = teacher_by_index.get(first_step_index - 1, {})
        prior_q = prior.get("qpos", [0.0] * N_JOINTS)
        prior_qvel = prior.get("qvel", [0.0] * N_JOINTS)
        prior_ctrl = prior.get("ctrl", [0.0] * N_JOINTS)
        entry = {
            "phase": "visual_step",
            "cycle": i + 1,
            "png_ref": f["png_path"].name,
            "png_sha256": f["png_sha256"],
            "frame_id": fid,
            "frame_monotonic_time_s": base_time + i * 0.05,
            "frame_age_ms": 10.0 + i,
            "error_px": f["error_px"],
            "q": prior_q,
            "qvel": prior_qvel,
            "prev_ctrl": prior_ctrl,
            "next_ctrl": f["nominal"],
            "next_ctrl_is_nominal": True,
            "action_requested": f.get("action_requested", True),
        }
        actor.append(entry)
    terminal_png_path = ep / f"cycle-{len(frames) + 1:03d}.png"
    terminal_png_path.write_bytes(_make_png_bytes(
        target_center=(400, 300),
        cursor_radius=15,
        target_radius=5,
    ))
    terminal_png_sha256 = sha256_file(terminal_png_path)
    terminal_error = _recompute_error(terminal_png_path)
    last_visual_step = teacher_by_index[frames[-1]["first_step_index"] + SUBSTEPS_PER_FRAME - 1]
    actor.append({
        "phase": "success_terminal",
        "cycle": len(frames) + 1,
        "png_ref": terminal_png_path.name,
        "png_sha256": terminal_png_sha256,
        "frame_id": frames[-1]["frame_id"] + 1,
        "frame_monotonic_time_s": base_time + len(frames) * 0.05,
        "frame_age_ms": 5.0,
        "error_px": terminal_error,
        "q": last_visual_step["qpos"],
        "qvel": last_visual_step["qvel"],
        "prev_ctrl": last_visual_step["ctrl"],
        "next_ctrl": None,
        "next_ctrl_is_nominal": False,
        "action_requested": False,
    })
    return actor


def _make_planner(seed_frames: dict[int, dict], teacher_by_index: dict[int, dict]) -> list[dict]:
    planner: list[dict] = []
    nominal_delta = [0.001] + [0.0] * (N_JOINTS - 1)
    for fid, info in sorted(seed_frames.items()):
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
            agg_force = sum(float(a.get("normal_force_n", 0.0)) for a in (consumed_rec.get("aggregate") or {}).values() if isinstance(a, dict))
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


def _make_touch(contact_times: list[float]) -> list[dict]:
    return [
        {
            "payload": {
                "cmd": "touch",
                "schema_version": "touch.v1",
                "sequence": i + 1,
                "simulation_time_s": t,
                "contacts": [{"finger_id": "index", "x_m": 0.07, "y_m": 0.006, "normal_force_n": 0.8}],
            },
            "ack": {"type": "touch_ok", "sequence": i + 1},
        }
        for i, t in enumerate(contact_times)
    ]


def _make_fixture(tmp_path: Path, seed: int = 7, action_frames: int = 4, contacts_end: int = 300, target_xs: list[int] | None = None):
    ep = tmp_path / f"seed-{seed:03d}"
    ep.mkdir()

    if target_xs is None:
        target_xs = [430, 415, 402, 400][:action_frames]
    else:
        target_xs = target_xs[:action_frames]
    cursor_center = (400, 300)
    frames = []
    for i, tx in enumerate(target_xs):
        png_path = ep / f"cycle-{i+1:03d}.png"
        png_path.write_bytes(_make_png_bytes(
            target_center=(tx, 300),
            cursor_radius=15,
            target_radius=5,
        ))
        png_sha = sha256_file(png_path)
        frames.append({
            "frame_id": i + 2,
            "first_step_index": 11 + i * SUBSTEPS_PER_FRAME,
            "png_path": png_path,
            "png_sha256": png_sha,
            "error_px": _recompute_error(png_path),
            "ctrl": [0.01 * (i + 1)] * N_JOINTS,
            "nominal": [0.1 * (i + 1)] * N_JOINTS,
        })

    seed_frames = {
        f["frame_id"]: {"first_step_index": f["first_step_index"], "ctrl": f["ctrl"]}
        for f in frames
    }
    total_steps = 600
    summary = _base_summary(seed, action_frames=action_frames + 1, steps=total_steps)
    summary["initial_error_px"] = frames[0]["error_px"]
    summary["final_error_px"] = frames[-1]["error_px"]
    with open(ep / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)
    with open(ep / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"controller_kind": "privileged_visual_teacher", "training_eligible": False, "seed": seed}, f)

    teacher = _make_teacher(seed_frames, total_steps=total_steps, contacts_end=contacts_end)
    teacher_by_index = {r["step_index"]: r for r in teacher}
    _write_jsonl(ep / "teacher.jsonl", teacher)
    _write_jsonl(ep / "actor.jsonl", _make_actor(frames, teacher_by_index, ep, seed))
    _write_jsonl(ep / "planner.jsonl", _make_planner(seed_frames, teacher_by_index))
    contact_times = [r["sim_time_s"] for r in teacher if r.get("frame_id") is not None and r.get("aggregate")]
    _write_jsonl(ep / "touch.jsonl", _make_touch(contact_times[:5]))
    return ep, frames


class TestAuditPure:
    def test_passing_fixture(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        audit, samples = audit_episode(ep, tmp_path)
        assert audit.accepted, (audit.rejection_reasons, audit.validation_issues)
        assert len(samples) == len(frames)

    def test_nominal_vs_actual_label(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        audit, samples = audit_episode(ep, tmp_path)
        assert audit.accepted
        assert len(samples) >= 2
        for s in samples:
            labels = s["labels"]["executed_ctrl_rad"]
            assert len(labels) == SUBSTEPS_PER_FRAME
            assert labels[0] != s["labels"]["nominal_endpoint_rad_for_reference_only"]

    def test_action_window_semantics(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        audit, samples = audit_episode(ep, tmp_path)
        assert audit.accepted
        for s in samples:
            w = s["action_window"]
            last_step_index = w["end_exclusive_step_index"] - 1
            assert w["consumed_prev_sim_time_s"] == (w["first_step_index"] - 1) * DT_S
            assert w["end_sim_time_s"] == last_step_index * DT_S
            assert w["end_sim_time_s"] != (last_step_index + 1) * DT_S
            assert w["end_exclusive_step_index"] - w["first_step_index"] == SUBSTEPS_PER_FRAME

    def test_shifted_future_state_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[0]["q"] = [99.0] * N_JOINTS
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert audit.causality_mismatches > 0
        assert any("q does not match" in i["reason"] for i in audit.validation_issues)

    def test_mismatched_png_sha_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[0]["png_sha256"] = "0" * 64
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert audit.png_failed >= 1
        assert any("sha mismatch" in i["reason"] for i in audit.validation_issues)

    def test_bad_png_shape_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        tiny = _make_png_bytes(width=2, height=2)
        Path(frames[0]["png_path"]).write_bytes(tiny)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[0]["png_sha256"] = sha256_file(frames[0]["png_path"])
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert audit.png_failed >= 1
        assert any("shape" in i["reason"] for i in audit.validation_issues)

    def test_nonfinite_clock_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[0]["frame_monotonic_time_s"] = float("nan")
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("monotonic clock" in i["reason"] for i in audit.validation_issues)

    def test_backward_image_clock_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[1]["frame_monotonic_time_s"] = actor[0]["frame_monotonic_time_s"] - 1.0
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("monotonic clock not increasing" in i["reason"] for i in audit.validation_issues)

    def test_unsafe_step_hidden_by_summary_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        teacher = load_jsonl(ep / "teacher.jsonl")
        # summary still claims physics_unsafe=false; a hidden unsafe step must be caught
        teacher[50]["checks"]["pad_force<=5N"] = False
        teacher[50]["ok"] = False
        _write_jsonl(ep / "teacher.jsonl", teacher)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        reasons = [i["reason"] for i in audit.validation_issues]
        assert any("pad_force<=5N" in r and "not True" in r for r in reasons)

    def test_missing_teacher_steps_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        teacher = load_jsonl(ep / "teacher.jsonl")
        first_frame_id = min(r.get("frame_id") for r in teacher if r.get("frame_id") is not None)
        teacher = [r for r in teacher if not (r.get("frame_id") == first_frame_id and r["step_index"] == min(s["step_index"] for s in teacher if s.get("frame_id") == first_frame_id))]
        _write_jsonl(ep / "teacher.jsonl", teacher)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        reasons = [i["reason"] for i in audit.validation_issues]
        assert any("25 steps" in r or "aligned teacher steps" in r for r in reasons)

    def test_forged_raw_error_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[0]["error_px"] = 999.0
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("recorded error" in i["reason"] for i in audit.validation_issues)

    def test_forged_touch_x_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        touch = load_jsonl(ep / "touch.jsonl")
        touch[0]["payload"]["contacts"][0]["x_m"] = 0.071
        _write_jsonl(ep / "touch.jsonl", touch)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("x_m mismatch" in i["reason"] for i in audit.validation_issues)

    def test_bad_ack_type_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        touch = load_jsonl(ep / "touch.jsonl")
        touch[0]["ack"]["type"] = "touch_fail"
        _write_jsonl(ep / "touch.jsonl", touch)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("ack type" in i["reason"] for i in audit.validation_issues)

    def test_no_force_intents_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        planner = load_jsonl(ep / "planner.jsonl")
        planner = [p for p in planner if p.get("kind") != "force_substep_intent"]
        _write_jsonl(ep / "planner.jsonl", planner)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("missing force_substep_intent" in i["reason"] for i in audit.validation_issues)

    def test_no_terminal_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor = [a for a in actor if a.get("phase") != "success_terminal"]
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("missing terminal" in i["reason"] for i in audit.validation_issues)

    def test_terminal_action_requested_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[-1]["action_requested"] = True
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("terminal action_requested not false" in i["reason"] for i in audit.validation_issues)

    def test_duplicate_actor_frame_id_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[1]["frame_id"] = actor[0]["frame_id"]
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("duplicate frame_id" in i["reason"] for i in audit.validation_issues)

    def test_png_path_wrong_name_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        actor = load_jsonl(ep / "actor.jsonl")
        actor[0]["png_ref"] = "cycle-999.png"
        _write_jsonl(ep / "actor.jsonl", actor)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        reasons = [i["reason"] for i in audit.validation_issues]
        assert any("PNG name" in r or "missing PNG" in r for r in reasons)

    def test_sim_time_mismatch_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        teacher = load_jsonl(ep / "teacher.jsonl")
        teacher[5]["sim_time_s"] = 999.0
        _write_jsonl(ep / "teacher.jsonl", teacher)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("sim_time_s" in i["reason"] for i in audit.validation_issues)

    def test_nonfinite_ctrl_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        teacher = load_jsonl(ep / "teacher.jsonl")
        teacher[5]["ctrl"][0] = float("inf")
        _write_jsonl(ep / "teacher.jsonl", teacher)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("ctrl invalid" in i["reason"] for i in audit.validation_issues)

    def test_contact_force_over_limit_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        teacher = load_jsonl(ep / "teacher.jsonl")
        for r in teacher:
            if r.get("aggregate"):
                r["contacts"][0]["normal_force_n"] = 6.0
                r["aggregate"]["index"]["normal_force_n"] = 6.0
                break
        _write_jsonl(ep / "teacher.jsonl", teacher)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("> 5" in i["reason"] or "force" in i["reason"].lower() for i in audit.validation_issues)

    def test_penetration_over_tol_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        teacher = load_jsonl(ep / "teacher.jsonl")
        for r in teacher:
            if r.get("aggregate"):
                r["contacts"][0]["dist_m"] = -0.002
                break
        _write_jsonl(ep / "teacher.jsonl", teacher)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("penetration" in i["reason"] for i in audit.validation_issues)

    def test_initial_error_too_low_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path, target_xs=[405, 400, 395, 390])
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("initial error" in i["reason"] for i in audit.validation_issues)

    def test_last3_error_too_high_rejected(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path, action_frames=4, target_xs=[430, 415, 405, 450])
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("last3 errors" in i["reason"] for i in audit.validation_issues)

    def test_empty_contact_final_span_too_short_rejected(self, tmp_path: Path):
        # contacts end at 550 => only 50 steps (0.1s) of empty contact tail before step 600
        ep, frames = _make_fixture(tmp_path, contacts_end=550)
        audit, samples = audit_episode(ep, tmp_path)
        assert not audit.accepted
        assert any("empty-contact span" in i["reason"] for i in audit.validation_issues)

    def test_force_fields_excluded_and_valid_masks(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        audit, samples = audit_episode(ep, tmp_path)
        assert audit.accepted
        for s in samples:
            ai = s["actor_input"]
            assert "force" not in ai
            assert "jacobian" not in ai
            assert "oracle" not in ai
            assert "reward" not in ai
            assert "outcome" not in ai
            assert "seed" not in ai
            assert "class" not in ai
            assert ai["fsr_N4"] is None
            assert ai["motor_current_A16"] is None
            assert ai["wrist"] is None
            assert all(v is False for v in ai["valid"]["fsr_valid4"])
            assert all(v is False for v in ai["valid"]["motor_current_valid16"])
            assert ai["valid"]["wrist_valid"] is False
            assert "frame_monotonic_time_s" not in ai

    @pytest.mark.parametrize("seed,expected", [
        (0, "train"), (1, "validation"), (2, "train"), (3, "train"), (4, "test"),
        (5, "train"), (6, "train"), (7, "validation"), (8, "train"), (9, "train"),
        (10, "test"), (11, "train"), (12, "rejected"),
    ])
    def test_split_groups_no_leak(self, seed: int, expected: str):
        assert split_of(seed) == expected

    def test_raw_png_readonly(self, tmp_path: Path):
        ep, frames = _make_fixture(tmp_path)
        before = sha256_file(frames[0]["png_path"])
        audit, samples = audit_episode(ep, tmp_path)
        after = sha256_file(frames[0]["png_path"])
        assert before == after


class TestAuditCLI:
    def _run_cli(self, pilot: Path, acceptance: Path, repo_root: Path, out: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "audit_leap_v1_pilot.py"),
            "--pilot-dir", str(pilot),
            "--acceptance", str(acceptance),
            "--repo-root", str(repo_root),
            "--out", str(out),
        ] + (extra or [])
        env = os.environ.copy()
        env["PYTHONPATH"] = "src:."
        return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)

    def test_cli_runs_on_fixture_collection(self, tmp_path: Path):
        pilot = tmp_path / "pilot"
        pilot.mkdir()
        for seed in range(12):
            _make_fixture(pilot, seed)
        # synthetic source file, asset manifest, and native binary for global hash checks
        src_path = tmp_path / "src" / "embodied_trackpad" / "leap_v1" / "fake_source.py"
        src_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.write_text("# frozen v1 pilot source\n")
        src_sha = sha256_file(src_path)
        src_rel = "src/embodied_trackpad/leap_v1/fake_source.py"
        asset_path = tmp_path / "assets" / "manifests" / "leap-v1-full.json"
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_text('{"files": []}')
        asset_sha = sha256_file(asset_path)
        nb_path = tmp_path / ".local" / "build" / "TrackpadSandbox.app" / "Contents" / "MacOS" / "TrackpadSandbox"
        nb_path.parent.mkdir(parents=True, exist_ok=True)
        nb_path.write_text("native")
        nb_sha = sha256_file(nb_path)
        prospective_index = [
            {"seed": s, "group": split_of(s)}
            for s in range(12)
        ]
        force_config = {
            "target_force_n": 0.8,
            "gain_m_per_ns": 0.004,
            "max_normal_speed_m_s": 0.002,
            "max_joint_step_rad": 0.0002,
            "max_cycle_delta_rad": 0.005,
        }
        force_governor_config = {
            "substeps_per_frame": 25,
            "target_force_n": 0.8,
            "tangential_stop_force_n": 1.1,
            "per_substep_total_joint_delta_cap_rad": 0.0002,
            "max_cycle_delta_rad": 0.005,
        }
        manifest = {
            "source_commit": "abc123",
            "source_hash_verified": {src_rel: src_sha},
            "asset_manifest_path": "assets/manifests/leap-v1-full.json",
            "asset_manifest_sha256": asset_sha,
            "native_binary_path": ".local/build/TrackpadSandbox.app/Contents/MacOS/TrackpadSandbox",
            "native_binary_sha256": nb_sha,
            "planner_kind": "force-governed",
            "controller_kind": "privileged_visual_teacher",
            "force_config": force_config,
            "force_governor_config": force_governor_config,
            "seeds_attempt_order": list(range(12)),
            "prospective_split": {
                "group_key": "seed modulo 12",
                "train": [0, 2, 3, 5, 6, 8, 9, 11],
                "validation": [1, 7],
                "test": [4, 10],
                "prospective_index": prospective_index,
            },
            "raw_flags": {"training_eligible": False, "privileged_visual_teacher": True},
        }
        with open(pilot / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with open(pilot / "batch-summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "source_commit": "abc123",
                "asset_manifest_sha256": asset_sha,
                "native_binary_sha256": nb_sha,
                "planner_kind": "force-governed",
                "controller_kind": "privileged_visual_teacher",
                "collection_wall_s": 2.0,
                "attempted_seeds": list(range(12)),
                "successful_seeds": list(range(12)),
            }, f)
        acceptance = {
            "source_commit": "abc123",
            "asset_manifest": {"sha256": asset_sha, "path": "assets/manifests/leap-v1-full.json"},
            "native_binary": {"sha256": nb_sha, "path": ".local/build/TrackpadSandbox.app/Contents/MacOS/TrackpadSandbox"},
            "config": {
                "planner_kind": "force-governed",
                "seeds": list(range(12)),
                "force_config": force_config,
                "force_governor_config": force_governor_config,
            },
            "controller_kind": "privileged_visual_teacher",
            "prospective_split": {"train": [0, 2, 3, 5, 6, 8, 9, 11], "validation": [1, 7], "test": [4, 10]},
            "source_hashes": {src_rel: src_sha},
        }
        acc_path = tmp_path / "accepted.json"
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump(acceptance, f)
        out = tmp_path / "out"
        result = self._run_cli(pilot, acc_path, tmp_path, out)
        assert result.returncode == 0, result.stderr
        assert (out / "sample_index.jsonl").exists()
        assert (out / "audit.json").exists()
        assert (out / "dataset_manifest.json").exists()
        with open(out / "dataset_manifest.json", "r", encoding="utf-8") as f:
            dm = json.load(f)
        assert dm["total_action_frames"] == 48
        assert dm["accepted_episodes"] == 12
        assert dm["split_counts"]["train"] == 32
        assert dm["split_counts"]["validation"] == 8
        assert dm["split_counts"]["test"] == 8

    def test_global_source_hash_mismatch_rejects(self, tmp_path: Path):
        pilot = tmp_path / "pilot"
        pilot.mkdir()
        _make_fixture(pilot, 0)
        # create a tiny synthetic source file referenced by the manifest
        src_rel = "tmp-audit-source.txt"
        src_path = tmp_path / src_rel
        src_path.write_text("hello")
        real_hash = sha256_file(src_path)
        manifest = {
            "source_commit": "abc123",
            "asset_manifest_path": "assets/manifests/leap-v1-full.json",
            "asset_manifest_sha256": "0" * 64,
            "native_binary_path": ".local/build/TrackpadSandbox.app/Contents/MacOS/TrackpadSandbox",
            "native_binary_sha256": "1" * 64,
            "planner_kind": "force-governed",
            "controller_kind": "privileged_visual_teacher",
            "seeds_attempt_order": [0],
            "prospective_split": {"prospective_index": [{"seed": 0, "group": "train"}]},
            "raw_flags": {"training_eligible": False, "privileged_visual_teacher": True},
            "source_hash_verified": {src_rel: "0" * 64},
        }
        with open(pilot / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with open(pilot / "batch-summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "source_commit": "abc123",
                "asset_manifest_sha256": "0" * 64,
                "native_binary_sha256": "1" * 64,
                "planner_kind": "force-governed",
                "controller_kind": "privileged_visual_teacher",
                "collection_wall_s": 1.0,
                "attempted_seeds": [0],
                "successful_seeds": [0],
            }, f)
        acceptance = {
            "source_commit": "abc123",
            "asset_manifest": {"sha256": "0" * 64},
            "native_binary": {"sha256": "1" * 64},
            "config": {"planner_kind": "force-governed", "seeds": [0]},
            "controller_kind": "privileged_visual_teacher",
            "prospective_split": {"train": [0], "validation": [], "test": []},
            "source_hashes": {src_rel: real_hash},
        }
        acc_path = tmp_path / "accepted.json"
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump(acceptance, f)
        out = tmp_path / "out"
        result = self._run_cli(pilot, acc_path, tmp_path, out)
        assert result.returncode != 0, result.stderr
        with open(out / "audit.json", "r", encoding="utf-8") as f:
            audit = json.load(f)
        assert any("manifest source hash mismatch" in g for g in audit["global_issues"])


class TestManifestBuilder:
    def test_build_manifest_split_counts(self):
        audits = []
        samples = [
            {"split": "train", "target_class": 0},
            {"split": "train", "target_class": 2},
            {"split": "validation", "target_class": 1},
            {"split": "test", "target_class": 4},
        ]
        acceptance = {
            "source_commit": "abc", "asset_manifest": {"sha256": "0" * 64},
            "native_binary": {"sha256": "1" * 64}, "config": {"planner_kind": "x"},
            "controller_kind": "teacher",
        }
        info = {"collection_wall_s": 1.0, "episode_wall_sum_s": 0.9, "worker_invocation_elapsed_s": None, "auditor_wall_s": 0.1}
        dm = build_dataset_manifest(audits, samples, acceptance, "msha", "bsha", info)
        assert dm["split_counts"]["train"] == 2
        assert dm["split_counts"]["validation"] == 1
        assert dm["split_counts"]["test"] == 1
