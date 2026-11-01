"""Pure offline helpers for auditing and indexing LEAP v1 force-governed pilot episodes.

No environment launch, no physics stepping, no solver imports at import time.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from embodied_trackpad.leap_v1.pilot_provenance import validate_capture_timing
from embodied_trackpad.leap_v1.visual_servo import decode_png_bytes, extract_visual_estimate

N_JOINTS = 16
SUBSTEPS_PER_FRAME = 25
DT_S = 0.002
EXPECTED_PNG_W = 800
EXPECTED_PNG_H = 600
MAX_PAD_FORCE_N = 5.0
PENETRATION_TOL_M = 0.001
RELEASE_MIN_EMPTY_S = 0.5
MAX_TERMINAL_ERROR_PX = 3.0
MIN_INITIAL_ERROR_PX = 10.0
MIN_ERROR_REDUCTION = 0.5

TRAIN_SEEDS = {0, 2, 3, 5, 6, 8, 9, 11}
VAL_SEEDS = {1, 7}
TEST_SEEDS = {4, 10}

# near_diagonal_v1 curriculum groups (whole episodes; seeds 0..3 are the short
# 12px diagonal offsets, seeds 4..7 the matching 18px diagonals).
DIAGONAL_TRAIN_SEEDS = {0, 1, 2, 3}
DIAGONAL_VAL_SEEDS = {4, 6}
DIAGONAL_TEST_SEEDS = {5, 7}
DEFAULT_TASK_PROFILE = "near_pointing_v1"
KNOWN_TASK_PROFILES = ("near_pointing_v1", "near_diagonal_v1")
COLLECTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _validate_collection_request(task_profile: str, collection_id: str | None) -> None:
    """Shared request-gate for profile-scoped audit/manifest identity."""
    if task_profile not in KNOWN_TASK_PROFILES:
        raise ValueError(f"unknown task_profile {task_profile!r}")
    if task_profile == "near_diagonal_v1" and (
            not isinstance(collection_id, str)
            or not COLLECTION_ID_RE.fullmatch(collection_id)):
        raise ValueError(
            "near_diagonal_v1 requires a nonempty collection_id matching [A-Za-z0-9_-]+")

REQUIRED_STEP_CHECKS = {
    "pre_step_state_finite",
    "finite_state",
    "raw_contacts_valid",
    "pad_force<=5N",
    "penetration<=tol",
    "no_unintended_contact",
    "joint_excursion<=tol",
    "no_new_warnings",
}
PERMITTED_CONTACT_KINDS = {"tip_pad_top"}


def split_of(seed: int, *, task_profile: str = DEFAULT_TASK_PROFILE) -> str:
    """Whole-episode split for a seed under a task profile.

    The default profile (near_pointing_v1) preserves the historical axial
    mapping exactly. An unknown profile is rejected, never silently treated
    as axial.
    """
    if task_profile == "near_pointing_v1":
        train, val, test = TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS
    elif task_profile == "near_diagonal_v1":
        train, val, test = DIAGONAL_TRAIN_SEEDS, DIAGONAL_VAL_SEEDS, DIAGONAL_TEST_SEEDS
    else:
        return "rejected"
    if seed in train:
        return "train"
    if seed in val:
        return "validation"
    if seed in test:
        return "test"
    return "rejected"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i}: invalid JSON") from exc
    return out


def _is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _finite_len16(xs: Any) -> bool:
    return isinstance(xs, list) and len(xs) == N_JOINTS and all(_is_finite_number(v) for v in xs)


def _rel_path_from_repo(full_path: str | Path, repo_root: Path) -> str:
    p = Path(full_path).resolve()
    root = repo_root.resolve()
    try:
        rel = p.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path {full_path} is outside repo root {repo_root}") from exc
    s = str(rel)
    if s.startswith("..") or os.path.isabs(s):
        raise ValueError(f"escaped path {s}")
    return s


def _episode_dir_size(ep: Path) -> int:
    total = 0
    for entry in os.scandir(ep):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def _final_release_empty_span_s(
    teacher: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    """Return (first_empty_sim_time_s, last_sim_time_s, span_s) for the trailing
    no-contact region. ``span_s`` is ``None`` when no trailing empty region exists.
    """
    first_empty: float | None = None
    last_sim: float | None = None
    for rec in reversed(teacher):
        sim = rec.get("sim_time_s")
        if last_sim is None and _is_finite_number(sim):
            last_sim = float(sim)
        if not rec.get("contacts") and not rec.get("aggregate"):
            if _is_finite_number(sim):
                first_empty = float(sim)
        else:
            break
    if first_empty is None or last_sim is None:
        return first_empty, last_sim, None
    return first_empty, last_sim, last_sim - first_empty


@dataclass
class EpisodeAudit:
    seed: int
    episode_dir: str
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    task_profile: str = DEFAULT_TASK_PROFILE
    collection_id: str | None = None
    accepted: bool = False
    rejection_reasons: list[str] = field(default_factory=list)
    validation_issues: list[dict[str, Any]] = field(default_factory=list)
    action_frames: int = 0
    terminal_frame: dict[str, Any] | None = None
    png_verified: int = 0
    png_failed: int = 0
    teacher_steps: int = 0
    actor_entries: int = 0
    timing_validated_frames: int = 0
    timing_legacy_frames: int = 0
    touch_messages: int = 0
    force_intents: int = 0
    force_intent_mismatches: int = 0
    causality_mismatches: int = 0
    first_frame_time: float | None = None
    last_frame_time: float | None = None
    first_sim_time: float | None = None
    last_sim_time: float | None = None
    raw_bytes_counted: int = 0
    summary_bytes: int = 0
    pre_summary_bytes: int = 0
    terminal_png_sha256: str | None = None
    raw_identity: dict[str, Any] = field(default_factory=dict)


def _record_issue(audit: EpisodeAudit, scope: str, reason: str, details: dict[str, Any] | None = None) -> None:
    audit.validation_issues.append({"scope": scope, "reason": reason, "details": details or {}})


def _reject(audit: EpisodeAudit, reason: str) -> None:
    audit.rejection_reasons.append(reason)
    audit.accepted = False


def _validate_summary(audit: EpisodeAudit, metadata: dict[str, Any], acceptance: dict[str, Any] | None) -> bool:
    s = audit.summary
    ok = True
    required = [
        "seed", "near_pointing_success", "physics_unsafe", "child_reaped",
        "release_measured", "steps", "wall_s", "bytes", "stop_reason",
        "controller_kind", "planner_kind", "training_eligible", "accepted_frames",
    ]
    for k in required:
        if k not in s:
            _record_issue(audit, "summary", f"missing field {k}")
            ok = False
    if s.get("seed") != audit.seed:
        _record_issue(audit, "summary", f"seed mismatch summary={s.get('seed')} dir={audit.seed}")
        ok = False
    if s.get("near_pointing_success") is not True:
        _record_issue(audit, "summary", "near_pointing_success not true")
        ok = False
    if s.get("physics_unsafe") is not False:
        _record_issue(audit, "summary", "physics_unsafe not false")
        ok = False
    if s.get("child_reaped") is not True:
        _record_issue(audit, "summary", "child_reaped not true")
        ok = False
    if s.get("release_measured") is not True:
        _record_issue(audit, "summary", "release_measured not true")
        ok = False
    if s.get("stop_reason") != "success":
        _record_issue(audit, "summary", f"stop_reason={s.get('stop_reason')!r}")
        ok = False
    if s.get("training_eligible") is not False:
        _record_issue(audit, "summary", "training_eligible not false")
        ok = False
    if metadata.get("seed") != audit.seed:
        _record_issue(audit, "metadata", f"metadata seed mismatch")
        ok = False
    if metadata.get("training_eligible") is not False:
        _record_issue(audit, "metadata", "metadata training_eligible not false")
        ok = False
    ctrl_kind = s.get("controller_kind")
    if ctrl_kind != metadata.get("controller_kind"):
        _record_issue(audit, "metadata", f"controller_kind mismatch summary/metadata")
        ok = False
    if acceptance is not None:
        if ctrl_kind != acceptance.get("controller_kind"):
            _record_issue(audit, "summary", f"controller_kind {ctrl_kind!r} != acceptance")
            ok = False
        if s.get("planner_kind") != acceptance.get("config", {}).get("planner_kind"):
            _record_issue(audit, "summary", f"planner_kind mismatch acceptance")
            ok = False
    return ok


def _validate_teacher_steps(teacher: list[dict[str, Any]], audit: EpisodeAudit) -> dict[int, dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    prev_index = 0
    for rec in teacher:
        idx = rec.get("step_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            _record_issue(audit, "teacher", f"bad step_index {idx!r}")
            continue
        if prev_index == 0:
            if idx != 1:
                _record_issue(audit, "teacher", f"first step_index {idx} != 1")
        elif idx != prev_index + 1:
            _record_issue(audit, "teacher", f"non-contiguous step_index {prev_index}->{idx}")
        prev_index = idx
        if idx in by_index:
            _record_issue(audit, "teacher", f"duplicate step_index {idx}")
        by_index[idx] = rec

        sim_t = rec.get("sim_time_s")
        if not _is_finite_number(sim_t):
            _record_issue(audit, "teacher", f"step {idx} sim_time_s invalid")
        else:
            expected = idx * DT_S
            if not math.isclose(float(sim_t), expected, abs_tol=1e-9, rel_tol=0.0):
                _record_issue(audit, "teacher", f"step {idx} sim_time_s {sim_t} != {expected}")

        if not _finite_len16(rec.get("ctrl")):
            _record_issue(audit, "teacher", f"step {idx} ctrl invalid")
        if not _finite_len16(rec.get("qpos")):
            _record_issue(audit, "teacher", f"step {idx} qpos invalid")
        if not _finite_len16(rec.get("qvel")):
            _record_issue(audit, "teacher", f"step {idx} qvel invalid")
        cs = rec.get("ctrl_shape")
        if not (isinstance(cs, list) and cs == [N_JOINTS]):
            _record_issue(audit, "teacher", f"step {idx} ctrl_shape {cs!r}")

        checks = rec.get("checks", {})
        missing = REQUIRED_STEP_CHECKS - set(checks.keys())
        if missing:
            _record_issue(audit, "teacher", f"step {idx} missing checks {sorted(missing)}")
        for k in REQUIRED_STEP_CHECKS:
            if checks.get(k) is not True:
                _record_issue(audit, "teacher", f"step {idx} check {k} not True")

        if rec.get("ok") is not True:
            _record_issue(audit, "teacher", f"step {idx} ok not True")
        if rec.get("failure_reasons"):
            _record_issue(audit, "teacher", f"step {idx} has failure_reasons")
        wd = rec.get("warning_delta")
        if not isinstance(wd, list) or len(wd) != 7 or any(v != 0 for v in wd):
            _record_issue(audit, "teacher", f"step {idx} warning_delta not seven zeros")

        raw_total_force = 0.0
        contacts = rec.get("contacts")
        if not isinstance(contacts, list):
            _record_issue(audit, "teacher", f"step {idx} contacts not a list")
            contacts = []
        for c in contacts:
            if not isinstance(c, dict):
                _record_issue(audit, "teacher", f"step {idx} contact not a dict")
                continue
            if c.get("kind") not in PERMITTED_CONTACT_KINDS:
                _record_issue(audit, "teacher", f"step {idx} contact kind {c.get('kind')!r}")
            fn = c.get("normal_force_n")
            if not _is_finite_number(fn):
                _record_issue(audit, "teacher", f"step {idx} contact force non-finite")
            elif fn < -1e-9:
                _record_issue(audit, "teacher", f"step {idx} contact force negative")
            elif fn > MAX_PAD_FORCE_N + 1e-9:
                _record_issue(audit, "teacher", f"step {idx} contact force {fn} > {MAX_PAD_FORCE_N}N")
            else:
                raw_total_force += float(fn)
            dist = c.get("dist_m")
            if not _is_finite_number(dist):
                _record_issue(audit, "teacher", f"step {idx} contact dist non-finite")
            elif dist < -PENETRATION_TOL_M - 1e-9:
                _record_issue(audit, "teacher", f"step {idx} penetration {dist} < -{PENETRATION_TOL_M}m")
            expected_shapes = {"force6": 6, "frame": 9, "pos_world": 3, "pad_local": 3}
            for key, exp_len in expected_shapes.items():
                vec = c.get(key)
                if not isinstance(vec, list) or len(vec) != exp_len or not all(_is_finite_number(v) for v in vec):
                    _record_issue(audit, "teacher", f"step {idx} contact {key} not {exp_len} finite vector")
        if raw_total_force > MAX_PAD_FORCE_N + 1e-9:
            _record_issue(audit, "teacher", f"step {idx} total pad force {raw_total_force} > {MAX_PAD_FORCE_N}N")

        aggregate = rec.get("aggregate")
        if not isinstance(aggregate, dict):
            _record_issue(audit, "teacher", f"step {idx} aggregate not a dict")
        else:
            agg_total_force = 0.0
            for fid, agg in aggregate.items():
                if not isinstance(fid, str):
                    _record_issue(audit, "teacher", f"step {idx} aggregate key not string")
                    continue
                if not isinstance(agg, dict):
                    _record_issue(audit, "teacher", f"step {idx} aggregate {fid} not a dict")
                    continue
                afn = agg.get("normal_force_n")
                if not _is_finite_number(afn):
                    _record_issue(audit, "teacher", f"step {idx} aggregate force non-finite")
                elif afn < -1e-9:
                    _record_issue(audit, "teacher", f"step {idx} aggregate force negative")
                elif afn > MAX_PAD_FORCE_N + 1e-9:
                    _record_issue(audit, "teacher", f"step {idx} aggregate force > {MAX_PAD_FORCE_N}N")
                else:
                    agg_total_force += float(afn)
                if not _is_finite_number(agg.get("x_m")) or not _is_finite_number(agg.get("y_m")):
                    _record_issue(audit, "teacher", f"step {idx} aggregate x/y non-finite")
            if contacts and not math.isclose(raw_total_force, agg_total_force, abs_tol=1e-9, rel_tol=0.0):
                _record_issue(audit, "teacher", f"step {idx} aggregate force sum {agg_total_force} != raw total {raw_total_force}")

    return by_index


def _teacher_by_frame(teacher: list[dict[str, Any]], audit: EpisodeAudit) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for rec in teacher:
        fid = rec.get("frame_id")
        if fid is None:
            continue
        if not isinstance(fid, int) or isinstance(fid, bool):
            _record_issue(audit, "teacher", f"bad frame_id {fid!r}")
            continue
        by_frame.setdefault(fid, []).append(rec)
    for fid, steps in by_frame.items():
        if len(steps) != SUBSTEPS_PER_FRAME:
            _record_issue(audit, "teacher", f"frame {fid} has {len(steps)} steps, expected {SUBSTEPS_PER_FRAME}")
        idxs = [r["step_index"] for r in steps]
        if idxs and (max(idxs) - min(idxs) + 1 != len(idxs) or any(idxs[i] + 1 != idxs[i + 1] for i in range(len(idxs) - 1))):
            _record_issue(audit, "teacher", f"frame {fid} step indices not contiguous")
        times = [r.get("sim_time_s") for r in steps]
        if len(times) == SUBSTEPS_PER_FRAME:
            expected = times[0] + np.arange(SUBSTEPS_PER_FRAME) * DT_S
            if not np.allclose(times, expected, atol=1e-9):
                _record_issue(audit, "teacher", f"frame {fid} step times not contiguous dt={DT_S}")
    return by_frame


def _validate_force_intents(
    planner: list[dict[str, Any]],
    teacher_by_index: dict[int, dict[str, Any]],
    audit: EpisodeAudit,
) -> dict[int, dict[str, Any]]:
    intents: dict[int, dict[str, Any]] = {}
    last_intended = 0
    for rec in planner:
        if rec.get("kind") != "force_substep_intent":
            continue
        audit.force_intents += 1
        intended = rec.get("intended_next_step_index")
        consumed = rec.get("consumed_record_step_index")
        if not isinstance(intended, int) or not isinstance(consumed, int):
            _record_issue(audit, "planner", "force_intent missing integer indices")
            continue
        if intended != consumed + 1:
            _record_issue(audit, "planner", f"intent causality broken {consumed}->{intended}")
            continue
        if intended in intents:
            _record_issue(audit, "planner", f"duplicate intended step {intended}")
        intents[intended] = rec
        if intended <= last_intended:
            _record_issue(audit, "planner", f"intended indices not strictly increasing {intended}")
        last_intended = intended

        t_intended = teacher_by_index.get(intended)
        t_consumed = teacher_by_index.get(consumed)
        if t_intended is None or t_consumed is None:
            _record_issue(audit, "planner", f"intent references missing teacher steps {consumed}/{intended}")
            continue

        if rec.get("frame_id") != t_intended.get("frame_id"):
            audit.force_intent_mismatches += 1
            _record_issue(audit, "planner", f"intent frame_id {rec.get('frame_id')} != intended step {intended} frame_id {t_intended.get('frame_id')}")

        if rec.get("consumed_record_sim_time_s") != t_consumed.get("sim_time_s"):
            _record_issue(audit, "planner", f"intent consumed sim_time mismatch {consumed}")
        if rec.get("prior_ctrl") != t_consumed.get("ctrl"):
            audit.force_intent_mismatches += 1
            _record_issue(audit, "planner", f"prior_ctrl mismatch at consumed {consumed}")

        measured = rec.get("measured_force_n")
        agg_force = 0.0
        for agg in (t_consumed.get("aggregate") or {}).values():
            if isinstance(agg, dict):
                agg_force += float(agg.get("normal_force_n", 0.0))
        if not _is_finite_number(measured) or abs(measured - agg_force) > 1e-9:
            _record_issue(audit, "planner", f"measured_force mismatch at consumed {consumed}: {measured} vs {agg_force}")

        if rec.get("corrected_ctrl") != t_intended.get("ctrl"):
            audit.force_intent_mismatches += 1
            _record_issue(audit, "planner", f"corrected_ctrl mismatch at intended {intended}")
    return intents


def _validate_touch(
    touch: list[dict[str, Any]],
    teacher_by_sim_time: dict[float, dict[str, Any]],
    audit: EpisodeAudit,
) -> None:
    seq = 0
    for rec in touch:
        audit.touch_messages += 1
        payload = rec.get("payload")
        ack = rec.get("ack")
        if not isinstance(payload, dict):
            _record_issue(audit, "touch", "payload not a dict")
            payload = {}
        if not isinstance(ack, dict):
            _record_issue(audit, "touch", "ack not a dict")
            ack = {}
        if payload.get("cmd") != "touch":
            _record_issue(audit, "touch", f"payload cmd {payload.get('cmd')!r}")
        if payload.get("schema_version") != "touch.v1":
            _record_issue(audit, "touch", f"schema_version {payload.get('schema_version')!r}")
        if ack.get("type") != "touch_ok":
            _record_issue(audit, "touch", f"ack type {ack.get('type')!r}")
        pseq = payload.get("sequence")
        aseq = ack.get("sequence")
        if not isinstance(pseq, int) or pseq != seq + 1:
            _record_issue(audit, "touch", f"sequence gap {seq}->{pseq}")
        else:
            seq = pseq
        if pseq != aseq:
            _record_issue(audit, "touch", f"payload/ack sequence mismatch {pseq}!={aseq}")
        sim_t = payload.get("simulation_time_s")
        if not _is_finite_number(sim_t):
            _record_issue(audit, "touch", "missing simulation_time_s")
            continue
        step = teacher_by_sim_time.get(float(sim_t))
        if step is None:
            _record_issue(audit, "touch", f"touch sim_t {sim_t} has no exact teacher step")
            continue
        p_contacts = payload.get("contacts")
        if not isinstance(p_contacts, list):
            _record_issue(audit, "touch", f"contacts not a list at {sim_t}")
            continue
        aggregate = step.get("aggregate")
        if not isinstance(aggregate, dict):
            _record_issue(audit, "touch", f"aggregate not a dict at {sim_t}")
            aggregate = {}
        if not p_contacts and aggregate:
            _record_issue(audit, "touch", f"empty touch payload hides nonempty aggregate at {sim_t}")
        if len(p_contacts) != len(aggregate):
            _record_issue(audit, "touch", f"contact count mismatch at {sim_t}: payload {len(p_contacts)} aggregate {len(aggregate)}")
        payload_fids: set[str] = set()
        for pc in p_contacts:
            if not isinstance(pc, dict):
                _record_issue(audit, "touch", f"contact not a dict at {sim_t}")
                continue
            fid = pc.get("finger_id")
            if not isinstance(fid, str):
                _record_issue(audit, "touch", f"touch finger_id invalid at {sim_t}")
                continue
            if fid in payload_fids:
                _record_issue(audit, "touch", f"duplicate finger_id {fid} at {sim_t}")
            payload_fids.add(fid)
            if fid not in aggregate:
                _record_issue(audit, "touch", f"touch finger {fid} missing aggregate at {sim_t}")
                continue
            a = aggregate[fid]
            if not isinstance(a, dict):
                _record_issue(audit, "touch", f"aggregate entry for {fid} not a dict at {sim_t}")
                continue
            for key in ("x_m", "y_m", "normal_force_n"):
                pv = pc.get(key)
                av = a.get(key)
                if not _is_finite_number(pv):
                    _record_issue(audit, "touch", f"{key} non-finite for {fid} at {sim_t}")
                elif not _is_finite_number(av):
                    _record_issue(audit, "touch", f"{key} aggregate non-finite for {fid} at {sim_t}")
                elif abs(float(pv) - float(av)) > 1e-9:
                    _record_issue(audit, "touch", f"{key} mismatch for {fid} at {sim_t}")
        missing_fids = set(aggregate.keys()) - payload_fids
        if missing_fids:
            _record_issue(audit, "touch", f"missing fingers {sorted(missing_fids)} at {sim_t}")
    audit.touch_sequences = seq


def _recompute_visual_error(png_path: Path) -> tuple[float, float] | None:
    rgb = decode_png_bytes(png_path.read_bytes())
    if rgb.shape != (EXPECTED_PNG_H, EXPECTED_PNG_W, 3):
        raise ValueError(f"shape {rgb.shape}")
    est = extract_visual_estimate(rgb)
    return est.error_px


def _resolve_png_path(png_ref: Any, ep: Path, root: Path) -> Path | None:
    """Resolve a capture png_ref to a candidate path.

    Absolute refs keep legacy behavior. Relative refs are episode-basename
    (legacy) or repo-relative (native-capture export, e.g.
    ``.local/v1-visual/<collection>/seed-002/cycle-001.png``); the repo-root
    fallback applies only when the file actually exists there. Later checks
    pin the resolved file to the exact episode directory, so other episodes,
    path traversal and outside-root paths are still rejected.
    """
    if not isinstance(png_ref, str) or not png_ref:
        return None
    if Path(png_ref).is_absolute():
        return Path(png_ref)
    ep_candidate = ep / png_ref
    if ep_candidate.exists():
        return ep_candidate
    root_candidate = root / png_ref
    if root_candidate.exists():
        return root_candidate
    return ep_candidate


def _validate_png(
    a: dict[str, Any],
    ep: Path,
    root: Path,
    audit: EpisodeAudit,
) -> tuple[Path, tuple[float, float] | None] | None:
    fid = a.get("frame_id")
    png_ref = a.get("png_ref", "")
    png_path = _resolve_png_path(png_ref, ep, root)
    if png_path is None or not png_path.exists():
        _record_issue(audit, "png", f"frame {fid} missing PNG {png_ref}")
        audit.png_failed += 1
        return None
    try:
        rel_png = _rel_path_from_repo(png_path, root)
    except ValueError as exc:
        _record_issue(audit, "png", f"frame {fid} path escape: {exc}")
        audit.png_failed += 1
        return None
    expected_cycle = f"cycle-{a.get('cycle'):03d}.png"
    if png_path.name != expected_cycle:
        _record_issue(audit, "png", f"frame {fid} PNG name {png_path.name!r} != {expected_cycle!r}")
        audit.png_failed += 1
        return None
    if png_path.resolve().parent != ep.resolve():
        _record_issue(audit, "png", f"frame {fid} PNG not in episode dir")
        audit.png_failed += 1
        return None
    expected_sha = a.get("png_sha256")
    actual_sha = sha256_file(png_path)
    if expected_sha != actual_sha:
        _record_issue(audit, "png", f"frame {fid} sha mismatch")
        audit.png_failed += 1
        return None
    try:
        error_xy = _recompute_visual_error(png_path)
    except Exception as exc:
        _record_issue(audit, "png", f"frame {fid} decode/extract error: {exc}")
        audit.png_failed += 1
        return None
    audit.png_verified += 1
    return png_path, error_xy


def audit_episode(
    episode_dir: str | Path,
    repo_root: str | Path,
    acceptance: dict[str, Any] | None = None,
    *,
    task_profile: str = DEFAULT_TASK_PROFILE,
    collection_id: str | None = None,
) -> tuple[EpisodeAudit, list[dict[str, Any]]]:
    """Return audit object and list of candidate sample-index entries.

    ``task_profile``/``collection_id`` scope the episode identity; the default
    reproduces legacy axial behavior exactly. Validation of the requested
    profile happens before any expensive per-image work.
    """
    _validate_collection_request(task_profile, collection_id)
    ep = Path(episode_dir)
    root = Path(repo_root)
    audit = EpisodeAudit(seed=-1, episode_dir=str(ep), summary={},
                         task_profile=task_profile, collection_id=collection_id)
    samples: list[dict[str, Any]] = []

    summary_path = ep / "summary.json"
    metadata_path = ep / "metadata.json"
    if not summary_path.exists():
        audit.rejection_reasons.append("missing summary.json")
        return audit, samples
    if not metadata_path.exists():
        audit.rejection_reasons.append("missing metadata.json")
        return audit, samples
    with open(summary_path, "r", encoding="utf-8") as f:
        audit.summary = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        audit.metadata = json.load(f)
    audit.seed = int(audit.summary.get("seed", -1))
    audit.teacher_steps = int(audit.summary.get("steps", 0))
    audit.raw_bytes_counted = _episode_dir_size(ep)
    audit.summary_bytes = summary_path.stat().st_size
    audit.pre_summary_bytes = audit.raw_bytes_counted - audit.summary_bytes

    if not _validate_summary(audit, audit.metadata, acceptance):
        _reject(audit, "summary/metadata validation failed")

    # Profile/collection identity gate BEFORE expensive image validation.
    # Legacy axial: an absent episode profile is allowed; any present profile
    # that conflicts with the request is rejected (never silently axial).
    meta_profile = audit.metadata.get("task_profile")
    sum_profile = audit.summary.get("task_profile")
    profile_conflict = False
    if task_profile == DEFAULT_TASK_PROFILE:
        for present in (meta_profile, sum_profile):
            if present is not None and present != DEFAULT_TASK_PROFILE:
                _reject(audit, f"task_profile conflict: episode declares {present!r} "
                               f"but request is {DEFAULT_TASK_PROFILE!r}")
                profile_conflict = True
    else:
        if meta_profile != task_profile or sum_profile != task_profile:
            _reject(audit, f"task_profile mismatch: metadata={meta_profile!r} "
                           f"summary={sum_profile!r} request={task_profile!r}")
            profile_conflict = True
    if acceptance is not None:
        acc_profile = (acceptance.get("config") or {}).get("task_profile")
        if task_profile == "near_diagonal_v1":
            if acc_profile != task_profile:
                _reject(audit, f"acceptance task_profile {acc_profile!r} != request {task_profile!r}")
                profile_conflict = True
        elif acc_profile is not None and acc_profile != task_profile:
            _reject(audit, f"acceptance task_profile {acc_profile!r} != request {task_profile!r}")
            profile_conflict = True
    if collection_id is not None:
        for src_name, doc in (("metadata", audit.metadata),
                              ("summary", audit.summary)):
            present = doc.get("collection_id")
            if present is not None and present != collection_id:
                _reject(audit, f"collection_id conflict in {src_name}: "
                               f"{present!r} != {collection_id!r}")
                profile_conflict = True
    if split_of(audit.seed, task_profile=task_profile) == "rejected":
        _reject(audit, f"seed {audit.seed} outside {task_profile} split")
        profile_conflict = True
    if profile_conflict:
        return audit, samples

    # Load JSONL
    try:
        actor = load_jsonl(ep / "actor.jsonl")
        teacher = load_jsonl(ep / "teacher.jsonl")
        planner = load_jsonl(ep / "planner.jsonl")
        touch = load_jsonl(ep / "touch.jsonl")
    except (FileNotFoundError, ValueError) as exc:
        _reject(audit, f"load error: {exc}")
        return audit, samples

    audit.actor_entries = len(actor)
    teacher_by_index = _validate_teacher_steps(teacher, audit)
    teacher_by_frame = _teacher_by_frame(teacher, audit)
    teacher_by_sim_time = {float(r["sim_time_s"]): r for r in teacher if _is_finite_number(r.get("sim_time_s"))}

    if teacher_by_index:
        idxs = sorted(teacher_by_index)
        audit.first_sim_time = teacher_by_index[idxs[0]].get("sim_time_s")
        audit.last_sim_time = teacher_by_index[idxs[-1]].get("sim_time_s")

    visual_steps = [r for r in teacher if r.get("frame_id") is not None]
    intents = _validate_force_intents(planner, teacher_by_index, audit)
    missing_intents = [r["step_index"] for r in visual_steps if r["step_index"] not in intents]
    if missing_intents:
        _record_issue(audit, "planner", f"missing force_substep_intent for visual steps {missing_intents[:5]}")
    if len(intents) != len(visual_steps):
        _record_issue(audit, "planner", f"intent count {len(intents)} != visual steps {len(visual_steps)}")

    _validate_touch(touch, teacher_by_sim_time, audit)

    # Process actor frames
    seen_frame_ids: set[int] = set()
    prev_actor: dict[str, Any] | None = None
    visual_errors: list[float] = []
    terminal_error_xy: tuple[float, float] | None = None
    terminal_index: int | None = None
    last_visual_steps: list[dict[str, Any]] | None = None

    timing_required = task_profile == "near_diagonal_v1"
    for idx, a in enumerate(actor):
        try:
            timing_info = validate_capture_timing(
                a, audit.summary, audit.metadata, required=timing_required)
        except ValueError as exc:
            _record_issue(audit, "actor",
                          f"frame {a.get('frame_id')!r} capture timing invalid: {exc}")
            continue
        if timing_info["clock_offset_s"] is None:
            audit.timing_legacy_frames += 1
        else:
            audit.timing_validated_frames += 1
        phase = a.get("phase")
        if phase == "success_terminal":
            if audit.terminal_frame is not None:
                _record_issue(audit, "actor", "duplicate terminal frame")
            audit.terminal_frame = a
            terminal_index = idx
            png_res = _validate_png(a, ep, root, audit)
            if png_res:
                png_path, terminal_error_xy = png_res
                audit.terminal_png_sha256 = sha256_file(png_path)
            if a.get("action_requested") is not False:
                _record_issue(audit, "actor", "terminal action_requested not false")
            if a.get("next_ctrl") is not None:
                _record_issue(audit, "actor", "terminal next_ctrl not None")
            tfid = a.get("frame_id")
            if teacher_by_frame.get(tfid):
                _record_issue(audit, "actor", f"terminal frame_id {tfid} has teacher steps")
            continue
        if phase != "visual_step":
            _record_issue(audit, "actor", f"unexpected actor phase {phase}")
            continue
        fid = a.get("frame_id")
        if not isinstance(fid, int) or isinstance(fid, bool):
            _record_issue(audit, "actor", f"bad frame_id {fid!r}")
            continue
        if fid in seen_frame_ids:
            _record_issue(audit, "actor", f"duplicate frame_id {fid}")
            continue
        seen_frame_ids.add(fid)

        cycle = a.get("cycle")
        if not isinstance(cycle, int) or cycle <= 0:
            _record_issue(audit, "actor", f"bad cycle {cycle!r}")

        mono = a.get("frame_monotonic_time_s")
        if not _is_finite_number(mono):
            _record_issue(audit, "actor", f"frame {fid} missing monotonic clock")
        else:
            mono_f = float(mono)
            if prev_actor is not None and mono_f <= float(prev_actor.get("frame_monotonic_time_s", -math.inf)):
                _record_issue(audit, "actor", f"frame {fid} monotonic clock not increasing")
            if audit.first_frame_time is None:
                audit.first_frame_time = mono_f
            audit.last_frame_time = mono_f

        fa = a.get("frame_age_ms")
        if not _is_finite_number(fa) or float(fa) < 0.0:
            _record_issue(audit, "actor", f"frame {fid} frame_age_ms invalid")

        if prev_actor is not None:
            if fid <= int(prev_actor.get("frame_id", -math.inf)):
                _record_issue(audit, "actor", f"frame_id {fid} not increasing")
            if isinstance(cycle, int) and cycle <= int(prev_actor.get("cycle", -math.inf)):
                _record_issue(audit, "actor", f"cycle {cycle} not increasing")

        if not _finite_len16(a.get("q")):
            _record_issue(audit, "actor", f"frame {fid} q invalid")
        if not _finite_len16(a.get("qvel")):
            _record_issue(audit, "actor", f"frame {fid} qvel invalid")
        if not _finite_len16(a.get("prev_ctrl")):
            _record_issue(audit, "actor", f"frame {fid} prev_ctrl invalid")

        png_res = _validate_png(a, ep, root, audit)
        if png_res is None:
            prev_actor = a
            continue
        png_path, error_xy = png_res
        rel_png = _rel_path_from_repo(png_path, root)

        rec_err = a.get("error_px")
        if error_xy is None:
            _record_issue(audit, "actor", f"frame {fid} visual estimate failed")
        else:
            norm = math.hypot(*error_xy)
            if not _is_finite_number(rec_err) or abs(float(rec_err) - norm) > 1e-9:
                _record_issue(audit, "actor", f"frame {fid} recorded error {rec_err} != recomputed {norm}")
            visual_errors.append(float(rec_err))

        steps = teacher_by_frame.get(fid)
        if steps is None or len(steps) != SUBSTEPS_PER_FRAME:
            _record_issue(audit, "actor", f"frame {fid} missing aligned teacher steps")
            prev_actor = a
            continue
        first_step = steps[0]
        last_step = steps[-1]
        prior_idx = first_step["step_index"] - 1
        prior_step = teacher_by_index.get(prior_idx)
        if prior_step is None:
            _record_issue(audit, "actor", f"frame {fid} missing prior step {prior_idx}")
            prev_actor = a
            continue

        if a.get("q") != prior_step.get("qpos"):
            audit.causality_mismatches += 1
            _record_issue(audit, "actor", f"frame {fid} q does not match prior step {prior_idx}")
        if a.get("qvel") != prior_step.get("qvel"):
            audit.causality_mismatches += 1
            _record_issue(audit, "actor", f"frame {fid} qvel does not match prior step {prior_idx}")
        if a.get("prev_ctrl") != prior_step.get("ctrl"):
            audit.causality_mismatches += 1
            _record_issue(audit, "actor", f"frame {fid} prev_ctrl does not match prior step {prior_idx}")

        if a.get("action_requested") is not True:
            _record_issue(audit, "actor", f"frame {fid} action_requested not true")
            prev_actor = a
            continue

        audit.action_frames += 1
        executed_ctrl = [step["ctrl"] for step in steps]
        prev_ctrl = a["prev_ctrl"]
        delta = [[step_ctrl[i] - prev_ctrl[i] for i in range(N_JOINTS)] for step_ctrl in executed_ctrl]
        sample = {
            "episode_id": audit.seed,
            "frame_id": fid,
            "split": split_of(audit.seed, task_profile=audit.task_profile),
            "target_class": audit.seed if audit.task_profile != DEFAULT_TASK_PROFILE
                            else audit.seed % 12,
            "actor_input": {
                "png_ref": rel_png,
                "png_sha256": sha256_file(png_path),
                "error_xy_px": list(error_xy) if error_xy is not None else None,
                "q_rad16": a["q"],
                "qvel_rad_s16": a["qvel"],
                "previous_ctrl_rad16": a["prev_ctrl"],
                "valid": {
                    "png_valid": True,
                    "visual_valid": error_xy is not None,
                    "q_valid": _finite_len16(a["q"]),
                    "qvel_valid": _finite_len16(a["qvel"]),
                    "previous_ctrl_valid": _finite_len16(a["prev_ctrl"]),
                    "fsr_valid4": [False] * 4,
                    "motor_current_valid16": [False] * 16,
                    "wrist_valid": False,
                },
                "fsr_N4": None,
                "motor_current_A16": None,
                "wrist": None,
            },
            "action_window": {
                "consumed_prev_step_index": prior_idx,
                "first_step_index": first_step["step_index"],
                "end_exclusive_step_index": last_step["step_index"] + 1,
                "consumed_prev_sim_time_s": prior_step["sim_time_s"],
                "first_sim_time_s": first_step["sim_time_s"],
                "end_sim_time_s": last_step["sim_time_s"],
                "dt_s": DT_S,
            },
            "labels": {
                "executed_ctrl_rad": executed_ctrl,
                "delta_from_frame_previous_ctrl_rad": delta,
                "nominal_endpoint_rad_for_reference_only": a.get("next_ctrl"),
            },
            "metadata": {
                "controller_kind": audit.metadata.get("controller_kind"),
                "planner_kind": audit.summary.get("planner_kind"),
                "native_capture_clock_id": "native_app_monotonic",
                "frame_monotonic_time_s": a["frame_monotonic_time_s"],
                "clock_offset_s": None,
                "clock_uncertainty_s": None,
                "simulation_clock_id": "episode_simulation_relative",
                "simulation_clock_epoch_note": audit.summary.get("clock_note"),
            },
            "training_eligible": False,
            "pending_coordinator": True,
        }
        if timing_info["clock_offset_s"] is not None:
            sample["metadata"].update(timing_info)
        if audit.task_profile != DEFAULT_TASK_PROFILE:
            sample["task_profile"] = audit.task_profile
            sample["collection_id"] = audit.collection_id
            sample["target_key"] = f"{audit.task_profile}:{audit.seed}"
            sample["episode_key"] = (
                f"{audit.collection_id}:{audit.task_profile}:seed-{audit.seed:03d}")
        samples.append(sample)
        prev_actor = a
        last_visual_steps = steps

    if audit.terminal_frame is None:
        _record_issue(audit, "actor", "missing terminal frame")
    else:
        t = audit.terminal_frame
        # Terminal must be unique and the final actor record.
        if terminal_index != len(actor) - 1:
            _record_issue(audit, "actor", "terminal is not the last actor record")
        # Finite positive clock/order against previous actor.
        if prev_actor is None:
            _record_issue(audit, "actor", "terminal has no preceding visual actor")
        else:
            mono = t.get("frame_monotonic_time_s")
            if not _is_finite_number(mono):
                _record_issue(audit, "actor", "terminal missing monotonic clock")
            else:
                mono_f = float(mono)
                if mono_f <= float(prev_actor.get("frame_monotonic_time_s", -math.inf)):
                    _record_issue(audit, "actor", "terminal monotonic clock not increasing")
            tfid = t.get("frame_id")
            if not isinstance(tfid, int) or isinstance(tfid, bool):
                _record_issue(audit, "actor", f"terminal bad frame_id {tfid!r}")
            elif tfid <= int(prev_actor.get("frame_id", -math.inf)):
                _record_issue(audit, "actor", "terminal frame_id not increasing")
            tcycle = t.get("cycle")
            if not isinstance(tcycle, int) or tcycle <= int(prev_actor.get("cycle", -math.inf)):
                _record_issue(audit, "actor", "terminal cycle not increasing")
        # State against the last executed (last visual) StepRecord.
        if last_visual_steps is None:
            _record_issue(audit, "actor", "terminal has no preceding visual StepRecord")
        else:
            last_step = last_visual_steps[-1]
            if t.get("q") != last_step.get("qpos"):
                audit.causality_mismatches += 1
                _record_issue(audit, "actor", "terminal q does not match last visual StepRecord")
            if t.get("qvel") != last_step.get("qvel"):
                audit.causality_mismatches += 1
                _record_issue(audit, "actor", "terminal qvel does not match last visual StepRecord")
            if t.get("prev_ctrl") != last_step.get("ctrl"):
                audit.causality_mismatches += 1
                _record_issue(audit, "actor", "terminal prev_ctrl does not match last visual StepRecord")
        # Terminal PNG-derived error is validated and included as the third final success frame.
        if terminal_error_xy is None:
            _record_issue(audit, "actor", "terminal visual estimate failed")
        else:
            rec_err = t.get("error_px")
            norm = math.hypot(*terminal_error_xy)
            if not _is_finite_number(rec_err) or abs(float(rec_err) - norm) > 1e-9:
                _record_issue(audit, "actor", f"terminal recorded error {rec_err} != recomputed {norm}")
            visual_errors.append(float(rec_err))
        if visual_errors:
            if visual_errors[0] < MIN_INITIAL_ERROR_PX:
                _record_issue(audit, "actor", f"initial error {visual_errors[0]} < {MIN_INITIAL_ERROR_PX}")
            last3 = visual_errors[-3:]
            if any(e > MAX_TERMINAL_ERROR_PX for e in last3):
                _record_issue(audit, "actor", f"last3 errors {last3} exceed {MAX_TERMINAL_ERROR_PX}")
            if visual_errors[0] > 0.0 and visual_errors[-1] / visual_errors[0] > 1.0 - MIN_ERROR_REDUCTION:
                _record_issue(audit, "actor", f"error reduction insufficient: {visual_errors[0]} -> {visual_errors[-1]}")
        first_empty, last_sim, span = _final_release_empty_span_s(teacher)
        if span is None or span < RELEASE_MIN_EMPTY_S - 1e-9:
            _record_issue(audit, "actor", f"final empty-contact span < {RELEASE_MIN_EMPTY_S}s")

    # Counts agreement: summary accepted_frames must equal the full actor record count.
    if len(actor) != int(audit.summary.get("accepted_frames", -1)):
        _record_issue(audit, "summary", f"accepted_frames {audit.summary.get('accepted_frames')} != actor records {len(actor)}")
    if len(teacher) != audit.teacher_steps:
        _record_issue(audit, "summary", f"summary steps {audit.teacher_steps} != teacher rows {len(teacher)}")

    # Raw identity snapshot
    audit.raw_identity = {
        "episode_dir": str(ep),
        "raw_bytes_counted": audit.raw_bytes_counted,
        "summary_bytes": audit.summary_bytes,
        "pre_summary_bytes": audit.pre_summary_bytes,
        "summary_sha256": sha256_file(summary_path),
        "files": [
            {"path": str(p.relative_to(ep)), "sha256": sha256_file(p)}
            for p in sorted(ep.iterdir()) if p.is_file()
        ],
    }

    audit.accepted = not audit.rejection_reasons and not audit.validation_issues
    return audit, samples


def build_dataset_manifest(
    audits: list[EpisodeAudit],
    samples: list[dict[str, Any]],
    acceptance: dict[str, Any],
    manifest_sha256: str,
    batch_summary_sha256: str,
    collection_info: dict[str, Any],
    *,
    task_profile: str = DEFAULT_TASK_PROFILE,
    collection_id: str | None = None,
) -> dict[str, Any]:
    """Build dataset_manifest.json content.

    Scoped (non-default) profiles stamp task_profile/collection_id identity
    and validate that audits/samples form one homogeneous collection;
    conflicting or mislabeled input raises ValueError instead of emitting a
    mislabeled manifest. Default arguments reproduce the legacy manifest.
    """
    _validate_collection_request(task_profile, collection_id)
    scoped = task_profile != DEFAULT_TASK_PROFILE
    seen_keys: set[str] = set()
    if scoped:
        accepted_by_key: dict[str, EpisodeAudit] = {}
        for a in audits:
            if a.task_profile != task_profile or a.collection_id != collection_id:
                raise ValueError(
                    f"mixed collection identity: audit {a.episode_dir} has "
                    f"profile={a.task_profile!r} collection={a.collection_id!r}")
            if split_of(a.seed, task_profile=task_profile) == "rejected":
                raise ValueError(f"audit seed {a.seed} outside {task_profile} split")
            ekey = f"{collection_id}:{task_profile}:seed-{a.seed:03d}"
            if ekey in accepted_by_key:
                raise ValueError(f"duplicate audit episode key {ekey}")
            accepted_by_key[ekey] = a
        for s in samples:
            if s.get("task_profile") != task_profile or s.get("collection_id") != collection_id:
                raise ValueError("sample task_profile/collection_id conflict with manifest request")
            split = split_of(s["episode_id"], task_profile=task_profile)
            if split == "rejected" or s.get("split") != split:
                raise ValueError(
                    f"sample split {s.get('split')!r} != {split!r} for seed {s['episode_id']}")
            if s.get("target_class") != s["episode_id"]:
                raise ValueError(f"sample target_class != episode_id for seed {s['episode_id']}")
            target_key = f"{task_profile}:{s['episode_id']}"
            episode_key = f"{collection_id}:{task_profile}:seed-{s['episode_id']:03d}"
            if s.get("target_key") != target_key or s.get("episode_key") != episode_key:
                raise ValueError(f"sample scoped keys wrong for seed {s['episode_id']}")
            audit = accepted_by_key.get(episode_key)
            if audit is None or audit.accepted is not True:
                raise ValueError(f"sample references missing/unaccepted audit {episode_key}")
            seen_keys.add(target_key)
        # NOTE: no global coverage requirement here: a manifest must
        # truthfully represent failed/partial collections (even zero accepted
        # samples). Full eight-class coverage is a separate CLI acceptance gate.
    elif any("task_profile" in s for s in samples) or any(
            a.task_profile != DEFAULT_TASK_PROFILE for a in audits):
        raise ValueError("scoped samples/audits in a legacy manifest request")
    split_counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0, "rejected": 0}
    for s in samples:
        split_counts[s["split"]] = split_counts.get(s["split"], 0) + 1
    target_classes = sorted({s["target_class"] for s in samples})
    validated_frames = sum(a.timing_validated_frames for a in audits)
    legacy_frames = sum(a.timing_legacy_frames for a in audits)
    invalid_frames = (sum(a.actor_entries for a in audits)
                      - validated_frames - legacy_frames)
    timing_meta = scoped or validated_frames > 0 or invalid_frames > 0
    if invalid_frames > 0:
        availability = "incomplete"  # invalid frames were rejected, never approved
    elif not validated_frames and not legacy_frames:
        availability = "unavailable"  # zero observed; missing timing is NOT legacy
    elif not legacy_frames:
        availability = "validated"
    elif not validated_frames:
        availability = "legacy"  # missing/unknown calibration
    else:
        availability = "mixed"
    per_episode = []
    total_raw = 0
    total_pre = 0
    total_sim = 0.0
    for audit in audits:
        total_raw += audit.raw_bytes_counted
        total_pre += audit.pre_summary_bytes
        total_sim += float(audit.summary.get("steps", 0)) * DT_S
        entry = {
            "seed": audit.seed,
            "split": split_of(audit.seed, task_profile=audit.task_profile),
            "accepted": audit.accepted,
            "action_frames": audit.action_frames,
            "actor_entries": audit.actor_entries,
            "teacher_steps": audit.teacher_steps,
            "wall_s": audit.summary.get("wall_s"),
            "sim_s": float(audit.summary.get("steps", 0)) * DT_S,
            "raw_bytes_counted": audit.raw_bytes_counted,
            "pre_summary_bytes": audit.pre_summary_bytes,
            "summary_bytes": audit.summary_bytes,
            "png_verified": audit.png_verified,
            "png_failed": audit.png_failed,
            "force_intents": audit.force_intents,
            "force_intent_mismatches": audit.force_intent_mismatches,
            "causality_mismatches": audit.causality_mismatches,
            "touch_messages": audit.touch_messages,
            "rejection_reasons": audit.rejection_reasons,
            "issue_count": len(audit.validation_issues),
            "raw_identity_summary_sha256": audit.raw_identity.get("summary_sha256"),
            "terminal_png_sha256": audit.terminal_png_sha256,
        }
        if scoped:
            entry["task_profile"] = audit.task_profile
            entry["collection_id"] = audit.collection_id
            entry["target_key"] = f"{audit.task_profile}:{audit.seed}"
            entry["episode_key"] = (
                f"{audit.collection_id}:{audit.task_profile}:seed-{audit.seed:03d}")
        if timing_meta:
            entry["timing_validated_frames"] = audit.timing_validated_frames
            entry["timing_legacy_frames"] = audit.timing_legacy_frames
            entry["timing_invalid_frames"] = (
                audit.actor_entries - audit.timing_validated_frames
                - audit.timing_legacy_frames)
        per_episode.append(entry)
    manifest = {
        "created_utc": None,
        "source_commit": acceptance.get("source_commit"),
        "asset_manifest_sha256": acceptance.get("asset_manifest", {}).get("sha256"),
        "native_binary_sha256": acceptance.get("native_binary", {}).get("sha256"),
        "planner_kind": acceptance.get("config", {}).get("planner_kind"),
        "controller_kind": acceptance.get("controller_kind"),
        "raw_flags": {
            "training_eligible": False,
            "privileged_visual_teacher": True,
        },
        "training_eligible": False,
        "pending_coordinator": True,
        "input_manifest_sha256": manifest_sha256,
        "input_batch_summary_sha256": batch_summary_sha256,
        "total_episodes": len(audits),
        "accepted_episodes": sum(1 for a in audits if a.accepted),
        "rejected_episodes": sum(1 for a in audits if not a.accepted),
        "total_action_frames": len(samples),
        "split_counts": split_counts,
        "target_classes": target_classes,
        "raw_bytes_counted_including_summaries": total_raw,
        "pre_summary_bytes_reported": total_pre,
        "total_sim_s": round(total_sim, 3),
        "collection_wall_s": collection_info.get("collection_wall_s"),
        "episode_wall_sum_s": collection_info.get("episode_wall_sum_s"),
        "worker_invocation_elapsed_s": collection_info.get("worker_invocation_elapsed_s"),
        "auditor_wall_s": collection_info.get("auditor_wall_s"),
        "per_episode": per_episode,
        "limitations": [
            "raw episodes remain training_eligible=false; coordinator audit required before training",
            "visual error_xy_px is image-derived and may be null if cursor/target not located",
            "fsr, motor current and wrist are missing sensors, values are null with false valid masks",
            "force/Jacobian/oracle coordinates/reward/outcome are not in actor_input",
            "action labels are actual executed StepRecord.ctrl, not nominal endpoints",
            "absolute native monotonic capture clock is metadata, not a trained actor feature",
            "clock offset and uncertainty are unavailable for this offline export",
        ],
    }
    if scoped:
        manifest["task_profile"] = task_profile
        manifest["collection_id"] = collection_id
        manifest["target_keys"] = sorted(seen_keys)
    if timing_meta:
        manifest["capture_timing"] = {
            "validated_frames": validated_frames,
            "legacy_frames": legacy_frames,
            "invalid_frames": invalid_frames,
            "frame_availability": availability,
        }
        clock_note = ("clock offset and uncertainty are unavailable "
                      "for this offline export")
        replacements = {
            "validated": ("capture timing validated per frame against the "
                          "summary calibration; clock offset/uncertainty are "
                          "measured values, not fabricated"),
            "mixed": ("clock offset and uncertainty are unavailable for "
                      "legacy frames in this mixed export"),
            "incomplete": ("capture timing invalid for some frames; those "
                           "frames were rejected and never approved for export"),
        }
        replacement = replacements.get(availability)
        if replacement is not None:
            manifest["limitations"] = [
                replacement if lim == clock_note else lim
                for lim in manifest["limitations"]]
    return manifest
