#!/usr/bin/env python3
"""Offline audit and candidate sample-index builder for LEAP v1 pilot episodes.

Reads raw episodes from a pilot collection directory and writes:
  - sample_index.jsonl  (one entry per action frame)
  - audit.json          (per-episode validation details)
  - dataset_manifest.json
  - report.md

No physics stepping, no app launch, no network access.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from embodied_trackpad.leap_v1 import pilot_collection, pilot_dataset, pilot_provenance
from embodied_trackpad.leap_v1.pilot_collection import (
    validate_diagonal_collection_identity,
)
from embodied_trackpad.leap_v1.pilot_dataset import (
    DEFAULT_TASK_PROFILE,
    _validate_collection_request,
    audit_episode,
    build_dataset_manifest,
    sha256_file,
    split_of,
)
from embodied_trackpad.leap_v1.pilot_provenance import (
    validate_teacher_configuration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

DIAGONAL_PROFILE = pilot_collection.DIAGONAL_PROFILE
DIAGONAL_PLANNED_SEEDS = list(pilot_collection.EXPECTED_SEEDS)
DIAGONAL_SEED_SET = set(DIAGONAL_PLANNED_SEEDS)
PRIVILEGED_CONTROLLER_KIND = "privileged_visual_teacher"


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


def _seed_dir_seed(d: Path) -> int | None:
    """Parse the seed number from a seed-NNN directory name.

    Returns ``None`` for malformed names (never raises), so callers report
    issues instead of crashing on ``int()``/``split_of``.
    """
    parts = d.name.split("-")
    if len(parts) != 2 or parts[0] != "seed" or not parts[1]:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _rehash_source_files(manifest: dict[str, Any], acceptance: dict[str, Any], repo_root: Path) -> list[str]:
    issues = []
    claimed = manifest.get("source_hash_verified")
    accepted = acceptance.get("source_hashes")
    if not isinstance(claimed, dict) or not claimed:
        issues.append("manifest source_hash_verified missing or empty")
        claimed = {}
    if not isinstance(accepted, dict) or not accepted:
        issues.append("acceptance source_hashes missing or empty")
        accepted = {}
    for rel, h in claimed.items():
        if not _valid_sha256(h):
            issues.append(f"manifest source hash invalid: {rel}")
    for rel, h in accepted.items():
        if not _valid_sha256(h):
            issues.append(f"acceptance source hash invalid: {rel}")
    all_paths = set(claimed.keys()) | set(accepted.keys())
    for rel in sorted(all_paths):
        p = repo_root / rel
        if not p.exists():
            issues.append(f"source file missing: {rel}")
            continue
        actual = sha256_file(p)
        if rel in claimed and claimed[rel] != actual:
            issues.append(f"manifest source hash mismatch: {rel}")
        if rel in accepted and accepted[rel] != actual:
            issues.append(f"acceptance source hash mismatch: {rel}")
    for rel in sorted(set(accepted.keys()) - set(claimed.keys())):
        issues.append(f"manifest missing source hash for accepted file: {rel}")
    for rel in sorted(set(claimed.keys()) - set(accepted.keys())):
        issues.append(f"acceptance missing source hash for manifest file: {rel}")
    return issues


def _verify_artifacts(manifest: dict[str, Any], batch_summary: dict[str, Any], acceptance: dict[str, Any], repo_root: Path) -> list[str]:
    """Source-file rehash plus asset-manifest/native-binary hash checks.

    Shared by the legacy and diagonal global checks; every claim is checked
    against the actual files under ``repo_root`` with no cross-document
    fallback, so a conflict can never be hidden.
    """
    issues = []

    issues += _rehash_source_files(manifest, acceptance, repo_root)

    acc_asset = acceptance.get("asset_manifest", {}) or {}
    asset_rel = Path(manifest.get("asset_manifest_path") or acc_asset.get("path", "assets/manifests/leap-v1-full.json"))
    asset_path = repo_root / asset_rel
    asset_claim_manifest = manifest.get("asset_manifest_sha256")
    asset_claim_acceptance = acc_asset.get("sha256")
    asset_valid_m = _valid_sha256(asset_claim_manifest)
    asset_valid_a = _valid_sha256(asset_claim_acceptance)
    if not asset_valid_m:
        issues.append("asset manifest hash missing or invalid in manifest")
    if not asset_valid_a:
        issues.append("asset manifest hash missing or invalid in acceptance")
    if not asset_path.exists():
        issues.append(f"asset manifest missing: {asset_rel}")
    else:
        actual = sha256_file(asset_path)
        if asset_valid_m and actual != asset_claim_manifest:
            issues.append("asset manifest hash mismatch vs manifest claim")
        if asset_valid_a and actual != asset_claim_acceptance:
            issues.append("asset manifest hash mismatch vs acceptance claim")
        if asset_valid_m and asset_valid_a and asset_claim_manifest != asset_claim_acceptance:
            issues.append("asset manifest manifest-vs-acceptance hash conflict")

    acc_nb = acceptance.get("native_binary", {}) or {}
    nb_rel = Path(manifest.get("native_binary_path") or acc_nb.get("path", ".local/build/TrackpadSandbox.app/Contents/MacOS/TrackpadSandbox"))
    nb_path = repo_root / nb_rel
    nb_claim_manifest = manifest.get("native_binary_sha256")
    nb_claim_acceptance = acc_nb.get("sha256")
    nb_valid_m = _valid_sha256(nb_claim_manifest)
    nb_valid_a = _valid_sha256(nb_claim_acceptance)
    if not nb_valid_m:
        issues.append("native binary hash missing or invalid in manifest")
    if not nb_valid_a:
        issues.append("native binary hash missing or invalid in acceptance")
    if not nb_path.exists():
        issues.append(f"native binary missing: {nb_rel}")
    else:
        actual = sha256_file(nb_path)
        if nb_valid_m and actual != nb_claim_manifest:
            issues.append("native binary hash mismatch vs manifest claim")
        if nb_valid_a and actual != nb_claim_acceptance:
            issues.append("native binary hash mismatch vs acceptance claim")
        if nb_valid_m and nb_valid_a and nb_claim_manifest != nb_claim_acceptance:
            issues.append("native binary manifest-vs-acceptance hash conflict")

    return issues


def _verify_global_diagonal(
    manifest: dict[str, Any],
    batch_summary: dict[str, Any],
    acceptance: dict[str, Any],
    repo_root: Path,
    seed_dirs: list[Path],
    collection_id: str | None,
) -> list[str]:
    """near_diagonal_v1 global checks.

    Collection identity (profile/collection/source-commit/seed orderings/
    frozen splits/every prospective index) comes from the pure identity
    helper; source/artifact rehashing is shared with the legacy check. The
    full frozen teacher configuration is validated across manifest +
    batch summary + acceptance through the existing provenance validator
    via a metadata.planner wrapper, and the png backend/version must be
    declared identically in manifest, batch summary and acceptance.config
    (no manifest-or-summary fallback can hide a conflict). The
    privileged-visual-teacher controller kind is required in manifest,
    batch summary and acceptance without fallback. Directory coverage is
    exactly seed-000..seed-007; malformed or out-of-profile directory
    names are reported, never crashed on.
    """
    issues = []

    issues += validate_diagonal_collection_identity(manifest, batch_summary, acceptance, collection_id)
    issues += _verify_artifacts(manifest, batch_summary, acceptance, repo_root)

    for doc, name in ((manifest, "manifest"), (batch_summary, "batch-summary"), (acceptance, "acceptance")):
        if doc.get("controller_kind") != PRIVILEGED_CONTROLLER_KIND:
            issues.append(
                f"{name} controller_kind must be {PRIVILEGED_CONTROLLER_KIND!r}, "
                f"got {doc.get('controller_kind')!r}")

    acc_asset = acceptance.get("asset_manifest", {}) or {}
    acc_nb = acceptance.get("native_binary", {}) or {}
    for key, label, acc_claim in (
        ("asset_manifest_sha256", "asset manifest", acc_asset.get("sha256")),
        ("native_binary_sha256", "native binary", acc_nb.get("sha256")),
    ):
        claim_m = manifest.get(key)
        claim_b = batch_summary.get(key)
        if not _valid_sha256(claim_b):
            issues.append(f"batch-summary {label} hash missing or invalid")
        else:
            if _valid_sha256(acc_claim) and claim_b != acc_claim:
                issues.append(f"batch-summary {label} hash != acceptance claim")
            if _valid_sha256(claim_m) and claim_b != claim_m:
                issues.append(f"batch-summary {label} hash != manifest claim")

    wrapper_summary = {
        "planner_kind": batch_summary.get("planner_kind"),
        "contact_mode": batch_summary.get("contact_mode"),
        "force_config": batch_summary.get("force_config"),
        "force_governor_config": batch_summary.get("force_governor_config"),
    }
    wrapper_metadata = {
        "planner": {
            "planner_kind": manifest.get("planner_kind"),
            "force_config": manifest.get("force_config"),
            "force_governor_config": manifest.get("force_governor_config"),
        },
        "contact_mode": manifest.get("contact_mode"),
    }
    try:
        validate_teacher_configuration(wrapper_summary, wrapper_metadata, acceptance, required=True)
    except ValueError as exc:
        issues.append(f"teacher configuration mismatch: {exc}")

    backend_docs = (
        (manifest, "manifest"),
        (batch_summary, "batch-summary"),
        ((acceptance.get("config") or {}), "acceptance.config"),
    )
    backends: set[str] = set()
    versions: set[str] = set()
    for doc, name in backend_docs:
        backend = doc.get("png_backend")
        version = doc.get("png_backend_version")
        if not isinstance(backend, str) or not backend:
            issues.append(f"{name} png_backend missing or invalid")
        else:
            backends.add(backend)
        if not isinstance(version, str) or not version:
            issues.append(f"{name} png_backend_version missing or invalid")
        else:
            versions.add(version)
    if len(backends) > 1:
        issues.append("png_backend mismatch across manifest/batch-summary/acceptance.config")
    if len(versions) > 1:
        issues.append("png_backend_version mismatch across manifest/batch-summary/acceptance.config")

    parsed = [_seed_dir_seed(d) for d in seed_dirs]
    malformed = sorted(d.name for d, s in zip(seed_dirs, parsed) if s is None)
    if malformed:
        issues.append(f"malformed seed directory names: {malformed}")
    actual_seeds = sorted(s for s in parsed if s is not None)
    if actual_seeds != DIAGONAL_PLANNED_SEEDS:
        issues.append(f"seed directories {actual_seeds} do not match 0..7")

    return issues


def _verify_global(
    manifest: dict[str, Any],
    batch_summary: dict[str, Any],
    acceptance: dict[str, Any],
    repo_root: Path,
    seed_dirs: list[Path],
    *,
    task_profile: str = DEFAULT_TASK_PROFILE,
    collection_id: str | None = None,
) -> list[str]:
    if task_profile != DEFAULT_TASK_PROFILE:
        return _verify_global_diagonal(manifest, batch_summary, acceptance, repo_root, seed_dirs, collection_id)

    issues = []

    if manifest.get("source_commit") != acceptance.get("source_commit"):
        issues.append("manifest source_commit mismatch")
    if batch_summary.get("source_commit") != acceptance.get("source_commit"):
        issues.append("batch-summary source_commit mismatch")

    issues += _verify_artifacts(manifest, batch_summary, acceptance, repo_root)

    planner_kind = manifest.get("planner_kind") or batch_summary.get("planner_kind")
    if planner_kind != acceptance.get("config", {}).get("planner_kind"):
        issues.append("planner_kind mismatch")
    ctrl_kind = manifest.get("controller_kind") or batch_summary.get("controller_kind")
    if ctrl_kind != acceptance.get("controller_kind"):
        issues.append("controller_kind mismatch")

    acc_force_config = (acceptance.get("config", {}) or {}).get("force_config")
    man_force_config = manifest.get("force_config")
    if acc_force_config is not None and man_force_config != acc_force_config:
        issues.append("force_config mismatch")
    acc_gov_config = (acceptance.get("config", {}) or {}).get("force_governor_config")
    man_gov_config = manifest.get("force_governor_config")
    if acc_gov_config is not None and man_gov_config != acc_gov_config:
        issues.append("force_governor_config mismatch")

    seeds = manifest.get("seeds_attempt_order", [])
    expected_seeds = acceptance.get("config", {}).get("seeds", [])
    if seeds != list(range(12)):
        issues.append("manifest seeds_attempt_order not exactly [0..11]")
    if expected_seeds and seeds != expected_seeds:
        issues.append("manifest seeds differ from acceptance order")

    parsed = [_seed_dir_seed(d) for d in seed_dirs]
    malformed = sorted(d.name for d, s in zip(seed_dirs, parsed) if s is None)
    if malformed:
        issues.append(f"malformed seed directory names: {malformed}")
    actual_seeds = sorted(s for s in parsed if s is not None)
    if actual_seeds != list(range(12)):
        issues.append(f"seed directories {actual_seeds} do not match 0..11")

    prospective = manifest.get("prospective_split", {})
    acc_split = acceptance.get("prospective_split", {})
    for split_name in ("train", "validation", "test"):
        if sorted(prospective.get(split_name, [])) != sorted(acc_split.get(split_name, [])):
            issues.append(f"prospective_split {split_name} mismatch")

    split_map = {}
    for split_name in ("train", "validation", "test"):
        for s in acc_split.get(split_name, []):
            split_map[s] = split_name
    prospective_index = (
        prospective.get("prospective_index")
        or batch_summary.get("prospective_index")
        or manifest.get("prospective_index", [])
    )
    seen_pi_seeds: set[int] = set()
    for entry in prospective_index:
        seed = entry.get("seed")
        group = entry.get("group")
        if not isinstance(seed, int) or seed not in range(12):
            issues.append(f"prospective_index seed {seed} out of 0..11")
        if seed in seen_pi_seeds:
            issues.append(f"prospective_index duplicate seed {seed}")
        seen_pi_seeds.add(seed)
        if split_map.get(seed) != group:
            issues.append(f"seed {seed} group {group!r} != acceptance split {split_map.get(seed)!r}")
    if seen_pi_seeds != set(range(12)):
        issues.append("prospective_index does not cover seeds 0..11 exactly once")

    raw_flags = manifest.get("raw_flags", {})
    if raw_flags.get("training_eligible") is not False:
        issues.append("raw training_eligible not false")

    attempted = batch_summary.get("attempted_seeds", [])
    successful = batch_summary.get("successful_seeds", [])
    if attempted != list(range(12)) or successful != list(range(12)):
        issues.append("batch-summary attempted/successful seeds not exactly 0..11")

    return issues


def _reconcile_diagonal(
    audits: list[Any],
    seed_dirs: list[Path],
    batch_summary: dict[str, Any],
    collection_backend: Any,
    collection_version: Any,
) -> dict[str, Any]:
    """Post-audit truth: batch successful_seeds vs actually accepted seeds.

    Missing directories, failed episodes and extra accepted episodes are
    reported truthfully (never fabricated); every episode summary/metadata
    must carry the collection-level png backend/version. The returned dict
    includes the ``issues`` list used for the no-sample export gate.
    """
    issues: list[str] = []
    parsed = [_seed_dir_seed(d) for d in seed_dirs]
    observed = sorted(s for s in parsed if s is not None)
    accepted_seeds = sorted({a.seed for a in audits if a.accepted})
    rejected_seeds = sorted({a.seed for a in audits if not a.accepted})

    successful = batch_summary.get("successful_seeds")
    if not isinstance(successful, list):
        issues.append("batch-summary successful_seeds must be a list")
        successful = []
    successful_clean = sorted({s for s in successful if isinstance(s, int) and not isinstance(s, bool)})
    missing_episodes = [s for s in successful_clean if s not in accepted_seeds]
    extra_episodes = [s for s in accepted_seeds if s not in successful_clean]
    if missing_episodes:
        issues.append(f"batch successful seeds {missing_episodes} not accepted after episode audit")
    if extra_episodes:
        issues.append(f"episodes accepted beyond batch successful seeds: {extra_episodes}")

    for a in audits:
        if a.summary.get("png_backend") != collection_backend:
            issues.append(f"episode seed-{a.seed:03d} summary png_backend != collection backend")
        if a.metadata.get("png_backend") != collection_backend:
            issues.append(f"episode seed-{a.seed:03d} metadata png_backend != collection backend")
        if a.summary.get("png_backend_version") != collection_version:
            issues.append(f"episode seed-{a.seed:03d} summary png_backend_version != collection version")
        if a.metadata.get("png_backend_version") != collection_version:
            issues.append(f"episode seed-{a.seed:03d} metadata png_backend_version != collection version")

    return {
        "planned_seeds": list(DIAGONAL_PLANNED_SEEDS),
        "observed_seed_dirs": observed,
        "missing_seed_dirs": [s for s in DIAGONAL_PLANNED_SEEDS if s not in observed],
        "malformed_seed_dirs": sorted(d.name for d, s in zip(seed_dirs, parsed) if s is None),
        "rejected_seeds": rejected_seeds,
        "accepted_seeds": accepted_seeds,
        "batch_successful_seeds": successful_clean,
        "missing_episodes": missing_episodes,
        "extra_episodes": extra_episodes,
        "issues": issues,
    }


def _auditor_source_sha256() -> dict[str, str]:
    """SHA-256 of the actually loaded auditor modules and this CLI.

    Paths come from the loaded module objects and this file, never from a
    fixture ``--repo-root``, so the auditor identity is always the real
    code that produced the outputs.
    """
    paths = {
        "tools/audit_leap_v1_pilot.py": Path(__file__).resolve(),
        "src/embodied_trackpad/leap_v1/pilot_dataset.py": Path(pilot_dataset.__file__).resolve(),
        "src/embodied_trackpad/leap_v1/pilot_provenance.py": Path(pilot_provenance.__file__).resolve(),
        "src/embodied_trackpad/leap_v1/pilot_collection.py": Path(pilot_collection.__file__).resolve(),
    }
    return {rel: sha256_file(path) for rel, path in paths.items()}


def _episode_report(a: Any) -> dict[str, Any]:
    """Serialize one episode audit, tolerating early-rejection audits.

    Fields that are only computed partway through auditing (e.g.
    ``touch_sequences``, set by the touch validator) are read with
    ``getattr`` defaults so a rejected episode never crashes the report.
    """
    return {
        "seed": a.seed,
        "episode_dir": a.episode_dir,
        "accepted": a.accepted,
        "rejection_reasons": a.rejection_reasons,
        "validation_issues": a.validation_issues,
        "summary": a.summary,
        "metadata": a.metadata,
        "action_frames": getattr(a, "action_frames", 0),
        "terminal_frame_present": getattr(a, "terminal_frame", None) is not None,
        "terminal_png_sha256": getattr(a, "terminal_png_sha256", None),
        "png_verified": getattr(a, "png_verified", 0),
        "png_failed": getattr(a, "png_failed", 0),
        "teacher_steps": getattr(a, "teacher_steps", 0),
        "actor_entries": getattr(a, "actor_entries", 0),
        "touch_messages": getattr(a, "touch_messages", 0),
        "touch_sequences": getattr(a, "touch_sequences", 0),
        "force_intents": getattr(a, "force_intents", 0),
        "force_intent_mismatches": getattr(a, "force_intent_mismatches", 0),
        "causality_mismatches": getattr(a, "causality_mismatches", 0),
        "first_frame_time": getattr(a, "first_frame_time", None),
        "last_frame_time": getattr(a, "last_frame_time", None),
        "first_sim_time": getattr(a, "first_sim_time", None),
        "last_sim_time": getattr(a, "last_sim_time", None),
        "raw_bytes_counted": getattr(a, "raw_bytes_counted", 0),
        "pre_summary_bytes": getattr(a, "pre_summary_bytes", 0),
        "summary_bytes": getattr(a, "summary_bytes", 0),
        "raw_identity": getattr(a, "raw_identity", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit LEAP v1 pilot episodes.")
    parser.add_argument("--pilot-dir", required=True, help="Directory containing manifest.json and seed-NNN folders")
    parser.add_argument("--acceptance", default=".local/v1-visual/accepted-governed-pilot-collection.json", help="Collection acceptance file")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root used to validate relative PNG paths")
    parser.add_argument("--out", required=True, help="Output directory for audit artifacts")
    parser.add_argument("--worker-elapsed-s", type=float, default=None, help="Optional previous worker invocation wall time (s) to preserve separately")
    parser.add_argument("--task-profile", default=DEFAULT_TASK_PROFILE, help="Task profile: near_pointing_v1 (legacy default) or near_diagonal_v1")
    parser.add_argument("--collection-id", default=None, help="Collection identifier (required for near_diagonal_v1)")
    args = parser.parse_args()

    try:
        _validate_collection_request(args.task_profile, args.collection_id)
    except ValueError as exc:
        print(f"invalid collection request: {exc}", file=sys.stderr)
        return 2

    pilot_dir = Path(args.pilot_dir)
    repo_root = Path(args.repo_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = pilot_dir / "manifest.json"
    batch_summary_path = pilot_dir / "batch-summary.json"
    acceptance_path = Path(args.acceptance)

    if not manifest_path.exists():
        print(f"missing {manifest_path}", file=sys.stderr)
        return 1
    if not batch_summary_path.exists():
        print(f"missing {batch_summary_path}", file=sys.stderr)
        return 1
    if not acceptance_path.exists():
        print(f"missing {acceptance_path}", file=sys.stderr)
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(batch_summary_path, "r", encoding="utf-8") as f:
        batch_summary = json.load(f)
    with open(acceptance_path, "r", encoding="utf-8") as f:
        acceptance = json.load(f)

    manifest_sha = sha256_file(manifest_path)
    batch_summary_sha = sha256_file(batch_summary_path)

    seed_dirs = sorted([d for d in pilot_dir.iterdir() if d.is_dir() and d.name.startswith("seed-")])
    diagonal = args.task_profile != DEFAULT_TASK_PROFILE
    if not seed_dirs and not diagonal:
        print("no seed-NNN directories found", file=sys.stderr)
        return 1

    auditor_start = time.perf_counter()
    global_issues = _verify_global(
        manifest, batch_summary, acceptance, repo_root, seed_dirs,
        task_profile=args.task_profile, collection_id=args.collection_id,
    )

    audits = []
    episode_samples: list[list[dict]] = []
    for ep_dir in seed_dirs:
        seed_num = _seed_dir_seed(ep_dir)
        if diagonal and seed_num not in DIAGONAL_SEED_SET:
            # Out-of-profile or malformed directories are reported by the
            # global checks; never fabricate an audit for them.
            continue
        print(f"Auditing {ep_dir.name} ...", file=sys.stderr)
        audit, samples = audit_episode(
            ep_dir, repo_root, acceptance,
            task_profile=args.task_profile, collection_id=args.collection_id,
        )
        audits.append(audit)
        episode_samples.append(samples)

    reconciliation: dict[str, Any] | None = None
    reconcile_issues: list[str] = []
    if diagonal:
        reconciliation = _reconcile_diagonal(
            audits, seed_dirs, batch_summary,
            manifest.get("png_backend"), manifest.get("png_backend_version"),
        )
        reconcile_issues = reconciliation["issues"]

    # Sample export gate: global identity/config/source failures and
    # post-audit reconciliation failures export NO samples, even when they
    # were only detected after auditing. Individual failed episodes keep
    # their failed denominators in the manifest but contribute no samples.
    export_ok = not global_issues and not reconcile_issues
    all_samples: list[dict] = []
    if export_ok:
        for audit, samples in zip(audits, episode_samples):
            if audit.accepted:
                all_samples.extend(samples)

    auditor_wall = round(time.perf_counter() - auditor_start, 3)

    collection_wall = float(batch_summary.get("collection_wall_s", 0.0))
    episode_wall_sum = round(sum(float(a.summary.get("wall_s", 0.0)) for a in audits), 3)
    collection_info = {
        "collection_wall_s": collection_wall,
        "episode_wall_sum_s": episode_wall_sum,
        "worker_invocation_elapsed_s": args.worker_elapsed_s,
        "auditor_wall_s": auditor_wall,
    }

    # Write sample_index.jsonl only from accepted episodes
    sample_index_path = out_dir / "sample_index.jsonl"
    with open(sample_index_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")
    sample_index_sha256 = sha256_file(sample_index_path)

    # Build audit.json
    audit_report = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit": acceptance.get("source_commit"),
        "manifest_sha256": manifest_sha,
        "batch_summary_sha256": batch_summary_sha,
        "input_manifest_path": str(manifest_path),
        "input_batch_summary_path": str(batch_summary_path),
        "sample_index_sha256": sample_index_sha256,
        "global_issues": global_issues,
        "auditor_wall_s": auditor_wall,
        "episodes": [_episode_report(a) for a in audits],
    }
    if diagonal:
        audit_report["task_profile"] = args.task_profile
        audit_report["collection_id"] = args.collection_id
        audit_report["collection_source_commit"] = acceptance.get("source_commit")
        audit_report["auditor_source_sha256"] = _auditor_source_sha256()
        audit_report["reconciliation"] = reconciliation
    with open(out_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    # Build dataset_manifest.json
    dataset_manifest = build_dataset_manifest(
        audits, all_samples, acceptance, manifest_sha, batch_summary_sha, collection_info,
        task_profile=args.task_profile, collection_id=args.collection_id,
    )
    dataset_manifest["created_utc"] = audit_report["created_utc"]
    dataset_manifest["sample_index_sha256"] = sample_index_sha256
    if diagonal:
        dataset_manifest["collection_source_commit"] = acceptance.get("source_commit")
        dataset_manifest["auditor_source_sha256"] = audit_report["auditor_source_sha256"]
        dataset_manifest["reconciliation"] = reconciliation
    if global_issues or reconcile_issues:
        dataset_manifest["accepted_episodes"] = 0
        dataset_manifest["rejected_episodes"] = len(audits)
        dataset_manifest["total_action_frames"] = 0
        dataset_manifest["split_counts"] = {"train": 0, "validation": 0, "test": 0, "rejected": 0}
        dataset_manifest["target_classes"] = []
    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    # Build report.md
    lines = [
        "# Pilot data audit report",
        "",
        f"- created_utc: {audit_report['created_utc']}",
        f"- source_commit: {audit_report['source_commit']}",
        f"- manifest_sha256: {manifest_sha}",
        f"- batch_summary_sha256: {batch_summary_sha}",
        f"- sample_index_sha256: {sample_index_sha256}",
        f"- global_issues: {global_issues}",
        f"- auditor_wall_s: {auditor_wall}",
        f"- total_episodes: {dataset_manifest['total_episodes']}",
        f"- accepted_episodes: {dataset_manifest['accepted_episodes']}",
        f"- rejected_episodes: {dataset_manifest['rejected_episodes']}",
        f"- total_action_frames: {dataset_manifest['total_action_frames']}",
        f"- split_counts: {dataset_manifest['split_counts']}",
        f"- target_classes: {dataset_manifest['target_classes']}",
        f"- raw_bytes_counted_including_summaries: {dataset_manifest['raw_bytes_counted_including_summaries']}",
        f"- pre_summary_bytes_reported: {dataset_manifest['pre_summary_bytes_reported']}",
        f"- total_sim_s: {dataset_manifest['total_sim_s']}",
        f"- collection_wall_s: {dataset_manifest['collection_wall_s']}",
        f"- episode_wall_sum_s: {dataset_manifest['episode_wall_sum_s']}",
        "",
        "## Per-episode",
        "",
        "| seed | split | accepted | action_frames | steps | png_ok | png_fail | force_intents | force_mismatch | causality_mismatch | touch | issues |",
        "|------|-------|----------|---------------|-------|--------|----------|---------------|----------------|--------------------|-------|--------|",
    ]
    for a in audits:
        lines.append(
            f"| {a.seed} | {split_of(a.seed, task_profile=args.task_profile)} | "
            f"{a.accepted} | {a.action_frames} | {a.teacher_steps} | {a.png_verified} | {a.png_failed} | "
            f"{a.force_intents} | {a.force_intent_mismatches} | {a.causality_mismatches} | {a.touch_messages} | {len(a.validation_issues)} |"
        )
    if diagonal and reconciliation is not None:
        lines += [
            "",
            "## Reconciliation (near_diagonal_v1)",
            "",
            f"- planned_seeds: {reconciliation['planned_seeds']}",
            f"- observed_seed_dirs: {reconciliation['observed_seed_dirs']}",
            f"- missing_seed_dirs: {reconciliation['missing_seed_dirs']}",
            f"- malformed_seed_dirs: {reconciliation['malformed_seed_dirs']}",
            f"- rejected_seeds: {reconciliation['rejected_seeds']}",
            f"- accepted_seeds: {reconciliation['accepted_seeds']}",
            f"- batch_successful_seeds: {reconciliation['batch_successful_seeds']}",
            f"- missing_episodes: {reconciliation['missing_episodes']}",
            f"- extra_episodes: {reconciliation['extra_episodes']}",
            f"- reconcile_issues: {reconcile_issues}",
            f"- collection_source_commit: {acceptance.get('source_commit')}",
        ]
    lines += ["", "## Limitations", ""]
    for lim in dataset_manifest["limitations"]:
        lines.append(f"- {lim}")
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(all_samples)} action frames to {sample_index_path}", file=sys.stderr)
    print(f"Audit report: {out_dir / 'report.md'}", file=sys.stderr)
    ok = not global_issues and not reconcile_issues and all(a.accepted for a in audits)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
