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

from embodied_trackpad.leap_v1.pilot_dataset import (
    audit_episode,
    build_dataset_manifest,
    sha256_file,
    split_of,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value.lower())


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


def _verify_global(
    manifest: dict[str, Any],
    batch_summary: dict[str, Any],
    acceptance: dict[str, Any],
    repo_root: Path,
    seed_dirs: list[Path],
) -> list[str]:
    issues = []

    if manifest.get("source_commit") != acceptance.get("source_commit"):
        issues.append("manifest source_commit mismatch")
    if batch_summary.get("source_commit") != acceptance.get("source_commit"):
        issues.append("batch-summary source_commit mismatch")

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

    actual_seeds = sorted(int(d.name.split("-")[1]) for d in seed_dirs if d.name.startswith("seed-"))
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit LEAP v1 pilot episodes.")
    parser.add_argument("--pilot-dir", required=True, help="Directory containing manifest.json and seed-NNN folders")
    parser.add_argument("--acceptance", default=".local/v1-visual/accepted-governed-pilot-collection.json", help="Collection acceptance file")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="Repository root used to validate relative PNG paths")
    parser.add_argument("--out", required=True, help="Output directory for audit artifacts")
    parser.add_argument("--worker-elapsed-s", type=float, default=None, help="Optional previous worker invocation wall time (s) to preserve separately")
    args = parser.parse_args()

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
    if not seed_dirs:
        print("no seed-NNN directories found", file=sys.stderr)
        return 1

    auditor_start = time.perf_counter()
    global_issues = _verify_global(manifest, batch_summary, acceptance, repo_root, seed_dirs)

    audits = []
    all_samples: list[dict] = []

    for ep_dir in seed_dirs:
        print(f"Auditing {ep_dir.name} ...", file=sys.stderr)
        audit, samples = audit_episode(ep_dir, repo_root, acceptance)
        audits.append(audit)
        if audit.accepted and not global_issues:
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
        "episodes": [
            {
                "seed": a.seed,
                "episode_dir": a.episode_dir,
                "accepted": a.accepted,
                "rejection_reasons": a.rejection_reasons,
                "validation_issues": a.validation_issues,
                "summary": a.summary,
                "metadata": a.metadata,
                "action_frames": a.action_frames,
                "terminal_frame_present": a.terminal_frame is not None,
                "terminal_png_sha256": a.terminal_png_sha256,
                "png_verified": a.png_verified,
                "png_failed": a.png_failed,
                "teacher_steps": a.teacher_steps,
                "actor_entries": a.actor_entries,
                "touch_messages": a.touch_messages,
                "touch_sequences": a.touch_sequences,
                "force_intents": a.force_intents,
                "force_intent_mismatches": a.force_intent_mismatches,
                "causality_mismatches": a.causality_mismatches,
                "first_frame_time": a.first_frame_time,
                "last_frame_time": a.last_frame_time,
                "first_sim_time": a.first_sim_time,
                "last_sim_time": a.last_sim_time,
                "raw_bytes_counted": a.raw_bytes_counted,
                "pre_summary_bytes": a.pre_summary_bytes,
                "summary_bytes": a.summary_bytes,
                "raw_identity": a.raw_identity,
            }
            for a in audits
        ],
    }
    with open(out_dir / "audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    # Build dataset_manifest.json
    dataset_manifest = build_dataset_manifest(
        audits, all_samples, acceptance, manifest_sha, batch_summary_sha, collection_info
    )
    dataset_manifest["created_utc"] = audit_report["created_utc"]
    dataset_manifest["sample_index_sha256"] = sample_index_sha256
    if global_issues:
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
            f"| {a.seed} | {split_of(a.seed)} | "
            f"{a.accepted} | {a.action_frames} | {a.teacher_steps} | {a.png_verified} | {a.png_failed} | "
            f"{a.force_intents} | {a.force_intent_mismatches} | {a.causality_mismatches} | {a.touch_messages} | {len(a.validation_issues)} |"
        )
    lines += ["", "## Limitations", ""]
    for lim in dataset_manifest["limitations"]:
        lines.append(f"- {lim}")
    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {len(all_samples)} action frames to {sample_index_path}", file=sys.stderr)
    print(f"Audit report: {out_dir / 'report.md'}", file=sys.stderr)
    ok = not global_issues and all(a.accepted for a in audits)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
