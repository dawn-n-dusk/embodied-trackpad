"""Pure in-memory collection-identity validation for near_diagonal_v1 audits.

No filesystem access, no mutation of inputs, no actor/feature semantics.
``validate_diagonal_collection_identity`` returns a list of issue strings
(``[]`` means the collection identity is consistent). Source-file rehashing,
teacher-config parity, png backend/version and seed-directory checks are
reused from the existing validators by a later CLI integration step.
"""
from __future__ import annotations

import re
from typing import Any

from embodied_trackpad.leap_v1.pilot_dataset import COLLECTION_ID_RE, split_of

DIAGONAL_PROFILE = "near_diagonal_v1"
EXPECTED_SEEDS = list(range(8))
SEED_SET = set(EXPECTED_SEEDS)
_SPLIT_NAMES = ("train", "validation", "test")
_SOURCE_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _frozen_diagonal_split() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {name: [] for name in _SPLIT_NAMES}
    for seed in EXPECTED_SEEDS:
        group = split_of(seed, task_profile=DIAGONAL_PROFILE)
        if group in out:
            out[group].append(seed)
    return out


def _check_exact_seed_order(doc: dict[str, Any], key: str, name: str,
                            issues: list[str]) -> None:
    value = doc.get(key)
    if not isinstance(value, list):
        issues.append(f"{name}.{key} must be a list")
        return
    for i, seed in enumerate(value):
        if not _is_int(seed):
            issues.append(
                f"{name}.{key}[{i}] must be an integer, got {seed!r}")
            return
    if value != EXPECTED_SEEDS:
        issues.append(f"{name}.{key} must be exactly {EXPECTED_SEEDS}")


def _check_prospective_split(doc_ps: Any, name: str,
                             expected: dict[str, list[int]],
                             issues: list[str]) -> None:
    if not isinstance(doc_ps, dict):
        issues.append(f"{name} must be an object")
        return
    for split in _SPLIT_NAMES:
        value = doc_ps.get(split)
        where = f"{name}.{split}"
        if not isinstance(value, list):
            issues.append(f"{where} must be a list")
            continue
        clean = True
        local: list[int] = []
        for seed in value:
            if not _is_int(seed) or seed not in SEED_SET:
                issues.append(f"{where} has unknown or bool seed {seed!r}")
                clean = False
                continue
            if seed in local:
                issues.append(f"{where} has duplicate seed {seed}")
                clean = False
            local.append(seed)
        if clean and sorted(local) != expected[split]:
            issues.append(f"{where} does not match the frozen diagonal split")


def _check_prospective_index(value: Any, where: str, issues: list[str]) -> None:
    if not isinstance(value, list):
        issues.append(f"{where} must be a list")
        return
    seen: set[int] = set()
    for i, entry in enumerate(value):
        if not isinstance(entry, dict):
            issues.append(f"{where}[{i}] must be an object")
            continue
        seed = entry.get("seed")
        if not _is_int(seed) or seed not in SEED_SET:
            issues.append(
                f"{where}[{i}].seed must be an integer in 0..7, got {seed!r}")
            continue
        if seed in seen:
            issues.append(f"{where} has duplicate seed {seed}")
        seen.add(seed)
        frozen = split_of(seed, task_profile=DIAGONAL_PROFILE)
        if entry.get("group") != frozen:
            issues.append(
                f"{where}[{i}] group {entry.get('group')!r} != frozen split "
                f"{frozen!r} for seed {seed}")
    if seen != SEED_SET:
        issues.append(f"{where} must cover seeds 0..7 exactly once")


def validate_diagonal_collection_identity(
    manifest: Any,
    batch_summary: Any,
    acceptance: Any,
    collection_id: Any,
) -> list[str]:
    """Return issue strings for diagonal collection identity (``[]`` = valid).

    Checks profile/collection_id/source_commit consistency across the three
    documents, exact 0..7 seed orderings, successful-seed subset semantics,
    frozen split parity derived from :func:`pilot_dataset.split_of`, and every
    provided ``prospective_index`` independently (at least one required).
    Inputs are not mutated and no exception escapes for malformed containers.
    """
    issues: list[str] = []
    collection_ok = isinstance(collection_id, str) and bool(
        COLLECTION_ID_RE.fullmatch(collection_id))
    if not collection_ok:
        issues.append(
            f"collection_id {collection_id!r} is not a safe identifier "
            f"matching [A-Za-z0-9_-]+")

    man = manifest if isinstance(manifest, dict) else None
    bat = batch_summary if isinstance(batch_summary, dict) else None
    acc = acceptance if isinstance(acceptance, dict) else None
    if man is None:
        issues.append("manifest must be an object")
    if bat is None:
        issues.append("batch_summary must be an object")
    if acc is None:
        issues.append("acceptance must be an object")
    acc_cfg = acc.get("config") if acc is not None else None
    if not isinstance(acc_cfg, dict):
        issues.append("acceptance.config must be an object")
        acc_cfg = None

    # task_profile / collection_id identity --------------------------------
    for doc, name in ((man, "manifest"), (bat, "batch_summary")):
        if doc is None:
            continue
        if doc.get("task_profile") != DIAGONAL_PROFILE:
            issues.append(
                f"{name}.task_profile must be {DIAGONAL_PROFILE!r}, got "
                f"{doc.get('task_profile')!r}")
        if collection_ok and doc.get("collection_id") != collection_id:
            issues.append(
                f"{name}.collection_id {doc.get('collection_id')!r} != "
                f"requested {collection_id!r}")
    if acc is not None and collection_ok and \
            acc.get("collection_id") != collection_id:
        issues.append(
            f"acceptance.collection_id {acc.get('collection_id')!r} != "
            f"requested {collection_id!r}")
    if acc_cfg is not None and \
            acc_cfg.get("task_profile") != DIAGONAL_PROFILE:
        issues.append(
            f"acceptance.config.task_profile must be {DIAGONAL_PROFILE!r}, "
            f"got {acc_cfg.get('task_profile')!r}")

    # source_commit ----------------------------------------------------------
    commits: dict[str, str] = {}
    for doc, name in ((man, "manifest"), (bat, "batch_summary"),
                      (acc, "acceptance")):
        if doc is None:
            continue
        commit = doc.get("source_commit")
        if not isinstance(commit, str) or not _SOURCE_COMMIT_RE.fullmatch(commit):
            issues.append(
                f"{name}.source_commit must be full 40-hex, got {commit!r}")
        else:
            commits[name] = commit
    if len(set(commits.values())) > 1:
        issues.append(f"source_commit mismatch across documents: {commits}")

    # seed ordering ----------------------------------------------------------
    if man is not None:
        _check_exact_seed_order(man, "seeds_attempt_order", "manifest", issues)
    if acc_cfg is not None:
        _check_exact_seed_order(acc_cfg, "seeds", "acceptance.config", issues)
    if bat is not None:
        _check_exact_seed_order(bat, "attempted_seeds", "batch_summary",
                                issues)
        successful = bat.get("successful_seeds")
        if not isinstance(successful, list):
            issues.append("batch_summary.successful_seeds must be a list")
        else:
            seen: set[int] = set()
            for seed in successful:
                if not _is_int(seed):
                    issues.append(
                        f"batch_summary.successful_seeds must contain "
                        f"integers, got {seed!r}")
                    continue
                if seed not in SEED_SET:
                    issues.append(
                        f"batch_summary.successful_seeds seed {seed} "
                        f"outside 0..7")
                if seed in seen:
                    issues.append(
                        f"batch_summary.successful_seeds duplicate seed {seed}")
                seen.add(seed)

    # frozen prospective splits ----------------------------------------------
    expected = _frozen_diagonal_split()
    if man is not None:
        _check_prospective_split(man.get("prospective_split"),
                                 "manifest.prospective_split", expected,
                                 issues)
    if acc is not None:
        _check_prospective_split(acc.get("prospective_split"),
                                 "acceptance.prospective_split", expected,
                                 issues)

    # prospective_index (each provided one independently) --------------------
    provided = 0
    if man is not None:
        man_ps = man.get("prospective_split")
        if isinstance(man_ps, dict) and "prospective_index" in man_ps:
            provided += 1
            _check_prospective_index(
                man_ps["prospective_index"],
                "manifest.prospective_split.prospective_index", issues)
        if "prospective_index" in man:
            provided += 1
            _check_prospective_index(man["prospective_index"],
                                     "manifest.prospective_index", issues)
    if bat is not None and "prospective_index" in bat:
        provided += 1
        _check_prospective_index(bat["prospective_index"],
                                 "batch_summary.prospective_index", issues)
    if provided == 0:
        issues.append(
            "no prospective_index provided; at least one complete index "
            "covering seeds 0..7 is required")

    return issues
