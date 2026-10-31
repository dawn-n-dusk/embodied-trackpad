"""Pinned LEAP Hand v1 Full asset integrity (stdlib-only shared module).

Single source of truth for the pinned 14-file MuJoCo Menagerie ``leap_hand``
right-hand asset set, the committed sha256/bytes manifest
``assets/manifests/leap-v1-full.json``, and manifest/local-file verification.
Shared by ``tools/fetch_leap_v1_assets.py`` and
``embodied_trackpad.leap_v1.model`` so downloading and MuJoCo loading use one
verifier and fetching/validation never requires importing MuJoCo.

Integrity model: the committed manifest is the ONLY authority for expected
paths, sha256 and byte sizes. Nothing in this module re-authors or overwrites
manifest hashes from downloaded content. The generic path-safety and verified
download primitives are imported unchanged from the v2 module
``embodied_trackpad.leap_assets`` (not modified); only the v1-specific
constants and manifest checks live here.

Licensing: ``leap_hand/LICENSE`` at the pinned Menagerie commit is the MIT
License (Copyright 2023 Ananye Agarwal); a verbatim copy is retained alongside
the assets. Meshes are NOT redistributed by this repository; they are fetched
on demand into the git-ignored download root.
"""
from __future__ import annotations

from pathlib import Path

from embodied_trackpad.leap_assets import (  # noqa: F401  (re-exported)
    LeapAssetError,
    _validate_entry,
    download,
    download_verified,
    sha256_of,
    validate_dest_path,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_ROOT = REPO_ROOT / "assets" / "downloads" / "leap_v1_full"
MANIFEST_JSON = REPO_ROOT / "assets" / "manifests" / "leap-v1-full.json"

SCHEMA_VERSION = "leap-v1-full-assets.v1"
UPSTREAM_REPOSITORY = "https://github.com/google-deepmind/mujoco_menagerie"
UPSTREAM_SUBDIRECTORY = "leap_hand"
PINNED_COMMIT = "a662de603224c47123b0783821c58ad4402ad3e3"
COMMIT_DATE_UTC = "2026-10-31T00:08:38Z"
# Historical locator pin from the task contract; verified on 2026-10-31 via the
# GitHub compare API to carry byte-identical leap_hand content vs PINNED_COMMIT.
LOCATOR_COMMIT = "f3b6013629c7b94f8ee93fd977b161fa8350f9bc"
RAW_BASE = f"{UPSTREAM_REPOSITORY.replace('github.com', 'raw.githubusercontent.com')}/{PINNED_COMMIT}/{UPSTREAM_SUBDIRECTORY}"

# Pinned upstream source whitelist: leap_hand-relative upstream path ->
# download-root-relative local path. Only the RIGHT hand MJCF + the 10 OBJ
# meshes it references plus license/readme/changelog are required (the left
# hand, scene wrappers, mount mesh and preview PNG are intentionally not
# downloaded). This literal list is the only permitted download source set.
_MESHES = (
    "base.obj",
    "distal.obj",
    "medial.obj",
    "palm_right.obj",
    "proximal.obj",
    "thumb_base.obj",
    "thumb_distal.obj",
    "thumb_proximal.obj",
    "thumb_tip.obj",
    "tip.obj",
)
FILES = {
    "LICENSE": "right_hand/LICENSE",
    "README.md": "right_hand/README.md",
    "CHANGELOG.md": "right_hand/CHANGELOG.md",
    "right_hand.xml": "right_hand/right_hand.xml",
    **{f"assets/{m}": f"right_hand/assets/{m}" for m in _MESHES},
}

REQUIRED_PATHS = frozenset(FILES.values())

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # task budget: 100 MiB total


def load_and_validate_manifest(manifest_path: Path) -> dict:
    """Load the committed v1 manifest and validate its version, pinned source
    and the exact required 14-file set. Raises LeapAssetError with a precise
    version/source/path message on any structural problem."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise LeapAssetError(f"manifest missing: {manifest_path}")
    import json

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise LeapAssetError(f"manifest is not valid JSON: {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise LeapAssetError(f"manifest is not a JSON object: {manifest_path}")
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise LeapAssetError(
            f"manifest schema_version mismatch: {version!r} (expected {SCHEMA_VERSION!r})"
        )
    upstream = manifest.get("upstream")
    commit = upstream.get("pinned_commit") if isinstance(upstream, dict) else None
    if commit != PINNED_COMMIT:
        raise LeapAssetError(
            f"manifest pinned_commit mismatch: {commit!r} (expected {PINNED_COMMIT})"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise LeapAssetError(f"manifest lists no file entries: {manifest_path}")
    seen: set[str] = set()
    duplicates: list[str] = []
    for index, entry in enumerate(files):
        rel = _validate_entry(entry, index)
        if rel in seen:
            duplicates.append(rel)
        seen.add(rel)
    if duplicates:
        raise LeapAssetError(f"manifest contains duplicate entries: {sorted(duplicates)}")
    missing_required = sorted(REQUIRED_PATHS - seen)
    unexpected = sorted(seen - REQUIRED_PATHS)
    if missing_required or unexpected:
        raise LeapAssetError(
            "manifest file set mismatch: "
            f"missing required={missing_required} unexpected={unexpected}"
        )
    total = sum(entry["bytes"] for entry in files)
    if total > MAX_DOWNLOAD_BYTES:
        raise LeapAssetError(
            f"manifest total bytes {total} exceeds download budget {MAX_DOWNLOAD_BYTES}"
        )
    return manifest


def verify_manifest(
    manifest_path: Path = MANIFEST_JSON, download_root: Path = DOWNLOAD_ROOT
) -> dict:
    """Verify downloaded v1 assets against the committed pinned manifest.

    Validates manifest structure (schema version, pinned commit, exact
    14-file set, path safety) and then each local file (no symlinks, no
    escape from the download root, byte size and sha256 match). Raises
    LeapAssetError listing every missing/unsafe/size/hash failure."""
    manifest = load_and_validate_manifest(manifest_path)
    download_root = Path(download_root)
    root_resolved = download_root.resolve()
    missing: list[str] = []
    unsafe: list[str] = []
    size_mismatch: list[str] = []
    hash_mismatch: list[str] = []
    for entry in manifest["files"]:
        rel = entry["path"]
        path = download_root / rel
        if path.is_symlink() or any(
            ancestor.is_symlink()
            for ancestor in path.parents
            if ancestor != download_root and ancestor.is_relative_to(download_root)
        ):
            unsafe.append(f"{rel} (symlink)")
            continue
        if not path.resolve().is_relative_to(root_resolved):
            unsafe.append(f"{rel} (escapes download root)")
            continue
        if not path.is_file():
            missing.append(rel)
            continue
        if path.stat().st_size != entry["bytes"]:
            size_mismatch.append(rel)
        if sha256_of(path) != entry["sha256"]:
            hash_mismatch.append(rel)
    if missing or unsafe or size_mismatch or hash_mismatch:
        raise LeapAssetError(
            "manifest verification failed: "
            f"missing={missing} unsafe={unsafe} "
            f"size_mismatch={size_mismatch} hash_mismatch={hash_mismatch}"
        )
    return manifest
