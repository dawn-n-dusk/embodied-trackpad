"""Pinned LEAP Hand v2 Advanced asset integrity (stdlib-only shared module).

Single source of truth for the pinned 13-file asset set, the committed
sha256/bytes manifest, and manifest/local-file verification. Shared by
``tools/fetch_leap_assets.py`` and ``embodied_trackpad.leap_sim.model`` so that
downloading and MuJoCo loading use one verifier and fetching/validation never
requires importing MuJoCo.

Integrity model: the committed manifest ``assets/manifests/leap-selected.json``
is the ONLY authority for expected paths, sha256 and byte sizes. Nothing in
this module re-authors or overwrites manifest hashes from downloaded content.

Licensing: the nested upstream ``leap_v2/LICENSE`` is Apache-2.0 and covers the
upstream software. The official CAD page https://v2-adv.leaphand.com/leap_cad
lists CC BY-NC-SA; this repository infers no blanket relicensing or
redistribution permission for individual assets. Meshes are NOT redistributed
by this repository; they are fetched on demand into the git-ignored download
root and the upstream LICENSE is retained alongside them.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_ROOT = REPO_ROOT / "assets" / "downloads" / "leap_hand_v2_adv"
MANIFEST_JSON = REPO_ROOT / "assets" / "manifests" / "leap-selected.json"

SCHEMA_VERSION = "leap-selected-assets.v1"
UPSTREAM_REPOSITORY = "https://github.com/leap-hand/LEAP_Hand_V2_Adv_API"
PINNED_COMMIT = "a0936196e8fa0bcefff045d6fc02523d6da591f2"
COMMIT_DATE_UTC = "2025-06-13T17:36:35Z"
RAW_BASE = f"https://raw.githubusercontent.com/leap-hand/LEAP_Hand_V2_Adv_API/{PINNED_COMMIT}"
CAD_LICENSE_URL = "https://v2-adv.leaphand.com/leap_cad"

# Pinned upstream source whitelist: repo-relative upstream path ->
# download-root-relative local path. Only the RIGHT hand URDF + its referenced
# meshes are required (left hand is a mirror duplicate and the telekinesis/
# subtree repeats the same files). This literal list is the only permitted
# download source set; it is never read from external configuration.
FILES = {
    "leap_v2/LICENSE": "LICENSE",
    "README.md": "README.upstream.md",
    "leap_v2/leap_v2/leap_v2_right/robot.urdf": "leap_v2_right/robot.urdf",
    "leap_v2/leap_v2/leap_v2_right/no_finger_frame_mirror.stl": "leap_v2_right/no_finger_frame_mirror.stl",
    "leap_v2/leap_v2/leap_v2_right/4_finger_frame_mirror.stl": "leap_v2_right/4_finger_frame_mirror.stl",
    "leap_v2/leap_v2/leap_v2_right/axle_4_gear_together.stl": "leap_v2_right/axle_4_gear_together.stl",
    "leap_v2/leap_v2/leap_v2_right/axle_4_gear_together_mir.stl": "leap_v2_right/axle_4_gear_together_mir.stl",
    "leap_v2/leap_v2/leap_v2_right/finger_v2_with_motor.stl": "leap_v2_right/finger_v2_with_motor.stl",
    "leap_v2/leap_v2/leap_v2_right/pip_dip_soft_parts.stl": "leap_v2_right/pip_dip_soft_parts.stl",
    "leap_v2/leap_v2/leap_v2_right/dip_tip_soft_upward_cut.stl": "leap_v2_right/dip_tip_soft_upward_cut.stl",
    "leap_v2/leap_v2/leap_v2_right/thumb_frame_mirror.stl": "leap_v2_right/thumb_frame_mirror.stl",
    "leap_v2/leap_v2/leap_v2_right/thumb_tip_hard.stl": "leap_v2_right/thumb_tip_hard.stl",
    "leap_v2/leap_v2/leap_v2_right/thumb_with_motor.stl": "leap_v2_right/thumb_with_motor.stl",
}

REQUIRED_PATHS = frozenset(FILES.values())

_HEX_DIGITS = frozenset("0123456789abcdef")


class LeapAssetError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 - pinned public host
        return resp.read()


def validate_dest_path(root: Path, dest: Path) -> None:
    """Validate that ``dest`` is a non-symlink path contained in ``root``
    BEFORE any network fetch or filesystem write. Rejects escaping paths and
    any symlink component at or below ``root`` (including a symlinked
    ``leap_v2_right`` directory or a symlinked destination file)."""
    root = Path(root)
    dest = Path(dest)
    root_resolved = root.resolve()
    try:
        rel_parts = dest.relative_to(root).parts
    except ValueError:
        rel_parts = ()
    if not rel_parts or ".." in rel_parts:
        raise LeapAssetError(f"destination escapes download root: {dest}")
    current = root
    for part in rel_parts:
        current = current / part
        if current.is_symlink():
            raise LeapAssetError(f"destination path contains a symlink: {current}")
    if not dest.resolve().is_relative_to(root_resolved):
        raise LeapAssetError(f"destination escapes download root after resolution: {dest}")


def download_verified(
    url: str,
    dest: Path,
    *,
    root: Path,
    expected_sha256: str,
    expected_bytes: int,
    fetch=None,
) -> None:
    """Fetch ``url`` and install it as ``dest`` inside ``root`` only after the
    pinned sha256/bytes match. The destination path is validated against
    ``root`` (no symlinks, no escape) BEFORE any fetch or write; the payload
    goes to a uniquely created temporary file in the validated destination
    directory and is atomically renamed on success. On any failure the
    temporary file is removed, the existing ``dest`` (if any) is left
    untouched, and LeapAssetError is raised. ``fetch`` defaults to
    :func:`download` (network); tests inject a stub."""
    dest = Path(dest)
    validate_dest_path(root, dest)  # before any network or writes
    payload = (fetch or download)(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        actual = sha256_of(tmp)
        if len(payload) != expected_bytes or actual != expected_sha256:
            raise LeapAssetError(
                f"downloaded payload failed pinned verification for {dest.name}: "
                f"expected sha256={expected_sha256} bytes={expected_bytes}, "
                f"got sha256={actual} bytes={len(payload)}; keeping existing file"
            )
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _validate_entry(entry: object, index: int) -> str:
    """Validate one manifest entry; return its verified relative path."""
    if not isinstance(entry, dict):
        raise LeapAssetError(f"manifest entry #{index} is not an object: {entry!r}")
    rel = entry.get("path")
    if not isinstance(rel, str) or not rel:
        raise LeapAssetError(f"manifest entry #{index} has an empty or non-string path: {rel!r}")
    if "\\" in rel:
        raise LeapAssetError(f"manifest entry #{index} path must use '/' separators: {rel!r}")
    pure = PurePosixPath(rel)
    if pure.is_absolute():
        raise LeapAssetError(f"manifest entry #{index} absolute path not allowed: {rel!r}")
    if ".." in pure.parts:
        raise LeapAssetError(f"manifest entry #{index} parent-directory escape not allowed: {rel!r}")
    digest = entry.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in _HEX_DIGITS for c in digest)
    ):
        raise LeapAssetError(f"manifest entry #{index} ({rel}) has an invalid sha256: {digest!r}")
    size = entry.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise LeapAssetError(f"manifest entry #{index} ({rel}) has an invalid byte size: {size!r}")
    return rel


def load_and_validate_manifest(manifest_path: Path) -> dict:
    """Load the committed manifest and validate its version, pinned source and
    the exact required 13-file set. Raises LeapAssetError with a precise
    version/source/path message on any structural problem."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise LeapAssetError(f"manifest missing: {manifest_path}")
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
            f"manifest pinned_commit mismatch: {commit!r} (expected {PINNED_COMMIT!r})"
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
    return manifest


def verify_manifest(
    manifest_path: Path = MANIFEST_JSON, download_root: Path = DOWNLOAD_ROOT
) -> dict:
    """Verify downloaded assets against the committed pinned manifest.

    Validates manifest structure (schema version, pinned commit, exact
    13-file set, path safety) and then each local file (no symlinks, no
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
