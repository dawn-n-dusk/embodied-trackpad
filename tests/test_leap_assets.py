"""Integrity tests for the pinned LEAP asset manifest, shared verifier,
verified-atomic download helper, and the fetch tool's no-rewrite policy."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from embodied_trackpad import leap_assets
from embodied_trackpad.leap_assets import LeapAssetError

REPO_ROOT = Path(__file__).resolve().parents[1]
# Read-only real-cache verification defaults to this repo's own download root;
# set LEAP_TEST_REAL_CACHE to point at another existing cache (e.g. a sibling
# worktree) when this one has no downloaded assets.
REAL_CACHE = Path(os.environ.get("LEAP_TEST_REAL_CACHE") or leap_assets.DOWNLOAD_ROOT)


def _build_tree(tmp_path: Path):
    """Create a synthetic 13-file download tree plus a matching manifest."""
    root = tmp_path / "downloads"
    entries = []
    for index, rel in enumerate(sorted(leap_assets.REQUIRED_PATHS)):
        payload = f"synthetic-payload-{index}-{rel}".encode()
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        entries.append(
            {
                "path": rel,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    manifest = {
        "schema_version": leap_assets.SCHEMA_VERSION,
        "upstream": {
            "repository": leap_assets.UPSTREAM_REPOSITORY,
            "pinned_commit": leap_assets.PINNED_COMMIT,
        },
        "files": entries,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return root, manifest_path, manifest


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _load_fetch_tool():
    spec = importlib.util.spec_from_file_location(
        "fetch_leap_assets", REPO_ROOT / "tools" / "fetch_leap_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_13_file_set_verifies(tmp_path):
    root, manifest_path, manifest = _build_tree(tmp_path)
    result = leap_assets.verify_manifest(manifest_path, root)
    assert len(result["files"]) == 13
    assert len(manifest["files"]) == 13


def test_committed_manifest_matches_pinned_whitelist():
    manifest = leap_assets.load_and_validate_manifest(leap_assets.MANIFEST_JSON)
    assert len(manifest["files"]) == 13
    assert {e["path"] for e in manifest["files"]} == leap_assets.REQUIRED_PATHS
    assert manifest["upstream"]["pinned_commit"] == leap_assets.PINNED_COMMIT


def test_tampered_payload_rejected(tmp_path):
    root, manifest_path, _ = _build_tree(tmp_path)
    target = root / "LICENSE"
    target.write_bytes(target.read_bytes() + b"tampered")
    with pytest.raises(LeapAssetError, match="hash_mismatch"):
        leap_assets.verify_manifest(manifest_path, root)


def test_tampered_manifest_hash_rejected(tmp_path):
    root, _, manifest = _build_tree(tmp_path)
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(LeapAssetError, match="hash_mismatch"):
        leap_assets.verify_manifest(_write_manifest(tmp_path, manifest), root)


def test_tampered_byte_size_rejected(tmp_path):
    root, _, manifest = _build_tree(tmp_path)
    manifest["files"][1]["bytes"] += 1
    with pytest.raises(LeapAssetError, match="size_mismatch"):
        leap_assets.verify_manifest(_write_manifest(tmp_path, manifest), root)


def test_missing_file_on_disk_rejected(tmp_path):
    root, manifest_path, _ = _build_tree(tmp_path)
    (root / "leap_v2_right" / "robot.urdf").unlink()
    with pytest.raises(LeapAssetError, match="missing="):
        leap_assets.verify_manifest(manifest_path, root)


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(LeapAssetError, match="manifest missing"):
        leap_assets.verify_manifest(tmp_path / "nope.json", tmp_path)


def test_empty_entries_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["files"] = []
    with pytest.raises(LeapAssetError, match="no file entries"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_missing_required_entry_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["files"] = manifest["files"][:-1]
    with pytest.raises(LeapAssetError, match="missing required"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_unexpected_entry_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["files"].append({"path": "leap_v2_right/extra.stl", "sha256": "a" * 64, "bytes": 4})
    with pytest.raises(LeapAssetError, match="unexpected"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_duplicate_entry_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["files"] = manifest["files"][:-1] + [manifest["files"][0], manifest["files"][0]]
    with pytest.raises(LeapAssetError, match="duplicate"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_absolute_path_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["files"][0]["path"] = "/etc/passwd"
    with pytest.raises(LeapAssetError, match="absolute path"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_dotdot_path_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["files"][0]["path"] = "leap_v2_right/../../escape.stl"
    with pytest.raises(LeapAssetError, match="escape"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_invalid_sha256_field_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["files"][0]["sha256"] = "not-a-hash"
    with pytest.raises(LeapAssetError, match="invalid sha256"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_bad_schema_version_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["schema_version"] = "leap-selected-assets.v0"
    with pytest.raises(LeapAssetError, match="schema_version mismatch"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_bad_pinned_commit_rejected(tmp_path):
    _, _, manifest = _build_tree(tmp_path)
    manifest["upstream"]["pinned_commit"] = "0" * 40
    with pytest.raises(LeapAssetError, match="pinned_commit mismatch"):
        leap_assets.load_and_validate_manifest(_write_manifest(tmp_path, manifest))


def test_symlink_file_rejected(tmp_path):
    root, manifest_path, _ = _build_tree(tmp_path)
    target = root / "LICENSE"
    payload = target.read_bytes()
    target.unlink()
    real = tmp_path / "real_LICENSE"
    real.write_bytes(payload)
    target.symlink_to(real)
    with pytest.raises(LeapAssetError, match="symlink"):
        leap_assets.verify_manifest(manifest_path, root)


def test_symlink_directory_escape_rejected(tmp_path):
    root, manifest_path, _ = _build_tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = root / "leap_v2_right"
    moved.rename(outside / "leap_v2_right")
    moved.symlink_to(outside / "leap_v2_right", target_is_directory=True)
    with pytest.raises(LeapAssetError, match="symlink|escapes"):
        leap_assets.verify_manifest(manifest_path, root)


def test_download_verified_success(tmp_path):
    payload = b"pinned payload"
    root = tmp_path / "downloads"
    dest = root / "LICENSE"
    leap_assets.download_verified(
        "https://example.invalid/x",
        dest,
        root=root,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
        fetch=lambda url: payload,
    )
    assert dest.read_bytes() == payload


def test_download_verified_mismatch_is_atomic(tmp_path):
    good = b"good pinned content"
    root = tmp_path / "downloads"
    dest = root / "LICENSE"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(good)
    with pytest.raises(LeapAssetError, match="failed pinned verification"):
        leap_assets.download_verified(
            "https://example.invalid/x",
            dest,
            root=root,
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
            expected_bytes=8,
            fetch=lambda url: b"tampered download",
        )
    assert dest.read_bytes() == good
    assert [p for p in dest.parent.iterdir() if p.name != "LICENSE"] == []


def test_download_verified_rejects_symlinked_parent_before_fetch(tmp_path):
    root = tmp_path / "downloads"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary.txt"
    canary.write_text("must survive")
    (root / "leap_v2_right").symlink_to(outside, target_is_directory=True)

    def boom(url):  # must never run: validation happens before any fetch
        raise AssertionError("fetch attempted before path validation")

    with pytest.raises(LeapAssetError, match="symlink"):
        leap_assets.download_verified(
            "https://example.invalid/x",
            root / "leap_v2_right" / "robot.urdf",
            root=root,
            expected_sha256="0" * 64,
            expected_bytes=1,
            fetch=boom,
        )
    assert canary.read_text() == "must survive"
    assert [p.name for p in outside.iterdir()] == ["canary.txt"]


def test_download_verified_rejects_symlink_dest_before_fetch(tmp_path):
    root = tmp_path / "downloads"
    root.mkdir()
    canary = tmp_path / "canary.txt"
    canary.write_text("must survive")
    (root / "LICENSE").symlink_to(canary)
    with pytest.raises(LeapAssetError, match="symlink"):
        leap_assets.download_verified(
            "https://example.invalid/x",
            root / "LICENSE",
            root=root,
            expected_sha256="0" * 64,
            expected_bytes=1,
            fetch=lambda url: pytest.fail("fetch attempted before path validation"),
        )
    assert canary.read_text() == "must survive"


def test_download_verified_rejects_escape_before_fetch(tmp_path):
    root = tmp_path / "downloads"
    root.mkdir()
    with pytest.raises(LeapAssetError, match="escapes download root"):
        leap_assets.download_verified(
            "https://example.invalid/x",
            root / ".." / "evil.stl",
            root=root,
            expected_sha256="0" * 64,
            expected_bytes=1,
            fetch=lambda url: pytest.fail("fetch attempted before path validation"),
        )


def test_download_verified_ignores_preexisting_tmp_symlink(tmp_path):
    # A stale symlink at the legacy fixed temp name must not be followed or
    # overwritten through; the payload goes to a uniquely created temp file.
    root = tmp_path / "downloads"
    root.mkdir()
    canary = tmp_path / "canary.txt"
    canary.write_text("must survive")
    legacy_tmp = root / "LICENSE.tmp"
    legacy_tmp.symlink_to(canary)
    payload = b"pinned payload"
    dest = root / "LICENSE"
    leap_assets.download_verified(
        "https://example.invalid/x",
        dest,
        root=root,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
        fetch=lambda url: payload,
    )
    assert dest.read_bytes() == payload
    assert canary.read_text() == "must survive"
    assert legacy_tmp.is_symlink()


def test_fetch_tool_reuses_and_never_rewrites_manifest(tmp_path):
    root, manifest_path, _ = _build_tree(tmp_path)
    before = manifest_path.read_bytes()

    def boom(url):  # pragma: no cover - must never be called for matching files
        raise AssertionError("network fetch attempted for reusable files")

    tool = _load_fetch_tool()
    assert tool.main([], manifest_path=manifest_path, download_root=root, fetch=boom) == 0
    assert manifest_path.read_bytes() == before


def test_fetch_tool_mismatch_exits_nonzero_and_preserves_manifest(tmp_path):
    root, manifest_path, manifest = _build_tree(tmp_path)
    before = manifest_path.read_bytes()
    entry = next(e for e in manifest["files"] if e["path"] == "LICENSE")
    (root / "LICENSE").unlink()  # force a re-download of this one file

    tool = _load_fetch_tool()
    rc = tool.main(
        [], manifest_path=manifest_path, download_root=root, fetch=lambda url: b"evil payload"
    )
    assert rc == 1
    assert manifest_path.read_bytes() == before
    assert not (root / "LICENSE").exists()
    assert [p for p in root.iterdir() if p.name.startswith("LICENSE")] == []
    assert entry["sha256"] != hashlib.sha256(b"evil payload").hexdigest()


def test_fetch_tool_dir_symlink_no_fetch_no_outside_write(tmp_path):
    root, manifest_path, _ = _build_tree(tmp_path)
    before = manifest_path.read_bytes()
    # Replace the real mesh directory with a symlink to an outside directory
    # containing only a canary; nothing may be fetched or written through it.
    (root / "leap_v2_right").rename(tmp_path / "real_meshes")
    outside = tmp_path / "outside"
    outside.mkdir()
    canary = outside / "canary.txt"
    canary.write_text("must survive")
    (root / "leap_v2_right").symlink_to(outside, target_is_directory=True)
    fetched = []

    tool = _load_fetch_tool()
    rc = tool.main(
        [],
        manifest_path=manifest_path,
        download_root=root,
        fetch=lambda url: fetched.append(url) or b"payload",
    )
    assert rc == 1
    assert fetched == []
    assert canary.read_text() == "must survive"
    assert [p.name for p in outside.iterdir()] == ["canary.txt"]
    assert manifest_path.read_bytes() == before


def test_fetch_tool_check_mode_shares_verifier(tmp_path):
    root, manifest_path, _ = _build_tree(tmp_path)
    tool = _load_fetch_tool()
    assert tool.main(["--check"], manifest_path=manifest_path, download_root=root) == 0
    (root / "README.upstream.md").write_bytes(b"tampered")
    assert tool.main(["--check"], manifest_path=manifest_path, download_root=root) == 1


@pytest.mark.skipif(not REAL_CACHE.is_dir(), reason="real asset cache not available")
def test_real_cache_matches_committed_manifest():
    manifest = leap_assets.verify_manifest(leap_assets.MANIFEST_JSON, REAL_CACHE)
    assert len(manifest["files"]) == 13
