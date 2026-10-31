"""Stdlib-only tests for the pinned LEAP v1 Full asset manifest and verifier."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from embodied_trackpad.leap_v1 import assets
from embodied_trackpad.leap_v1.assets import LeapAssetError

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "assets" / "manifests" / "leap-v1-full.json"


def _write_files(root: Path, entries, corrupt: str | None = None) -> None:
    for e in entries:
        dest = root / e["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = b"\x00" * e["bytes"]
        if e["path"] == corrupt:
            payload = b"\xff" * e["bytes"]
        dest.write_bytes(payload)


def _manifest_entries():
    return json.loads(MANIFEST.read_text())["files"]


def test_committed_manifest_validates():
    manifest = assets.load_and_validate_manifest(MANIFEST)
    assert manifest["schema_version"] == "leap-v1-full-assets.v1"
    assert manifest["upstream"]["pinned_commit"] == assets.PINNED_COMMIT
    paths = {e["path"] for e in manifest["files"]}
    assert paths == assets.REQUIRED_PATHS
    assert len(paths) == 14
    # right-hand only: no left-hand, mount, scene or preview files
    assert not any("left" in p or "mount" in p or "scene" in p for p in paths)
    total = sum(e["bytes"] for e in manifest["files"])
    assert total <= assets.MAX_DOWNLOAD_BYTES


def test_files_whitelist_matches_manifest():
    assert set(assets.FILES.values()) == {
        e["path"] for e in _manifest_entries()
    }
    assert len(assets.FILES) == 14


def test_manifest_rejects_bad_version_and_commit(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["schema_version"] = "bogus"
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(LeapAssetError, match="schema_version"):
        assets.load_and_validate_manifest(bad)
    data = json.loads(MANIFEST.read_text())
    data["upstream"]["pinned_commit"] = "0" * 40
    bad.write_text(json.dumps(data))
    with pytest.raises(LeapAssetError, match="pinned_commit"):
        assets.load_and_validate_manifest(bad)


def test_manifest_rejects_missing_and_extra_files(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["files"] = data["files"][1:]
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(LeapAssetError, match="missing required"):
        assets.load_and_validate_manifest(bad)
    data = json.loads(MANIFEST.read_text())
    data["files"].append(dict(data["files"][0]))
    bad.write_text(json.dumps(data))
    with pytest.raises(LeapAssetError, match="duplicate"):
        assets.load_and_validate_manifest(bad)


def test_manifest_rejects_path_escape(tmp_path):
    data = json.loads(MANIFEST.read_text())
    data["files"][0]["path"] = "../escape.xml"
    bad = tmp_path / "m.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(LeapAssetError, match="escape"):
        assets.load_and_validate_manifest(bad)


def test_verify_local_files(tmp_path):
    entries = _manifest_entries()
    # Correct content (hashes recomputed into a scratch manifest) verifies.
    import hashlib

    scratch = json.loads(MANIFEST.read_text())
    for e in scratch["files"]:
        payload = hashlib.sha256(e["path"].encode()).digest()
        e["bytes"] = len(payload)
        e["sha256"] = hashlib.sha256(payload).hexdigest()
        dest = tmp_path / e["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(scratch))
    assert assets.verify_manifest(mpath, tmp_path)["files"]
    # Corrupt one byte -> verification must fail loudly.
    victim = tmp_path / scratch["files"][0]["path"]
    victim.write_bytes(b"\x00" * victim.stat().st_size)
    with pytest.raises(LeapAssetError, match="hash_mismatch|size_mismatch"):
        assets.verify_manifest(mpath, tmp_path)
    # Missing file -> fail.
    (tmp_path / scratch["files"][1]["path"]).unlink()
    with pytest.raises(LeapAssetError, match="missing"):
        assets.verify_manifest(mpath, tmp_path)


def test_download_verified_stubbed_fetch(tmp_path):
    payload = b"pinned v1 bytes"
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "right_hand" / "x.obj"
    assets.download_verified(
        "https://example.invalid/x", dest,
        root=tmp_path, expected_sha256=sha, expected_bytes=len(payload),
        fetch=lambda url: payload,
    )
    assert dest.read_bytes() == payload
    with pytest.raises(LeapAssetError, match="failed pinned verification"):
        assets.download_verified(
            "https://example.invalid/x", dest,
            root=tmp_path, expected_sha256="0" * 64, expected_bytes=len(payload),
            fetch=lambda url: payload,
        )
    assert dest.read_bytes() == payload  # existing file untouched


def test_dest_path_safety(tmp_path):
    with pytest.raises(LeapAssetError):
        assets.validate_dest_path(tmp_path, tmp_path / ".." / "escape")
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(LeapAssetError, match="symlink"):
        assets.validate_dest_path(tmp_path, link / "x")


@pytest.mark.skipif(not assets.DOWNLOAD_ROOT.is_dir(), reason="v1 asset cache not available")
def test_local_cache_matches_manifest():
    manifest = assets.verify_manifest()
    assert len(manifest["files"]) == 14


SRC_DIR = REPO_ROOT / "src"
FETCH_TOOL = REPO_ROOT / "tools" / "fetch_leap_v1_assets.py"


def _run_no_sitepackages(args, **kwargs):
    """Run the current interpreter with -S (no site-packages => no mujoco/numpy)."""
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-S", *args],
        capture_output=True,
        text=True,
        timeout=120,
        **kwargs,
    )


def test_assets_importable_without_optional_physics_deps():
    """Regression: leap_v1.assets must import on a foundation-only install.

    The package __init__ must stay lazy: importing the stdlib-only asset
    module must not pull in mujoco/numpy (coordinator reproduction failed
    with ModuleNotFoundError: mujoco before the fix).
    """
    snippet = (
        "import sys; sys.path.insert(0, {src!r});"
        "import embodied_trackpad.leap_v1.assets as a;"
        "assert a.PINNED_COMMIT and a.REQUIRED_PATHS;"
        "print('assets-ok')"
    ).format(src=str(SRC_DIR))
    proc = _run_no_sitepackages(["-c", snippet])
    assert proc.returncode == 0, proc.stderr
    assert "assets-ok" in proc.stdout
    assert "mujoco" not in proc.stderr


def test_lazy_model_exports_raise_missing_dep_without_physics():
    """Under -S, requesting a model export surfaces the missing optional dep."""
    snippet = (
        "import sys; sys.path.insert(0, {src!r});"
        "import embodied_trackpad.leap_v1 as pkg;"
        "ok = False\n"
        "try:\n"
        "    pkg.load_hand\n"
        "except ModuleNotFoundError as exc:\n"
        "    ok = exc.name in ('mujoco', 'numpy')\n"
        "assert ok, 'expected ModuleNotFoundError for mujoco/numpy'\n"
        "print('lazy-ok')"
    ).format(src=str(SRC_DIR))
    proc = _run_no_sitepackages(["-c", snippet])
    assert proc.returncode == 0, proc.stderr
    assert "lazy-ok" in proc.stdout


def test_fetch_tool_runs_without_optional_physics_deps():
    """The downloader CLI must work with no site-packages physics deps."""
    proc = _run_no_sitepackages([str(FETCH_TOOL), "--help"])
    assert proc.returncode == 0, proc.stderr
    assert "--check" in proc.stdout


@pytest.mark.skipif(not assets.DOWNLOAD_ROOT.is_dir(), reason="v1 asset cache not available")
def test_fetch_tool_check_without_optional_physics_deps():
    proc = _run_no_sitepackages([str(FETCH_TOOL), "--check"])
    assert proc.returncode == 0, proc.stderr
    assert "14 files match" in proc.stdout
