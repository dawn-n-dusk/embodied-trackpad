"""Fetch the pinned LEAP Hand v2 Advanced model assets into the ignored download root.

The upstream repository ``leap-hand/LEAP_Hand_V2_Adv_API`` ships a URDF plus
STL meshes for the v2 Advanced hand. Meshes are NOT redistributed by this
repository; they are fetched on demand and verified against the committed
sha256 manifest ``assets/manifests/leap-selected.json``.

Integrity model: the committed manifest is the ONLY authority for the 13
pinned files (paths, sha256, bytes). This script NEVER rewrites, re-authors or
overwrites the manifest or its hashes. Default mode validates the manifest,
reuses any local file that already matches the pinned hash, and otherwise
downloads to a temporary file, verifies the expected sha256/bytes, and only
then atomically replaces the local file. Any verification mismatch exits
nonzero and leaves existing files untouched.

Licensing: the nested upstream ``leap_v2/LICENSE`` is Apache-2.0 and covers the
upstream software (a verbatim copy is retained alongside the assets). The
official CAD page https://v2-adv.leaphand.com/leap_cad lists CC BY-NC-SA; no
blanket relicensing or redistribution permission is inferred for individual
assets. Assets stay in the git-ignored download root and are never
redistributed by this repository.

Usage:
    python tools/fetch_leap_assets.py            # download + verify against the pinned manifest
    python tools/fetch_leap_assets.py --check    # verify local files against the manifest only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_trackpad import leap_assets  # noqa: E402
from embodied_trackpad.leap_assets import LeapAssetError  # noqa: E402


def main(
    argv: list[str] | None = None,
    *,
    manifest_path: Path | None = None,
    download_root: Path | None = None,
    fetch=None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only, no downloads")
    args = parser.parse_args(argv)

    manifest_path = Path(manifest_path) if manifest_path else leap_assets.MANIFEST_JSON
    download_root = Path(download_root) if download_root else leap_assets.DOWNLOAD_ROOT

    try:
        manifest = leap_assets.load_and_validate_manifest(manifest_path)
        if args.check:
            leap_assets.verify_manifest(manifest_path, download_root)
            print(f"OK: {len(manifest['files'])} files match {manifest_path.name}")
            return 0

        entries = {entry["path"]: entry for entry in manifest["files"]}
        for upstream_path, local_rel in leap_assets.FILES.items():
            entry = entries[local_rel]
            dest = download_root / local_rel
            # Reject symlink/escaping destinations before any read, fetch or write.
            leap_assets.validate_dest_path(download_root, dest)
            if dest.is_file() and leap_assets.sha256_of(dest) == entry["sha256"]:
                print(f"reuse {local_rel} (matches pinned sha256)")
                continue
            url = f"{leap_assets.RAW_BASE}/{upstream_path}"
            leap_assets.download_verified(
                url,
                dest,
                root=download_root,
                expected_sha256=entry["sha256"],
                expected_bytes=entry["bytes"],
                fetch=fetch,
            )
            try:
                shown = dest.relative_to(REPO_ROOT)
            except ValueError:
                shown = dest
            print(f"fetched {url} -> {shown} (sha256 verified)")
        leap_assets.verify_manifest(manifest_path, download_root)
    except LeapAssetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(manifest['files'])} files verified against {manifest_path.name}; "
        "manifest left unchanged"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
