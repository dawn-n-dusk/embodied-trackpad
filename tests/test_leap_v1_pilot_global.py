"""Global-level audit rejection tests for LEAP v1 pilot collections.

No physics stepping, no app launch, no live data collection.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from embodied_trackpad.leap_v1.pilot_dataset import sha256_file, split_of
from tests.test_leap_v1_pilot_data_audit import _make_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]

FORCE_CONFIG = {
    "target_force_n": 0.8,
    "gain_m_per_ns": 0.004,
    "max_normal_speed_m_s": 0.002,
    "max_joint_step_rad": 0.0002,
    "max_cycle_delta_rad": 0.005,
}
FORCE_GOVERNOR_CONFIG = {
    "substeps_per_frame": 25,
    "target_force_n": 0.8,
    "tangential_stop_force_n": 1.1,
    "per_substep_total_joint_delta_cap_rad": 0.0002,
    "max_cycle_delta_rad": 0.005,
}
TRAIN_SEEDS = [0, 2, 3, 5, 6, 8, 9, 11]
VAL_SEEDS = [1, 7]
TEST_SEEDS = [4, 10]


def _run_cli(pilot: Path, acceptance: Path, repo_root: Path, out: Path, extra: list[str] | None = None) -> subprocess.CompletedProcess:
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


def _make_valid_collection(tmp_path: Path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    for seed in range(12):
        _make_fixture(pilot, seed)

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

    prospective_index = [{"seed": s, "group": split_of(s)} for s in range(12)]
    manifest = {
        "source_commit": "abc123",
        "source_hash_verified": {src_rel: src_sha},
        "asset_manifest_path": "assets/manifests/leap-v1-full.json",
        "asset_manifest_sha256": asset_sha,
        "native_binary_path": ".local/build/TrackpadSandbox.app/Contents/MacOS/TrackpadSandbox",
        "native_binary_sha256": nb_sha,
        "planner_kind": "force-governed",
        "controller_kind": "privileged_visual_teacher",
        "force_config": FORCE_CONFIG,
        "force_governor_config": FORCE_GOVERNOR_CONFIG,
        "seeds_attempt_order": list(range(12)),
        "prospective_split": {
            "group_key": "seed modulo 12",
            "train": TRAIN_SEEDS,
            "validation": VAL_SEEDS,
            "test": TEST_SEEDS,
            "prospective_index": prospective_index,
        },
        "raw_flags": {"training_eligible": False, "privileged_visual_teacher": True},
    }
    (pilot / "manifest.json").write_text(json.dumps(manifest))

    batch_summary = {
        "source_commit": "abc123",
        "asset_manifest_sha256": asset_sha,
        "native_binary_sha256": nb_sha,
        "planner_kind": "force-governed",
        "controller_kind": "privileged_visual_teacher",
        "collection_wall_s": 2.0,
        "attempted_seeds": list(range(12)),
        "successful_seeds": list(range(12)),
    }
    (pilot / "batch-summary.json").write_text(json.dumps(batch_summary))

    acceptance = {
        "source_commit": "abc123",
        "asset_manifest": {"sha256": asset_sha, "path": "assets/manifests/leap-v1-full.json"},
        "native_binary": {"sha256": nb_sha, "path": ".local/build/TrackpadSandbox.app/Contents/MacOS/TrackpadSandbox"},
        "config": {
            "planner_kind": "force-governed",
            "seeds": list(range(12)),
            "force_config": FORCE_CONFIG,
            "force_governor_config": FORCE_GOVERNOR_CONFIG,
        },
        "controller_kind": "privileged_visual_teacher",
        "prospective_split": {"train": TRAIN_SEEDS, "validation": VAL_SEEDS, "test": TEST_SEEDS},
        "source_hashes": {src_rel: src_sha},
    }
    acc_path = tmp_path / "accepted.json"
    acc_path.write_text(json.dumps(acceptance))

    return pilot, acc_path, tmp_path


class TestGlobalAcceptance:
    def test_valid_collection_passes(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode == 0, result.stderr
        with open(out / "dataset_manifest.json", "r", encoding="utf-8") as f:
            dm = json.load(f)
        assert dm["accepted_episodes"] == 12
        assert dm["total_action_frames"] == 48
        assert dm["sample_index_sha256"] is not None

    def test_asset_manifest_hash_conflict_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["asset_manifest_sha256"] = "0" * 64
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("asset manifest manifest-vs-acceptance hash conflict" in g for g in audit["global_issues"])

    def test_native_binary_hash_conflict_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["native_binary_sha256"] = "0" * 64
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("native binary manifest-vs-acceptance hash conflict" in g for g in audit["global_issues"])

    def test_changed_force_config_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["force_config"] = {**FORCE_CONFIG, "target_force_n": 1.5}
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("force_config mismatch" in g for g in audit["global_issues"])

    def test_changed_force_governor_config_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["force_governor_config"] = {**FORCE_GOVERNOR_CONFIG, "substeps_per_frame": 50}
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("force_governor_config mismatch" in g for g in audit["global_issues"])

    def test_duplicate_seed_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["seeds_attempt_order"] = [0, 0] + list(range(2, 12))
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("seeds_attempt_order not exactly [0..11]" in g for g in audit["global_issues"])

    def test_missing_seed_directory_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        import shutil
        shutil.rmtree(pilot / "seed-011")
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("seed directories" in g and "do not match 0..11" in g for g in audit["global_issues"])

    def test_swapped_whole_class_split_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["prospective_split"]["train"] = [0, 2, 3, 5, 6, 8, 9, 4]
        manifest["prospective_split"]["test"] = [10, 11]
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("prospective_split train mismatch" in g for g in audit["global_issues"])
        assert any("prospective_split test mismatch" in g for g in audit["global_issues"])

    def test_changed_actual_source_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        src_path = repo_root / "src" / "embodied_trackpad" / "leap_v1" / "fake_source.py"
        src_path.write_text("# tampered\n")
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("manifest source hash mismatch" in g for g in audit["global_issues"])
        assert any("acceptance source hash mismatch" in g for g in audit["global_issues"])

    def test_missing_source_file_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        src_path = repo_root / "src" / "embodied_trackpad" / "leap_v1" / "fake_source.py"
        src_path.unlink()
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("source file missing" in g for g in audit["global_issues"])

    def test_missing_manifest_source_hash_rejects(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["source_hash_verified"] = {}
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        audit = json.loads((out / "audit.json").read_text())
        assert any("manifest missing source hash for accepted file" in g for g in audit["global_issues"])

    def test_global_failure_exports_no_samples(self, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        manifest["asset_manifest_sha256"] = "0" * 64
        (pilot / "manifest.json").write_text(json.dumps(manifest))
        out = tmp_path / "out"
        result = _run_cli(pilot, acc_path, repo_root, out)
        assert result.returncode != 0
        sample_index = out / "sample_index.jsonl"
        assert sample_index.exists()
        assert sample_index.stat().st_size == 0
        dm = json.loads((out / "dataset_manifest.json").read_text())
        assert dm["accepted_episodes"] == 0
        assert dm["total_action_frames"] == 0


@pytest.fixture(scope="module")
def audit_mod():
    repo_src = str(REPO_ROOT / "src")
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "audit_leap_v1_pilot", str(REPO_ROOT / "tools" / "audit_leap_v1_pilot.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGlobalClaimGuards:
    def test_rehash_null_source_maps_returns_issues(self, audit_mod, tmp_path: Path):
        issues = audit_mod._rehash_source_files(
            {"source_hash_verified": None}, {"source_hashes": None}, tmp_path
        )
        assert any("manifest source_hash_verified missing or empty" in i for i in issues)
        assert any("acceptance source_hashes missing or empty" in i for i in issues)

    def test_rehash_invalid_source_hash_returns_issues(self, audit_mod, tmp_path: Path):
        manifest = {"source_hash_verified": {"src/fake.py": "notahexhash"}}
        acceptance = {"source_hashes": {"src/fake.py": "0" * 62}}
        issues = audit_mod._rehash_source_files(manifest, acceptance, tmp_path)
        assert any("manifest source hash invalid" in i for i in issues)
        assert any("acceptance source hash invalid" in i for i in issues)

    def test_verify_global_asset_native_claims_required(self, audit_mod, tmp_path: Path):
        pilot, acc_path, repo_root = _make_valid_collection(tmp_path)
        manifest = json.loads((pilot / "manifest.json").read_text())
        batch_summary = json.loads((pilot / "batch-summary.json").read_text())
        acceptance = json.loads(acc_path.read_text())
        seed_dirs = sorted([d for d in pilot.iterdir() if d.is_dir() and d.name.startswith("seed-")])
        manifest["asset_manifest_sha256"] = None
        manifest["native_binary_sha256"] = "gg" * 32
        acceptance["asset_manifest"]["sha256"] = ""
        del acceptance["native_binary"]["sha256"]
        issues = audit_mod._verify_global(manifest, batch_summary, acceptance, repo_root, seed_dirs)
        assert any("asset manifest hash missing or invalid" in i for i in issues)
        assert any("native binary hash missing or invalid" in i for i in issues)
