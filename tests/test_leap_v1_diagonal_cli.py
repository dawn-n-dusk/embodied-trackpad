"""Functional CLI tests for the near_diagonal_v1 audit adaptation.

Compact synthetic fixtures only: the legacy 12-seed collection builder from
test_leap_v1_pilot_global is converted into a diagonal collection with the
_enrich_clock/_enrich_teacher_config helpers from test_leap_v1_profile_export.
No real episodes, no physics, no network.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from embodied_trackpad.leap_v1.pilot_dataset import (  # noqa: E402
    sha256_file,
    split_of,
)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tests" / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fx_global = _load("pilot_global_fixtures", "test_leap_v1_pilot_global.py")
fx_profile = _load("pilot_profile_export_fixtures", "test_leap_v1_profile_export.py")

DIAG = "near_diagonal_v1"
AXIAL = "near_pointing_v1"
COLL = "diagdev-A_1"
GOOD_COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
SCRIPT = REPO_ROOT / "tools" / "audit_leap_v1_pilot.py"
DIAGONAL_EXTRA = ["--task-profile", DIAG, "--collection-id", COLL]


def _run_cli(pilot: Path, acc: Path, repo_root: Path, out: Path,
             extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(SCRIPT),
        "--pilot-dir", str(pilot),
        "--acceptance", str(acc),
        "--repo-root", str(repo_root),
        "--out", str(out),
    ] + (extra if extra is not None else DIAGONAL_EXTRA)
    env = os.environ.copy()
    env["PYTHONPATH"] = "src:."
    return subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)


def _write_docs(pilot: Path, acc_path: Path, man: dict, batch: dict, accept: dict) -> None:
    (pilot / "manifest.json").write_text(json.dumps(man))
    (pilot / "batch-summary.json").write_text(json.dumps(batch))
    acc_path.write_text(json.dumps(accept))


def _make_diagonal_collection(tmp_path: Path):
    """Legacy 12-seed collection converted to a valid diagonal collection."""
    pilot, acc_path, repo_root = fx_global._make_valid_collection(tmp_path)
    for seed in range(8, 12):
        shutil.rmtree(pilot / f"seed-{seed:03d}")
    for seed in range(8):
        ep = pilot / f"seed-{seed:03d}"
        fx_profile._enrich_clock(ep)
        fx_profile._enrich_teacher_config(ep)
        for name in ("metadata.json", "summary.json"):
            p = ep / name
            d = json.loads(p.read_text())
            d.update(task_profile=DIAG, collection_id=COLL)
            p.write_text(json.dumps(d))
    md = json.loads((pilot / "seed-000/summary.json").read_text())
    cfg = {k: copy.deepcopy(md[k]) for k in (
        "contact_mode", "planner_kind", "force_config", "force_governor_config",
        "png_backend", "png_backend_version")}
    man = json.loads((pilot / "manifest.json").read_text())
    batch = json.loads((pilot / "batch-summary.json").read_text())
    accept = json.loads(acc_path.read_text())
    idx = [{"seed": i, "group": split_of(i, task_profile=DIAG)} for i in range(8)]
    # Deep-copy cfg per document: nested config dicts must not be shared, so
    # a batch-only mutation cannot alias into manifest/acceptance.
    man.update(task_profile=DIAG, collection_id=COLL, source_commit=GOOD_COMMIT,
               **copy.deepcopy(cfg))
    batch.update(task_profile=DIAG, collection_id=COLL, source_commit=GOOD_COMMIT,
                 **copy.deepcopy(cfg))
    man.update(
        seeds_attempt_order=list(range(8)),
        prospective_split={"train": [0, 1, 2, 3], "validation": [4, 6],
                           "test": [5, 7], "prospective_index": idx})
    batch.update(attempted_seeds=list(range(8)), successful_seeds=list(range(8)))
    accept.update(source_commit=GOOD_COMMIT, collection_id=COLL,
                  prospective_split={"train": [0, 1, 2, 3], "validation": [4, 6],
                                     "test": [5, 7]})
    accept["config"].update(task_profile=DIAG, seeds=list(range(8)),
                            **copy.deepcopy(cfg))
    _write_docs(pilot, acc_path, man, batch, accept)
    return pilot, acc_path, repo_root, man, batch, accept


class TestDiagonalCLIValid:
    def test_valid_collection_32_samples_16_8_8(self, tmp_path):
        pilot, acc, repo, *_ = _make_diagonal_collection(tmp_path)
        out = tmp_path / "out"
        r = _run_cli(pilot, acc, repo, out)
        assert r.returncode == 0, r.stderr
        assert "Traceback" not in r.stderr
        dm = json.loads((out / "dataset_manifest.json").read_text())
        assert dm["task_profile"] == DIAG and dm["collection_id"] == COLL
        assert dm["accepted_episodes"] == 8 and dm["total_episodes"] == 8
        assert dm["total_action_frames"] == 32
        assert dm["split_counts"] == {"train": 16, "validation": 8, "test": 8, "rejected": 0}
        assert dm["target_classes"] == list(range(8))
        assert dm["collection_source_commit"] == GOOD_COMMIT
        assert dm["source_commit"] == GOOD_COMMIT  # legacy capture field retained
        hashes = dm["auditor_source_sha256"]
        assert set(hashes) == {
            "tools/audit_leap_v1_pilot.py",
            "src/embodied_trackpad/leap_v1/pilot_dataset.py",
            "src/embodied_trackpad/leap_v1/pilot_provenance.py",
            "src/embodied_trackpad/leap_v1/pilot_collection.py",
        }
        assert all(len(h) == 64 for h in hashes.values())
        assert hashes["tools/audit_leap_v1_pilot.py"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        rec = dm["reconciliation"]
        assert rec["planned_seeds"] == list(range(8))
        assert rec["accepted_seeds"] == list(range(8))
        assert rec["missing_seed_dirs"] == [] and rec["rejected_seeds"] == []
        lines = [ln for ln in (out / "sample_index.jsonl").read_text().splitlines() if ln.strip()]
        assert len(lines) == 32
        first = json.loads(lines[0])
        assert first["task_profile"] == DIAG and first["collection_id"] == COLL
        audit = json.loads((out / "audit.json").read_text())
        assert audit["collection_source_commit"] == GOOD_COMMIT
        assert audit["auditor_source_sha256"] == hashes


class TestDiagonalCLIRejections:
    def _reject(self, tmp_path, mutate):
        pilot, acc, repo, man, batch, accept = _make_diagonal_collection(tmp_path)
        # Docs share nested config dicts in the fixture; isolate before mutating.
        man, batch, accept = copy.deepcopy((man, batch, accept))
        mutate(man, batch, accept)
        _write_docs(pilot, acc, man, batch, accept)
        out = tmp_path / "out"
        r = _run_cli(pilot, acc, repo, out)
        assert r.returncode != 0
        assert "Traceback" not in r.stderr, r.stderr
        assert (out / "sample_index.jsonl").stat().st_size == 0
        dm = json.loads((out / "dataset_manifest.json").read_text())
        assert dm["accepted_episodes"] == 0 and dm["total_action_frames"] == 0
        return r, json.loads((out / "audit.json").read_text())

    def test_wrong_task_profile_rejected(self, tmp_path):
        _, audit = self._reject(tmp_path, lambda m, b, a: m.update(task_profile=AXIAL))
        assert any("task_profile" in g for g in audit["global_issues"])

    def test_wrong_collection_id_rejected(self, tmp_path):
        _, audit = self._reject(tmp_path, lambda m, b, a: a.update(collection_id="other-coll"))
        assert any("collection_id" in g for g in audit["global_issues"])

    def test_teacher_config_conflict_no_traceback(self, tmp_path):
        r, audit = self._reject(
            tmp_path, lambda m, b, a: b["force_config"].update(target_force_n=0.8))
        assert any("teacher configuration mismatch" in g for g in audit["global_issues"])

    def test_backend_version_conflict_rejected(self, tmp_path):
        _, audit = self._reject(tmp_path, lambda m, b, a: m.update(png_backend="stdlib"))
        assert any("png_backend" in g for g in audit["global_issues"])

    def test_batch_asset_hash_conflict_rejected(self, tmp_path):
        _, audit = self._reject(
            tmp_path, lambda m, b, a: b.update(asset_manifest_sha256="f" * 64))
        assert any("batch-summary asset manifest hash" in g for g in audit["global_issues"])

    def test_batch_controller_kind_conflict_rejected(self, tmp_path):
        _, audit = self._reject(
            tmp_path, lambda m, b, a: b.update(controller_kind="open_loop"))
        assert any("controller_kind" in g for g in audit["global_issues"])

    def test_swapped_frozen_groups_rejected(self, tmp_path):
        def swap(m, b, a):
            m["prospective_split"].update(train=[4, 6], validation=[0, 1, 2, 3])
            a["prospective_split"].update(train=[4, 6], validation=[0, 1, 2, 3])
        _, audit = self._reject(tmp_path, swap)
        assert any("prospective_split" in g for g in audit["global_issues"])

    def test_claim_fewer_success_extra_accepted_rejected(self, tmp_path):
        # successful_seeds=0..6 while all 8 episodes audit clean: the truthful
        # report is an extra accepted episode, never a fabricated missing one.
        _, audit = self._reject(tmp_path, lambda m, b, a: b.update(successful_seeds=list(range(7))))
        rec = audit["reconciliation"]
        assert rec["missing_episodes"] == []
        assert rec["extra_episodes"] == [7]
        assert any("accepted beyond batch successful seeds" in i for i in rec["issues"])


class TestDiagonalCLITruthfulPartial:
    def test_missing_directory_keeps_planned8_no_ready(self, tmp_path):
        pilot, acc, repo, *_ = _make_diagonal_collection(tmp_path)
        shutil.rmtree(pilot / "seed-006")
        out = tmp_path / "out"
        r = _run_cli(pilot, acc, repo, out)
        assert r.returncode != 0
        assert "Traceback" not in r.stderr
        audit = json.loads((out / "audit.json").read_text())
        assert any("do not match 0..7" in g for g in audit["global_issues"])
        rec = audit["reconciliation"]
        assert rec["planned_seeds"] == list(range(8))
        assert rec["observed_seed_dirs"] == [0, 1, 2, 3, 4, 5, 7]
        assert rec["missing_seed_dirs"] == [6]
        assert rec["accepted_seeds"] == [0, 1, 2, 3, 4, 5, 7]
        assert rec["rejected_seeds"] == []
        dm = json.loads((out / "dataset_manifest.json").read_text())
        assert dm["total_episodes"] == 7  # failed denominators preserved
        assert dm["accepted_episodes"] == 0 and dm["total_action_frames"] == 0
        assert (out / "sample_index.jsonl").stat().st_size == 0

    def test_malformed_seed_dir_name_reports_not_crashes(self, tmp_path):
        pilot, acc, repo, *_ = _make_diagonal_collection(tmp_path)
        (pilot / "seed-007").rename(pilot / "seed-abc")
        out = tmp_path / "out"
        r = _run_cli(pilot, acc, repo, out)
        assert r.returncode != 0
        assert "Traceback" not in r.stderr
        audit = json.loads((out / "audit.json").read_text())
        assert any("malformed seed directory names" in g for g in audit["global_issues"])
        assert audit["reconciliation"]["malformed_seed_dirs"] == ["seed-abc"]

    def test_early_rejected_episode_serializes_without_traceback(self, tmp_path):
        # Corrupt one episode's planner teacher config so audit_episode
        # rejects it before the touch validator runs (touch_sequences is
        # never set); the CLI must still report without a traceback.
        pilot, acc, repo, *_ = _make_diagonal_collection(tmp_path)
        meta_path = pilot / "seed-003/metadata.json"
        meta = json.loads(meta_path.read_text())
        meta["planner"]["force_config"]["target_force_n"] = 0.8
        meta_path.write_text(json.dumps(meta))
        out = tmp_path / "out"
        r = _run_cli(pilot, acc, repo, out)
        assert r.returncode != 0
        assert "Traceback" not in r.stderr, r.stderr
        audit = json.loads((out / "audit.json").read_text())
        ep = next(e for e in audit["episodes"] if e["seed"] == 3)
        assert ep["accepted"] is False
        assert ep["touch_sequences"] == 0  # safe serialization default
        assert audit["reconciliation"]["rejected_seeds"] == [3]
        assert audit["reconciliation"]["missing_episodes"] == [3]

    def test_missing_collection_id_rejected_before_outputs(self, tmp_path):
        pilot, acc, repo, *_ = _make_diagonal_collection(tmp_path)
        out = tmp_path / "out"
        r = _run_cli(pilot, acc, repo, out, extra=["--task-profile", DIAG])
        assert r.returncode != 0
        assert not (out / "audit.json").exists()
        assert not (out / "sample_index.jsonl").exists()


class TestLegacyDefaultCLIUnchanged:
    def test_old_cli_48_samples_32_8_8(self, tmp_path):
        pilot, acc, repo = fx_global._make_valid_collection(tmp_path)
        out = tmp_path / "out"
        r = _run_cli(pilot, acc, repo, out, extra=[])
        assert r.returncode == 0, r.stderr
        dm = json.loads((out / "dataset_manifest.json").read_text())
        assert dm["total_action_frames"] == 48
        assert dm["accepted_episodes"] == 12
        assert dm["split_counts"] == {"train": 32, "validation": 8, "test": 8, "rejected": 0}
        assert "task_profile" not in dm and "collection_id" not in dm
        assert "auditor_source_sha256" not in dm and "reconciliation" not in dm
        lines = [ln for ln in (out / "sample_index.jsonl").read_text().splitlines() if ln.strip()]
        assert len(lines) == 48
        assert all("task_profile" not in json.loads(ln) for ln in lines[:5])
