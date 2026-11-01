#!/usr/bin/env python3
"""Stdlib-only admission tests for scripts/pi05_model_load_smoke.py.

Covers _admit gates with the real accepted weight report when present, a
temporary git HEAD, and a synthetic SLURM_JOB_ID. Without the .local report
(CI), a purely software 29-row synthetic report with the same expected total
bytes exercises the same gates; only the actual-report SHA pin test skips.
Never calls _load_and_count and never imports JAX/Flax/OpenPI.
Run directly: python3 tests/test_pi05_model_load_smoke.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pi05_model_load_smoke.py"
ACCEPTED_REPORT = ROOT / ".local" / "pi05-weights-stage-20261101" / "stage-report-resume-20261101.json"
ACCEPTED_SHA256 = "1e1b20ce412018d754734f767fa66fe9c3a90f412610e185c7a249c9c8f21b2d"
PINNED_HEAD = "215abfb217dbac7d5f1273282331b9b1866c0479"
EXPECTED_BYTES = 12441749581


class _Skip(unittest.SkipTest):
    """Clean skip for both the direct runner and pytest/unittest discovery."""


def _synthetic_report_data():
    """Purely software fixture: 29 unique verified rows summing to EXPECTED_BYTES."""
    rows = [
        {"name": f"synthetic/object/{i:02d}", "state": "verified", "size": 1, "actual_size": 1}
        for i in range(28)
    ]
    rows.append({"name": "synthetic/object/blob", "state": "verified",
                 "size": EXPECTED_BYTES - 28, "actual_size": EXPECTED_BYTES - 28})
    return {
        "audit": rows,
        "totals": {"verified_count": 29, "total_count": 29, "verified_bytes": EXPECTED_BYTES},
        "stop_reason": "all_verified",
    }


def _load_module():
    spec = importlib.util.spec_from_file_location("pi05_model_load_smoke", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load_module()


class _slurm_env:
    def __enter__(self):
        self._old = os.environ.get("SLURM_JOB_ID")
        os.environ["SLURM_JOB_ID"] = "smoke-test-job"

    def __exit__(self, *exc):
        if self._old is None:
            os.environ.pop("SLURM_JOB_ID", None)
        else:
            os.environ["SLURM_JOB_ID"] = self._old


def _fixture(tmp: Path, mutate=None, source="actual"):
    """Build a weight-report fixture. source="actual" uses the accepted report when
    present (else synthetic); source="synthetic" always builds the purely software
    29-row report so rejection tests behave identically in CI without .local data."""
    if source == "actual" and ACCEPTED_REPORT.exists():
        data = json.loads(ACCEPTED_REPORT.read_text())
    else:
        data = _synthetic_report_data()
    upstream = tmp / "upstream"
    (upstream / ".git").mkdir(parents=True)
    (upstream / ".git" / "HEAD").write_text(PINNED_HEAD + "\n")
    if mutate:
        mutate(data)
    raw = (json.dumps(data, indent=1) + "\n").encode()
    report = tmp / "weight-report.json"
    report.write_bytes(raw)
    output = tmp / "out" / "report.json"
    output.parent.mkdir(parents=True)
    return upstream, report, output, hashlib.sha256(raw).hexdigest()


def _parse(tmp, upstream, report, output, sha, deadline):
    return MOD._parse_args([
        "--upstream-root", str(upstream), "--weights-dir", str(tmp / "weights"),
        "--weight-report", str(report), "--expected-weight-report-sha256", sha,
        "--output-report", str(output), "--deadline-unix", str(deadline),
    ])


def _expect_reject(fn, needle=""):
    try:
        fn()
    except RuntimeError as exc:
        if needle and needle not in str(exc):
            raise AssertionError(f"error {exc!r} missing {needle!r}")
        return str(exc)
    raise AssertionError("expected RuntimeError, admission passed")


def test_accepted_report_sha_matches_pin():
    if not ACCEPTED_REPORT.exists():
        raise _Skip(f"accepted report missing: {ACCEPTED_REPORT}")
    digest = hashlib.sha256(ACCEPTED_REPORT.read_bytes()).hexdigest()
    assert digest == ACCEPTED_SHA256, "accepted report does not match pinned sha256"


def test_accepted_report_admitted():
    with _slurm_env(), tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        upstream, report, output, sha = _fixture(tmp)
        args = _parse(tmp, upstream, report, output, sha, time.time() + 600)
        digest, wr, head, remaining = MOD._admit(args)
        assert digest == sha and head == PINNED_HEAD and remaining > 0
        assert wr["totals"]["verified_count"] == 29
        assert wr["totals"]["total_count"] == 29
        assert wr["totals"]["verified_bytes"] == EXPECTED_BYTES
        assert wr["stop_reason"] == "all_verified"


def test_altered_sha_rejected():
    with _slurm_env(), tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        upstream, report, output, _ = _fixture(tmp, source="synthetic")
        args = _parse(tmp, upstream, report, output, "0" * 64, time.time() + 600)
        _expect_reject(lambda: MOD._admit(args), "sha256")


def test_incomplete_report_rejected():
    def drop_one(row):
        row["audit"].pop()

    with _slurm_env(), tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        upstream, report, output, sha = _fixture(tmp, drop_one, source="synthetic")
        args = _parse(tmp, upstream, report, output, sha, time.time() + 600)
        _expect_reject(lambda: MOD._admit(args), "audit")


def test_totals_mismatch_rejected():
    def tweak(row):
        row["totals"]["verified_count"] = 28

    with _slurm_env(), tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        upstream, report, output, sha = _fixture(tmp, tweak, source="synthetic")
        args = _parse(tmp, upstream, report, output, sha, time.time() + 600)
        _expect_reject(lambda: MOD._admit(args), "complete")


def test_missing_slurm_rejected():
    old = os.environ.pop("SLURM_JOB_ID", None)
    try:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            upstream, report, output, sha = _fixture(tmp, source="synthetic")
            args = _parse(tmp, upstream, report, output, sha, time.time() + 600)
            _expect_reject(lambda: MOD._admit(args), "SLURM_JOB_ID")
    finally:
        if old is not None:
            os.environ["SLURM_JOB_ID"] = old


def test_expired_deadline_rejected():
    with _slurm_env(), tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        upstream, report, output, sha = _fixture(tmp, source="synthetic")
        args = _parse(tmp, upstream, report, output, sha, time.time() - 5)
        _expect_reject(lambda: MOD._admit(args), "deadline")


def test_preexisting_output_rejected():
    with _slurm_env(), tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        upstream, report, output, sha = _fixture(tmp, source="synthetic")
        output.write_text("{}")
        args = _parse(tmp, upstream, report, output, sha, time.time() + 600)
        _expect_reject(lambda: MOD._admit(args), "already exists")


def test_help_and_gating_without_heavy_imports():
    help_run = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60
    )
    assert help_run.returncode == 0 and "--deadline-unix" in help_run.stdout
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        upstream, report, output, sha = _fixture(tmp, source="synthetic")
        env = {k: v for k, v in os.environ.items() if k != "SLURM_JOB_ID"}
        argv = [
            sys.executable, str(SCRIPT), "--upstream-root", str(upstream),
            "--weights-dir", str(tmp / "weights"), "--weight-report", str(report),
            "--expected-weight-report-sha256", sha, "--output-report", str(output),
            "--deadline-unix", str(time.time() + 600),
        ]
        run = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env)
        assert run.returncode == 1, run.stdout + run.stderr
        rep = json.loads(output.read_text())
        assert rep["loaded"] is False
        assert rep["error"]["type"] == "RuntimeError" and "SLURM_JOB_ID" in rep["error"]["message"]


def main():
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
        except _Skip as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - report any failure
            failed += 1
            print(f"FAIL {name}: {exc!r}")
        else:
            print(f"PASS {name}")
    print(f"{len(tests) - failed - skipped} passed, {skipped} skipped, {failed} failed of {len(tests)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
