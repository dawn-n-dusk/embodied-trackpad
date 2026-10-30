"""Dry-run CLI behavior: determinism, fixture semantics, failure checks."""

import json
from pathlib import Path

from embodied_trackpad.cli import main

PINNED_UTC = "2026-10-30T00:00:00Z"


def _tree_bytes(root: Path) -> dict:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_cli_deterministic_with_seed(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    args = ["--seed", "11", "--steps", "8", "--created-utc", PINNED_UTC]
    assert main([*args, "--out", str(out_a)]) == 0
    assert main([*args, "--out", str(out_b)]) == 0
    assert _tree_bytes(out_a) == _tree_bytes(out_b)


def test_cli_seed_changes_content(tmp_path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert main(["--seed", "1", "--steps", "8", "--created-utc", PINNED_UTC,
                 "--out", str(out_a)]) == 0
    assert main(["--seed", "2", "--steps", "8", "--created-utc", PINNED_UTC,
                 "--out", str(out_b)]) == 0
    tree_a = _tree_bytes(out_a)
    tree_b = _tree_bytes(out_b)
    assert tree_a.keys() == {k.replace("seed2", "seed1") for k in tree_b.keys()}
    actor_a = next(v for k, v in tree_a.items() if k.endswith("actor_events.jsonl"))
    actor_b = next(v for k, v in tree_b.items() if k.endswith("actor_events.jsonl"))
    assert actor_a != actor_b


def test_cli_fixture_manifest_semantics(tmp_path):
    out = tmp_path / "fixture"
    assert main(["--seed", "3", "--steps", "6", "--created-utc", PINNED_UTC,
                 "--out", str(out)]) == 0
    manifest = json.loads((out / "dryrun-seed3.fixture_manifest.json").read_text())

    assert manifest["evidence_class"] == "mock_software_fixture"
    assert manifest["training_eligible"] is False
    assert manifest["episode_status"] == "completed"
    assert manifest["schema_version"] == "trackpad.v2-draft"
    assert any("MOCK" in d for d in manifest["disclaimers"])
    assert manifest["negative_checks_ok"] is True
    assert len(manifest["negative_checks"]) == 12
    assert all(c["rejected"] and c["ok"] for c in manifest["negative_checks"])
    codes = {c["code"] for c in manifest["negative_checks"]}
    assert {
        "embodiment_mismatch", "actuator_order_mismatch", "layout_mismatch",
        "nonfinite", "out_of_range", "expired", "goal_mismatch", "clock_domain",
        "command_type", "missing_evidence", "calibration_mismatch",
    } <= codes
    # no absolute paths leak into the fixture manifest
    assert str(out) not in json.dumps(manifest)

    episode = out / manifest["episode_dir"]
    actor_lines = (episode / "actor_events.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["kind"] for line in actor_lines]
    assert "observation" in kinds and "action_chunk" in kinds
    # the all-zero action chunk from step 0 was recorded and accepted
    chunks = [
        json.loads(line)["payload"] for line in actor_lines
        if json.loads(line)["kind"] == "action_chunk"
    ]
    assert any(
        all(all(v == 0.0 for v in row) for row in c["values"]) for c in chunks
    )
    # the interrupted final attempt remains recorded without success
    outcomes = [
        json.loads(line)["payload"] for line in actor_lines
        if json.loads(line)["kind"] == "attempt_outcome"
    ]
    assert any(o["motion_status"] == "interrupted" for o in outcomes)
    assert not any(
        o["motion_status"] == "interrupted" and o["visible_outcome"] == "observed_success"
        for o in outcomes
    )


def test_cli_rejects_zero_steps(tmp_path):
    try:
        main(["--steps", "0", "--out", str(tmp_path / "x")])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("expected SystemExit for --steps 0")
