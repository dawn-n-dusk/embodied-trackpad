"""Bounded CPU probe for the LEAP v1 Full baseline (hand-only).

Loads the pinned Menagerie leap_hand right-hand model (16 independent
channels, four fingers, fixed palm), writes the compiled-model inventory, and
executes the scripted diagnostics in ``embodied_trackpad.leap_v1.diagnostics``
(16 individual channel sweeps, index/middle spread, flex/return) purely
through ctrl + mj_step. Renders real MuJoCo PNGs and writes traces/metrics to
the output directory, which must not already exist. Exit code is nonzero when
the diagnostics verdict is not PASS.

Usage:
    python tools/leap_v1_model_probe.py --out .local/v1-baseline/probe
    python tools/leap_v1_model_probe.py --out .local/v1-baseline/tiny --tiny
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_trackpad.leap_sim.pad_env import dump_json, sha256_file  # noqa: E402
from embodied_trackpad.leap_v1 import diagnostics, load_hand, model_inventory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="fresh output directory")
    parser.add_argument("--tiny", action="store_true", help="sweep 2 channels only (smoke)")
    args = parser.parse_args(argv)

    import mujoco
    import numpy as np

    started = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    hand = load_hand()
    inv = model_inventory(hand)

    out = args.out
    if out.exists():
        print(f"error: output directory already exists: {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    dump_json(out / "inventory.json", inv)

    channels = ("if_mcp", "if_pip") if args.tiny else tuple(diagnostics.MOTOR_JOINTS)
    result = diagnostics.run_motion_diagnostics(hand, out_dir=out, render=True, channels=channels)

    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(out))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    summary = {
        "task": "leap-v1-full hand-only diagnostics",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": " ".join(sys.argv),
        "versions": {
            "python": sys.version.split()[0],
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "verdict": result.metrics["verdict"],
        "failure_reasons": result.metrics["failure_reasons"],
        "steps_executed": result.metrics["steps_executed"],
        "sim_seconds": result.metrics["sim_seconds"],
        "physics_seconds_wall": result.physics_seconds,
        "gates": result.metrics["gates"],
        "reported": result.metrics["reported"],
        "warnings": result.metrics["warnings"],
        "output_files": files,
        "pad_contact": "NOT RUN (deferred to a separate bounded task)",
        "native_gui_online_feedback": "NOT RUN",
        "training": "NOT RUN",
    }
    dump_json(out / "result.json", summary)
    print(json.dumps({"verdict": summary["verdict"], "out": str(out)}, indent=1))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
