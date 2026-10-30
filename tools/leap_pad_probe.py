"""Bounded LEAP v2 Advanced static-pad contact probe (physics only).

Runs at most ``--max-candidates`` clear -> press -> hold -> slide -> lift
episodes against a STATIC 160 x 110 mm pad with the accepted 17-motor hand,
stopping at the first fully passing candidate or when the wall-clock physics
budget (default 300 s of mj_step time) is consumed. Every candidate gets a
fresh output directory; nothing is overwritten and failed candidates keep
their metrics, raw contacts, trace and renders for review.

All motion goes through data.ctrl + mj_step (reset is the only qpos write).
Contact truth is the actual geom-pair contact array plus mj_contactForce
(compressive-positive normal force, never abs). touch.v1-compatible messages
are recorded for every contact step but the native app is NOT launched.

Usage:
    python tools/leap_pad_probe.py [--out-root .local/leap-pad/run-001]
                                   [--max-candidates 16] [--physics-budget-s 300]
                                   [--tiny] [--only c01]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_trackpad.leap_sim import DERIVED_URDF, MANIFEST_JSON  # noqa: E402
from embodied_trackpad.leap_sim.pad_env import (  # noqa: E402
    PadEpisodeConfig,
    load_pad_scene,
    run_pad_episode,
    scene_integrity,
    sha256_file,
)

MAX_CANDIDATES = 16
PHYSICS_BUDGET_S = 300.0


def candidate_grid() -> list[PadEpisodeConfig]:
    """Ordered candidates; execution stops at the first PASS.

    Tuning rationale: press depth (mcpf/dip over-target) sets the normal
    force; pad top height sets clearance and contact onset; slide_mcps sign
    moves the index tip AWAY from the middle finger (positive mcps approaches
    it, see kinematic survey in .local/explore2.py).
    """
    grid = [
        PadEpisodeConfig(candidate_id="c01", press_mcpf=0.12, press_dip=0.45, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c02", press_mcpf=0.16, press_dip=0.45, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c03", press_mcpf=0.20, press_dip=0.45, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c04", press_mcpf=0.12, press_dip=0.30, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c05", press_mcpf=0.16, press_dip=0.30, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c06", press_mcpf=0.12, press_dip=0.60, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c07", press_mcpf=0.20, press_dip=0.30, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c08", press_mcpf=0.12, press_dip=0.45, slide_mcps=-0.16),
        PadEpisodeConfig(candidate_id="c09", press_mcpf=0.16, press_dip=0.45, slide_mcps=-0.28),
        PadEpisodeConfig(candidate_id="c10", press_mcpf=0.22, press_dip=0.45, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c11", press_mcpf=0.25, press_dip=0.45, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c12", press_mcpf=0.16, press_dip=0.60, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c13", press_mcpf=0.20, press_dip=0.60, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c14", press_mcpf=0.10, press_dip=0.45, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c15", press_mcpf=0.14, press_dip=0.55, slide_mcps=-0.22),
        PadEpisodeConfig(candidate_id="c16", press_mcpf=0.24, press_dip=0.35, slide_mcps=-0.24),
    ]
    return grid[:MAX_CANDIDATES]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_progress(path: Path, **fields) -> None:
    payload = {"task": "leap-pad-contact", "updated_utc": _utcnow(), **fields}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    parser.add_argument("--out-root", type=Path,
                        default=REPO_ROOT / ".local" / "leap-pad" / f"run-{stamp}")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--physics-budget-s", type=float, default=PHYSICS_BUDGET_S)
    parser.add_argument("--only", type=str, default=None,
                        help="run only this candidate id (still budgeted)")
    parser.add_argument("--tiny", action="store_true",
                        help="short 0.8 s clearance-only smoke run; NOT an acceptance candidate")
    parser.add_argument("--progress", type=Path, default=REPO_ROOT / ".local" / "progress.json")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / ".local" / "pad-report.json")
    args = parser.parse_args()

    args.out_root = args.out_root.resolve()
    if args.out_root.exists():
        print(f"error: output root already exists: {args.out_root}", file=sys.stderr)
        return 2
    args.out_root.mkdir(parents=True)

    started = _utcnow()
    _write_progress(
        args.progress,
        stage="probe_started",
        out_root=str(args.out_root),
        candidates_attempted=0,
        physics_seconds_used=0.0,
        notes=["scene loading"],
    )

    scenes: dict[tuple, object] = {}

    def scene_for(cfg: PadEpisodeConfig):
        key = tuple(cfg.pad_center)
        if key not in scenes:
            scenes[key] = load_pad_scene(pad_center=cfg.pad_center)
        return scenes[key]

    first_scene = scene_for(candidate_grid()[0])
    integrity = scene_integrity(first_scene)
    (args.out_root / "scene-integrity.json").write_text(json.dumps(integrity, indent=2) + "\n")

    candidates = candidate_grid()
    if args.only:
        candidates = [c for c in candidates if c.candidate_id == args.only]
        if not candidates:
            print(f"error: unknown candidate id {args.only}", file=sys.stderr)
            return 2
    candidates = candidates[: args.max_candidates]

    if args.tiny:
        cfg = PadEpisodeConfig(
            candidate_id="tiny-clearance-smoke",
            clear1_end=0.4, press_ramp_end=0.45, press_hold_end=0.5,
            slide_end=0.55, lift_ramp_end=0.6, t_end=0.8,
            press_mcpf=0.05, press_dip=0.05, slide_mcps=0.0,
            smoke=True,
        )
        out = args.out_root / "tiny"
        res = run_pad_episode(scene_for(cfg), cfg, out_dir=out, render=True)
        summary = {
            "tiny": True,
            "verdict": res.metrics["verdict"],
            "checks": res.metrics["checks"],
            "physics_seconds_wall": res.physics_seconds,
        }
        print(json.dumps(summary, indent=2))
        return 0 if res.metrics["gates"]["finite"] else 1

    physics_used = 0.0
    wall0 = time.perf_counter()
    attempted: list[dict] = []
    passing_dir = None
    verdict = "BUDGET_EXHAUSTED"
    for cfg in candidates:
        if physics_used >= args.physics_budget_s:
            break
        out = args.out_root / f"cand-{cfg.candidate_id}"
        res = run_pad_episode(scene_for(cfg), cfg, out_dir=out, render=True)
        physics_used += res.physics_seconds
        m = res.metrics
        attempted.append(
            {
                "candidate_id": cfg.candidate_id,
                "dir": str(out.relative_to(REPO_ROOT)),
                "verdict": m["verdict"],
                "failure_reasons": m["failure_reasons"],
                "physics_seconds_wall": res.physics_seconds,
                "press_peak_n": m["contact"]["press_peak_n"],
                "max_sustained_s": m["contact"]["max_sustained_s_ge_0.05N"],
                "slide_net_displacement_m": m["contact"]["slide_net_displacement_m"],
                "max_penetration_m": m["checks"]["max_penetration_m"],
                "max_pad_normal_force_all_n": m["checks"]["max_pad_normal_force_all_n"],
                "max_pad_normal_force_tip_eligible_n": m["checks"]["max_pad_normal_force_tip_eligible_n"],
                "unintended_contact_steps": m["checks"]["unintended_contact_steps"],
                "unintended_contact_events": m["checks"]["unintended_contact_events"],
                "device_press_thresholds": m["device_press_thresholds"],
            }
        )
        _write_progress(
            args.progress,
            stage="candidate_done",
            out_root=str(args.out_root),
            candidates_attempted=len(attempted),
            physics_seconds_used=physics_used,
            last_candidate=cfg.candidate_id,
            last_verdict=m["verdict"],
            last_failures=m["failure_reasons"],
        )
        print(f"[{cfg.candidate_id}] {m['verdict']} "
              f"peak={m['contact']['press_peak_n']:.3f}N "
              f"sustain={m['contact']['max_sustained_s_ge_0.05N']:.3f}s "
              f"slide={m['contact']['slide_net_displacement_m']*1000:.2f}mm "
              f"pen={m['checks']['max_penetration_m']*1000:.3f}mm "
              f"fails={m['failure_reasons']}", flush=True)
        if m["verdict"] == "PASS":
            passing_dir = str(out.relative_to(REPO_ROOT))
            verdict = "PASS"
            break
    else:
        if attempted and physics_used < args.physics_budget_s and len(attempted) >= len(candidates):
            verdict = "ALL_CANDIDATES_FAILED"
    if attempted and verdict not in ("PASS", "ALL_CANDIDATES_FAILED") and len(attempted) >= len(candidates):
        verdict = "ALL_CANDIDATES_FAILED"

    report = {
        "probe": "leap-v2-adv-static-pad-contact-cpu",
        "verdict": verdict,
        "passing_candidate_dir": passing_dir,
        "started_utc": started,
        "finished_utc": _utcnow(),
        "command": " ".join([sys.executable, *sys.argv]),
        "runtime": {
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "budget": {
            "max_candidates": args.max_candidates,
            "physics_budget_s": args.physics_budget_s,
            "physics_seconds_used": physics_used,
            "wall_seconds_total": time.perf_counter() - wall0,
        },
        "hashes": {
            "model_py_sha256": sha256_file(REPO_ROOT / "src/embodied_trackpad/leap_sim/model.py"),
            "pad_env_py_sha256": sha256_file(REPO_ROOT / "src/embodied_trackpad/leap_sim/pad_env.py"),
            "probe_py_sha256": sha256_file(Path(__file__).resolve()),
            "manifest_sha256": sha256_file(MANIFEST_JSON),
            "upstream_urdf_sha256": sha256_file(
                REPO_ROOT / "assets/downloads/leap_hand_v2_adv/leap_v2_right/robot.urdf"
            ),
            "derived_urdf_sha256": sha256_file(DERIVED_URDF),
        },
        "scene_integrity": integrity,
        "candidates": attempted,
        "coverage": (
            "Each candidate: clear->press->hold->slide->lift with all 17 ctrl channels "
            "commanded every step; every-step gate maxima recorded (finite/limit/coupling/"
            "penetration/pad force/unintended contacts/warnings); full state trace sampled "
            "every 10 steps plus endpoints; raw contact log per contact step; touch.v1 "
            "messages per contact step; renders at initial/press/slide/lift from executed state."
        ),
        "limitations": [
            "Physics only: the native GUI is NOT launched; touch.v1 messages are recorded, not delivered.",
            "Actuator gains/damping are simulation placeholders; normal-force values are not hardware-calibrated.",
            "Rigid meshes, eight assembly-pair exclusions and 1:1 PIP/DIP couplings remain approximations (see docs/leap-model.md).",
            "Single fixed hand/pad transform per candidate; pad is static by construction (worldbody geom).",
            "Hand-only replay traces are not accepted training data.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    _write_progress(
        args.progress,
        stage="probe_finished",
        out_root=str(args.out_root),
        candidates_attempted=len(attempted),
        physics_seconds_used=physics_used,
        verdict=verdict,
        passing_candidate_dir=passing_dir,
    )
    print(json.dumps({"verdict": verdict, "passing": passing_dir,
                      "candidates": len(attempted),
                      "physics_seconds_used": round(physics_used, 2)}, indent=2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
