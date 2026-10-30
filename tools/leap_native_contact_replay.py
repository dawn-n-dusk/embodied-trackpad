#!/usr/bin/env python3
"""Offline replay of accepted LEAP pad contacts into the native TrackpadSandbox GUI.

This is an OFFLINE replay of a previously accepted physics run
(leap-pad-contact run-accept-001/cand-c01): the recorded touch.v1 frames are
sent over the existing SandboxClient JSONL wire with only envelope changes —
``cmd="touch"`` is added and the recorded 0-based sequence is renumbered to the
wire's 1-based sequence (coordinator-authorized transport adapter; the
source_sequence is preserved in evidence, and x/y/force/finger/time payloads
are bit-identical). No feedback was used to choose the recorded actions; no
synthetic frames, coordinate edits, force edits, or GUI state writes are
performed. The actor (normal role) run never queries privileged state; a
separate evaluator-role process replays the identical stream and its state
evidence is saved to a separate file. Not an online policy, not
training-eligible data.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mac_sandbox_client import SandboxClient, SandboxError  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WT_ROOT = os.path.dirname(REPO_ROOT)
DEFAULT_CAND_DIR = os.path.join(
    WT_ROOT, "leap-pad-contact", ".local", "leap-pad", "run-accept-001", "cand-c01")
DEFAULT_PAD_REPORT = os.path.join(WT_ROOT, "leap-pad-contact", ".local", "pad-report.json")
DEFAULT_BINARY = os.path.join(
    WT_ROOT, "mac-sandbox", ".local", "build", "TrackpadSandbox.app",
    "Contents", "MacOS", "TrackpadSandbox")
# Frozen at task start (codesign --verify --deep --strict PASS; entitlement
# com.apple.security.app-sandbox=true only). Enforced for the default binary.
DEFAULT_BINARY_SHA256 = "f64822ed6ac206bf95e4e06a039a6553ed7bfcf82306b402fc31ba41d3a4b33d"
INPUT_NAMES = ["metrics.json", "touch-v1.json", "raw-contacts.json", "trace.npz"]
SEED = 7
N_FRAMES = 105
PACE = 1.0  # wall seconds per sim second; input stays far above the 0.8 s watchdog
PRESS_MARK_S, SLIDE_MARK_S = 1.5, 2.6
GAIN_PX_PER_M = 4000.0  # macos/TrackpadSandbox PadConfig.gainPxPerM
CURSOR_TOL_PX = 0.05
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class ReplayError(Exception):
    """Protocol/run failure carrying the partial run record as evidence."""

    def __init__(self, msg, rec):
        super().__init__(msg)
        self.rec = rec


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_and_validate(cand_dir, pad_report_path):
    """Freeze input hashes and assert the accepted-run gates before any launch."""
    inputs = {n: {"path": os.path.join(cand_dir, n),
                  "sha256": sha256(os.path.join(cand_dir, n))} for n in INPUT_NAMES}
    inputs["pad-report.json"] = {"path": pad_report_path, "sha256": sha256(pad_report_path)}
    with open(os.path.join(cand_dir, "metrics.json")) as fh:
        metrics = json.load(fh)
    with open(os.path.join(cand_dir, "touch-v1.json")) as fh:
        frames = json.load(fh)
    with open(pad_report_path) as fh:
        pad = json.load(fh)
    assert metrics["verdict"] == "PASS" and metrics["episode_complete"] is True, "metrics not accepted"
    assert all(metrics["gates"].values()), "not all physics gates pass"
    ts = metrics["checks"]["touch_stream"]
    assert ts["frames"] == len(frames) == N_FRAMES and ts["monotonic"] is True, "touch stream incomplete"
    assert pad["verdict"] == "PASS" and pad["passing_candidate_dir"].endswith("cand-c01")
    seqs = [f["sequence"] for f in frames]
    assert seqs == list(range(len(frames))), "recorded sequences not contiguous from 0"
    times = [f["simulation_time_s"] for f in frames]
    assert all(b >= a for a, b in zip(times, times[1:])), "non-monotonic frame times"
    forces = []
    for f in frames:
        assert f["schema_version"] == "touch.v1"
        assert isinstance(f["contacts"], list) and len(f["contacts"]) <= 10
        for c in f["contacts"]:
            assert c["finger_id"] and isinstance(c["finger_id"], str)
            assert abs(c["x_m"]) <= 0.08 + 1e-12 and abs(c["y_m"]) <= 0.055 + 1e-12
            frc = c["normal_force_n"]
            assert frc == frc and 0.0 <= frc != float("inf")
            forces.append(frc)
    # Sampled-stream checks only: the 105-frame peak is NOT the 7600-step peak.
    assert max(forces) >= 1.0, "sampled stream never crosses the 1 N press threshold"
    assert min(forces) <= 0.6, "sampled stream never reaches the 0.6 N release threshold"
    return inputs, metrics, frames


def expected_cursor_delta(frames):
    """Contact-derived cursor motion: the device model (DeviceModel.swift:200-201)
    adds (dx_m, dy_m) * gain to cursor px only across consecutive frames where
    the same finger stays in contact; +y_m increases cursorYPx (+y up view)."""
    dx = dy = 0.0
    prev = None
    for f in frames:
        cur = {c["finger_id"]: c for c in f["contacts"]}
        if prev is not None:
            for fid, c in cur.items():
                if fid in prev:
                    dx += (c["x_m"] - prev[fid]["x_m"]) * GAIN_PX_PER_M
                    dy += (c["y_m"] - prev[fid]["y_m"]) * GAIN_PX_PER_M
        prev = cur
    return {"dx_px": dx, "dy_px": dy}


def codesign_receipt(binary):
    r = subprocess.run(["codesign", "--verify", "--deep", "--strict", binary],
                       capture_output=True, text=True)
    ent = subprocess.run(["codesign", "-d", "--entitlements", "-", binary],
                         capture_output=True, text=True)
    return {"verify_rc": r.returncode, "verify_ok": r.returncode == 0,
            "entitlements": (ent.stdout + ent.stderr).strip()}


def replay(evaluator, out_dir, frames, binary, want_captures):
    """One app run replaying all frozen frames. Any error ack, wrong ack
    sequence, or privileged ack field fails the run with evidence attached."""
    rec = {"role": "evaluator" if evaluator else "normal", "seed": SEED,
           "started_utc": utc_now(), "acks_accepted": 0, "acks": [],
           "captures": {}, "states": {}}
    start_wall = time.monotonic()
    marks = {"press": PRESS_MARK_S, "slide": SLIDE_MARK_S}
    done = set()
    try:
        with SandboxClient(binary=binary, out_dir=out_dir, evaluator=evaluator) as c:
            rec["process"] = {"pid": c.proc.pid, "content_size": c.content_size}
            rec["reset"] = c.reset(seed=SEED)
            if evaluator:
                rec["states"]["initial"] = c.state()
            if want_captures:
                rec["captures"]["before"] = _cap(c, "normal-before.png", start_wall, 0.0)
            for i, fr in enumerate(frames):
                target = start_wall + fr["simulation_time_s"] * PACE
                if time.monotonic() < target:
                    time.sleep(target - time.monotonic())
                # Envelope adapter: add cmd and renumber sequence 0-based -> 1-based.
                wire_seq = fr["sequence"] + 1
                msg = dict(fr)
                msg["cmd"] = "touch"
                msg["sequence"] = wire_seq
                ack = c.raw(msg)
                rec["acks"].append({"i": i, "source_sequence": fr["sequence"],
                                    "wire_sequence": wire_seq,
                                    "simulation_time_s": fr["simulation_time_s"],
                                    "wall_offset_s": time.monotonic() - start_wall,
                                    "ack": ack})
                if ack.get("type") != "touch_ok":
                    raise ReplayError(f"frame {i} rejected: {ack}", rec)
                if ack.get("sequence") != wire_seq or set(ack) - {"type", "sequence"}:
                    raise ReplayError(f"frame {i} ack contract violated: {ack}", rec)
                rec["acks_accepted"] += 1
                sim = fr["simulation_time_s"]
                for name, mark in marks.items():
                    if name not in done and sim >= mark:
                        done.add(name)
                        if want_captures:
                            rec["captures"][name] = _cap(c, f"normal-{name}.png", start_wall, sim)
                        if evaluator:
                            rec["states"][name] = c.state()
            if want_captures:
                rec["captures"]["release"] = _cap(c, "normal-release.png", start_wall,
                                                  frames[-1]["simulation_time_s"])
            if evaluator:
                rec["states"]["final"] = c.state()
            rec["stop"] = c.stop()
            rec["quit_rc"] = c.quit()
            rec["proc_poll_after_quit"] = c.proc.poll()
    except ReplayError:
        raise
    except Exception as e:
        raise ReplayError(f"{type(e).__name__}: {e}", rec) from e
    rec["finished_utc"] = utc_now()
    rec["wall_seconds"] = time.monotonic() - start_wall
    return rec


def _cap(c, name, start_wall, sim):
    r = c.capture(save_as=name)
    return {"path": r["path"], "sha256": hashlib.sha256(r["png"]).hexdigest(),
            "frame_id": r["frame_id"], "width": r["width"], "height": r["height"],
            "originating_sim_time_s": sim, "obs_simulation_time_s": r["simulation_time_s"],
            "app_monotonic_time_s": r["monotonic_time_s"],
            "capture_wall_offset_s": time.monotonic() - start_wall, "capture_utc": utc_now()}


def fresh_path(base):
    if not os.path.exists(base):
        return base
    stem, ext = os.path.splitext(base)
    for k in range(2, 100):
        p = f"{stem}-{k}{ext}"
        if not os.path.exists(p):
            return p
    raise SystemExit(f"no fresh path available near {base}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand-dir", default=DEFAULT_CAND_DIR)
    ap.add_argument("--pad-report", default=DEFAULT_PAD_REPORT)
    ap.add_argument("--binary", default=DEFAULT_BINARY)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, ".local", "native-contact"))
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"refusing to reuse existing output dir: {args.out}")
    os.makedirs(args.out)
    report_path = fresh_path(os.path.join(REPO_ROOT, ".local", "native-contact-report.json"))
    report = {"probe": "leap-native-contact-replay", "tool": os.path.abspath(__file__),
              "tool_sha256": sha256(os.path.abspath(__file__)),
              "python": sys.version, "started_utc": utc_now(), "seed": SEED,
              "command": " ".join([sys.executable, os.path.abspath(__file__),
                                   "--cand-dir", args.cand_dir, "--pad-report", args.pad_report,
                                   "--binary", args.binary, "--out", args.out])}
    normal_path = os.path.join(args.out, "normal-run.json")
    evaluator_path = os.path.join(args.out, "evaluator.json")
    try:
        inputs, metrics, frames = load_and_validate(args.cand_dir, args.pad_report)
        report["inputs"] = inputs
        report["source_hashes"] = {
            "mac_sandbox_client_py_sha256": sha256(
                os.path.join(REPO_ROOT, "tools", "mac_sandbox_client.py")),
            "sandbox_source": {n: sha256(os.path.join(
                REPO_ROOT, "macos", "TrackpadSandbox", "Sources", n))
                for n in ("main.swift", "DeviceModel.swift")}}
        report["binary"] = {"path": args.binary, "sha256": sha256(args.binary),
                            "codesign": codesign_receipt(args.binary)}
        assert report["binary"]["codesign"]["verify_ok"], "codesign verify failed"
        if os.path.abspath(args.binary) == os.path.abspath(DEFAULT_BINARY):
            assert report["binary"]["sha256"] == DEFAULT_BINARY_SHA256, "binary hash changed since freeze"
        report["derived"] = {
            "expected_cursor_delta": expected_cursor_delta(frames),
            "sampled_max_force_n": max(c["normal_force_n"] for f in frames for c in f["contacts"]),
            "note": "sampled 105-frame max is not the 7600-step all-step peak "
                    f"{metrics['checks']['max_pad_normal_force_all_n']}"}
        try:
            normal = replay(False, args.out, frames, args.binary, want_captures=True)
        except ReplayError as e:
            normal = e.rec
            normal["error"] = str(e)
        with open(normal_path, "w") as fh:
            json.dump(normal, fh, indent=2)
        if "error" not in normal:
            try:
                evaluator = replay(True, args.out, frames, args.binary, want_captures=False)
            except ReplayError as e:
                evaluator = e.rec
                evaluator["error"] = str(e)
        else:
            evaluator = {"role": "evaluator", "skipped": "normal run failed"}
        with open(evaluator_path, "w") as fh:
            json.dump(evaluator, fh, indent=2)

        gates = []

        def gate(name, ok, observed):
            gates.append({"name": name, "ok": bool(ok), "observed": observed})

        gate("normal_acks_105_accepted", normal.get("acks_accepted") == N_FRAMES,
             normal.get("acks_accepted"))
        gate("evaluator_acks_105_accepted", evaluator.get("acks_accepted") == N_FRAMES,
             evaluator.get("acks_accepted"))
        for name in ("before", "press", "slide", "release"):
            cap = normal.get("captures", {}).get(name, {})
            p = cap.get("path", "")
            ok = (os.path.exists(p) and os.path.getsize(p) > 1000
                  and open(p, "rb").read(8) == PNG_MAGIC)
            gate(f"png_{name}_captured", ok,
                 {"path": p, "size": os.path.getsize(p) if os.path.exists(p) else None})
        states = evaluator.get("states", {})
        fin = states.get("final", {})
        gate("evaluator_press_held_at_mark", states.get("press", {}).get("button_down") is True,
             states.get("press", {}).get("button_down"))
        gate("evaluator_final_button_up", fin.get("button_down") is False, fin.get("button_down"))
        gate("evaluator_release_count>=1", fin.get("release_count", 0) >= 1,
             fin.get("release_count"))
        init = states.get("initial", {})
        exp = report["derived"]["expected_cursor_delta"]
        got = {"dx_px": fin.get("cursor", {}).get("x", float("nan")) - init.get("cursor", {}).get("x", float("nan")),
               "dy_px": fin.get("cursor", {}).get("y", float("nan")) - init.get("cursor", {}).get("y", float("nan"))}
        err = [abs(got["dx_px"] - exp["dx_px"]), abs(got["dy_px"] - exp["dy_px"])]
        gate("cursor_delta_within_0.05px", err[0] <= CURSOR_TOL_PX and err[1] <= CURSOR_TOL_PX,
             {"expected": exp, "evaluator": got, "abs_error_px": err})
        report["gates"] = gates
        report["outcomes"] = {
            "release_count": fin.get("release_count"), "target_hits": fin.get("target_hits"),
            "drag_done": fin.get("drag_done"), "button_down_final": fin.get("button_down"),
            "press_held_at_mark": states.get("press", {}).get("button_down"),
            "note": "generic force-hysteresis release counted; target_hits and drag_done "
                    "are reported separately and no task success is claimed."}
        report["artifacts"] = {
            "normal_evidence": normal_path, "evaluator_evidence": evaluator_path,
            "captures": {k: v.get("path") for k, v in normal.get("captures", {}).items()},
            "normal_quit_rc": normal.get("quit_rc"),
            "normal_proc_poll_after_quit": normal.get("proc_poll_after_quit"),
            "evaluator_quit_rc": evaluator.get("quit_rc"),
            "evaluator_proc_poll_after_quit": evaluator.get("proc_poll_after_quit"),
            "normal_wall_seconds": normal.get("wall_seconds"),
            "evaluator_wall_seconds": evaluator.get("wall_seconds")}
        failures = [g for g in gates if not g["ok"]]
        report["status"] = "failed" if failures else "ok"
        if failures:
            report["gate_failures"] = failures
    except Exception as e:
        report["status"] = "failed"
        report["error"] = f"{type(e).__name__}: {e}"
    report["limitations"] = [
        "Offline replay of recorded physics contacts; no feedback/policy chose actions; not training-eligible.",
        "Recorded sequences 0..104 are renumbered to wire sequences 1..105 (coordinator-authorized transport "
        "envelope adapter); source_sequence is preserved per frame in evidence; x/y/force/finger/time payloads "
        "are bit-identical to the frozen touch-v1.json.",
        "PNGs are saved for independent coordinator inspection; no pixel analysis performed in this call.",
        "Sampled 105-frame force max (1.0538767 N) is not the 7600-step all-step peak (1.0543157 N); no equality asserted.",
        "Wall pacing is approximately 1.0x sim time with capture-induced drift; wall and sim clocks recorded separately.",
        "Evaluator state is saved separately (evaluator.json); it never enters the actor channel.",
        "Protocol role boundary is not an OS security claim; app is ad-hoc signed with app-sandbox entitlement only.",
    ]
    report["finished_utc"] = utc_now()
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({"status": report["status"], "report": report_path,
                      "error": report.get("error"),
                      "gate_failures": [g["name"] for g in report.get("gate_failures", [])],
                      "outcomes": report.get("outcomes")}, indent=2))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
