#!/usr/bin/env python3
"""Stdlib client for the TrackpadSandbox native Mac test GUI.

Manages the app subprocess lifecycle over inherited stdin/stdout JSONL only.
No TCP, no AppleScript, no AX automation, no host pointer/keyboard injection.
All reads are bounded (select-based deadlines) and teardown always reaps the
child, so a hung app can never block the caller or leave a zombie.

Boundary contract:
  - Actor input wire: {"cmd":"touch","schema_version":"touch.v1","sequence":int,
    "simulation_time_s":float,"contacts":[{"finger_id":str,"x_m":float,
    "y_m":float,"normal_force_n":float}]}. x/y are meters about pad center,
    +x right, +y up; an omitted finger means lifted. This wire matches the
    physics worker's contact format so a real-physics caller can later supply
    contacts without scripted click coordinates.
  - touch acknowledgements carry acceptance/sequence only - never cursor,
    layout, or task-success truth.
  - reset(seed, task_profile=None) optionally selects one NAMED fixed
    benchmark profile ("near_pointing_v1": fixed 32x32 target at one of 12
    published offsets from the reset cursor). Names only, never coordinates;
    the ack echoes the profile name but no layout/cursor truth.
  - Observation: {"type":"observation","frame_id":int,"width":int,"height":int,
    "monotonic_time_s":float,"simulation_time_s":float,"png_base64":str} -
    pixels of the app's own content view only, a declared monotonic capture
    clock, and the separate last simulation time. No desktop capture exists.
  - Evaluator state (cursor, release/target-hit counts, drag object placement,
    layout) is a separate PRIVILEGED "state" query, never part of actor pixels.
  - Diagnostics (sandbox probes) run only in an explicit --diagnostics launch
    mode with a caller-designated fresh canary root; the normal client exposes
    only the contact/capture lifecycle plus the privileged queries.
  - Synthetic smoke sequences generated here are ineligible for training.
"""
import argparse
import base64
import json
import math
import os
import select
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_BINARY = os.path.join(
    REPO_ROOT, ".local", "build", "TrackpadSandbox.app", "Contents", "MacOS", "TrackpadSandbox")
DEFAULT_OUT = os.path.join(REPO_ROOT, ".local", "output")
CANARY_NAME = "sandbox-canary.txt"

PAD_HALF_W = 0.08      # 0.16 m pad, pad-centered coordinates
PAD_HALF_H = 0.055     # 0.11 m pad
GAIN = 4000.0          # px per meter

# Named fixed benchmark reset profile (see macos/TrackpadSandbox/Sources/
# main.swift): the click target becomes a 32x32 blue square centred at a
# FIXED offset from the reset cursor (400,300), selected by seed % 12 in
# exactly this published order (logical px, +x right / +y up). The target
# never follows the cursor and distances never shrink during an episode.
NEAR_PROFILE_ID = "near_pointing_v1"
DIAGONAL_PROFILE_ID = "near_diagonal_v1"
NEAR_OFFSETS_PX = (
    (12, 0), (18, 0), (24, 0),
    (-12, 0), (-18, 0), (-24, 0),
    (0, 12), (0, 18), (0, 24),
    (0, -12), (0, -18), (0, -24),
)
DIAGONAL_OFFSETS_PX = (
    (12, 12), (-12, 12), (-12, -12), (12, -12),
    (18, 18), (-18, 18), (-18, -18), (18, -18),
)
NEAR_TARGET_SIZE = 32
RESET_CURSOR_PX = (400.0, 300.0)


def near_target_rect(seed):
    """Expected privileged click_target rect (x, y, w, h) for a near seed."""
    dx, dy = NEAR_OFFSETS_PX[seed % 12]
    return {
        "x": RESET_CURSOR_PX[0] + dx - NEAR_TARGET_SIZE / 2,
        "y": RESET_CURSOR_PX[1] + dy - NEAR_TARGET_SIZE / 2,
        "w": float(NEAR_TARGET_SIZE), "h": float(NEAR_TARGET_SIZE),
    }


def diagonal_target_rect(seed):
    """Expected privileged click_target rect for a diagonal seed."""
    dx, dy = DIAGONAL_OFFSETS_PX[seed % 8]
    return {
        "x": RESET_CURSOR_PX[0] + dx - NEAR_TARGET_SIZE / 2,
        "y": RESET_CURSOR_PX[1] + dy - NEAR_TARGET_SIZE / 2,
        "w": float(NEAR_TARGET_SIZE), "h": float(NEAR_TARGET_SIZE),
    }


class SandboxError(RuntimeError):
    pass


class SandboxClient:
    def __init__(self, binary=APP_BINARY, out_dir=DEFAULT_OUT,
                 diagnostics=False, canary_root=None, evaluator=False,
                 startup_timeout=15.0):
        if not os.path.exists(binary):
            raise SandboxError(f"app not built: {binary}; run macos/TrackpadSandbox/build.sh")
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        args = [binary]
        if diagnostics:
            if not canary_root:
                raise SandboxError("diagnostics mode requires canary_root")
            args += ["--diagnostics", "--canary-root", canary_root]
        if evaluator:
            args += ["--evaluator"]
        self.diagnostics = diagnostics
        self.evaluator = evaluator
        self.proc = subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0)
        self._buf = b""
        self.seq = 0
        self.sim_time = 0.0
        try:
            hello = self._read(timeout=startup_timeout)
        except Exception:
            self._teardown()
            raise
        if hello.get("type") != "hello":
            self._teardown()
            raise SandboxError(f"expected hello, got {hello}")
        self.content_size = (hello["content_size"]["w"], hello["content_size"]["h"])

    # ---- bounded IO ---------------------------------------------------------
    def _read(self, timeout=30.0):
        """Read exactly one JSONL response with a hard wall-clock deadline."""
        deadline = time.monotonic() + timeout
        while True:
            i = self._buf.find(b"\n")
            if i >= 0:
                line = self._buf[:i]
                self._buf = self._buf[i + 1:]
                return json.loads(line.decode("utf-8"))
            if self.proc.poll() is not None and not self._buf:
                raise SandboxError(f"app exited (rc={self.proc.returncode})")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SandboxError("timeout waiting for app response")
            r, _, _ = select.select([self.proc.stdout], [], [], min(remaining, 1.0))
            if not r:
                continue
            chunk = os.read(self.proc.stdout.fileno(), 1 << 20)
            if not chunk:
                if self._buf.strip():
                    raise SandboxError("partial line at EOF: app output truncated")
                raise SandboxError("app closed output")
            self._buf += chunk

    def _send(self, obj, timeout=30.0):
        self.proc.stdin.write(json.dumps(obj).encode() + b"\n")
        self.proc.stdin.flush()
        return self._read(timeout=timeout)

    def _teardown(self):
        """Close stdin, then terminate/kill until the child is reaped."""
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    # ---- normal lifecycle (contact/capture only) -----------------------------
    def raw(self, obj, timeout=30.0):
        """Send a verbatim object (used to exercise malformed/replayed input)."""
        return self._send(obj, timeout=timeout)

    def raw_line(self, text, timeout=30.0):
        self.proc.stdin.write(text.encode() + b"\n")
        self.proc.stdin.flush()
        return self._read(timeout=timeout)

    def reset(self, seed=1, task_profile=None):
        """Reset the episode; optionally select a named fixed benchmark profile.

        task_profile is a profile NAME only (currently "near_pointing_v1" or
        "near_diagonal_v1"), never caller-provided coordinates. Unknown
        profiles are rejected by the app atomically before any state is reset.
        The acknowledgement echoes the profile name but carries no
        layout/cursor truth.
        """
        msg = {"cmd": "reset", "seed": seed}
        if task_profile is not None:
            msg["task_profile"] = task_profile
        r = self._send(msg)
        assert r["type"] == "reset_ok", r
        return r

    def touch(self, contacts, dt=0.016):
        """Advance simulation time and submit one touch.v1 frame.

        contacts: list of (finger_id, x_m, y_m, normal_force_n). Mirrors the
        shared robot-input wire; callers supply contacts, not coordinates.
        Returns the bare acknowledgement (acceptance/sequence only).
        """
        self.seq += 1
        self.sim_time += dt
        return self._send({
            "cmd": "touch", "schema_version": "touch.v1", "sequence": self.seq,
            "simulation_time_s": round(self.sim_time, 6),
            "contacts": [
                {"finger_id": f, "x_m": x, "y_m": y, "normal_force_n": n}
                for (f, x, y, n) in contacts]})

    def capture(self, save_as=None):
        r = self._send({"cmd": "capture"})
        assert r["type"] == "observation", r
        png = base64.b64decode(r["png_base64"])
        path = None
        if save_as:
            path = os.path.join(self.out_dir, save_as)
            with open(path, "wb") as fh:
                fh.write(png)
        return {"frame_id": r["frame_id"], "width": r["width"], "height": r["height"],
                "monotonic_time_s": r["monotonic_time_s"],
                "simulation_time_s": r["simulation_time_s"], "png": png, "path": path}

    def stop(self):
        r = self._send({"cmd": "stop"})
        assert r["type"] == "stopped", r
        return r

    # ---- privileged evaluator query (never actor-visible) --------------------
    def state(self):
        """PRIVILEGED evaluator query; requires evaluator=True at launch."""
        r = self._send({"cmd": "state"})
        assert r["type"] == "state" and r.get("privileged") is True, r
        return r

    # ---- diagnostics-only probes ---------------------------------------------
    def probe_file(self):
        if not self.diagnostics:
            raise SandboxError("probe_file is diagnostics-mode only")
        return self._send({"cmd": "probe_file"})

    def probe_net(self, port=9):
        if not self.diagnostics:
            raise SandboxError("probe_net is diagnostics-mode only")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise SandboxError("port must be an int in 1...65535")
        return self._send({"cmd": "probe_net", "port": port})

    def quit(self, timeout=10.0):
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._teardown()
            raise SandboxError("app did not exit on EOF")
        return self.proc.returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._teardown()
        return False


# ---- bounded synthetic smoke (INELIGIBLE FOR TRAINING) ------------------------
def _slide_to(c, tx, ty, force=0.2, fid="a"):
    """Move the cursor to (tx, ty) px using lift/reposition hops that stay
    inside the physical pad bounds (evaluator-guided synthetic input)."""
    for _ in range(40):
        cur = c.state()["cursor"]
        dx, dy = tx - cur["x"], ty - cur["y"]
        if abs(dx) < 1 and abs(dy) < 1:
            return
        dist = max(1e-9, math.hypot(dx, dy))
        step = min(1.0, 120.0 / dist)  # <=120 px per hop = 30 mm
        mx, my = dx * step / GAIN, dy * step / GAIN
        n = 10
        for i in range(1, n + 1):
            c.touch([(fid, mx * i / n, my * i / n, force)])
        c.touch([])
    raise SandboxError("slide_to did not converge")


def _drag_by(c, dx, dy, force=1.5, fid="a"):
    """Continuous held drag of (dx, dy) px without lifting the finger.

    The finger starts at the press origin (0,0); the caller's press frame must
    already be at (0,0) so no frame exceeds the per-sample delta bound.
    """
    n = max(4, int(math.hypot(dx, dy) / 12))
    for i in range(1, n + 1):
        c.touch([(fid, (dx / GAIN) * i / n, (dy / GAIN) * i / n, force)])


def run_smoke(out_dir=DEFAULT_OUT):
    report = {"checks": [], "samples": [], "synthetic": True,
              "training_eligible": False}

    def check(name, ok, detail=""):
        report["checks"].append({"name": name, "ok": bool(ok), "detail": detail})

    with SandboxClient(out_dir=out_dir, evaluator=True) as c:
        c.reset(seed=7)
        base = c.capture("smoke-0-reset.png")
        report["samples"].append(base["path"])

        # Acknowledgement carries no cursor truth.
        ack = c.touch([("a", 0.0, 0.0, 0.2)])
        check("touch_ok_no_truth",
              ack.get("type") == "touch_ok" and "cursor" not in ack
              and "button_down" not in ack, json.dumps(ack))

        # First contact must not teleport.
        st0 = c.state()
        c.touch([])  # lift before repositioning (a jump while down is rejected)
        c.touch([("a", 0.07, 0.05, 0.2)])
        check("first_touch_no_jump", c.state()["cursor"] == st0["cursor"])

        # Relative move + pixel change (continuous contact from the origin).
        for i in range(1, 9):
            c.touch([("a", 0.07 + i * 0.001, 0.05 + i * 0.0005, 0.2)])
        c.touch([])
        st1 = c.state()
        dx, dy = st1["cursor"]["x"] - st0["cursor"]["x"], st1["cursor"]["y"] - st0["cursor"]["y"]
        check("relative_move", abs(dx - 32) < 1e-6 and abs(dy - 16) < 1e-6, f"d=({dx},{dy})")
        moved = c.capture("smoke-1-moved.png")
        report["samples"].append(moved["path"])
        check("pixels_change_on_move", moved["png"] != base["png"])
        check("capture_clock_monotonic",
              moved["monotonic_time_s"] > base["monotonic_time_s"]
              and moved["simulation_time_s"] >= base["simulation_time_s"])

        # Click ON the target: release counted AND target hit.
        tgt = c.state()["click_target"]
        _slide_to(c, tgt["x"] + tgt["w"] / 2, tgt["y"] + tgt["h"] / 2)
        c.touch([("a", 0.0, 0.0, 1.5)])
        c.touch([("a", 0.0, 0.0, 0.1)])
        c.touch([])
        st2 = c.state()
        check("click_target_hit", st2["target_hits"] == 1 and st2["release_count"] == 1,
              f"hits={st2['target_hits']} releases={st2['release_count']}")
        clicked = c.capture("smoke-2-clicked.png")
        report["samples"].append(clicked["path"])
        check("pixels_change_on_click", clicked["png"] != moved["png"])

        # Negative: click OUTSIDE the target counts a release but no hit.
        _slide_to(c, 400, 560 if tgt["y"] + tgt["h"] / 2 < 300 else 40)
        c.touch([("a", 0.0, 0.0, 1.5)])
        c.touch([("a", 0.0, 0.0, 0.1)])
        c.touch([])
        st3 = c.state()
        check("outside_click_no_hit", st3["target_hits"] == 1 and st3["release_count"] == 2,
              f"hits={st3['target_hits']} releases={st3['release_count']}")

        # Negative: press OUTSIDE the object, slide through it, release at the
        # destination - must NOT complete the drag.
        st = c.state()
        src, dst = st["drag_source"], st["drag_destination"]
        obj0 = dict(st["drag_object"])
        _slide_to(c, src["x"] + src["w"] / 2, min(590.0, src["y"] + src["h"] + 60))
        cur = c.state()["cursor"]
        _drag_by(c, dst["x"] + dst["w"] / 2 - cur["x"], dst["y"] + dst["h"] / 2 - cur["y"])
        c.touch([])
        st4 = c.state()
        check("false_drag_rejected", st4["drag_done"] is False
              and st4["drag_object"] == obj0, f"object={st4['drag_object']}")

        # Positive: press ON the object, drag to destination; the object is
        # actually moved and retained at its observed final position.
        _slide_to(c, obj0["x"], obj0["y"])
        c.touch([("a", 0.0, 0.0, 1.5)])  # arm on object
        _drag_by(c, dst["x"] + dst["w"] / 2 - obj0["x"], dst["y"] + dst["h"] / 2 - obj0["y"])
        c.touch([])
        st5 = c.state()
        o = st5["drag_object"]
        in_dst = dst["x"] <= o["x"] <= dst["x"] + dst["w"] and dst["y"] <= o["y"] <= dst["y"] + dst["h"]
        check("drag_done_by_object_placement", st5["drag_done"] is True and in_dst
              and o != obj0, f"object={o}")
        dragged = c.capture("smoke-3-dragged.png")
        report["samples"].append(dragged["path"])
        check("pixels_change_on_drag", dragged["png"] != clicked["png"])

        # Scroll.
        off0 = c.state()["scroll_offset_px"]
        for i in range(1, 11):
            c.touch([("a", -0.02, i * 0.001, 0.3), ("b", 0.02, i * 0.001, 0.3)])
        c.touch([])
        off1 = c.state()["scroll_offset_px"]
        check("scroll_offset_changes", off1 > off0, f"offset {off0} -> {off1}")
        report["samples"].append(c.capture("smoke-4-scrolled.png")["path"])

        # Wall-clock watchdog: hold the button, go silent >0.8 s of real time.
        c.touch([("a", 0.0, 0.0, 1.5)])
        check("button_held_before_watchdog", c.state()["button_down"] is True)
        rc_before = c.state()["release_count"]
        time.sleep(1.2)
        st6 = c.state()
        check("watchdog_cancels_held_button",
              st6["button_down"] is False and st6["release_count"] == rc_before
              and st6["press_suppressed"] is True,
              f"down={st6['button_down']} releases={st6['release_count']}")
        # Press stays suppressed until a fresh lift.
        c.touch([("a", 0.01, 0.0, 1.5)])
        check("press_suppressed_until_lift", c.state()["button_down"] is False)
        c.touch([])
        c.touch([("a", 0.0, 0.0, 1.5)])
        check("fresh_press_after_lift", c.state()["button_down"] is True)

        # Stop clears held button; app exits on EOF.
        c.stop()
        check("stop_clears_button", c.state()["button_down"] is False)
        rc = c.quit()
        check("clean_exit_on_eof", rc == 0, f"rc={rc}")

    report["ok"] = all(ch["ok"] for ch in report["checks"])
    return report


def main():
    ap = argparse.ArgumentParser(description="TrackpadSandbox client")
    ap.add_argument("command", choices=["smoke", "hold", "probe"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.command == "smoke":
        report = run_smoke(args.out)
        print(json.dumps(report, indent=2))
        sys.exit(0 if report["ok"] else 1)
    if args.command == "hold":
        # Launch and keep the window visible for coordinator inspection.
        c = SandboxClient(out_dir=args.out)
        c.reset(seed=7)
        print(f"TrackpadSandbox visible (pid {c.proc.pid}); Ctrl+C to close.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            c.quit()
    if args.command == "probe":
        # Diagnostics-only launch: fixed fresh canary root, fixed canary name.
        canary_root = os.path.join(REPO_ROOT, ".local", "diag-canary")
        os.makedirs(canary_root, exist_ok=True)
        canary = os.path.join(canary_root, CANARY_NAME)
        if os.path.exists(canary):
            os.remove(canary)
        with SandboxClient(out_dir=args.out, diagnostics=True, canary_root=canary_root) as c:
            print(json.dumps(c.probe_file()))
            print("canary written outside container:", os.path.exists(canary))
            print(json.dumps(c.probe_net(9)))
            print(json.dumps(c.probe_net(65000)))
            c.quit()
        # Normal mode: probes are unavailable.
        with SandboxClient(out_dir=args.out) as c:
            print(json.dumps(c.raw({"cmd": "probe_file"})))
            print(json.dumps(c.raw({"cmd": "probe_net", "port": 9})))
            c.quit()


if __name__ == "__main__":
    main()
