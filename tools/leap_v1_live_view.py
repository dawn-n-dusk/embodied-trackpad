"""Local-only LIVE viewer for the LEAP v1 Full hand simulation (scripted physics).

Serves a small HTTP page bound to 127.0.0.1 that streams freshly rendered
MuJoCo frames of the pinned Menagerie leap_hand right-hand model while the
existing diagnostics script (per-channel sweep, index/middle spread,
flex/return from ``embodied_trackpad.leap_v1.diagnostics.build_script``)
drives all 16 channels through ``ctrl`` + ``mujoco.mj_step``. The only live
qpos write is the all-zero open pose at each episode reset. There is NO
trackpad contact and NO learned policy in this view; the UI states so.

Freshness is physics-based: every frame carries the wall-clock instant it was
rendered and a monotonically increasing frame id. Server and page both report
STALLED when the newest rendered frame is older than ``STALE_AFTER_S`` seconds
and STOPPED when the server/state cannot be reached; the page keeps polling and
redisplays instead of freezing under a LIVE label. No prerecorded frames are
ever served or labeled live.

The process enforces its own maximum lifetime (default 1200 s): at the
deadline the physics loop stops, the HTTP server shuts down, the port is
released and a result receipt is written. SIGTERM/SIGINT trigger the same
clean shutdown.

Usage:
    PYTHONPATH=<repo>/src python tools/leap_v1_live_view.py \
        --port 8768 --lifetime-s 1200 --out-dir .local/leap-v1-live
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import struct
import sys
import threading
import time
import traceback
import zlib
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Numerical-thread cap: hard assignment (an inherited value above 2 must not
# survive); must be set before numpy/mujoco are imported (in run()).
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "2"

WIDTH = 640
HEIGHT = 480
MAX_FPS = 8.0
STALE_AFTER_S = 2.0
DEFAULT_PORT = 8768
PORT_ATTEMPTS = 10
DEFAULT_LIFETIME_S = 1200
BIND_HOST = "127.0.0.1"

MODE_LABEL = "scripted physics; hand only; no learned policy"
SCOPE_LABELS = ("no trackpad contact", "uncalibrated upstream defaults", "no prerecorded frames")

STATUS_LIVE = "LIVE"
STATUS_STALLED = "STALLED"
STATUS_STOPPED = "STOPPED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_frame_status(
    frame_age_s: float | None,
    server_running: bool,
    stale_after_s: float = STALE_AFTER_S,
) -> str:
    """Physics-freshness status; never LIVE for stale or absent frames."""
    if not server_running:
        return STATUS_STOPPED
    if frame_age_s is None:
        return STATUS_STALLED
    if frame_age_s > stale_after_s:
        return STATUS_STALLED
    return STATUS_LIVE


def png_bytes(rgb) -> bytes:
    """Stdlib 8-bit RGB PNG encoder.

    Byte-identical chunk assembly to ``leap_sim.pad_env.write_png`` (accepted
    writer), returning bytes instead of writing a file.
    """
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


def build_state(
    *,
    run: dict,
    frame: dict | None,
    server_running: bool,
    now_unix: float | None = None,
) -> dict:
    """/state.json payload; freshness derives from rendered-frame identity."""
    now = time.time() if now_unix is None else now_unix
    frame_age = None if frame is None else max(0.0, now - frame["rendered_unix"])
    status = classify_frame_status(frame_age, server_running)
    return {
        "run_id": run["run_id"],
        "pid": run["pid"],
        "bind": f"{BIND_HOST}:{run['port']}",
        "started_utc": run["started_utc"],
        "deadline_utc": run["deadline_utc"],
        "remaining_lifetime_s": max(0.0, run["deadline_unix"] - now),
        "status": status,
        "frame_id": None if frame is None else frame["frame_id"],
        "frame_rendered_unix": None if frame is None else frame["rendered_unix"],
        "frame_age_s": frame_age,
        "stale_after_s": STALE_AFTER_S,
        "episode": None if frame is None else frame["episode"],
        "sim_time_s": None if frame is None else frame["episode_time_s"],
        "sim_time_total_s": None if frame is None else frame["sim_time_total_s"],
        "phase": None if frame is None else frame["phase"],
        "physics_steps": None if frame is None else frame["physics_steps"],
        "measured_fps": None if frame is None else frame["measured_fps"],
        "target_fps": MAX_FPS,
        "resolution": [WIDTH, HEIGHT],
        "timestep_s": run["timestep_s"],
        "embodiment_id": run["embodiment_id"],
        "calibration_id": run["calibration_id"],
        "label": MODE_LABEL,
        "scope": list(SCOPE_LABELS),
        "frame_source": "newly rendered mujoco physics frames (ctrl + mj_step); no prerecorded media",
    }


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LEAP v1 Full - live scripted physics</title>
<style>
 body { font-family: -apple-system, Helvetica, Arial, sans-serif; background:#f4f6f8;
        color:#1c2733; margin:0; padding:16px; }
 h1 { font-size:18px; margin:0 0 4px; }
 .chips span { display:inline-block; background:#e3ecf5; border:1px solid #b9cbdd;
        border-radius:10px; padding:2px 8px; margin:2px 4px 2px 0; font-size:12px; }
 #status { font-weight:700; padding:4px 10px; border-radius:6px; color:#fff; }
 .LIVE { background:#1e8e3e; } .STALLED { background:#b26a00; } .STOPPED { background:#c5221f; }
 #view { background:#dfe5ec; border:1px solid #9fb2c3; margin-top:8px; width:640px; height:auto;
        max-width:100%; display:block; box-sizing:border-box; }
 table { border-collapse:collapse; margin-top:8px; font-size:13px; max-width:100%;
        overflow-wrap:anywhere; }
 td { padding:2px 10px 2px 0; font-variant-numeric:tabular-nums; overflow-wrap:anywhere;
        word-break:break-word; }
 td.k { color:#5b6b7b; white-space:nowrap; }
 #note { font-size:12px; color:#5b6b7b; margin-top:8px; max-width:100%; overflow-wrap:anywhere; }
</style>
</head>
<body>
<h1>LEAP Hand v1 Full - live MuJoCo view <span id="status" class="STOPPED">CONNECTING</span></h1>
<div class="chips">
 <span>__MODE_LABEL__</span><span>__SCOPE_0__</span><span>__SCOPE_1__</span><span>__SCOPE_2__</span>
</div>
<img id="view" width="640" height="480" alt="live simulation frame">
<table>
 <tr><td class="k">run</td><td id="run">-</td><td class="k">deadline in</td><td id="deadline">-</td></tr>
 <tr><td class="k">episode</td><td id="episode">-</td><td class="k">phase</td><td id="phase">-</td></tr>
 <tr><td class="k">sim time</td><td id="simt">-</td><td class="k">displayed frame</td><td id="frame">-</td></tr>
 <tr><td class="k">displayed frame age</td><td id="age">-</td><td class="k">measured fps</td><td id="fps">-</td></tr>
</table>
<div id="note">Frames are newly rendered MuJoCo physics frames fetched one by one
from /frame.png (scripted ctrl + mj_step; hand only; no learned policy, no
trackpad contact). Only a newly decoded displayed frame refreshes freshness:
LIVE requires a displayed frame younger than 2 s; an independent watchdog shows
STALLED after &gt;2 s without a new displayed frame (even if metadata still
answers) and STOPPED when the server is gone; polling retries automatically.
No prerecorded frames are ever labeled live.</div>
<script>
const STALE_AFTER_MS = 2000;        // displayed-frame freshness threshold
const POLL_MS = 150;                // <= 8 displayed fps
const FETCH_TIMEOUT_MS = 1500;      // bounded per-request timeout
const view = document.getElementById('view');
const statusEl = document.getElementById('status');
let displayedFrameId = -1;
let lastDisplayedWallMs = 0;        // set ONLY when a new frame is decoded+shown
let currentBlobUrl = null;
let inFlight = false;               // no overlapping /frame.png polls
let consecutiveFailures = 0;

function setStatus(s, detail) {
  statusEl.textContent = s + (detail ? ' - ' + detail : '');
  statusEl.className = s;
}

async function pollFrame() {
  if (inFlight) return;             // never overlap frame requests
  inFlight = true;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const r = await fetch('/frame.png', {cache: 'no-store', signal: ctrl.signal});
    if (!r.ok) throw new Error('http ' + r.status);
    const fid = parseInt(r.headers.get('X-Frame-Id') || '-1', 10);
    const simt = parseFloat(r.headers.get('X-Sim-Time-S') || 'NaN');
    const blob = await r.blob();
    if (fid !== displayedFrameId) {
      const url = URL.createObjectURL(blob);
      const probe = new Image();
      probe.src = url;
      await probe.decode();         // fully decoded BEFORE it may be displayed
      if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = url;
      view.src = url;               // only now does freshness advance
      displayedFrameId = fid;
      lastDisplayedWallMs = Date.now();
      if (Number.isFinite(simt)) document.getElementById('simt').textContent = simt.toFixed(2) + ' s';
      document.getElementById('frame').textContent = fid;
    }
    consecutiveFailures = 0;
  } catch (e) {
    consecutiveFailures += 1;       // aborts, http errors, decode failures
  } finally {
    clearTimeout(timer);
    inFlight = false;
  }
}

async function pollState() {        // metadata only; never marks LIVE by itself
  try {
    const r = await fetch('/state.json', {cache: 'no-store'});
    if (!r.ok) throw new Error('http ' + r.status);
    const s = await r.json();
    document.getElementById('run').textContent = s.run_id;
    document.getElementById('deadline').textContent = Math.max(0, s.remaining_lifetime_s).toFixed(0) + ' s';
    document.getElementById('episode').textContent = s.episode == null ? '-' : s.episode;
    document.getElementById('phase').textContent = s.phase || '-';
    document.getElementById('fps').textContent = s.measured_fps == null ? '-' : s.measured_fps.toFixed(1);
    if (s.status === 'STOPPED') consecutiveFailures = Math.max(consecutiveFailures, 2);
  } catch (e) { /* frame loop + watchdog own the status */ }
}

function watchdog() {               // independent of any request state
  const ageMs = lastDisplayedWallMs ? Date.now() - lastDisplayedWallMs : Infinity;
  document.getElementById('age').textContent =
    ageMs === Infinity ? '-' : (ageMs / 1000).toFixed(2) + ' s';
  if (consecutiveFailures >= 3) {
    setStatus('STOPPED', 'server unreachable');
  } else if (ageMs > STALE_AFTER_MS) {
    setStatus('STALLED', 'no new displayed frame for >2 s');
  } else {
    setStatus('LIVE');
  }
}

setInterval(pollFrame, POLL_MS);    // retries automatically after any failure
setInterval(pollState, 1000);
setInterval(watchdog, 500);
pollFrame(); pollState(); watchdog();
</script>
</body>
</html>
"""


def render_index() -> bytes:
    html = (
        INDEX_HTML.replace("__MODE_LABEL__", MODE_LABEL)
        .replace("__SCOPE_0__", SCOPE_LABELS[0])
        .replace("__SCOPE_1__", SCOPE_LABELS[1])
        .replace("__SCOPE_2__", SCOPE_LABELS[2])
    )
    return html.encode("utf-8")


def script_ctrl_at(phases, t: float):
    """ctrl vector at script time t; mirrors diagnostics._ctrl_at semantics."""
    for ph in phases:
        if ph.t0 - 1e-12 <= t < ph.t1 or (t >= phases[-1].t1 and ph is phases[-1]):
            if ph.kind == "ramp":
                a = min(max((t - ph.t0) / (ph.t1 - ph.t0), 0.0), 1.0)
                return (1 - a) * ph.base + a * ph.target, ph
            return ph.base.copy(), ph
    return phases[-1].base.copy(), phases[-1]


class LiveSim:
    """Physics + render loop for the scripted LEAP v1 episode (thread target)."""

    def __init__(self, args, run: dict, cond: threading.Condition, stop: threading.Event):
        self.args = args
        self.run = run
        self.cond = cond  # guards self.latest
        self.stop = stop
        self.latest: dict | None = None
        self.render_times: deque = deque(maxlen=64)
        self.stop_reason: str | None = None  # set when the physics loop exits
        self.frames_rendered = 0
        self.episodes_completed = 0
        self.sim_time_total = 0.0

    def _bright_scene_hook(self, spec) -> None:
        import mujoco

        spec.add_texture(
            name="skybox",
            type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
            rgb1=[0.93, 0.95, 1.0],
            rgb2=[0.55, 0.62, 0.75],
            width=64,
            height=64,
        )
        spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
        spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]

    def _publish(self, **fields) -> None:
        with self.cond:
            if self.latest is None:
                self.latest = {}
            self.latest.update(fields)
            self.cond.notify_all()

    def loop(self) -> None:
        import mujoco
        import numpy as np

        from embodied_trackpad.leap_sim.pad_env import apply_ctrl_checked
        from embodied_trackpad.leap_v1 import (
            CALIBRATION_ID,
            EMBODIMENT_ID,
            FINGERS,
            TIP_SITES,
            load_hand,
            open_pose_qpos,
        )
        from embodied_trackpad.leap_v1.diagnostics import build_script

        hand = load_hand(spec_hook=self._bright_scene_hook)
        m = hand.model
        self.run["embodiment_id"] = EMBODIMENT_ID
        self.run["calibration_id"] = CALIBRATION_ID
        self.run["timestep_s"] = float(m.opt.timestep)

        d = mujoco.MjData(m)
        phases = build_script(m)
        episode_len = float(phases[-1].t1)
        dt = float(m.opt.timestep)

        renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
        cam = mujoco.MjvCamera()

        def reset_episode() -> None:
            mujoco.mj_resetData(m, d)
            d.qpos[:] = open_pose_qpos(m)  # the only live qpos write; reset-only
            apply_ctrl_checked(m, d, np.zeros(m.nu))
            mujoco.mj_forward(m, d)

        reset_episode()
        tips0 = np.stack([d.site(TIP_SITES[f]).xpos for f in FINGERS])
        centre = 0.5 * (tips0.mean(axis=0) + d.body("palm").xpos)
        cam.lookat = [float(v) for v in centre]
        cam.distance = 0.38
        cam.azimuth = 115.0
        cam.elevation = -25.0

        episode = 1
        episode_t = 0.0
        physics_steps = 0
        frame_id = 0
        render_period = 1.0 / min(self.args.fps, MAX_FPS)
        deadline_unix = self.run["deadline_unix"]

        wall0 = time.perf_counter()
        next_step_wall = wall0
        next_render_wall = wall0

        while not self.stop.is_set():
            if time.time() >= deadline_unix:
                break
            desired, phase = script_ctrl_at(phases, episode_t)
            apply_ctrl_checked(m, d, desired)
            mujoco.mj_step(m, d)
            physics_steps += 1
            episode_t += dt
            self.sim_time_total += dt

            now_wall = time.perf_counter()
            if now_wall >= next_render_wall:
                mujoco.mj_forward(m, d)  # observation instant matches rendered state
                renderer.update_scene(d, cam)
                png = png_bytes(renderer.render())
                frame_id += 1
                self.frames_rendered = frame_id
                self.render_times.append(time.time())
                rts = self.render_times
                fps = (len(rts) - 1) / (rts[-1] - rts[0]) if len(rts) > 1 else 0.0
                self._publish(
                    frame_id=frame_id,
                    png=png,
                    sha256=hashlib.sha256(png).hexdigest(),
                    rendered_unix=time.time(),
                    episode=episode,
                    episode_time_s=episode_t,
                    sim_time_total_s=self.sim_time_total,
                    phase=phase.name,
                    physics_steps=physics_steps,
                    measured_fps=fps,
                )
                next_render_wall = now_wall + render_period

            if episode_t >= episode_len - 1e-9:
                reset_episode()
                self.episodes_completed += 1
                episode += 1
                episode_t = 0.0

            next_step_wall += dt
            lag = time.perf_counter() - next_step_wall
            if lag > 0.25:
                next_step_wall = time.perf_counter()  # resync; never spiral
            else:
                sleep_s = next_step_wall - time.perf_counter()
                if sleep_s > 0:
                    self.stop.wait(timeout=min(sleep_s, 0.05))

        self.stop_reason = "signal" if self.stop.is_set() else "lifetime_expired"
        renderer.close()
        with self.cond:
            self.cond.notify_all()


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, run: dict, sim: LiveSim, cond: threading.Condition):
        super().__init__(addr, handler)
        self.run = run
        self.sim = sim
        self.cond = cond
        self.index_bytes = render_index()


class Handler(BaseHTTPRequestHandler):
    server: ViewerServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quiet; launcher log records lifecycle
        pass

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send(HTTPStatus.OK, self.server.index_bytes, "text/html; charset=utf-8")
        elif path == "/state.json":
            with self.server.cond:
                frame = None if self.server.sim.latest is None else dict(self.server.sim.latest)
            if frame is not None:
                frame.pop("png", None)
            running = self.server.sim.stop_reason is None and not self.server.sim.stop.is_set()
            payload = build_state(run=self.server.run, frame=frame, server_running=running)
            self._send(HTTPStatus.OK, json.dumps(payload).encode(), "application/json")
        elif path == "/frame.png":
            with self.server.cond:
                frame = None if self.server.sim.latest is None else dict(self.server.sim.latest)
            if frame is None:
                self._send(HTTPStatus.SERVICE_UNAVAILABLE, b"no frame yet", "text/plain")
                return
            self._send(
                HTTPStatus.OK,
                frame["png"],
                "image/png",
                {
                    "X-Frame-Id": str(frame["frame_id"]),
                    "X-Frame-Sha256": frame["sha256"],
                    "X-Frame-Rendered-Unix": f"{frame['rendered_unix']:.3f}",
                    "X-Sim-Time-S": f"{frame['sim_time_total_s']:.3f}",
                },
            )
        elif path == "/stream":
            self._send(
                HTTPStatus.GONE,
                b"display transport is /frame.png polling; multipart stream removed",
                "text/plain",
            )
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _stream(self) -> None:  # removed display transport; see /stream route
        raise NotImplementedError


def bind_server(args, run: dict, sim: LiveSim, cond: threading.Condition) -> ViewerServer:
    last_err: OSError | None = None
    for port in range(args.port, args.port + PORT_ATTEMPTS):
        try:
            srv = ViewerServer((BIND_HOST, port), Handler, run, sim, cond)
            run["port"] = port
            return srv
        except OSError as exc:  # port busy: try the next one, never kill anyone
            last_err = exc
    raise RuntimeError(
        f"no free port in {args.port}..{args.port + PORT_ATTEMPTS - 1} on {BIND_HOST}"
    ) from last_err


def run(args) -> int:
    started_wall = time.time()
    run = {
        "run_id": f"leapv1live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}",
        "pid": os.getpid(),
        "port": None,
        "started_utc": utc_now_iso(),
        "deadline_unix": started_wall + args.lifetime_s,
        "deadline_utc": datetime.fromtimestamp(
            started_wall + args.lifetime_s, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "lifetime_s": args.lifetime_s,
        "embodiment_id": None,
        "calibration_id": None,
        "timestep_s": None,
    }
    cond = threading.Condition()
    stop = threading.Event()
    sim = LiveSim(args, run, cond, stop)
    server = bind_server(args, run, sim, cond)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_path = Path(__file__).resolve()
    identity = dict(run)
    identity.update(
        {
            "bind_host": BIND_HOST,
            "url": f"http://{BIND_HOST}:{run['port']}/",
            "fps_cap": min(args.fps, MAX_FPS),
            "resolution": [WIDTH, HEIGHT],
            "stale_after_s": STALE_AFTER_S,
            "label": MODE_LABEL,
            "python": sys.executable,
            "tool": tool_path.name,
            "tool_sha256": hashlib.sha256(tool_path.read_bytes()).hexdigest(),
        }
    )
    (out_dir / "run.json").write_text(json.dumps(identity, indent=2) + "\n")
    (out_dir / "viewer.pid").write_text(f"{os.getpid()}\n")

    def on_signal(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def physics_target() -> None:
        try:
            sim.loop()
        except Exception:
            traceback.print_exc()
            if sim.stop_reason is None:
                sim.stop_reason = "error"
            stop.set()

    physics = threading.Thread(target=physics_target, name="physics", daemon=True)
    physics.start()

    def shutdown_when_physics_ends() -> None:
        # Deadline-based: shuts the server down even if the physics thread hung.
        while sim.stop_reason is None and time.time() < run["deadline_unix"]:
            time.sleep(0.2)
        server.shutdown()

    monitor = threading.Thread(
        target=shutdown_when_physics_ends, name="expiry-monitor", daemon=True
    )
    monitor.start()
    print(
        f"[leap-v1-live] serving {identity['url']} run_id={run['run_id']} "
        f"pid={os.getpid()} deadline_utc={run['deadline_utc']}",
        flush=True,
    )
    server.serve_forever(poll_interval=0.2)
    physics.join(timeout=5.0)
    server.server_close()  # releases the port

    result = {
        "run_id": run["run_id"],
        "pid": os.getpid(),
        "port": run["port"],
        "started_utc": run["started_utc"],
        "deadline_utc": run["deadline_utc"],
        "stopped_utc": utc_now_iso(),
        "stop_reason": sim.stop_reason or "deadline_forced",
        "lifetime_s": args.lifetime_s,
        "frames_rendered": sim.frames_rendered,
        "episodes_completed": sim.episodes_completed,
        "sim_time_total_s": round(sim.sim_time_total, 3),
        "measured_fps_final": (
            round(
                (len(sim.render_times) - 1) / (sim.render_times[-1] - sim.render_times[0]),
                3,
            )
            if len(sim.render_times) > 1
            else None
        ),
        "label": MODE_LABEL,
        "frame_source": "newly rendered mujoco physics frames; no prerecorded media",
        "embodiment_id": run["embodiment_id"],
        "calibration_id": run["calibration_id"],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[leap-v1-live] stopped reason={sim.stop_reason} frames={sim.frames_rendered} "
        f"episodes={sim.episodes_completed}",
        flush=True,
    )
    return 0


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Hard bounds: lifetime in (0, 1200] s, fps in (0, 8]; both finite."""
    if not math.isfinite(args.lifetime_s) or not (0 < args.lifetime_s <= 1200):
        parser.error("--lifetime-s must be finite and in (0, 1200]")
    if not math.isfinite(args.fps) or not (0 < args.fps <= MAX_FPS):
        parser.error(f"--fps must be finite and in (0, {MAX_FPS:g}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="first port to try on 127.0.0.1")
    parser.add_argument("--lifetime-s", type=float, default=DEFAULT_LIFETIME_S, help="hard maximum lifetime")
    parser.add_argument("--fps", type=float, default=MAX_FPS, help=f"render rate cap (<= {MAX_FPS})")
    parser.add_argument("--out-dir", type=Path, default=Path(".local/leap-v1-live"))
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
