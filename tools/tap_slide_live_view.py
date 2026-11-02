"""Continuous local MuJoCo scene viewer (SIM preview, no policy/training).

Renders the real 17-joint scene into bounded 800x600 overview/wrist buffers,
capped at 12 fps. The robot holds its initial pose; only the overview camera
responds to orbit/zoom commands. There is no lifetime deadline by default.
All endpoints bind to loopback. Runtime receipts/logs are private in .local/live.

Usage: python tools/tap_slide_live_view.py --port 8768
Optional --detach depends on the launcher's orphan-process policy. A managed
foreground session is supported. SIGTERM cleanly stops only this viewer.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

B_ROOT = Path(__file__).resolve().parents[1]
# Portable default: the current repo. The B worktree passes the A scene root
# via TAP_SLIDE_SCENE_ROOT because B intentionally has no scene.py.
A_ROOT = Path(os.environ.get("TAP_SLIDE_SCENE_ROOT", str(B_ROOT)))
SCENE_SRC = A_ROOT / "src" / "embodied_trackpad" / "tap_slide" / "scene.py"
sys.path.insert(0, str(A_ROOT / "src"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

LIVE_DIR = B_ROOT / ".local" / "live"
LOG_PATH = LIVE_DIR / "server.log"
RUNTIME_PATH = LIVE_DIR / "runtime.json"
HOST = "127.0.0.1"
PREFERRED_PORT = 8768
WIDTH, HEIGHT = 800, 600
MAX_FPS = 12.0
STALE_AFTER_S = 2.0
LIFETIME_S = 0.0  # 0 / None / non-finite / negative = unlimited (default)
WRIST_SWING_RAD = 0.0  # hard bound; wrist range is (-0.6, 0.6) rad
WRIST_SWITCH_HZ = 0.25
LOG_CAP_BYTES = 8 * 1024 * 1024

# Interactive overview free-camera bounds (camera-only, never physics).
CAM_DIST_MIN_M = 0.1
CAM_DIST_MAX_M = 2.0
CAM_ELEV_MIN_DEG = -89.0
CAM_ELEV_MAX_DEG = 89.0


def normalize_lifetime(value) -> float:
    """0 / None / non-finite / negative -> 0.0 (unlimited); else finite >0."""
    if value is None:
        return 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v) or v <= 0.0:
        return 0.0
    return v


def initial_overview_camera(config) -> dict:
    """Free-camera state matching the compiled fixed overview camera.

    Coordinator-verified MuJoCo freecam convention (mjv_updateScene):
    az=0,el=0 -> eye=lookat+(-1,0,0)*d; az=90,el=0 -> (0,-1,0)*d;
    az=0,el=-45 -> (-.707,0,+.707)*d. Inverted here:
    elevation=-degrees(asin(off_z/d)); azimuth=degrees(atan2(-off_y,-off_x))
    mod 360, with off = eye - lookat.
    """
    eye = np.asarray(config.overview_cam_pos, dtype=float)
    target = np.asarray(config.overview_cam_target, dtype=float)
    off = eye - target
    dist = float(np.linalg.norm(off))
    elev = -math.degrees(math.asin(off[2] / dist))
    azim = math.degrees(math.atan2(-off[1], -off[0])) % 360.0
    return {
        "azimuth": float(azim),
        "elevation": float(elev),
        "distance": float(dist),
        "lookat": [float(v) for v in target],
    }


CAMERA_PRESETS = {
    "reset": None,  # filled at runtime with initial_overview_camera
    "top": {"azimuth": 0.0, "elevation": -89.0, "distance": 0.48,
            "lookat": [-0.07, 0.02, 0.1]},
    "side": {"azimuth": 90.0, "elevation": -5.0, "distance": 0.4,
             "lookat": [-0.035, 0.0, 0.14]},
}


def _finite(cmd: dict, key: str):
    v = cmd.get(key)
    if v is None:
        return None
    v = float(v)  # raises TypeError/ValueError on bad types
    if not math.isfinite(v):
        raise ValueError(f"{key} must be finite")
    return v


def clamp_camera(state: dict) -> dict:
    state["distance"] = min(max(float(state["distance"]), CAM_DIST_MIN_M),
                            CAM_DIST_MAX_M)
    state["elevation"] = min(max(float(state["elevation"]), CAM_ELEV_MIN_DEG),
                             CAM_ELEV_MAX_DEG)
    state["azimuth"] = float(state["azimuth"]) % 360.0
    state["lookat"] = [float(v) for v in state["lookat"]]
    return state


def apply_camera_command(current: dict, cmd: dict,
                         presets: dict | None = None) -> dict:
    """Absolute {azimuth,elevation,distance}, delta {d_*}, or {preset: name}.

    Presets (reset/top/side) replace the whole camera state; deltas may
    combine with a preset. Raises ValueError on non-finite input or an empty
    command. Camera-only: physics is never touched.
    """
    if not isinstance(cmd, dict):
        raise ValueError("camera command must be a JSON object")
    preset_name = cmd.get("preset")
    if preset_name is not None:
        if not isinstance(preset_name, str) or preset_name not in (presets or {}):
            raise ValueError(f"unknown camera preset: {preset_name!r}")
        src = presets[preset_name]
        new = {k: float(src[k]) for k in ("azimuth", "elevation", "distance")}
        new["lookat"] = [float(v) for v in src["lookat"]]
        keys = ("azimuth", "elevation", "distance")
    else:
        new = {k: float(current[k]) for k in ("azimuth", "elevation", "distance")}
        new["lookat"] = [float(v) for v in current["lookat"]]
        keys = ("azimuth", "elevation", "distance")
    touched = preset_name is not None
    if "lookat" in cmd:
        if preset_name is not None:
            raise ValueError("lookat not allowed with preset")
        target = cmd["lookat"]
        if not isinstance(target, list) or len(target) != 3:
            raise ValueError("lookat must have three coordinates")
        target = [float(v) for v in target]
        if not all(math.isfinite(v) and abs(v) <= 0.5 for v in target):
            raise ValueError("lookat outside scene bounds")
        new["lookat"] = target
        touched = True
    for key in keys:
        absolute = _finite(cmd, key)
        delta = _finite(cmd, "d_" + key)
        if absolute is not None:
            if preset_name is not None:
                raise ValueError("absolute keys not allowed with preset")
            new[key] = absolute
            touched = True
        if delta is not None:
            new[key] = new[key] + delta
            touched = True
    if not touched:
        raise ValueError("empty camera command")
    return clamp_camera(new)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"{utc_now_iso()} {msg}\n"
    try:
        if LOG_PATH.exists() and LOG_PATH.stat().st_size > LOG_CAP_BYTES:
            return  # capped: stop appending rather than grow unbounded
        with open(LOG_PATH, "a") as fh:
            fh.write(line)
    except OSError:
        pass


def source_sha() -> str:
    return hashlib.sha256(SCENE_SRC.read_bytes()).hexdigest()


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, format="PNG")
    return buf.getvalue()


class LiveSim:
    """Physics + render thread over the real 17-DoF scene."""

    def __init__(self, run: dict, stop: threading.Event) -> None:
        self.run = run
        self.stop = stop
        self.lock = threading.Lock()
        self.latest: dict = {}
        from embodied_trackpad.tap_slide.scene import (
            InitialPoseConfig,
            load_scene,
            load_working_scene,
        )

        try:
            self.scene, winfo = load_working_scene()
            run["label"] = "working_mount_17j"
            run["mount_info"] = {
                "computed_wrist_pos": winfo["mount"]["computed_wrist_pos"],
                "palm_inclination_deg": winfo["mount"]["palm_inclination_deg"],
                "index_clearance_target_m": winfo["mount"]["index_clearance_target_m"],
                "reset_measurements": winfo["measured_working"],
                "pad_yaw_deg": self.scene.config.pad_yaw_deg,
                "pad_top_m": winfo["mount"]["pad_top_m"],
            }
            pose = {k: float(v) for k, v in winfo["pose"].items()}
        except Exception as exc:  # honest fallback, labelled
            self.scene = load_scene()
            run["label"] = "structural_pose_fallback_17j"
            run["mount_info"] = {"fallback_reason": f"{type(exc).__name__}: {exc}"}
            pose = InitialPoseConfig().qpos17()
        log(f"scene loaded: {run['label']} mujoco={mujoco.__version__}")

        m = self.scene.model
        self.data = mujoco.MjData(m)
        mujoco.mj_resetData(m, self.data)
        q = self.scene.open_pose_qpos()
        for i in range(m.njnt):
            name = m.joint(i).name
            if name in pose:
                q[m.jnt_qposadr[i]] = pose[name]
        self.data.qpos[:] = q
        # Hold via ACTUATOR names: actuator <j>_act targets joint <j>;
        # wrist actuator is LAST, wrist qpos is FIRST.
        self.data.ctrl[:] = 0.0
        for i in range(m.nu):
            aname = m.actuator(i).name
            jname = aname[:-4] if aname.endswith("_act") else aname
            jid = self.scene.joint_index(jname)
            self.data.ctrl[i] = q[m.jnt_qposadr[jid]]
        self._wrist_act = self.scene.wrist_actuator_id
        self._base_ctrl = self.data.ctrl.copy()
        mujoco.mj_forward(m, self.data)
        # NOTE: the Renderer is created inside loop() (the sim thread), not
        # here: on macOS the offscreen GL context is current only on the
        # thread that created it, and rendering from another thread hangs.
        # Interactive OVERVIEW free camera (camera-only; physics untouched).
        self.camera_state = initial_overview_camera(self.scene.config)
        self._initial_camera = dict(self.camera_state)
        CAMERA_PRESETS["reset"] = dict(self.camera_state)
        self.camera_revision = 0
        self.failed = False
        self.frame_id = 0

    def apply_camera_command(self, cmd: dict) -> dict:
        """Validate + clamp a camera command; applied by the render loop."""
        with self.lock:
            new = apply_camera_command(self.camera_state, cmd, CAMERA_PRESETS)
            self.camera_state = new
            self.camera_revision += 1
            return {**new, "revision": self.camera_revision}

    def loop(self) -> None:
        m = self.scene.model
        renderer = None
        freecam = mujoco.MjvCamera()
        freecam.type = mujoco.mjtCamera.mjCAMERA_FREE
        steps_per_frame = max(1, int(round((1.0 / MAX_FPS) / m.opt.timestep)))
        period = 1.0 / MAX_FPS
        try:
            renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
            while not self.stop.is_set():
                t_wall = time.time()
                deadline = self.run["deadline_unix"]
                if deadline is not None and t_wall >= deadline:
                    log("deadline reached; physics loop stopping")
                    break
                # Camera-only view change; never pauses physics.
                with self.lock:
                    rendered_revision = self.camera_revision
                    cam = {k: self.camera_state[k] for k in (
                        "azimuth", "elevation", "distance", "lookat")}
                freecam.azimuth = cam["azimuth"]
                freecam.elevation = cam["elevation"]
                freecam.distance = cam["distance"]
                freecam.lookat[:] = cam["lookat"]
                t_sim = float(self.data.time)
                self.data.ctrl[self._wrist_act] = self._base_ctrl[self._wrist_act] + (
                    WRIST_SWING_RAD * math.sin(2.0 * math.pi * WRIST_SWITCH_HZ * t_sim)
                )
                mujoco.mj_step(m, self.data, nstep=steps_per_frame)
                t_sim = float(self.data.time)
                renderer.update_scene(self.data, camera=freecam)
                overview = png_bytes(renderer.render())
                renderer.update_scene(self.data, camera=self.scene.wrist_cam_id)
                wrist = png_bytes(renderer.render())
                self.frame_id += 1
                with self.lock:
                    self.latest = {
                        "frame_id": self.frame_id,
                        "sim_time": t_sim,
                        "rendered_wall": time.time(),
                        "frames": {"overview": overview, "wrist": wrist},
                        "wrist_pitch_qpos": float(
                            self.data.qpos[m.jnt_qposadr[self.scene.wrist_joint_id]]
                        ),
                        "camera": {**cam, "revision": rendered_revision},
                    }
                elapsed = time.time() - t_wall
                self.stop.wait(max(0.0, period - elapsed))
        except Exception:
            self.failed = True
            log("render worker exception:\n" + traceback.format_exc(limit=4))
        finally:
            try:
                if renderer is not None:
                    renderer.close()
            except Exception as exc:
                log(f"renderer close failed (ignored): {exc}")


def classify(latest: dict, now: float, failed: bool = False) -> str:
    if failed:
        return "STOPPED"
    if not latest:
        return "STALLED"
    return "LIVE" if now - latest["rendered_wall"] <= STALE_AFTER_S else "STALLED"


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, run, sim):
        super().__init__(addr, handler)
        self.run = run
        self.sim = sim


class Handler(BaseHTTPRequestHandler):
    server: ViewerServer

    def log_message(self, fmt, *a):  # quiet; lifecycle lines go to server.log
        pass

    def _send(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _state(self):
        now = time.time()
        with self.server.sim.lock:
            latest = dict(self.server.sim.latest)
        frames = latest.pop("frames", None)
        camera = latest.pop("camera", None)
        deadline_unix = self.server.run["deadline_unix"]
        return {
            **latest,
            "status": classify(latest, now, self.server.sim.failed),
            "stale_after_s": STALE_AFTER_S,
            "server_age_s": round(now - self.server.run["started_unix"], 3),
            "frame_bytes": {k: len(v) for k, v in frames.items()} if frames else {},
            "camera": camera or {**self.server.sim.camera_state,
                                 "revision": self.server.sim.camera_revision},
            "deadline_utc": self.server.run["deadline_utc"],
            "deadline_unix": deadline_unix,
            "remaining_lifetime_s": (
                round(max(0.0, deadline_unix - now), 1)
                if deadline_unix is not None else None
            ),
            "continuous": deadline_unix is None,
            "label": self.server.run["label"],
            "source_sha256": self.server.run["source_sha256"],
            "mode": self.server.run["mode"],
            "mount_info": self.server.run["mount_info"],
            "camera_presets": CAMERA_PRESETS,
            "stream_id": self.server.run["start_utc"],
            "max_fps": MAX_FPS,
        }

    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/":
            self._send(200, (WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path in ("/viewer.css", "/viewer.js"):
            ctype = "text/css" if path.endswith(".css") else "text/javascript"
            self._send(200, (WEB_ROOT / path[1:]).read_bytes(), ctype + "; charset=utf-8")
        elif path == "/state.json":
            self._send(200, json.dumps(self._state()).encode(), "application/json")
        elif path in ("/overview.png", "/wrist.png"):
            key = path.strip("/").split(".")[0]
            with self.server.sim.lock:
                latest = self.server.sim.latest
                body = latest.get("frames", {}).get(key)
            if body is None:
                self._send(503, b"no frame yet", "text/plain")
            else:
                self._send(200, body, "image/png", {"X-Frame-Id": str(latest["frame_id"]), "X-Rendered-Wall": str(latest["rendered_wall"]), "X-Stream-Id": self.server.run["start_utc"], "X-Camera-Revision": str(latest["camera"]["revision"])})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        origin = self.headers.get("Origin")
        expected = f"http://{HOST}:{self.server.server_port}"
        if origin is not None and origin != expected:
            self._send(403, b"origin rejected", "text/plain")
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self._send(415, b"application/json required", "text/plain")
            return
        if path != "/camera.json":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("bad content length")
            cmd = json.loads(self.rfile.read(length))
            state = self.server.sim.apply_camera_command(cmd)
            self._send(200, json.dumps({"ok": True, "camera": state}).encode(),
                       "application/json")
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._send(400, json.dumps({"ok": False, "error": str(exc)}).encode(),
                       "application/json")


WEB_ROOT = B_ROOT / "tools" / "tap_slide_viewer"



def bind_port(preferred: int) -> int:
    import socket

    for port in range(preferred, preferred + 20):
        with socket.socket() as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((HOST, port))
                return port
            except OSError:
                continue  # owned by someone else; never kill, try next
    raise RuntimeError("no free port in range")


def write_runtime(run: dict) -> None:
    # An older instance finishing on another port must not erase the new owner.
    target = RUNTIME_PATH
    if run.get("status", "").startswith("stopped") and target.exists():
        try:
            owner = json.loads(target.read_text()).get("pid")
            if owner != run["pid"]:
                target = LIVE_DIR / f"runtime-{run['pid']}.json"
        except (OSError, ValueError):
            pass
    tmp = target.with_suffix(f".{run['pid']}.tmp")
    tmp.write_text(json.dumps(run, indent=2) + "\n")
    tmp.replace(target)


def service_main(args) -> int:
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    lifetime = normalize_lifetime(args.lifetime_s)  # 0 => unlimited
    deadline = started + lifetime if lifetime > 0.0 else None
    run = {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "started_unix": started,
        "start_utc": datetime.fromtimestamp(started, timezone.utc).isoformat(
            timespec="seconds"
        ),
        "deadline_unix": deadline,
        "deadline_utc": (
            datetime.fromtimestamp(deadline, timezone.utc).isoformat(timespec="seconds")
            if deadline is not None else None
        ),
        "lifetime_s": lifetime,
        "mode": (
            "SIM 调试演示（非训练） · 持续直播 · 手动停止"
            if deadline is None
            else "SIM 调试演示（非训练） · 限时"
        ),
        "continuous": deadline is None,
        "host": HOST,
        "port": None,
        "source_path": str(SCENE_SRC),
        "source_sha256": source_sha(),
        "service_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "web_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(WEB_ROOT.iterdir()) if p.is_file()},
        "scene_root": str(A_ROOT),
        "mujoco_version": mujoco.__version__,
        "label": None,
        "status": "starting",
        "log": str(LOG_PATH),
    }
    write_runtime(run)
    log(f"service start pid={os.getpid()} src_sha={run['source_sha256'][:16]}")

    stop = threading.Event()
    sim = LiveSim(run, stop)
    server_box: dict = {}

    def shutdown(signum, frame):
        log(f"signal {signum}; shutting down")
        stop.set()

        def _shutdown():
            time.sleep(0.2)
            srv = server_box.get("server")
            if srv is not None:
                try:
                    srv.shutdown()
                except Exception:
                    pass

        threading.Thread(target=_shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # survive terminal/CLI exit

    port = bind_port(args.port)
    run["port"] = port
    write_runtime(run)
    server = ViewerServer((HOST, port), Handler, run, sim)
    server_box["server"] = server
    run["url"] = f"http://{HOST}:{port}/"
    run["status"] = "running"
    write_runtime(run)
    log(f"listening on {HOST}:{port} "
        + (f"deadline={run['deadline_utc']}" if deadline is not None else "continuous (no deadline)"))

    sim_thread = threading.Thread(target=sim.loop, daemon=True)
    sim_thread.start()
    serve_thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.2}
    )
    serve_thread.daemon = True
    serve_thread.start()

    while not stop.is_set() and (deadline is None or time.time() < deadline):
        time.sleep(0.5)
    stop.set()
    try:
        server.shutdown()
        server.server_close()
    except Exception:
        pass
    sim_thread.join(timeout=5.0)
    if sim_thread.is_alive():  # renderer teardown hung; exit anyway
        log("sim thread did not join; forcing exit")
        os._exit(0)
    run["status"] = (
        "stopped_deadline" if deadline is not None and time.time() >= deadline
        else "stopped"
    )
    run["stopped_utc"] = utc_now_iso()
    write_runtime(run)
    log(f"service end status={run['status']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=PREFERRED_PORT)
    ap.add_argument(
        "--lifetime-s",
        type=float,
        default=LIFETIME_S,
        help="0 / negative / non-finite = unlimited continuous stream (default)",
    )
    ap.add_argument(
        "--detach",
        action="store_true",
        help="spawn a detached service process and exit (survives CLI exit)",
    )
    args = ap.parse_args()
    lifetime = normalize_lifetime(args.lifetime_s)
    if args.detach:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--port", str(args.port)]
        if lifetime > 0.0:  # unlimited: pass nothing (default is continuous)
            cmd += ["--lifetime-s", str(lifetime)]
        with open(LOG_PATH, "ab") as logf:
            # NOTE: no start_new_session. A fully detached session loses the
            # macOS WindowServer connection and MuJoCo offscreen GL rendering
            # blocks forever. Staying in the current session keeps rendering
            # working; SIGHUP is ignored in the child so the service still
            # outlives the CLI exit.
            proc = subprocess.Popen(
                cmd,
                cwd=str(B_ROOT),
                stdin=subprocess.DEVNULL,
                stdout=logf,
                stderr=logf,
                close_fds=True,
            )
        print(f"detached child pid={proc.pid}; waiting for runtime.json ...")
        for _ in range(100):
            time.sleep(0.2)
            if RUNTIME_PATH.exists():
                try:
                    run = json.loads(RUNTIME_PATH.read_text())
                except json.JSONDecodeError:
                    continue
                if run.get("pid") == proc.pid and run.get("status") == "running":
                    print(json.dumps({
                        "url": run["url"], "pid": run["pid"],
                        "start_utc": run["start_utc"],
                        "deadline_utc": run["deadline_utc"],
                        "continuous": run["continuous"],
                        "label": run["label"],
                        "source_sha256": run["source_sha256"],
                    }, indent=2, ensure_ascii=False))
                    return 0
        print("ERROR: service did not report running; see .local/live/server.log")
        return 1
    return service_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
