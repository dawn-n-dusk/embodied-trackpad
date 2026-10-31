"""Localhost-only observer for training-monitor run directories.

Serves a small GET-only page on 127.0.0.1 (port 8770, then nearby) that
follows a run directory populated through
``embodied_trackpad.monitor.MonitorPublisher``. The server re-reads and
re-verifies ``latest.json`` + ``frame.png`` on every poll (schema, counters,
PNG identity, SHA-256 content hash); invalid or mismatched data fails closed
and is never served as training. This process is an observer only: it does
not simulate, does not generate motor actions, and never substitutes the
scripted demo for training. All state labels are producer-reported until
independent integration acceptance.

Freshness rules (server and page):
  * freshness derives from the producer timestamps inside the last verified
    record; re-reading unchanged files never manufactures freshness;
  * a record already present at viewer startup is REPLAY (historical), never
    LIVE, even if its heartbeat looks fresh;
  * terminal lifecycle shows the final frame as ENDED/FAILED, never LIVE
    (terminal wins over historical REPLAY);
  * test_fixture / scripted_diagnostic sources show TEST / DEMO while fresh
    and STALLED when stale — they are never LIVE and never "training";
  * within one run_id, frame_seq must strictly increase: a lower sequence or
    a mutated record at the same sequence is rejected, keeping the last
    verified state; a new run_id resets sequence identity;
  * waiting with no source shows no video plus a Chinese explanatory label;
  * the page swaps the image only after a *new* frame is fully decoded, and
    only that decode advances browser-side freshness; an independent
    watchdog downgrades to STALLED/STOPPED with automatic retry.

Lifetime: ``--max-wall-s`` is the declared lifetime — finite, default 600 s,
accepted up to 86400 s when explicitly supplied; the declared value must
match the authorized job budget. The server also stops automatically once the
verified producer record reaches a terminal lifecycle (completed/failed),
after a short final-frame grace period (``TERMINAL_GRACE_S``) so the last
frame stays viewable. No-source startup stays bounded by the same deadline.
Expiry or SIGTERM/SIGINT shut the server down, release the port and write a
stop receipt (result.json).

Usage:
    PYTHONPATH=<repo>/src python tools/training_live_view.py \
        --run-dir <run directory> --port 8770 --max-wall-s 600 \
        --out-dir .local/training-monitor/server
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from embodied_trackpad.monitor import (  # noqa: E402
    LABEL_BASIS,
    MonitorProtocolError,
    derive_display,
    read_verified_record,
)

BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8770
PORT_ATTEMPTS = 10
DEFAULT_MAX_WALL_S = 600.0
MAX_WALL_S_CAP = 86400.0  # declared-lifetime ceiling; must match the authorized job budget
TERMINAL_GRACE_S = 30.0  # final-frame viewing grace after a terminal producer state
DEFAULT_STALE_AFTER_S = 10.0
MAX_STALE_AFTER_S = 120.0
DEFAULT_POLL_INTERVAL_S = 0.5

SCOPE_LABELS = (
    "observer only; no simulation; no motor actions",
    "producer-reported labels; pending independent acceptance",
    "never substitutes the scripted demo for training",
)

TERMINAL_LIFECYCLES = ("completed", "failed")


def stop_reason(now: float, deadline_unix: float, terminal_since: float | None,
                grace_s: float = TERMINAL_GRACE_S) -> str | None:
    """Stop decision: hard declared deadline, or terminal producer state plus
    the final-frame grace period. ``None`` means keep serving."""
    if now >= deadline_unix:
        return "deadline_expired"
    if terminal_since is not None and now >= terminal_since + grace_s:
        return "terminal_state"
    return None


def select_stop_reason(current: str, signaled: bool, box_reason: str) -> str:
    """Final receipt reason: an internal serve_forever error is preserved and
    never overwritten by the signal/deadline selection."""
    if current == "error":
        return "error"
    return "signal" if signaled else box_reason


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunMonitor:
    """Verified view of one run directory (no sockets; safe to unit-test)."""

    def __init__(self, run_dir: str | Path, *, stale_after_s: float = DEFAULT_STALE_AFTER_S):
        self.run_dir = Path(run_dir)
        self.stale_after_s = stale_after_s
        self.cond = threading.Condition()
        self.current = None  # last accepted Snapshot
        self.historical = False  # whether `current` predates this viewer
        self.last_error: str | None = None
        self.last_error_unix: float | None = None
        self.reads_total = 0
        self.reads_rejected = 0
        self.records_accepted = 0
        self.frames_served = 0
        # Set when a verified terminal lifecycle is first seen (grace timing).
        self.terminal_since_unix: float | None = None
        # Startup floor: anything already on disk is historical, never LIVE.
        self.startup_floor: tuple | None = None
        try:
            floor_snap = read_verified_record(self.run_dir)
        except (FileNotFoundError, MonitorProtocolError, OSError):
            floor_snap = None
        if floor_snap is not None:
            self.startup_floor = floor_snap.identity
            self.current = floor_snap
            self.historical = True
            self.records_accepted = 1
            if floor_snap.record["lifecycle"] in TERMINAL_LIFECYCLES:
                self.terminal_since_unix = time.time()

    def _is_historical(self, snap) -> bool:
        if self.startup_floor is None:
            return False
        floor_run, floor_seq, _ = self.startup_floor
        return snap.record["run_id"] == floor_run and snap.record["frame_seq"] <= floor_seq

    def poll_once(self, *, now_unix: float | None = None) -> str:
        """Re-read and re-verify the run directory once.

        Returns "accepted" (new identity), "unchanged" (same record;
        freshness untouched), "absent" (no record file) or "invalid".

        Within one run_id, frame_seq must strictly increase: a lower
        frame_seq is a regression and a changed record at the same frame_seq
        is a mutation — both are rejected as invalid and keep the last
        verified state. A new run_id resets sequence identity.
        """
        now = time.time() if now_unix is None else now_unix
        try:
            snap = read_verified_record(self.run_dir, now_unix=now)
        except FileNotFoundError:
            with self.cond:
                self.reads_total += 1
            return "absent"
        except (MonitorProtocolError, OSError) as exc:
            with self.cond:
                self.reads_total += 1
                self.reads_rejected += 1
                self.last_error = str(exc)
                self.last_error_unix = now
            return "invalid"
        with self.cond:
            self.reads_total += 1
            if self.current is not None:
                cur = self.current.record
                new = snap.record
                if new["run_id"] == cur["run_id"]:
                    if new["frame_seq"] < cur["frame_seq"]:
                        self.reads_rejected += 1
                        self.last_error = (
                            f"frame_seq regressed for run {new['run_id']!r}: "
                            f"{new['frame_seq']} < {cur['frame_seq']}"
                        )
                        self.last_error_unix = now
                        return "invalid"
                    if new["frame_seq"] == cur["frame_seq"]:
                        if new == cur and snap.frame_bytes == self.current.frame_bytes:
                            self.last_error = None
                            return "unchanged"  # metadata-only re-read: no freshness reset
                        self.reads_rejected += 1
                        self.last_error = (
                            f"record mutated at frame_seq {new['frame_seq']} "
                            f"for run {new['run_id']!r} without a new sequence"
                        )
                        self.last_error_unix = now
                        return "invalid"
            self.last_error = None
            terminal = snap.record["lifecycle"] in TERMINAL_LIFECYCLES
            if terminal and self.terminal_since_unix is None:
                self.terminal_since_unix = now  # grace clock starts at first terminal record
            elif not terminal:
                self.terminal_since_unix = None
            self.current = snap
            self.historical = self._is_historical(snap)
            self.records_accepted += 1
            self.cond.notify_all()
            return "accepted"

    def build_state(self, *, now_unix: float, server_running: bool, server_info: dict) -> dict:
        with self.cond:
            snap = self.current
            historical = self.historical
            last_error = self.last_error
            counters = {
                "reads_total": self.reads_total,
                "reads_rejected": self.reads_rejected,
                "records_accepted": self.records_accepted,
                "frames_served": self.frames_served,
                "last_error": last_error,
            }
        record = None if snap is None else snap.record
        heartbeat_age = None
        frame_age = None
        if record is not None:
            heartbeat_age = max(0.0, now_unix - record["producer"]["heartbeat_unix"])
            if record["frame"] is not None:
                frame_age = max(0.0, now_unix - record["frame"]["rendered_unix"])
        display = derive_display(
            record,
            heartbeat_age_s=heartbeat_age,
            frame_age_s=frame_age,
            stale_after_s=self.stale_after_s,
            historical=historical,
        )
        if not server_running:
            display = dict(display, status="STOPPED", live=False, training_active=False,
                           notice_zh="查看服务已停止")
        elif record is None and last_error is not None:
            display = dict(display, status="ERROR", live=False,
                           notice_zh="来源数据无效，已关闭显示：" + last_error[:120])
        state = {
            "schema": "training-monitor-view/1",
            "server": server_info,
            "status": display["status"],
            "activity": display["activity"],
            "training_active": display["training_active"],
            "training_declared": display["training_declared"],
            "live": display["live"],
            "source_badge": display["source_badge"],
            "notice_zh": display["notice_zh"],
            "label_basis": LABEL_BASIS,
            "historical": historical,
            "heartbeat_age_s": heartbeat_age,
            "frame_age_s": frame_age,
            "stale_after_s": self.stale_after_s,
            "record": record,
            "frame": None
            if record is None or record["frame"] is None
            else {
                "frame_seq": record["frame_seq"],
                "sha256": record["frame"]["sha256"],
                "width": record["frame"]["width"],
                "height": record["frame"]["height"],
                "rendered_unix": record["frame"]["rendered_unix"],
                "age_s": frame_age,
            },
            "counters": counters,
            "scope": list(SCOPE_LABELS),
        }
        return state

    def get_frame(self) -> tuple[bytes, dict] | None:
        """Current verified frame bytes + headers, or None when no video."""
        with self.cond:
            snap = self.current
            if snap is None or snap.frame_bytes is None:
                return None
            record = snap.record
            self.frames_served += 1
            return snap.frame_bytes, {
                "X-Run-Id": record["run_id"],
                "X-Frame-Seq": str(record["frame_seq"]),
                "X-Frame-Sha256": record["frame"]["sha256"],
                "X-Frame-Rendered-Unix": f"{record['frame']['rendered_unix']:.3f}",
                "X-Sim-Time-S": f"{record['sim_time_s']:.3f}",
                "X-Source-Kind": record["source_kind"],
                "X-Lifecycle": record["lifecycle"],
            }


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>训练监看 (training monitor, observer)</title>
<style>
 body { font-family: -apple-system, Helvetica, Arial, sans-serif; background:#f4f6f8;
        color:#1c2733; margin:0; padding:16px; }
 h1 { font-size:18px; margin:0 0 4px; }
 .chips span { display:inline-block; background:#e3ecf5; border:1px solid #b9cbdd;
        border-radius:10px; padding:2px 8px; margin:2px 4px 2px 0; font-size:12px; }
 #status { font-weight:700; padding:4px 10px; border-radius:6px; color:#fff; }
 .LIVE { background:#1e8e3e; } .STALLED { background:#b26a00; }
 .STOPPED, .FAILED, .ERROR { background:#c5221f; }
 .WAITING, .REPLAY, .ENDED { background:#5b6b7b; }
 .TEST, .DEMO { background:#7b1fa2; }
 #badge { font-weight:700; padding:4px 10px; border-radius:6px; color:#fff;
        background:#7b1fa2; display:none; }
 #notice { margin:6px 0; font-size:13px; color:#7a3c00; font-weight:600; }
 #view { background:#dfe5ec; border:1px solid #9fb2c3; margin-top:8px;
        width:min(640px,100%); height:auto; max-height:480px; object-fit:contain;
        display:none; box-sizing:border-box; }  /* hidden until first decoded frame */
 table { border-collapse:collapse; margin-top:8px; font-size:13px; max-width:100%;
        overflow-wrap:anywhere; }
 td { padding:2px 10px 2px 0; font-variant-numeric:tabular-nums; overflow-wrap:anywhere;
        word-break:break-word; }
 td.k { color:#5b6b7b; white-space:nowrap; }
 #note { font-size:12px; color:#5b6b7b; margin-top:8px; max-width:100%; overflow-wrap:anywhere; }
</style>
</head>
<body>
<h1>训练监看 <span id="status" class="STOPPED">CONNECTING</span>
 <span id="badge"></span></h1>
<div class="chips">
 <span>__SCOPE_0__</span><span>__SCOPE_1__</span><span>__SCOPE_2__</span>
</div>
<div id="notice"></div>
<img id="view" width="640" height="480" alt="monitor frame">
<table>
 <tr><td class="k">run</td><td id="run">-</td><td class="k">deadline in</td><td id="deadline">-</td></tr>
 <tr><td class="k">source</td><td id="source">-</td><td class="k">phase / activity</td><td id="phase">-</td></tr>
 <tr><td class="k">train step</td><td id="tstep">-</td><td class="k">gradient updates</td><td id="grad">-</td></tr>
 <tr><td class="k">checkpoint</td><td id="ckpt">-</td><td class="k">policy</td><td id="policy">-</td></tr>
 <tr><td class="k">world / envs</td><td id="world">-</td><td class="k">loss / reward</td><td id="metrics">-</td></tr>
 <tr><td class="k">sim time</td><td id="simt">-</td><td class="k">physics step</td><td id="pstep">-</td></tr>
 <tr><td class="k">displayed frame</td><td id="frame">-</td><td class="k">displayed frame age</td><td id="age">-</td></tr>
 <tr><td class="k">source frame seq</td><td id="fseq">-</td><td class="k">heartbeat age</td><td id="hb">-</td></tr>
 <tr><td class="k">records accepted</td><td id="racc">-</td><td class="k">reads rejected</td><td id="rrej">-</td></tr>
</table>
<div id="note">状态图例：LIVE 实时（真实来源的新鲜画面：训练或检查点评估；夹具/演示永远不是 LIVE）/ STALLED 停滞 / WAITING 等待来源 /
REPLAY 历史画面（非当前）/ ENDED 已结束 / FAILED 已失败 / TEST 测试影像（测试夹具，非真实训练）/ DEMO 脚本演示（非学习策略，非训练）。
本页面只读取训练器/渲染器发布到运行目录的最新帧与记录
（schema + 计数器 + PNG SHA-256 内容哈希逐次校验，无效即关闭显示）。只有新解码并实际显示的帧才推进浏览器端新鲜度；
独立看门狗在超时未显示新帧时降级为 STALLED（即使元数据仍在应答），服务不可达时为 STOPPED，并自动重试。
启动前已存在的画面标为 REPLAY（历史），lifecycle 结束/失败后保留最终画面并标为 ENDED/FAILED，绝不显示为 LIVE。
来源为 test_fixture / scripted_diagnostic 时始终显示 TEST / DEMO 徽章，不能呈现为真实训练。
所有阶段标签均为生产者上报，未经独立验收；本观察端不仿真、不产生动作。</div>
<script>
const STALE_AFTER_MS = __STALE_MS__;  // server-provided staleness threshold
const POLL_MS = 250;                  // frame poll period (one in flight max)
const FETCH_TIMEOUT_MS = 2000;        // bounded per-request timeout
const view = document.getElementById('view');
const statusEl = document.getElementById('status');
const badgeEl = document.getElementById('badge');
const TRANSPORT = new Set(['LIVE', 'STALLED']);  // client may only downgrade these
let serverStatus = 'STOPPED';
let displayedIdentity = '';           // run_id + '#' + frame_seq of the shown frame
let displayedSha = '';
let lastDisplayedWallMs = 0;          // set ONLY when a new frame is decoded+shown
let currentBlobUrl = null;
let inFlight = false;                 // no overlapping /frame.png polls
let stateInFlight = false;            // no overlapping /state.json polls
let consecutiveFailures = 0;

function setStatus(s) { statusEl.textContent = s; statusEl.className = s; }
function setText(id, v) { document.getElementById(id).textContent = v; }

async function pollFrame() {
  if (inFlight) return;
  inFlight = true;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const r = await fetch('/frame.png', {cache: 'no-store', signal: ctrl.signal});
    if (r.status === 503) { consecutiveFailures = 0; return; }  // no video: not an error
    if (!r.ok) throw new Error('http ' + r.status);
    const runId = r.headers.get('X-Run-Id') || '';
    const seq = r.headers.get('X-Frame-Seq') || '-';
    const sha = r.headers.get('X-Frame-Sha256') || '';
    const simt = parseFloat(r.headers.get('X-Sim-Time-S') || 'NaN');
    const identity = runId + '#' + seq;  // identity is run+seq, not pixels: a
    const blob = await r.blob();         // re-rendered identical image still refreshes
    if (identity !== displayedIdentity) {
      const url = URL.createObjectURL(blob);
      const probe = new Image();
      probe.src = url;
      await probe.decode();           // fully decoded BEFORE it may be displayed
      if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
      currentBlobUrl = url;
      view.src = url;                 // only now does freshness advance
      view.style.display = 'block';   // reveal only after a real decoded frame
      displayedIdentity = identity;
      displayedSha = sha;
      lastDisplayedWallMs = Date.now();
      setText('frame', seq + ' / ' + sha.slice(0, 12));
      if (Number.isFinite(simt)) setText('simt', simt.toFixed(2) + ' s');
    }
    consecutiveFailures = 0;
  } catch (e) {
    consecutiveFailures += 1;
  } finally {
    clearTimeout(timer);
    inFlight = false;
  }
}

async function pollState() {          // metadata only; never marks LIVE by itself
  if (stateInFlight) return;          // bounded: one in flight, abort on timeout
  stateInFlight = true;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
  try {
    const r = await fetch('/state.json', {cache: 'no-store', signal: ctrl.signal});
    if (!r.ok) throw new Error('http ' + r.status);
    const s = await r.json();
    serverStatus = s.status;
    const rec = s.record;
    setText('run', rec ? rec.run_id : '-');
    setText('deadline', Math.max(0, s.server.remaining_wall_s).toFixed(0) + ' s');
    setText('source', rec ? rec.source_kind : '-');
    setText('phase', rec ? rec.lifecycle + ' / ' + s.activity : s.activity);
    setText('tstep', rec && rec.train_step != null ? rec.train_step : 'null');
    setText('grad', rec && rec.gradient_updates != null ? rec.gradient_updates : 'null');
    setText('ckpt', rec && rec.checkpoint ? rec.checkpoint.version + ' / ' +
            rec.checkpoint.sha256.slice(0, 12) : '-');
    setText('policy', rec ? rec.policy_kind : '-');
    setText('world', rec ? rec.world_id + ' / ' + rec.environment_count : '-');
    const m = rec && rec.metrics ? rec.metrics : null;
    setText('metrics', m ? ('loss=' + (m.loss == null ? 'null' : m.loss) +
            ' reward=' + (m.reward == null ? 'null' : m.reward)) : '-');
    setText('pstep', rec ? rec.physics_step : '-');
    setText('fseq', s.frame ? s.frame.frame_seq : '-');
    setText('hb', s.heartbeat_age_s == null ? '-' : s.heartbeat_age_s.toFixed(1) + ' s');
    setText('racc', s.counters.records_accepted);
    setText('rrej', s.counters.reads_rejected);
    document.getElementById('notice').textContent = s.notice_zh || '';
    if (s.source_badge) { badgeEl.textContent = s.source_badge; badgeEl.style.display = 'inline-block'; }
    else { badgeEl.style.display = 'none'; }
    if (s.status === 'STOPPED') consecutiveFailures = Math.max(consecutiveFailures, 3);
  } catch (e) { /* frame loop + watchdog own the status */ }
  finally { clearTimeout(timer); stateInFlight = false; }
}

function watchdog() {                 // independent of any request state
  const ageMs = lastDisplayedWallMs ? Date.now() - lastDisplayedWallMs : Infinity;
  setText('age', ageMs === Infinity ? '-' : (ageMs / 1000).toFixed(2) + ' s');
  if (consecutiveFailures >= 3) {
    setStatus('STOPPED');
  } else if (serverStatus === 'TEST' || serverStatus === 'DEMO') {
    // decoded-frame freshness applies to fixtures too: badge retained, never LIVE
    setStatus((ageMs > STALE_AFTER_MS || !displayedIdentity) ? 'STALLED' : serverStatus);
  } else if (!TRANSPORT.has(serverStatus)) {
    setStatus(serverStatus);          // WAITING / REPLAY / ENDED / FAILED / ERROR verbatim
  } else if (ageMs > STALE_AFTER_MS || !displayedIdentity) {
    setStatus('STALLED');
  } else if (serverStatus === 'LIVE') {
    setStatus('LIVE');
  } else {
    setStatus('STALLED');
  }
}

setInterval(pollFrame, POLL_MS);      // retries automatically after any failure
setInterval(pollState, 1000);
setInterval(watchdog, 500);
pollFrame(); pollState(); watchdog();
</script>
</body>
</html>
"""


def render_index(stale_after_s: float = DEFAULT_STALE_AFTER_S) -> bytes:
    html = INDEX_HTML.replace("__STALE_MS__", str(int(stale_after_s * 1000)))
    for i, label in enumerate(SCOPE_LABELS):
        html = html.replace(f"__SCOPE_{i}__", label)
    return html.encode("utf-8")


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, monitor: RunMonitor, server_info: dict):
        super().__init__(addr, handler)
        self.monitor = monitor
        self.server_info = server_info
        self.index_bytes = render_index(monitor.stale_after_s)


class Handler(BaseHTTPRequestHandler):
    server: MonitorServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):  # quiet; server.json/result.json record lifecycle
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
            info = dict(self.server.server_info)
            info["remaining_wall_s"] = max(0.0, info["deadline_unix"] - time.time())
            info.pop("deadline_unix", None)
            state = self.server.monitor.build_state(
                now_unix=time.time(), server_running=True, server_info=info
            )
            self._send(HTTPStatus.OK, json.dumps(state).encode(), "application/json")
        elif path == "/frame.png":
            result = self.server.monitor.get_frame()
            if result is None:
                self._send(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    "no source frame (waiting / no video)",
                    "text/plain; charset=utf-8",
                )
                return
            payload, headers = result
            self._send(HTTPStatus.OK, payload, "image/png", headers)
        else:
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

    def _method_not_allowed(self) -> None:
        self._send(
            HTTPStatus.METHOD_NOT_ALLOWED,
            b"GET-only interface",
            "text/plain",
            {"Allow": "GET"},
        )

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed


def bind_server(args, monitor: RunMonitor, server_info: dict) -> MonitorServer:
    last_err: OSError | None = None
    for port in range(args.port, args.port + PORT_ATTEMPTS):
        try:
            srv = MonitorServer((BIND_HOST, port), Handler, monitor, server_info)
            server_info["port"] = port
            server_info["url"] = f"http://{BIND_HOST}:{port}/"
            server_info["bind"] = f"{BIND_HOST}:{port}"
            return srv
        except OSError as exc:  # port busy: try the next one, never kill anyone
            last_err = exc
    raise RuntimeError(
        f"no free port in {args.port}..{args.port + PORT_ATTEMPTS - 1} on {BIND_HOST}"
    ) from last_err


def run(args) -> int:
    started_wall = time.time()
    deadline_unix = started_wall + args.max_wall_s
    monitor = RunMonitor(args.run_dir, stale_after_s=args.stale_after_s)
    server_info = {
        "pid": os.getpid(),
        "bind": None,
        "url": None,
        "port": None,
        "started_utc": utc_now_iso(),
        "deadline_utc": datetime.fromtimestamp(deadline_unix, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "deadline_unix": deadline_unix,
        "max_wall_s": args.max_wall_s,
    }
    server = bind_server(args, monitor, server_info)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tool_path = Path(__file__).resolve()
    identity = {
        "pid": os.getpid(),
        "bind_host": BIND_HOST,
        "port": server_info["port"],
        "url": server_info["url"],
        "started_utc": server_info["started_utc"],
        "deadline_utc": server_info["deadline_utc"],
        "max_wall_s": args.max_wall_s,
        "terminal_grace_s": TERMINAL_GRACE_S,
        "stale_after_s": args.stale_after_s,
        "poll_interval_s": args.poll_interval_s,
        "run_dir": str(Path(args.run_dir).resolve()),
        "startup_floor": None
        if monitor.startup_floor is None
        else {
            "run_id": monitor.startup_floor[0],
            "frame_seq": monitor.startup_floor[1],
            "frame_sha256": monitor.startup_floor[2],
        },
        "python": sys.executable,
        "tool": tool_path.name,
        "tool_sha256": hashlib.sha256(tool_path.read_bytes()).hexdigest(),
        "scope": list(SCOPE_LABELS),
    }
    (out_dir / "server.json").write_text(json.dumps(identity, indent=2) + "\n")
    (out_dir / "viewer.pid").write_text(f"{os.getpid()}\n")

    stop = threading.Event()

    def on_signal(signum, frame):  # noqa: ARG001
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def poll_target() -> None:
        while not stop.is_set() and time.time() < deadline_unix:
            try:
                monitor.poll_once()
            except Exception:
                traceback.print_exc()
            stop.wait(timeout=args.poll_interval_s)

    poller = threading.Thread(target=poll_target, name="run-dir-poller", daemon=True)
    poller.start()

    final_reason_box = ["deadline_expired"]

    def expiry_target() -> None:
        # Deadline-based: shuts the server down even if the poller hangs;
        # a terminal producer state stops it after the final-frame grace.
        while not stop.is_set():
            r = stop_reason(time.time(), deadline_unix, monitor.terminal_since_unix)
            if r is not None:
                final_reason_box[0] = r
                break
            time.sleep(0.2)
        server.shutdown()

    expiry = threading.Thread(target=expiry_target, name="expiry-monitor", daemon=True)
    expiry.start()
    print(
        f"[training-monitor] serving {server_info['url']} "
        f"run_dir={Path(args.run_dir).resolve()} pid={os.getpid()} "
        f"deadline_utc={server_info['deadline_utc']}",
        flush=True,
    )
    final_reason = "deadline_expired"
    try:
        server.serve_forever(poll_interval=0.2)
    except Exception:
        final_reason = "error"
        raise
    finally:
        final_reason = select_stop_reason(
            final_reason, stop.is_set(), final_reason_box[0]
        )
        stop.set()
        poller.join(timeout=5.0)
        server.server_close()  # releases the port
        final = monitor.build_state(
            now_unix=time.time(), server_running=False,
            server_info={k: v for k, v in server_info.items() if k != "deadline_unix"},
        )
        result = {
            "pid": os.getpid(),
            "port": server_info["port"],
            "url": server_info["url"],
            "started_utc": server_info["started_utc"],
            "deadline_utc": server_info["deadline_utc"],
            "stopped_utc": utc_now_iso(),
            "stop_reason": final_reason,
            "max_wall_s": args.max_wall_s,
            "terminal_grace_s": TERMINAL_GRACE_S,
            "run_dir": str(Path(args.run_dir).resolve()),
            "startup_floor": identity["startup_floor"],
            "counters": final["counters"],
            "final_status": final["status"],
            "final_activity": final["activity"],
            "final_run_id": None
            if final["record"] is None
            else final["record"]["run_id"],
            "final_lifecycle": None
            if final["record"] is None
            else final["record"]["lifecycle"],
            "final_frame_seq": None
            if final["record"] is None
            else final["record"]["frame_seq"],
            "scope": list(SCOPE_LABELS),
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            f"[training-monitor] stopped reason={final_reason} "
            f"counters={final['counters']}",
            flush=True,
        )
    return 0


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Hard bounds: max-wall-s finite in (0, 86400] (default 600; an explicit
    declared lifetime must match the authorized job budget), stale-after in
    (0, 120]."""
    if not math.isfinite(args.max_wall_s) or not (0 < args.max_wall_s <= MAX_WALL_S_CAP):
        parser.error(f"--max-wall-s must be finite and in (0, {MAX_WALL_S_CAP:g}]")
    if not math.isfinite(args.stale_after_s) or not (0 < args.stale_after_s <= MAX_STALE_AFTER_S):
        parser.error(f"--stale-after-s must be finite and in (0, {MAX_STALE_AFTER_S:g}]")
    if not math.isfinite(args.poll_interval_s) or not (0.1 <= args.poll_interval_s <= 10.0):
        parser.error("--poll-interval-s must be finite and in [0.1, 10]")
    if not (1024 <= args.port <= 65535):
        parser.error("--port must be in [1024, 65535]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="run directory published via MonitorPublisher")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="first port to try on 127.0.0.1")
    parser.add_argument("--max-wall-s", type=float, default=DEFAULT_MAX_WALL_S,
                        help=f"declared finite lifetime (default {DEFAULT_MAX_WALL_S:g}, "
                             f"max {MAX_WALL_S_CAP:g}); must match the authorized job budget")
    parser.add_argument("--stale-after-s", type=float, default=DEFAULT_STALE_AFTER_S,
                        help=f"producer freshness threshold (<= {MAX_STALE_AFTER_S:g})")
    parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S,
                        help="run-directory re-read period")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(".local/training-monitor/server"))
    args = parser.parse_args(argv)
    validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
