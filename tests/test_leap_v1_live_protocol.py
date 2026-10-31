"""Protocol tests for tools/leap_v1_live_view.py (pure helpers, no MuJoCo).

These cover the viewer's freshness/status contract, state schema, HTML page
guarantees and the stdlib PNG encoder. They do NOT start the simulation or a
server; live behavior is verified separately and recorded under .local/.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")

TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "leap_v1_live_view.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("leap_v1_live_view", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def test_classify_frame_status_boundaries(tool):
    assert tool.classify_frame_status(0.0, True) == "LIVE"
    assert tool.classify_frame_status(1.99, True) == "LIVE"
    assert tool.classify_frame_status(2.0, True) == "LIVE"  # stale only when age EXCEEDS 2 s
    assert tool.classify_frame_status(2.01, True) == "STALLED"
    assert tool.classify_frame_status(30.0, True) == "STALLED"
    assert tool.classify_frame_status(None, True) == "STALLED"  # no frame yet is never LIVE
    assert tool.classify_frame_status(0.1, False) == "STOPPED"
    assert tool.classify_frame_status(None, False) == "STOPPED"


def _run_dict():
    return {
        "run_id": "test-run",
        "pid": 1234,
        "port": 8768,
        "started_utc": "2026-10-31T00:00:00Z",
        "deadline_utc": "2026-10-31T00:20:00Z",
        "deadline_unix": 1_000_000.0 + 1200.0,
        "timestep_s": 0.002,
        "embodiment_id": "leap_v1_full_right_menagerie_v1",
        "calibration_id": "uncalibrated_upstream_defaults_v1",
    }


def _frame_dict():
    return {
        "frame_id": 42,
        "rendered_unix": 1_000_100.0,
        "episode": 2,
        "episode_time_s": 3.5,
        "sim_time_total_s": 25.5,
        "phase": "sweep_mf_mcp_hold",
        "physics_steps": 12750,
        "measured_fps": 7.9,
    }


def test_build_state_schema_and_labels(tool):
    state = tool.build_state(run=_run_dict(), frame=_frame_dict(), server_running=True,
                             now_unix=1_000_100.5)
    for key in (
        "run_id", "pid", "bind", "started_utc", "deadline_utc", "remaining_lifetime_s",
        "status", "frame_id", "frame_rendered_unix", "frame_age_s", "stale_after_s",
        "episode", "sim_time_s", "sim_time_total_s", "phase", "physics_steps",
        "measured_fps", "target_fps", "resolution", "timestep_s",
        "embodiment_id", "calibration_id", "label", "scope", "frame_source",
    ):
        assert key in state, key
    assert state["label"] == "scripted physics; hand only; no learned policy"
    assert "newly rendered" in state["frame_source"]
    assert "prerecorded" in state["frame_source"]  # stated as absent
    assert state["status"] == "LIVE"
    assert state["frame_age_s"] == pytest.approx(0.5)
    assert state["stale_after_s"] == 2.0
    assert state["target_fps"] <= 8.0
    assert state["resolution"] == [640, 480]
    assert state["bind"] == "127.0.0.1:8768"


def test_build_state_stale_and_stopped(tool):
    stale = tool.build_state(run=_run_dict(), frame=_frame_dict(), server_running=True,
                             now_unix=1_000_103.0)
    assert stale["status"] == "STALLED"
    stopped = tool.build_state(run=_run_dict(), frame=_frame_dict(), server_running=False,
                               now_unix=1_000_100.5)
    assert stopped["status"] == "STOPPED"
    noframe = tool.build_state(run=_run_dict(), frame=None, server_running=True,
                               now_unix=1_000_100.5)
    assert noframe["status"] == "STALLED"
    assert noframe["frame_id"] is None


def test_index_html_contract(tool):
    html = tool.render_index().decode()
    assert "scripted physics; hand only; no learned policy" in html
    assert "STALLED" in html and "STOPPED" in html and "LIVE" in html
    assert "STALE_AFTER_MS = 2000" in html  # client-side 2 s freshness threshold
    assert "/state.json" in html and "/frame.png" in html
    # Display transport: fetch + blob decode, freshness only from displayed frames
    assert "AbortController" in html          # bounded per-request timeout
    assert "createObjectURL" in html and "revokeObjectURL" in html
    assert "probe.decode()" in html           # decoded BEFORE display
    assert "lastDisplayedWallMs" in html      # watchdog keyed to displayed frames
    assert "inFlight" in html                 # no overlapping frame polls
    assert 'src="/stream"' not in html        # no passive MJPEG display
    assert "No prerecorded frames are ever labeled live" in html
    assert "__MODE_LABEL__" not in html       # all placeholders substituted


def test_validate_args_bounds(tool, capsys):
    import argparse

    def check(*argv):
        parser = argparse.ArgumentParser()
        parser.add_argument("--lifetime-s", type=float, default=1200.0)
        parser.add_argument("--fps", type=float, default=8.0)
        args = parser.parse_args(list(argv))
        tool.validate_args(parser, args)

    check()  # defaults are valid
    check("--lifetime-s", "1200", "--fps", "8")
    for bad in ("0", "-1", "1200.01", "nan", "inf"):
        with pytest.raises(SystemExit):
            check("--lifetime-s", bad)
    for bad in ("0", "-0.5", "8.01", "nan", "inf"):
        with pytest.raises(SystemExit):
            check("--fps", bad)
    capsys.readouterr()  # drop parser.error messages


def test_png_bytes_roundtrip(tool):
    rgb = np.zeros((4, 3, 3), dtype=np.uint8)
    rgb[0, 0] = [255, 0, 0]
    rgb[3, 2] = [0, 128, 255]
    data = tool.png_bytes(rgb)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    # IHDR is the first chunk: length 13, width 3, height 4, 8-bit truecolor
    assert data[12:16] == b"IHDR"
    import struct

    w, h, bitdepth, color = struct.unpack(">IIBB", data[16:26])
    assert (w, h, bitdepth, color) == (3, 4, 8, 2)
    assert data.endswith(b"IEND\xaeB\x60\x82")


def test_script_ctrl_at_ramp_and_hold(tool):
    open_vec = np.zeros(2)
    target = np.array([1.0, 2.0])
    phases = [
        SimpleNamespace(name="hold_open", t0=0.0, t1=1.0, kind="hold", base=open_vec),
        SimpleNamespace(name="ramp_up", t0=1.0, t1=3.0, kind="ramp",
                        base=open_vec, target=target),
        SimpleNamespace(name="hold_target", t0=3.0, t1=4.0, kind="hold", base=target),
    ]
    v, ph = tool.script_ctrl_at(phases, 0.5)
    assert ph.name == "hold_open" and np.allclose(v, open_vec)
    v, ph = tool.script_ctrl_at(phases, 2.0)
    assert ph.name == "ramp_up" and np.allclose(v, [0.5, 1.0])
    v, ph = tool.script_ctrl_at(phases, 3.5)
    assert ph.name == "hold_target" and np.allclose(v, target)
    v, ph = tool.script_ctrl_at(phases, 99.0)  # past the end: clamps to final phase
    assert ph.name == "hold_target" and np.allclose(v, target)


def test_constants_match_task_contract(tool):
    assert tool.BIND_HOST == "127.0.0.1"
    assert tool.MAX_FPS <= 8.0
    assert tool.STALE_AFTER_S == 2.0
    assert tool.DEFAULT_LIFETIME_S == 1200
    assert tool.DEFAULT_PORT == 8768
    assert (tool.WIDTH, tool.HEIGHT) == (640, 480)
