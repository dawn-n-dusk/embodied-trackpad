"""Pure/mock tests for tools/leap_v1_near_probe.py.

No app launch, no real solver steps: the real planner/FrameGate/release logic
is exercised with SimpleNamespace fakes, stubbed step/capture callables and
fake clients. These tests claim NO closed-loop performance.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy", reason="numpy not installed")
pytest.importorskip("mujoco", reason="MuJoCo not installed")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import leap_v1_near_probe as probe  # noqa: E402
from embodied_trackpad.leap_v1 import visual_physics as vp  # noqa: E402
from embodied_trackpad.leap_v1.visual_servo import (  # noqa: E402
    ClockReconciler, FrameGate, FrameGateConfig, PixelEstimate, VisualEstimate,
)

FAKE_MAP = vp.NamedJointMap(vp.INDEX_JOINTS, (0, 1, 2, 3), (0, 1, 2, 3),
                            (0, 1, 2, 3), ((-2.0, 2.0),) * 4)
JAC = np.array([[0.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.0]])


def fake_env():
    return SimpleNamespace(
        model=SimpleNamespace(
            nu=16, nv=16, njnt=16,
            actuator_ctrlrange=np.tile(np.array([[-2.5, 2.5]]), (16, 1)),
            actuator=lambda i: SimpleNamespace(name=f"motor_{i}"),
            jnt_range=np.tile(np.array([[-2.0, 2.0]]), (16, 1)),
            jnt_qposadr=np.arange(16)),
        data=SimpleNamespace(ctrl=np.zeros(16), qpos=np.zeros(16),
                             qvel=np.zeros(16), time=1.0,
                             warning=[SimpleNamespace(number=0) for _ in range(8)]),
        scene=SimpleNamespace(pad_geom_id=0),
        index_map=FAKE_MAP,
        warn_counts=None)


def make_R(tmp_path, env=None, client=None):
    return {"t0": time.monotonic(), "out": tmp_path, "env": env, "client": client,
            "step_index": 0, "seq": 0, "cycle": 0, "ctrl": np.zeros(16),
            "z_hold": 0.1, "initial_error": None, "final_error": None,
            "errors": [], "success": False, "cleanup_records": [],
            "last_record": None,
            "teacher": open(tmp_path / "teacher.jsonl", "w"),
            "actor": open(tmp_path / "actor.jsonl", "w"),
            "planner": open(tmp_path / "planner.jsonl", "w"),
            "touch": open(tmp_path / "touch.jsonl", "w"),
            "reconciler": ClockReconciler(offset_s=0.0, uncertainty_s=0.001),
            "gate": FrameGate(FrameGateConfig(clock_uncertainty_ms=1.0))}


def close_R(R):
    for key in ("teacher", "actor", "planner", "touch"):
        if not R[key].closed:
            R[key].close()


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def make_est(ex, ey):
    cursor = PixelEstimate(True, 400.0, 300.0, 400.0, 300.0, 1.0, 200,
                           (390, 290, 410, 310))
    target = PixelEstimate(True, 400.0 + ex, 300.0 + ey, 400.0 + ex, 300.0 - ey,
                           1.0, 400, (390, 290, 410, 310))
    return VisualEstimate(cursor=cursor, target=target, image_shape=(800, 600))


def make_frame(fid, mono, tag=0):
    return {"frame_id": fid, "width": 800, "height": 600,
            "monotonic_time_s": mono, "simulation_time_s": 1.0,
            "png": b"fakepng" + bytes([tag]), "path": None}


def stub_loop(R, monkeypatch, captures):
    """Patch capture/step/send/jacobian for visual_phase; returns step recorder."""
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    steps = []

    def fake_step(R, ctrl, fid, phase):
        steps.append((fid, phase))
        return SimpleNamespace(aggregate={})

    monkeypatch.setattr(probe, "step", fake_step)
    return steps


def patch_run_env(monkeypatch, visual_reason="max_cycles_exhausted"):
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["last_record"] = {"aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "visual_phase", lambda R: visual_reason)


class FakeClient:
    def __init__(self, out_dir=None):
        self.proc = SimpleNamespace(pid=4242, returncode=0)
        self.stopped = False
        self.quit_called = False
        self._send = lambda obj, timeout=30.0: {"type": "touch_ok",
                                                "sequence": obj.get("sequence")}

    def reset(self, seed=1, task_profile=None):
        self.reset_args = (seed, task_profile)
        return {"type": "reset_ok"}

    def raw(self, payload):
        return {"type": "touch_ok", "sequence": payload["sequence"]}

    def capture(self, save_as=None):
        return {"frame_id": 1, "width": 800, "height": 600,
                "monotonic_time_s": 100.0, "simulation_time_s": 1.0,
                "png": b"fakepng-calib", "path": None}

    def stop(self):
        self.stopped = True
        return {"type": "stopped"}

    def quit(self, timeout=10.0):
        self.quit_called = True
        return 0


# --- 1. changed pixel error changes the actual full-16 planner ctrl ---------

def test_pixel_error_changes_full16_ctrl():
    ctrl = np.zeros(16)
    out_x, info_x = vp.plan_index_correction((40.0, 0.0), 0.0, JAC, ctrl,
                                             FAKE_MAP, probe.PLAN_CFG)
    out_y, _ = vp.plan_index_correction((0.0, 40.0), 0.0, JAC, ctrl,
                                        FAKE_MAP, probe.PLAN_CFG)
    out_x2, _ = vp.plan_index_correction((40.0, 0.0), 0.0, JAC, ctrl,
                                         FAKE_MAP, probe.PLAN_CFG)
    assert out_x.shape == (16,) and out_y.shape == (16,)
    assert not np.array_equal(out_x, out_y)  # different pixel error -> different plan
    assert np.array_equal(out_x, out_x2)     # deterministic for the same error
    assert np.array_equal(out_x[4:], ctrl[4:])  # only index channels are touched
    assert not np.array_equal(out_x[:4], ctrl[:4])
    assert abs(info_x["dq_rad"][0]) <= probe.PLAN_CFG.max_joint_delta_rad + 1e-12


# --- 2. stale/duplicate frame -> no visual stepping, rejection recorded -----

def test_stale_frame_produces_no_visual_step(tmp_path, monkeypatch):
    R = make_R(tmp_path, env=fake_env())
    captures = [(make_frame(7, 1000.00, 1), make_est(20.0, 0.0), 5.0),
                (make_frame(7, 1000.05, 2), make_est(20.0, 0.0), 5.0)]
    steps = stub_loop(R, monkeypatch, captures)
    reason = probe.visual_phase(R)
    close_R(R)
    assert reason.startswith("frame_gate_rejected") and "stale frame_id" in reason
    assert len(steps) == probe.VISUAL_SUBSTEPS  # only the first, accepted cycle
    actor = read_lines(tmp_path / "actor.jsonl")
    assert len(actor) == 2  # every capture recorded
    assert actor[0]["action_requested"] is True and actor[0]["next_ctrl"] is not None
    assert actor[1]["phase"] == "gate_rejected"
    assert actor[1]["action_requested"] is False and actor[1]["next_ctrl"] is None
    planner = read_lines(tmp_path / "planner.jsonl")
    assert any(p["kind"] == "gate_reject" for p in planner)


# --- 3. first unsafe physics step: logged exactly once, cleanup forbidden ---

def test_first_unsafe_step_logged_once(tmp_path):
    env = fake_env()
    R = make_R(tmp_path, env=env)
    with pytest.raises(vp.PhysicsStepError):
        probe.step(R, np.zeros(3), None, "visual")  # invalid ctrl shape
    close_R(R)
    lines = read_lines(tmp_path / "teacher.jsonl")
    assert len(lines) == 1
    assert lines[0]["ok"] is False and lines[0]["checks"]["ctrl_valid"] is False
    assert lines[0]["failure_reasons"] and lines[0]["phase"] == "visual"


def test_unsafe_physics_forbids_cleanup(tmp_path, monkeypatch):
    rec = vp.StepRecord(step_index=1, sim_time_s=0.002, frame_id=None,
                        ctrl=[0.0] * 16, qpos=[], qvel=[], contacts=[],
                        aggregate={}, warning_delta=[], checks={"ctrl_valid": False},
                        ok=False, failure_reasons=["ctrl_invalid: fake"])
    err = vp.PhysicsStepError(rec)
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)
    monkeypatch.setattr(probe, "prep_phase",
                        lambda R: (_ for _ in ()).throw(err))
    called = []
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: called.append(1) or True)
    summary = probe.run(1, tmp_path / "out")
    assert called == []  # no further physics after the first unsafe step
    assert summary["physics_unsafe"] is True
    assert summary["episode_safe_complete_released"] is False
    assert (tmp_path / "out" / "summary.json").exists()


# --- 4/5. run() always finalizes: cleanup failure and make_env failure ------

def test_cleanup_generic_failure_still_finalizes(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    patch_run_env(monkeypatch)

    def bad_cleanup(R):
        raise RuntimeError("boom")

    monkeypatch.setattr(probe, "cleanup_phase", bad_cleanup)
    summary = probe.run(3, tmp_path / "out")
    assert client.stopped and client.quit_called  # app always software-closed
    assert "cleanup_error:RuntimeError:boom" in summary["stop_reason"]
    assert summary["episode_safe_complete_released"] is False
    assert summary["release_measured"] is False
    assert summary["child_reaped"] is True and summary["child_returncode"] == 0
    on_disk = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert on_disk["stop_reason"] == summary["stop_reason"]
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["controller_kind"] == "privileged_visual_teacher"
    assert meta["training_eligible"] is False and meta["started_utc"]
    assert summary["completed_utc"] >= summary["started_utc"]


def test_make_env_failure_still_writes_summary(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("assets missing")

    monkeypatch.setattr(vp, "make_env", boom)
    called = []
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: called.append(1) or True)
    summary = probe.run(1, tmp_path / "out")
    assert called == []  # no env -> no cleanup physics
    assert summary["stop_reason"].startswith("error:RuntimeError")
    assert summary["child_pid"] is None and summary["child_reaped"] is False
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "metadata.json").exists()


# --- 6. terminal success frame is retained with null next_ctrl --------------

def test_terminal_capture_retained(tmp_path, monkeypatch):
    R = make_R(tmp_path, env=fake_env())
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), est, 5.0)
                for i, est in enumerate([make_est(20.0, 0.0), make_est(2.0, 0.0),
                                         make_est(2.0, 0.0), make_est(2.0, 0.0)], 1)]
    steps = stub_loop(R, monkeypatch, captures)
    reason = probe.visual_phase(R)
    close_R(R)
    assert reason == "success" and R["success"] is True
    assert len(steps) == 3 * probe.VISUAL_SUBSTEPS  # terminal frame moves nothing
    actor = read_lines(tmp_path / "actor.jsonl")
    assert len(actor) == 4  # every capture recorded incl. the terminal one
    last = actor[-1]
    assert last["phase"] == "success_terminal"
    assert last["next_ctrl"] is None and last["action_requested"] is False
    assert last["error_px"] == pytest.approx(2.0)


# --- 7. release window: span boundary + interrupted contact -----------------

def _recs(n, dt=0.002, t0=1.0, pad_at=None, self_at=None):
    out = []
    t = t0
    for i in range(n):
        contacts = []
        if pad_at is not None and i == pad_at:
            contacts.append({"kind": "tip_pad_top"})
        if self_at is not None and i == self_at:
            contacts.append({"kind": "hand_self"})
        out.append({"sim_time_s": t, "contacts": contacts})
        t += dt
    return out


def test_release_span_boundary():
    assert probe.release_from_records([]) is False
    assert probe.release_from_records(_recs(249)) is False  # spans 0.496 s
    assert probe.release_from_records(_recs(250)) is False  # spans 0.498 s
    assert probe.release_from_records(_recs(251)) is True   # spans 0.500 s
    assert probe.release_from_records(_recs(300)) is True
    # A pad contact inside the trailing window blocks the release...
    assert probe.release_from_records(_recs(300, pad_at=299)) is False
    assert probe.release_from_records(_recs(300, pad_at=100)) is False
    # ...but one before the window does not; hand_self never blocks.
    assert probe.release_from_records(_recs(300, pad_at=0)) is True
    assert probe.release_from_records(_recs(300, self_at=299)) is True


def test_release_rejects_gapped_or_nonfinite_times():
    recs = _recs(251)
    recs[100]["sim_time_s"] += 0.01  # non-contiguous gap
    assert probe.release_from_records(recs) is False
    recs = _recs(251)
    recs[0]["sim_time_s"] = "nan"  # sanitized string, not a finite float
    assert probe.release_from_records(recs) is False
    recs = _recs(251)
    recs[50]["sim_time_s"] = recs[49]["sim_time_s"]  # not strictly increasing
    assert probe.release_from_records(recs) is False


# --- 8. touch ack validation: payload+ack preserved, rejection stops --------

def test_touch_ack_rejection(tmp_path):
    class C:
        def __init__(self, ack):
            self.ack = ack
            self.calls = []

        def raw(self, payload):
            self.calls.append(payload)
            return self.ack

    bad = C({"type": "error", "error": "touch_rejected"})
    R = make_R(tmp_path, env=fake_env(), client=bad)
    with pytest.raises(probe.TouchRejected):
        probe.send_touch(R, {})
    close_R(R)
    lines = read_lines(tmp_path / "touch.jsonl")
    assert len(lines) == 1  # raw payload+ack preserved BEFORE the raise
    assert lines[0]["payload"]["sequence"] == 1
    assert lines[0]["payload"]["schema_version"] == "touch.v1"
    assert lines[0]["ack"] == {"type": "error", "error": "touch_rejected"}

    wrong_seq = C({"type": "touch_ok", "sequence": 99})
    R2 = make_R(tmp_path, env=fake_env(), client=wrong_seq)
    with pytest.raises(probe.TouchRejected):
        probe.send_touch(R2, {})
    close_R(R2)

    good = C({"type": "touch_ok", "sequence": 1})
    R3 = make_R(tmp_path, env=fake_env(), client=good)
    probe.send_touch(R3, {})  # matching touch_ok ack passes
    close_R(R3)


# --- 10. bounded teardown: stop _send capped at 1 s, quit(timeout=1.0) -------

class TimeoutCaptureClient:
    """Mock app client recording the actual _send/quit timeout arguments."""

    def __init__(self, out_dir=None):
        self.proc = SimpleNamespace(pid=777, returncode=0)
        self.send_calls = []  # (obj, timeout) as actually received
        self.quit_timeouts = []
        self.stopped = False

    def _send(self, obj, timeout=30.0):
        self.send_calls.append((obj, timeout))
        if obj.get("cmd") == "stop":
            return {"type": "stopped"}
        return {"type": "touch_ok", "sequence": obj.get("sequence")}

    def reset(self, seed=1, task_profile=None):
        return {"type": "reset_ok"}

    def raw(self, payload):
        return self._send(payload)

    def capture(self, save_as=None):
        return {"frame_id": 1, "width": 800, "height": 600,
                "monotonic_time_s": 100.0, "simulation_time_s": 1.0,
                "png": b"fakepng-calib", "path": None}

    def stop(self):
        self.stopped = True
        return self._send({"cmd": "stop"})

    def quit(self, timeout=10.0):
        self.quit_timeouts.append(timeout)
        return 0


def test_teardown_uses_bounded_timeouts(tmp_path, monkeypatch):
    client = TimeoutCaptureClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    patch_run_env(monkeypatch)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    summary = probe.run(5, tmp_path / "out")
    assert client.stopped
    stop_calls = [t for obj, t in client.send_calls if obj.get("cmd") == "stop"]
    assert stop_calls == [1.0]        # teardown _send capped, never the 30 s default
    assert client.quit_timeouts == [1.0]
    assert summary["teardown"]["send_timeout_s"] == 1.0
    assert summary["teardown"]["quit_timeout_s"] == 1.0
    assert summary["teardown"]["worst_case_s"] == 10.0
    assert summary["child_reaped"] is True
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["budgets"]["teardown_reserve_s"] == 10.0
    assert meta["budgets"]["teardown_bounds_s"]["quit_wait"] == 1.0


# --- 11. first-substep failure: planned frame stays intent, no completion ----

def test_first_step_failure_leaves_intent_only(tmp_path, monkeypatch):
    R = make_R(tmp_path, env=fake_env())
    rec = vp.StepRecord(step_index=1, sim_time_s=1.002, frame_id=1,
                        ctrl=[0.0] * 16, qpos=[], qvel=[], contacts=[],
                        aggregate={}, warning_delta=[],
                        checks={"finite_state": False}, ok=False,
                        failure_reasons=["non_finite_state"])
    err = vp.PhysicsStepError(rec)
    captures = [(make_frame(1, 1000.0, 1), make_est(20.0, 0.0), 5.0)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)

    def failing_step(R, ctrl, fid, phase):
        raise err

    monkeypatch.setattr(probe, "step", failing_step)
    with pytest.raises(vp.PhysicsStepError):
        probe.visual_phase(R)
    close_R(R)
    actor = read_lines(tmp_path / "actor.jsonl")
    assert len(actor) == 1
    line = actor[0]
    assert line["phase"] == "visual_step"
    assert line["action_requested"] is True   # planner INTENT, recorded pre-step
    assert line["next_ctrl"] is not None
    assert "action_executed" not in line      # no completion claim anywhere
    # No per-substep teacher StepRecord exists, so completion was never
    # established: intent without execution evidence.
    assert read_lines(tmp_path / "teacher.jsonl") == []


# --- 9. success is distinct from release; CLI exit needs both + reaping -----

def test_cli_exit_requires_success_release_and_reaping(tmp_path, monkeypatch):
    base = {"near_pointing_success": True,
            "episode_safe_complete_released": True, "child_reaped": True}
    calls = []
    monkeypatch.setattr(probe, "run",
                        lambda seed, out: calls.append((seed, out)) or dict(base))
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "a")]) == 0
    for field in base:
        monkeypatch.setattr(probe, "run",
                            lambda seed, out, f=field: {**base, f: False})
        assert probe.main(["--seed", "1", "--out",
                           str(tmp_path / f"b_{field}")]) == 1
    calls.clear()
    with pytest.raises(SystemExit):
        probe.main(["--seed", "-1", "--out", str(tmp_path / "neg")])
    assert calls == []  # negative seed rejected before env/app work
