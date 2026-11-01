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
from embodied_trackpad.leap_v1 import force_governor as fgv  # noqa: E402
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
            nu=16, nv=16, njnt=16, ngeom=16,
            actuator_ctrlrange=np.tile(np.array([[-2.5, 2.5]]), (16, 1)),
            actuator=lambda i: SimpleNamespace(name=f"motor_{i}"),
            jnt_range=np.tile(np.array([[-2.0, 2.0]]), (16, 1)),
            jnt_qposadr=np.arange(16),
            geom_bodyid=np.arange(16)),
        data=SimpleNamespace(ctrl=np.zeros(16), qpos=np.zeros(16),
                             qvel=np.zeros(16), time=1.0,
                             warning=[SimpleNamespace(number=0) for _ in range(8)]),
        scene=SimpleNamespace(pad_geom_id=0,
                              tip_geom_ids={"index": [10, 11]}),
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
                        lambda seed, out, planner_kind="dls",
                        task_profile="near_pointing_v1", contact_mode="standard":
                        calls.append((seed, out, task_profile, contact_mode)) or dict(base))
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "a")]) == 0
    for field in base:
        monkeypatch.setattr(probe, "run",
                            lambda seed, out, planner_kind="dls",
                            task_profile="near_pointing_v1",
                            contact_mode="standard", f=field:
                            {**base, f: False})
        assert probe.main(["--seed", "1", "--out",
                           str(tmp_path / f"b_{field}")]) == 1
    calls.clear()
    with pytest.raises(SystemExit):
        probe.main(["--seed", "-1", "--out", str(tmp_path / "neg")])
    assert calls == []  # negative seed rejected before env/app work


# --- 12. planner selection: default dls preserved, normal-priority wired -----

def test_default_planner_kind_is_dls(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    patch_run_env(monkeypatch)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    summary = probe.run(1, tmp_path / "out")
    assert summary["planner_kind"] == "dls"
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["planner"]["planner_kind"] == "dls"


def test_invalid_planner_kind_rejected_before_output(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="planner_kind"):
        probe.run(1, tmp_path / "nope", planner_kind="bogus")
    assert not (tmp_path / "nope").exists()  # no output/app work happened
    calls = []
    monkeypatch.setattr(probe, "run",
                        lambda seed, out, planner_kind="dls",
                        task_profile="near_pointing_v1": calls.append(1) or {})
    with pytest.raises(SystemExit):
        probe.main(["--seed", "1", "--out", str(tmp_path / "c"), "--planner", "bogus"])
    assert calls == [] and not (tmp_path / "c").exists()


def test_cli_planner_option_passed_through(tmp_path, monkeypatch):
    base = {"near_pointing_success": True,
            "episode_safe_complete_released": True, "child_reaped": True}
    seen = []
    monkeypatch.setattr(probe, "run",
                        lambda seed, out, planner_kind="dls",
                        task_profile="near_pointing_v1", contact_mode="standard":
                        seen.append(planner_kind) or dict(base))
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "a"),
                       "--planner", "normal-priority"]) == 0
    assert seen == ["normal-priority"]
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "b")]) == 0
    assert seen == ["normal-priority", "dls"]


def test_normal_priority_calls_new_planner_and_logs(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["z_hold"] = 0.1
        R["last_record"] = {"aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), est, 5.0)
                for i, est in enumerate([make_est(20.0, 0.0), make_est(2.0, 0.0),
                                         make_est(2.0, 0.0), make_est(2.0, 0.0)], 1)]
    spy = []
    real_plan = probe.npp.plan_normal_priority

    def wrapped(*a, **kw):
        out, info = real_plan(*a, **kw)
        spy.append(info)
        return out, info

    monkeypatch.setattr(probe.npp, "plan_normal_priority", wrapped)
    steps, sends = [], []
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: sends.append(agg))
    monkeypatch.setattr(probe, "step",
                        lambda R, ctrl, fid, phase: steps.append(1) or
                        SimpleNamespace(aggregate={"index": {"normal_force_n": 0.5}}))
    summary = probe.run(1, tmp_path / "out", planner_kind="normal-priority")
    assert len(spy) == 3  # terminal success frame requests no action
    assert len(steps) == 3 * probe.VISUAL_SUBSTEPS
    assert summary["planner_kind"] == "normal-priority"
    assert summary["near_pointing_success"] is True
    plans = [p for p in read_lines(tmp_path / "out" / "planner.jsonl")
             if p["kind"] == "plan"]
    assert plans and all(p["planner_kind"] == "normal-priority" for p in plans)
    for p in plans:
        assert {"requested_task", "predicted_task", "common_scale", "dq",
                "jac_3x4", "frame_id"} <= set(p)
        assert p["predicted_task"][2] == pytest.approx(p["requested_task"][2])
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["planner"]["planner_kind"] == "normal-priority"


def test_contact_loss_guard_stops_midcycle_normal_priority_only(tmp_path, monkeypatch):
    R = make_R(tmp_path, env=fake_env())
    R["planner_kind"] = "normal-priority"
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), make_est(20.0, 0.0), 5.0)
                for i in range(1, 8)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    sends = []
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: sends.append(agg))
    steps = []

    def fake_step(R, ctrl, fid, phase):
        steps.append(1)
        R["step_index"] += 1  # mirror the real step() bookkeeping
        # valid index contact for the first 10 substeps, then contact lost
        agg = {"index": {"normal_force_n": 0.5}} if len(steps) <= 10 else {}
        return SimpleNamespace(aggregate=agg)

    monkeypatch.setattr(probe, "step", fake_step)
    reason = probe.visual_phase(R)
    close_R(R)
    assert reason == "contact_loss"
    # 10 valid (guard reset) + 50 lost x 2 ms = 0.1 s -> stop at substep 60,
    # i.e. MID cycle 3 (cycle spans 51..75); no further visual step ran.
    assert len(steps) == 60
    assert len(sends) == 2  # cycles 1-2 sent; cycle 3 interrupted before send
    events = read_lines(tmp_path / "planner.jsonl")
    stop = [e for e in events if e["kind"] == "guard_stop"]
    assert len(stop) == 1
    assert stop[0]["cycle"] == 3 and stop[0]["step_index"] == 60
    assert stop[0]["lost_s"] == pytest.approx(0.1)
    assert "pre-action" in stop[0]["final_error_scope"]


def test_dls_path_has_no_guard(tmp_path, monkeypatch):
    R = make_R(tmp_path, env=fake_env())  # planner_kind defaults to dls
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i % 251), make_est(20.0, 0.0), 5.0)
                for i in range(1, probe.MAX_CYCLES + 2)]
    steps = stub_loop(R, monkeypatch, captures)  # aggregate always {}
    reason = probe.visual_phase(R)
    close_R(R)
    assert reason == "max_cycles_exhausted"  # empty aggregate never triggers dls
    assert len(steps) == probe.MAX_CYCLES * probe.VISUAL_SUBSTEPS
    assert not any(e["kind"] == "guard_stop"
                   for e in read_lines(tmp_path / "planner.jsonl"))


# --- force-admittance mode mock checks ---------------------------------------


def test_force_admittance_dispatches_helper_and_uses_corrected_ctrl(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["z_hold"] = 0.1
        R["last_record"] = {"step_index": 0, "sim_time_s": 1.0, "contacts": [], "aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), est, 5.0)
                for i, est in enumerate([make_est(20.0, 0.0), make_est(2.0, 0.0),
                                         make_est(2.0, 0.0), make_est(2.0, 0.0)], 1)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    force_calls = []
    update_calls = []

    def fake_update(env, last_record):
        update_calls.append(last_record.get("step_index"))
        return np.array([1.0, 0.0, 0.0, 0.0]), 1.0, [0.07, 0.0, 0.1]

    monkeypatch.setattr(probe, "force_mode_contact_update", fake_update)

    def fake_admittance(ctrl, anchor, row, force, dt, mapping, cfg):
        force_calls.append((ctrl[0], force))
        corrected = np.array(ctrl, dtype=float)
        corrected[0] += 0.0001  # deterministic correction
        info = {"force_error_n": 0.2, "requested_normal_speed_m_s": 0.0005,
                "requested_dz_m": 1e-6, "applied_dq": [1e-6, 0, 0, 0],
                "predicted_normal_dz_m": 1e-6, "common_scale": 1.0,
                "bound_limited": [], "teacher_only": True}
        return corrected, info

    monkeypatch.setattr(probe.fsv, "normal_force_admittance", fake_admittance)
    stepped_ctrls = []

    def fake_step(R, ctrl, fid, phase):
        R["step_index"] += 1
        R["last_record"] = {"step_index": R["step_index"], "sim_time_s": 1.0 + 0.002 * R["step_index"],
                            "contacts": [], "aggregate": {"index": {"normal_force_n": 1.0}}}
        stepped_ctrls.append(np.array(ctrl, dtype=float))
        return SimpleNamespace(aggregate=R["last_record"]["aggregate"])

    monkeypatch.setattr(probe, "step", fake_step)
    summary = probe.run(1, tmp_path / "out", planner_kind="force-admittance")
    assert summary["planner_kind"] == "force-admittance"
    actor = read_lines(tmp_path / "out" / "actor.jsonl")
    plans = [a for a in actor if a["phase"] == "visual_step"]
    assert all(p["next_ctrl_is_nominal"] is True for p in plans)
    # Each substep consumed the PRIOR record (step_index before increment).
    assert update_calls == list(range(0, 3 * probe.VISUAL_SUBSTEPS))
    # Every stepped ctrl is the helper-corrected one (first element increased).
    assert all(c[0] > 0.0 for c in stepped_ctrls)
    assert len(force_calls) == 3 * probe.VISUAL_SUBSTEPS
    # Nominal planar endpoint and actual stepped ctrl differ.
    nominal = np.array(plans[0]["next_ctrl"], dtype=float)
    assert not np.allclose(nominal, stepped_ctrls[0])
    intents = [e for e in read_lines(tmp_path / "out" / "planner.jsonl")
               if e["kind"] == "force_substep_intent"]
    assert len(intents) == 3 * probe.VISUAL_SUBSTEPS
    assert intents[0]["intended_next_step_index"] == 1


def test_force_admittance_intended_step_matches_actual_step(tmp_path, monkeypatch):
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["z_hold"] = 0.1
        R["last_record"] = {"step_index": 0, "sim_time_s": 1.0, "contacts": [], "aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    captures = [(make_frame(1, 1000.0, 1), make_est(20.0, 0.0), 5.0)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    monkeypatch.setattr(probe, "force_mode_contact_update",
                        lambda env, rec: (np.array([1.0, 0.0, 0.0, 0.0]), 1.0, [0.07, 0.0, 0.1]))

    def fake_admittance(ctrl, anchor, row, force, dt, mapping, cfg):
        corrected = np.array(ctrl, dtype=float)
        corrected[0] += 0.0001
        info = {"force_error_n": 0.2, "requested_normal_speed_m_s": 0.0005,
                "requested_dz_m": 1e-6, "applied_dq": [1e-6, 0, 0, 0],
                "predicted_normal_dz_m": 1e-6, "common_scale": 1.0,
                "bound_limited": [], "teacher_only": True}
        return corrected, info

    monkeypatch.setattr(probe.fsv, "normal_force_admittance", fake_admittance)

    def fake_step(R, ctrl, fid, phase):
        R["step_index"] += 1
        rec = {"step_index": R["step_index"], "sim_time_s": 1.0 + 0.002 * R["step_index"],
               "contacts": [], "aggregate": {"index": {"normal_force_n": 1.0}}}
        R["last_record"] = rec
        return SimpleNamespace(aggregate=rec["aggregate"])

    monkeypatch.setattr(probe, "step", fake_step)
    probe.run(1, tmp_path / "out", planner_kind="force-admittance")
    intents = [e for e in read_lines(tmp_path / "out" / "planner.jsonl")
               if e["kind"] == "force_substep_intent"]
    for intent in intents:
        # The intended next step index must be exactly the step that followed.
        assert intent["intended_next_step_index"] == intent["consumed_record_step_index"] + 1


def test_force_admittance_missing_contact_skips_helper_and_guard_stops(tmp_path, monkeypatch):
    R = make_R(tmp_path, env=fake_env())
    R["planner_kind"] = "force-admittance"
    R["last_record"] = {"step_index": 0, "sim_time_s": 1.0, "contacts": [], "aggregate": {}}
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), make_est(20.0, 0.0), 5.0)
                for i in range(1, 8)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    helper_calls = []
    monkeypatch.setattr(probe, "force_mode_contact_update",
                        lambda env, rec: helper_calls.append(1) or None)
    steps = []

    def fake_step(R, ctrl, fid, phase):
        steps.append(1)
        R["step_index"] += 1
        # index contact for first 10 substeps, then lost.
        agg = {"index": {"normal_force_n": 0.5}} if len(steps) <= 10 else {}
        R["last_record"] = {"step_index": R["step_index"], "sim_time_s": 1.0 + 0.002 * R["step_index"],
                            "contacts": [], "aggregate": agg}
        return SimpleNamespace(aggregate=agg)

    monkeypatch.setattr(probe, "step", fake_step)
    reason = probe.visual_phase(R)
    close_R(R)
    assert reason == "contact_loss"
    # Helper is consulted each substep but returns None; no fabricated force.
    assert len(helper_calls) == 60
    # 10 valid + 50 lost x 2 ms = 0.1 s -> stop at step 60 mid-cycle 3.
    assert len(steps) == 60
    events = read_lines(tmp_path / "planner.jsonl")
    assert not any(e["kind"] == "force_substep_intent" for e in events)
    stop = [e for e in events if e["kind"] == "guard_stop"]
    assert len(stop) == 1 and stop[0]["planner_kind"] == "force-admittance"


def test_force_mode_contact_update_static_stub(monkeypatch):
    env = fake_env()
    pad_id = env.scene.pad_geom_id
    index_gids = env.scene.tip_geom_ids["index"]
    # Two positive top-face contacts on the index tip, same body.
    rec = {
        "contacts": [
            {"geom1": pad_id, "geom2": index_gids[0], "body1": "world", "body2": "if_ds",
             "kind": "tip_pad_top", "normal_force_n": 1.0,
             "pos_world": [0.06, 0.0, 0.1]},
            {"geom1": index_gids[1], "geom2": pad_id, "body1": "if_ds", "body2": "world",
             "kind": "tip_pad_top", "normal_force_n": 3.0,
             "pos_world": [0.08, 0.0, 0.1]},
            {"geom1": pad_id, "geom2": 99, "body1": "world", "body2": "mf_ds",
             "kind": "tip_pad_top", "normal_force_n": 2.0,
             "pos_world": [0.0, 0.0, 0.1]},  # non-index: ignored
        ]
    }
    # Stub the Jacobian so no real MuJoCo call is needed.
    expected_row = np.array([0.5, 0.0, 0.25, 0.25])
    monkeypatch.setattr(probe, "contact_jacobian_at_point",
                        lambda env, point, body_id: np.vstack([expected_row] * 3))
    row, force, point = probe.force_mode_contact_update(env, rec)
    assert force == 4.0
    assert point == pytest.approx([0.075, 0.0, 0.1])
    assert np.allclose(row, expected_row)
    # Missing contacts -> None.
    assert probe.force_mode_contact_update(env, {"contacts": []}) is None
    # Inconsistent non-pad bodies -> None.
    bad = dict(rec)
    bad["contacts"] = [dict(rec["contacts"][0]),
                       {**dict(rec["contacts"][1]), "body1": "mf_ds"}]
    assert probe.force_mode_contact_update(env, bad) is None


# --- force-governed mode mock checks -----------------------------------------


def test_force_governed_dispatches_chain_and_uses_prior_record(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["z_hold"] = 0.1
        R["last_record"] = {"step_index": 0, "sim_time_s": 1.0, "contacts": [], "aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), est, 5.0)
                for i, est in enumerate([make_est(20.0, 0.0), make_est(2.0, 0.0),
                                         make_est(2.0, 0.0), make_est(2.0, 0.0)], 1)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    contact_calls = []
    monkeypatch.setattr(probe, "force_mode_contact_update",
                        lambda env, rec: contact_calls.append(rec.get("step_index")) or
                        (np.array([1.0, 0.0, 0.0, 0.0]), 1.0, [0.07, 0.0, 0.1]))
    normal_calls = []
    real_admittance = probe.fsv.normal_force_admittance

    def fake_admittance(ctrl, anchor, row, force, dt, mapping, cfg):
        normal_calls.append((ctrl[0], force))
        return real_admittance(ctrl, anchor, row, force, dt, mapping, cfg)

    monkeypatch.setattr(probe.fsv, "normal_force_admittance", fake_admittance)
    gov_calls = []
    real_compose = probe.fgv.compose_governed_substep

    def fake_compose(current, normal, frame_delta, anchor, force, mapping, cfg):
        gov_calls.append((current[0], normal[0], frame_delta[0], force))
        combined = np.array(current, dtype=float)
        combined[0] += 0.00005  # deterministic small combined step
        info = {"teacher_only": True, "gate": 0.5, "common_scale": 1.0,
                "bound_limited": [], "requested_tangent": [0.00005, 0, 0, 0],
                "applied_tangent": [0.00005, 0, 0, 0],
                "normal_delta": [0.0, 0, 0, 0], "total_delta": [0.00005, 0, 0, 0]}
        return combined, info

    monkeypatch.setattr(probe.fgv, "compose_governed_substep", fake_compose)
    stepped_ctrls = []

    def fake_step(R, ctrl, fid, phase):
        R["step_index"] += 1
        R["last_record"] = {"step_index": R["step_index"], "sim_time_s": 1.0 + 0.002 * R["step_index"],
                            "contacts": [], "aggregate": {"index": {"normal_force_n": 1.0}}}
        stepped_ctrls.append(np.array(ctrl, dtype=float))
        return SimpleNamespace(aggregate=R["last_record"]["aggregate"])

    monkeypatch.setattr(probe, "step", fake_step)
    summary = probe.run(1, tmp_path / "out", planner_kind="force-governed")
    assert summary["planner_kind"] == "force-governed"
    # Each substep consumed the prior record (step_index before increment).
    assert contact_calls == list(range(0, 3 * probe.VISUAL_SUBSTEPS))
    assert len(normal_calls) == 3 * probe.VISUAL_SUBSTEPS
    assert len(gov_calls) == 3 * probe.VISUAL_SUBSTEPS
    assert len(stepped_ctrls) == 3 * probe.VISUAL_SUBSTEPS
    # First stepped ctrl is near prev, not the nominal endpoint jump.
    actor = read_lines(tmp_path / "out" / "actor.jsonl")
    plans = [a for a in actor if a["phase"] == "visual_step"]
    assert plans and all(p["next_ctrl_is_nominal"] is True for p in plans)
    nominal_first = np.array(plans[0]["next_ctrl"], dtype=float)
    assert not np.allclose(nominal_first, stepped_ctrls[0])
    # Teacher StepRecords prove execution matches intent.
    intents = [e for e in read_lines(tmp_path / "out" / "planner.jsonl")
               if e["kind"] == "force_substep_intent"]
    assert len(intents) == 3 * probe.VISUAL_SUBSTEPS
    assert all("governor_info" in e and "nominal_frame_delta" in e for e in intents)
    # The actual stepped control equals the intent's corrected_ctrl.
    for intent, stepped in zip(intents, stepped_ctrls):
        assert np.allclose(stepped, intent["corrected_ctrl"])


def test_force_governed_high_force_zeroes_tangent(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["z_hold"] = 0.1
        R["last_record"] = {"step_index": 0, "sim_time_s": 1.0, "contacts": [], "aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    captures = [(make_frame(1, 1000.0, 1), make_est(20.0, 0.0), 5.0)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    # High force -> gate should be 0.
    monkeypatch.setattr(probe, "force_mode_contact_update",
                        lambda env, rec: (np.array([1.0, 0.0, 0.0, 0.0]), 1.2, [0.07, 0.0, 0.1]))

    def fake_step(R, ctrl, fid, phase):
        R["step_index"] += 1
        R["last_record"] = {"step_index": R["step_index"], "sim_time_s": 1.0 + 0.002 * R["step_index"],
                            "contacts": [], "aggregate": {"index": {"normal_force_n": 1.2}}}
        return SimpleNamespace(aggregate=R["last_record"]["aggregate"])

    monkeypatch.setattr(probe, "step", fake_step)
    probe.run(1, tmp_path / "out", planner_kind="force-governed")
    intents = [e for e in read_lines(tmp_path / "out" / "planner.jsonl")
               if e["kind"] == "force_substep_intent"]
    assert intents
    for intent in intents:
        assert intent["governor_info"]["gate"] == pytest.approx(0.0)
        assert intent["governor_info"]["applied_tangent"] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_force_governed_missing_contact_holds_ctrl_and_guard_stops(tmp_path, monkeypatch):
    R = make_R(tmp_path, env=fake_env())
    R["planner_kind"] = "force-governed"
    R["last_record"] = {"step_index": 0, "sim_time_s": 1.0, "contacts": [], "aggregate": {}}
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), make_est(20.0, 0.0), 5.0)
                for i in range(1, 8)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    contact_calls = []
    monkeypatch.setattr(probe, "force_mode_contact_update",
                        lambda env, rec: contact_calls.append(1) or None)
    gov_calls = []
    monkeypatch.setattr(probe.fgv, "compose_governed_substep",
                        lambda *a, **k: gov_calls.append(1) or (a[0], {}))
    steps = []

    def fake_step(R, ctrl, fid, phase):
        steps.append(1)
        R["step_index"] += 1
        agg = {"index": {"normal_force_n": 0.5}} if len(steps) <= 10 else {}
        R["last_record"] = {"step_index": R["step_index"], "sim_time_s": 1.0 + 0.002 * R["step_index"],
                            "contacts": [], "aggregate": agg}
        return SimpleNamespace(aggregate=agg)

    monkeypatch.setattr(probe, "step", fake_step)
    reason = probe.visual_phase(R)
    close_R(R)
    assert reason == "contact_loss"
    assert len(contact_calls) == 60
    assert gov_calls == []  # no force/tangent invented when contact missing
    assert len(steps) == 60
    events = read_lines(tmp_path / "planner.jsonl")
    assert not any(e["kind"] == "force_substep_intent" for e in events)
    stop = [e for e in events if e["kind"] == "guard_stop"]
    assert len(stop) == 1 and stop[0]["planner_kind"] == "force-governed"


def test_force_governed_no_full_nominal_jump_at_frame_start(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["z_hold"] = 0.1
        R["last_record"] = {"step_index": 0, "sim_time_s": 1.0, "contacts": [], "aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    captures = [(make_frame(1, 1000.0, 1), make_est(20.0, 0.0), 5.0)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    monkeypatch.setattr(probe, "force_mode_contact_update",
                        lambda env, rec: (np.array([1.0, 0.0, 0.0, 0.0]), 1.0, [0.07, 0.0, 0.1]))
    first_stepped = []

    def fake_step(R, ctrl, fid, phase):
        R["step_index"] += 1
        R["last_record"] = {"step_index": R["step_index"], "sim_time_s": 1.0 + 0.002 * R["step_index"],
                            "contacts": [], "aggregate": {"index": {"normal_force_n": 1.0}}}
        if not first_stepped:
            first_stepped.append(np.array(ctrl, dtype=float))
        return SimpleNamespace(aggregate=R["last_record"]["aggregate"])

    monkeypatch.setattr(probe, "step", fake_step)
    probe.run(1, tmp_path / "out", planner_kind="force-governed")
    actor = read_lines(tmp_path / "out" / "actor.jsonl")
    plan = [a for a in actor if a["phase"] == "visual_step"][0]
    nominal_jump = np.linalg.norm(np.array(plan["next_ctrl"], dtype=float) -
                                  np.array(plan["prev_ctrl"], dtype=float))
    actual_first = np.linalg.norm(first_stepped[0] - np.array(plan["prev_ctrl"], dtype=float))
    # Governed mode first substep is much smaller than the full nominal frame jump.
    assert actual_first < nominal_jump
    assert actual_first <= 0.0002 + 1e-12  # per-substep cap


# --- task-profile selection and diagonal seed validation ---------------------


def test_default_task_profile_is_near_pointing_and_allows_large_seed(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    patch_run_env(monkeypatch)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    summary = probe.run(100, tmp_path / "out")
    assert summary["task_profile"] == "near_pointing_v1"
    assert client.reset_args == (100, "near_pointing_v1")
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["task_profile"] == "near_pointing_v1"


def test_invalid_task_profile_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="task_profile"):
        probe.run(1, tmp_path / "nope", task_profile="bogus")
    assert not (tmp_path / "nope").exists()  # no output/app work happened


def test_diagonal_seed_out_of_range_rejected_before_output(tmp_path):
    for bad_seed in [-1, 8, 100]:
        with pytest.raises(ValueError, match="near_diagonal_v1 seed"):
            probe.run(bad_seed, tmp_path / f"nope{bad_seed}",
                      task_profile="near_diagonal_v1")
        assert not (tmp_path / f"nope{bad_seed}").exists()


def test_cli_invalid_task_profile_rejected_before_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(probe, "run",
                        lambda seed, out, planner_kind="dls",
                        task_profile="near_pointing_v1": calls.append(1) or {})
    with pytest.raises(SystemExit):
        probe.main(["--seed", "1", "--out", str(tmp_path / "a"),
                    "--task-profile", "bogus"])
    assert calls == [] and not (tmp_path / "a").exists()


def test_cli_diagonal_seed_rejected_before_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(probe, "run",
                        lambda seed, out, planner_kind="dls",
                        task_profile="near_pointing_v1": calls.append(1) or {})
    with pytest.raises(SystemExit):
        probe.main(["--seed", "8", "--out", str(tmp_path / "a"),
                    "--task-profile", "near_diagonal_v1"])
    assert calls == [] and not (tmp_path / "a").exists()


def test_cli_task_profile_option_passed_through(tmp_path, monkeypatch):
    base = {"near_pointing_success": True,
            "episode_safe_complete_released": True, "child_reaped": True}
    seen = []
    monkeypatch.setattr(probe, "run",
                        lambda seed, out, planner_kind="dls",
                        task_profile="near_pointing_v1", contact_mode="standard":
                        seen.append(task_profile) or dict(base))
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "a"),
                       "--task-profile", "near_diagonal_v1"]) == 0
    assert seen == ["near_diagonal_v1"]
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "b")]) == 0
    assert seen == ["near_diagonal_v1", "near_pointing_v1"]


def test_diagonal_profile_reaches_reset_and_records(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    patch_run_env(monkeypatch)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    summary = probe.run(3, tmp_path / "out", task_profile="near_diagonal_v1")
    assert client.reset_args == (3, "near_diagonal_v1")
    assert summary["task_profile"] == "near_diagonal_v1"
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["task_profile"] == "near_diagonal_v1"


# --- contact-mode selection --------------------------------------------------


def test_default_contact_mode_is_standard(tmp_path, monkeypatch):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    patch_run_env(monkeypatch)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    summary = probe.run(1, tmp_path / "out")
    assert summary["contact_mode"] == "standard"
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["contact_mode"] == "standard"


def test_invalid_contact_mode_rejected_before_output(tmp_path):
    with pytest.raises(ValueError, match="contact_mode"):
        probe.run(1, tmp_path / "nope", contact_mode="bogus")
    assert not (tmp_path / "nope").exists()


def test_pointing_light_rejected_unless_force_governed(tmp_path):
    with pytest.raises(ValueError, match="pointing-light"):
        probe.run(1, tmp_path / "nope", planner_kind="dls",
                  contact_mode="pointing-light")
    assert not (tmp_path / "nope").exists()


def test_cli_invalid_contact_mode_rejected_before_run(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(probe, "run",
                        lambda *a, **k: calls.append((a, k)) or {})
    with pytest.raises(SystemExit):
        probe.main(["--seed", "1", "--out", str(tmp_path / "a"),
                    "--contact-mode", "bogus"])
    assert calls == [] and not (tmp_path / "a").exists()


def test_cli_contact_mode_option_passed_through(tmp_path, monkeypatch):
    base = {"near_pointing_success": True,
            "episode_safe_complete_released": True, "child_reaped": True}
    seen = []

    def fake_run(seed, out, planner_kind="dls",
                 task_profile="near_pointing_v1", contact_mode="standard"):
        seen.append(contact_mode)
        return dict(base)

    monkeypatch.setattr(probe, "run", fake_run)
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "a"),
                       "--planner", "force-governed",
                       "--contact-mode", "pointing-light"]) == 0
    assert seen == ["pointing-light"]
    assert probe.main(["--seed", "1", "--out", str(tmp_path / "b")]) == 0
    assert seen == ["pointing-light", "standard"]


def _force_governed_run(tmp_path, monkeypatch, contact_mode):
    client = FakeClient()
    monkeypatch.setitem(sys.modules, "mac_sandbox_client",
                        SimpleNamespace(SandboxClient=lambda out_dir=None: client))
    monkeypatch.setattr(vp, "make_env", lambda: fake_env())
    monkeypatch.setattr(vp, "reset_open", lambda env: None)

    def fake_prep(R):
        R["z_hold"] = 0.1
        R["last_record"] = {"step_index": 0, "sim_time_s": 1.0,
                            "contacts": [], "aggregate": {}}

    monkeypatch.setattr(probe, "prep_phase", fake_prep)
    monkeypatch.setattr(probe, "cleanup_phase", lambda R: True)
    captures = [(make_frame(i, 1000.0 + 0.05 * i, i), est, 5.0)
                for i, est in enumerate([make_est(20.0, 0.0), make_est(2.0, 0.0),
                                         make_est(2.0, 0.0), make_est(2.0, 0.0)], 1)]
    it = iter(captures)
    monkeypatch.setattr(probe, "capture_estimate", lambda R: next(it))
    monkeypatch.setattr(probe, "site_jacobian", lambda env: JAC)
    monkeypatch.setattr(probe, "tip_z", lambda env: 0.1)
    monkeypatch.setattr(probe, "CYCLE_MIN_WALL_S", 0.0)
    monkeypatch.setattr(probe, "send_touch", lambda R, agg: None)
    monkeypatch.setattr(probe, "force_mode_contact_update",
                        lambda env, rec: (np.array([1.0, 0.0, 0.0, 0.0]),
                                          1.0, [0.07, 0.0, 0.1]))

    def fake_admittance(ctrl, anchor, row, force, dt, mapping, cfg):
        normal = np.array(ctrl, dtype=float)
        normal[0] += 0.00001
        info = {"force_error_n": force - 0.8,
                "requested_normal_speed_m_s": 0.0001,
                "requested_dz_m": 1e-6, "applied_dq": [1e-6, 0, 0, 0],
                "predicted_normal_dz_m": 1e-6, "common_scale": 1.0,
                "bound_limited": [], "teacher_only": True}
        return normal, info

    monkeypatch.setattr(probe.fsv, "normal_force_admittance", fake_admittance)

    def fake_step(R, ctrl, fid, phase):
        R["step_index"] += 1
        R["last_record"] = {"step_index": R["step_index"],
                            "sim_time_s": 1.0 + 0.002 * R["step_index"],
                            "contacts": [],
                            "aggregate": {"index": {"normal_force_n": 1.0}}}
        return SimpleNamespace(aggregate=R["last_record"]["aggregate"])

    monkeypatch.setattr(probe, "step", fake_step)
    captured_cfgs = []
    real_compose = probe.fgv.compose_governed_substep

    def spy_compose(current, normal, frame_delta, anchor, force, mapping, cfg):
        captured_cfgs.append(cfg)
        return real_compose(current, normal, frame_delta, anchor, force,
                            mapping, cfg)

    monkeypatch.setattr(probe.fgv, "compose_governed_substep", spy_compose)
    summary = probe.run(1, tmp_path / "out", planner_kind="force-governed",
                        contact_mode=contact_mode)
    return summary, captured_cfgs


def test_pointing_light_uses_lower_tangential_stop_force(tmp_path, monkeypatch):
    summary, cfgs = _force_governed_run(tmp_path, monkeypatch, "pointing-light")
    assert summary["planner_kind"] == "force-governed"
    assert summary["contact_mode"] == "pointing-light"
    assert cfgs
    assert all(cfg.tangential_stop_force_n == pytest.approx(0.9) for cfg in cfgs)
    assert all(cfg.target_force_n == pytest.approx(0.8) for cfg in cfgs)
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["contact_mode"] == "pointing-light"
    gov_cfg = meta["planner"]["force_governor_config"]
    assert gov_cfg["tangential_stop_force_n"] == pytest.approx(0.9)
    assert gov_cfg["target_force_n"] == pytest.approx(0.8)
    assert gov_cfg["substeps_per_frame"] == 25
    assert gov_cfg["per_substep_total_joint_delta_cap_rad"] == pytest.approx(0.0002)
    assert gov_cfg["max_cycle_delta_rad"] == pytest.approx(0.005)
    on_disk = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert on_disk["contact_mode"] == "pointing-light"
    assert on_disk["force_governor_config"]["tangential_stop_force_n"] == pytest.approx(0.9)


def test_standard_force_governed_retains_default_stop_force(tmp_path, monkeypatch):
    summary, cfgs = _force_governed_run(tmp_path, monkeypatch, "standard")
    assert summary["contact_mode"] == "standard"
    assert cfgs
    assert all(cfg.tangential_stop_force_n == pytest.approx(1.1) for cfg in cfgs)
    meta = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert meta["contact_mode"] == "standard"
    assert meta["planner"]["force_governor_config"]["tangential_stop_force_n"] == pytest.approx(1.1)
