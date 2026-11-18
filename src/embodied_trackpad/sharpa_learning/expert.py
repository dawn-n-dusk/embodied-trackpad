"""Scripted stage-1 expert + ONE real contact diagnostic.

The diagnostic executes a complete approach / contact / short slide / lift
with real per-step joint position targets at 120 Hz, gated by MEASURED
PhysX contact (impulse/dt), not by geometry gaps. Other fingers are kept
clear by curling them away; the curl sign is determined at runtime by a
measured wiggle probe (never assumed). All contacts (any finger, any pair)
are recorded with the actual collider/body pair, world contact position,
normal and impulse.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .env import ACTION_DIM, FINGERS, SharpaEnv, atomic_json

NON_INDEX = ("middle", "ring", "pinky", "thumb")
# Maximum commanded search depth BELOW the pad top surface (m).  This is a
# motion bound for the index fingertip reference point, not a validated
# elastic-travel guarantee.  Descent stops at the first measured PhysX
# index-pad force/pair, well before this bound if possible.  Record actual
# penetration, force and limits; never trust depth alone.
MAX_SEARCH_DEPTH_M = 0.006
SLIDE_OFFSET_M = (0.010, 0.0)   # pad-local slide (+x, +y)
CONTACT_FORCE_N = 0.5           # approach success gate (first real touch)
WIGGLE_RAD = 0.1


def _finger_curl_joints(dof_names, finger):
    keys = ("MCP", "PIP", "IP", "CMC")
    return [n for n in dof_names
            if finger in n and any(k in n for k in keys)
            and not n.startswith("panda_joint")]


class ContactDiagnosticExpert:
    """IK-waypoint expert for one approach/contact/slide/lift diagnostic."""

    def __init__(self, env: SharpaEnv):
        self.env = env
        self.cfg = env.scene_config
        self.dt = env.dt
        self.result = {"phases": [], "probe": {}, "ik": {}, "blockers": []}
        self._curl_sign = {}
        self._curl_joints = {}
        self._plan = None

    def _world_to_base(self, p_world):
        cfg = self.cfg
        ax, ay = cfg.base_xy
        c, s = math.cos(-cfg.base_yaw), math.sin(-cfg.base_yaw)
        dx, dy = p_world[0] - ax, p_world[1] - ay
        return np.array([c * dx - s * dy, s * dx + c * dy,
                         p_world[2] - cfg.table_top_z])

    def _solve_ik(self, tip_world, label):
        from embodied_trackpad.sharpa_sim import kinematics as kn
        cfg = self.cfg
        target_base = self._world_to_base(np.asarray(tip_world, dtype=float))
        seeds = [tuple(self.env.home_q7)]
        home = kn.solve_home(target_base, tuple(cfg.mount_pos0),
                             tuple(cfg.mount_rot0_wxyz),
                             tuple(cfg.mount_rot1_wxyz), seeds=seeds,
                             max_iter=500)
        self.result["ik"][label] = home.to_dict()
        self.result["ik"][label]["tip_target_world"] = [
            round(float(v), 6) for v in tip_world]
        if not home.success:
            self.result["blockers"].append(
                f"IK {label} did not converge "
                f"(pos_err={home.position_error_m:.6f})")
        return home

    def plan(self):
        cfg = self.cfg
        pad_top = cfg.table_top_z + cfg.pad_size[2]
        tip_home_w = np.array(cfg.home_tip_world, dtype=float)
        # Hover a few millimetres above the pad, then search below the top.
        hover_w = np.array([tip_home_w[0], tip_home_w[1],
                            pad_top + 0.003])
        touch_w = np.array([tip_home_w[0], tip_home_w[1],
                            pad_top - MAX_SEARCH_DEPTH_M])
        slide_w = touch_w + np.array(
            [SLIDE_OFFSET_M[0], SLIDE_OFFSET_M[1], 0.0])
        ik_hover = self._solve_ik(hover_w, "hover")
        ik_touch = self._solve_ik(touch_w, "touch")
        ik_slide = self._solve_ik(slide_w, "slide")
        if not (ik_hover.success and ik_touch.success and ik_slide.success):
            return False
        self._plan = {
            "home": np.asarray(self.env.home_q7, dtype=float),
            "hover": np.asarray(ik_hover.q, dtype=float),
            "touch": np.asarray(ik_touch.q, dtype=float),
            "slide": np.asarray(ik_slide.q, dtype=float),
        }
        return True

    def curl_command(self, finger, value):
        q = np.zeros(ACTION_DIM)
        names = self.env.action_names
        for n in self._curl_joints.get(finger, []):
            q[names.index(n)] = value
        return q

    def raise_others_command(self):
        q = np.zeros(ACTION_DIM)
        for f in NON_INDEX:
            q = q + self.curl_command(f, 1.1 * self._curl_sign.get(f, 1.0))
        # clip canonical hand commands to the measured articulation limits
        lo, hi = self.env.joint_limits
        margin = 0.01
        return np.clip(q, lo + margin, hi - margin)

    @staticmethod
    def _hold(q):
        return lambda phase_t, t: np.asarray(q, dtype=float)

    def _arm_spline(self, q_from, q_to, dur, hand_q=None):
        """Arm-only spline, optionally superimposed on a fixed hand pose.

        ``phase_t`` is elapsed time since the phase started (seconds), not
        the global simulation time.  ``t`` is provided for logging but not
        used for blending.
        """
        if hand_q is None:
            hand_q = np.zeros(ACTION_DIM)

        def fn(phase_t, t):
            u = min(1.0, max(0.0, phase_t / max(1e-9, dur)))
            s = u * u * (3 - 2 * u)
            v = hand_q.copy()
            v[:7] = q_from + (q_to - q_from) * s
            return v
        return fn

    def main_phases(self):
        """Post-probe phases: raise others, approach, press, slide, lift.

        The non-index hand curl is preserved through approach, press and
        slide so the index finger is the only finger near the pad.  Hand
        commands are never zeroed during the motion; the measured curl sign
        from the probe is used.
        """
        plan = self._plan
        raise_q = self.raise_others_command()

        lo, hi = self.env.joint_limits
        margin = 0.01

        def pose(q_arm):
            """Full 29-DoF command: arm q + raised non-index hand pose."""
            v = np.zeros(ACTION_DIM)
            v[:7] = q_arm
            v[7:] = raise_q[7:]
            return np.clip(v, lo + margin, hi - margin)

        hand_home = pose(plan["home"])
        return [
            ("raise_others", 1.0, self._hold(hand_home)),
            ("approach", 2.0, self._arm_spline(
                plan["home"], plan["hover"], 2.0, hand_q=hand_home)),
            ("press", 1.5, self._arm_spline(
                plan["hover"], plan["touch"], 1.5, hand_q=hand_home)),
            ("hold_touch", 0.5, self._hold(pose(plan["touch"]))),
            ("slide", 1.5, self._arm_spline(
                plan["touch"], plan["slide"], 1.5, hand_q=hand_home)),
            ("hold_slide", 0.5, self._hold(pose(plan["slide"]))),
            ("lift", 1.5, self._arm_spline(
                plan["slide"], plan["home"], 1.5, hand_q=hand_home)),
            ("settle_end", 0.5, self._hold(hand_home)),
        ]


def run_contact_diagnostic(stage: str, out_dir: str,
                           gpu_index: int = 0) -> int:
    """ONE real approach/contact/shortslide/lift with real joint targets."""
    from PIL import Image
    out = Path(out_dir).resolve()
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    result = {"schema": "sharpa-contact-diagnostic-v1",
              "status": "starting", "pid": os.getpid(), "uid": os.getuid(),
              "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "stage": str(stage),
              "contact_backend": "PhysX isaacsim.core.experimental.prims.RigidPrim get_contact_force_data(dt=physics_dt) actual-elastomer/DP/MP-bodies-vs-Trackpad/geom",
              "geometry_gap_proxy_used": False}
    atomic_json(out / "RESULT.json", result)

    env = SharpaEnv({"stage": stage, "out_dir": str(out),
                     "gpu_index": gpu_index})
    expert = ContactDiagnosticExpert(env)
    art = env.art
    names = env.action_names
    canon_to_art = env._canon_to_art
    logf = (out / "series.jsonl").open("w")
    contact_stats = {f: {"frames": 0, "max_force_n": 0.0, "pairs": set()}
                     for f in FINGERS}
    state = {
        "step": 0, "cam_tick": 0,
        "frame_counters": {k: 0 for k in env.products},
        "approach_contact_step": None,
        "first_contact_step": None,
        "lift_clear_step": None,
        "min_gap_mm": None,
    }
    contact_stop = {"q": None, "phase": None, "step": None}

    def run_phases(seq):
        """Execute (name, dur_s, fn) phases at 120 Hz with full logging.

        Positive index-pad force during press stops further descent: the
        current commanded pose is retained as the measured contact stop pose
        and the arm is held there through press/hold/slide before lifting.
        """
        nonlocal contact_stop
        for name, dur, fn in seq:
            n_steps = int(round(dur / env.dt))
            for k in range(n_steps):
                phase_t = k * env.dt
                t = state["step"] * env.dt
                q_canon = fn(phase_t, t)
                # Feedback hold: once contact is found, freeze the arm at the
                # measured stop pose for press/hold/slide phases; lift and
                # settle are allowed to move away.
                if contact_stop["q"] is not None and name in (
                        "press", "hold_touch", "slide", "hold_slide"):
                    q_canon = contact_stop["q"].copy()
                q_art = np.zeros(int(art.num_dofs))
                q_art[canon_to_art] = q_canon
                art.set_dof_position_targets(q_art)
                env.SM.step(steps=1,
                            update_fabric=env.SM.is_fabric_enabled())
                state["step"] += 1
                q_meas = np.asarray(art.get_dof_positions().numpy(),
                                    dtype=float).reshape(-1)
                dq_meas = np.asarray(art.get_dof_velocities().numpy(),
                                     dtype=float).reshape(-1)
                finite = bool(np.all(np.isfinite(q_meas))
                              and np.all(np.isfinite(dq_meas)))
                contact_frames = {}
                for f in FINGERS:
                    c = env._contact[f]
                    c.reset_window()
                    force, ok = c.read()
                    for ct in c.contacts:
                        pair = "|".join(sorted([ct["body0"], ct["body1"]]))
                        contact_stats[f]["pairs"].add(pair)
                    contact_frames[f] = {
                        "force_n": round(force, 5), "valid": bool(ok),
                        "contacts": c.contacts[-4:] if c.contacts else []}
                    if ok:
                        contact_stats[f]["frames"] += 1
                        contact_stats[f]["max_force_n"] = max(
                            contact_stats[f]["max_force_n"], force)
                index_force = contact_frames["index"]["force_n"]
                if state["first_contact_step"] is None \
                        and index_force >= CONTACT_FORCE_N:
                    state["first_contact_step"] = state["step"]
                    state["approach_contact_step"] = state["step"]
                    # Record the actual commanded pose at first measured
                    # contact, not the open-loop target.
                    contact_stop["q"] = q_canon.copy()
                    contact_stop["phase"] = name
                    contact_stop["step"] = state["step"]
                if name == "lift" and state["lift_clear_step"] is None \
                        and index_force < 0.2:
                    state["lift_clear_step"] = state["step"]
                tips_w = {f: [round(float(v), 5) for v in np.asarray(
                    env.tips[f].get_world_poses()[0]).reshape(-1)[:3]]
                    for f in FINGERS}
                gap_mm = (tips_w["index"][2] - (
                    env.table_z + env.pad_t)) * 1000.0
                state["min_gap_mm"] = gap_mm if state["min_gap_mm"] is None \
                    else min(state["min_gap_mm"], gap_mm)
                if state["step"] % 2 == 0 or state["step"] == 1:
                    logf.write(json.dumps({
                        "t": round(t, 4), "phase": name,
                        "finite_state": finite,
                        "q_cmd": [round(float(v), 5) for v in q_art],
                        "q_meas": [round(float(v), 5) for v in q_meas],
                        "dq_meas": [round(float(v), 4) for v in dq_meas],
                        "tip_world": tips_w,
                        "contacts": contact_frames,
                    }) + "\n")
                if state["step"] % 8 == 0:
                    state["cam_tick"] += 1
                    n0 = env.SM.get_num_physics_steps()
                    env.rep.orchestrator.step(delta_time=0,
                                              pause_timeline=False)
                    if state["cam_tick"] == 1:
                        result["render_physics_steps_observed"] = int(
                            env.SM.get_num_physics_steps() - n0)
                    for cam_name, ann in env.products.items():
                        data = ann.get_data()
                        arr = (data.get("data") if isinstance(data, dict)
                               else data)
                        if arr is not None and getattr(arr, "size", 0):
                            state["frame_counters"][cam_name] += 1
                            Image.fromarray(arr).save(
                                frames_dir /
                                f"{cam_name}_"
                                f"{state['frame_counters'][cam_name]:05d}.png")
                if state["step"] % 240 == 0:
                    logf.flush()
                    result["status"] = "running"
                    result["phase"] = name
                    result["sim_time"] = round(t, 3)
                    atomic_json(out / "RESULT.json", result)

    # ---- 1) reset + settle + measured curl-sign probe -------------------
    obs, info = env.reset(seed=0)
    probe_seq = [("settle", 1.0, ContactDiagnosticExpert._hold(
        np.concatenate([expert.env.home_q7, np.zeros(ACTION_DIM - 7)])))]
    for f in NON_INDEX:
        js = _finger_curl_joints(names, f)
        expert._curl_joints[f] = js
        if not js:
            expert.result["probe"][f] = {"joints": [], "sign": None}
            continue
        q0 = np.concatenate([env.home_q7, np.zeros(ACTION_DIM - 7)])
        q_wiggle = q0 + expert.curl_command(f, WIGGLE_RAD)
        q_back = q0.copy()
        probe_seq.append((f"probe_{f}_out", 0.5,
                          ContactDiagnosticExpert._hold(q_wiggle)))
        probe_seq.append((f"probe_{f}_back", 0.3,
                          ContactDiagnosticExpert._hold(q_back)))
    # The wiggle z-delta must be measured around the actual stepping, so we
    # execute the probe with inline hooks instead of the generic loop.
    z_probe = {}
    for item in probe_seq:
        name, dur, fn = item
        if name.startswith("probe_") and name.endswith("_out"):
            f = name.split("_")[1]
            z_probe[f] = [float(np.asarray(
                env.tips[f].get_world_poses()[0]).reshape(-1)[2]), None]
        run_phases([item])
        if name.startswith("probe_") and name.endswith("_out"):
            f = name.split("_")[1]
            z_probe[f][1] = float(np.asarray(
                env.tips[f].get_world_poses()[0]).reshape(-1)[2])
    for f in NON_INDEX:
        if f in z_probe and z_probe[f][1] is not None:
            dz = z_probe[f][1] - z_probe[f][0]
            expert._curl_sign[f] = 1.0 if dz >= 0 else -1.0
            expert.result["probe"][f] = {
                "joints": expert._curl_joints.get(f, []),
                "dz_m": round(dz, 6), "sign": expert._curl_sign[f],
                "wiggle_rad": WIGGLE_RAD}
        elif f not in z_probe:
            expert._curl_sign[f] = 1.0

    result["probe"] = expert.result["probe"]
    atomic_json(out / "RESULT.json", result)

    # ---- 2) IK waypoints + main approach/contact/slide/lift -------------
    t_plan0 = time.monotonic()
    if not expert.plan():
        result["status"] = "blocked"
        result["blockers"] = expert.result["blockers"]
        atomic_json(out / "RESULT.json", result)
        logf.close()
        return 2
    result["ik"] = expert.result["ik"]
    result["plan_seconds"] = round(time.monotonic() - t_plan0, 2)
    result["status"] = "running_main"
    atomic_json(out / "RESULT.json", result)

    t_start = time.monotonic()
    run_phases(expert.main_phases())
    logf.close()

    result["approach_contact_step"] = state["approach_contact_step"]
    result["first_contact_step"] = state["first_contact_step"]
    result["contact_stop"] = {
        "phase": contact_stop["phase"],
        "step": contact_stop["step"],
        "q_cmd": ([round(float(v), 6) for v in contact_stop["q"]]
                  if contact_stop["q"] is not None else None),
    }
    result["lift_clear_step"] = state["lift_clear_step"]
    result["min_gap_mm"] = (None if state["min_gap_mm"] is None
                            else round(state["min_gap_mm"], 3))
    result["contact_stats"] = {
        f: {"frames": contact_stats[f]["frames"],
            "max_force_n": round(contact_stats[f]["max_force_n"], 4),
            "pairs": sorted(contact_stats[f]["pairs"])}
        for f in FINGERS}
    # A "valid frame" is not contact: contact requires a positive force or
    # at least one reported pair from PhysX.
    result["any_real_contact"] = any(
        contact_stats[f]["max_force_n"] > 0.0 or contact_stats[f]["pairs"]
        for f in FINGERS)
    result["index_contact"] = state["approach_contact_step"] is not None
    result["wall_seconds"] = round(time.monotonic() - t_start, 2)
    result["physics_steps"] = int(env.SM.get_num_physics_steps())
    result["sim_time_final"] = round(
        float(env.SM.get_simulation_time()), 6)

    # stills + MP4 BEFORE SimulationApp.close (close never returns)
    env.rep.orchestrator.step(delta_time=0, pause_timeline=False)
    for cam_name, ann in env.products.items():
        data = ann.get_data()
        arr = data.get("data") if isinstance(data, dict) else data
        if arr is not None and getattr(arr, "size", 0):
            Image.fromarray(arr).save(out / f"{cam_name}.png")
    mp4 = {}
    for cam_name in env.products:
        frames = sorted(frames_dir.glob(f"{cam_name}_*.png"))
        if len(frames) >= 4:
            rc = subprocess.run(
                ["ffmpeg", "-y", "-framerate", "15", "-i",
                 str(frames_dir / f"{cam_name}_%05d.png"),
                 "-pix_fmt", "yuv420p", str(out / f"{cam_name}.mp4")],
                check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL).returncode
            mp4[cam_name] = {"returncode": int(rc), "frames": len(frames)}
        else:
            mp4[cam_name] = {"returncode": None, "frames": len(frames),
                             "error": "too few frames"}
    result["mp4_encode"] = mp4
    result["status"] = ("complete" if result["index_contact"]
                        else "complete_no_index_contact")
    result["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(out / "RESULT.json", result)
    env.close()
    return 0 if result["index_contact"] else 3


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()
    raise SystemExit(run_contact_diagnostic(
        args.stage, args.output, args.gpu_index))
