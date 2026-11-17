"""Headless Isaac Sim runner for the Sharpa trackpad scene (runs on Lim).

Uses the installed tensor Articulation API (isaacsim.core.experimental.prims)
for commands and MEASURED states: ordered DOF names from ``art.dof_names``,
``set_dof_position_targets`` for commands, ``get_dof_positions`` /
``get_dof_velocities`` for measured joint state. USD state attributes are not
used as measurement. The hand/flange link pose is measured through the
physics tensor view (RigidBodyView), never from USD xforms. Contact is a
labelled geometry-gap proxy; true contact penetration stays reported as
unavailable.

Home pose: the scene config carries a CHECKED IK solution (solved + FK
verified by the stage builder, see kinematics.py). The runner re-verifies
it against the measured link pose and records the mount residual.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from .config import SharpaSceneConfig
from .device import ContactEvent, TrackpadDeviceAdapter, VirtualGUI

EPISODE_PHASES = [
    ("settle", 2.0), ("sway_out", 3.0), ("hold", 2.0),
    ("sway_back", 3.0), ("settle2", 4.0),
]
# Small target offsets around the checked home (rad); no unchecked
# approach-to-contact motion. Index curl stays modest.
SWAY_DQ = (0.04, -0.03, 0.0, 0.05, 0.02, -0.04, 0.0)
INDEX_SWAY_RAD = 0.12


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _bounds(pb, t):
    for name, end in pb:
        if t < end:
            return name
    return "done"


def _blend(a, b, u):
    return [x + (y - x) * u for x, y in zip(a, b)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--episode-seconds", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cam-cadence", type=int, default=4)
    ap.add_argument("--gpu-uuid", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "sharpa-sim-run-v1", "pid": os.getpid(), "uid": os.getuid(),
        "gpu_uuid": args.gpu_uuid, "stage": str(args.stage), "seed": args.seed,
        "status": "starting", "scripted_controller": True,
        "learned_policy": False, "blockers": [],
        "contact_basis": "geometry_gap_proxy_not_measured_contact",
        "penetration": "unavailable",
    }
    atomic_json(out / "RESULT.json", result)

    cfg_path = args.config or args.stage.with_suffix(".config.json")
    config = SharpaSceneConfig.from_dict(json.loads(cfg_path.read_text()))

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "active_gpu": 0, "physics_gpu": 0,
                         "multi_gpu": False, "max_gpu_count": 1,
                         "limit_cpu_threads": 12,
                         "renderer": "RaytracedLighting",
                         "width": 960, "height": 720, "fast_shutdown": True})
    import numpy as np
    import omni.timeline
    import omni.usd
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Usd, UsdGeom, Gf
    from isaacsim.core.simulation_manager import SimulationManager as SM
    from isaacsim.core.experimental.prims import Articulation

    t0 = time.monotonic()
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(args.stage)):
        raise RuntimeError("open_stage failed")
    for _ in range(30):
        app.update()
    deadline = time.monotonic() + 120
    while ctx.get_stage_loading_status()[2] > 0:
        if time.monotonic() > deadline:
            raise TimeoutError("stage assets did not finish loading")
        app.update()
    stage = ctx.get_stage()
    result["startup_seconds"] = round(time.monotonic() - t0, 3)
    result["status"] = "stage_loaded"
    atomic_json(out / "RESULT.json", result)

    dt = config.physics_dt
    pad_w, pad_h, pad_t = config.pad_size
    table_z = config.table_top_z
    px, py = config.pad_xy
    env0 = "/World/envs/env_0"

    SM.setup_simulation(dt=dt, device="cpu")
    SM.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
    omni.timeline.get_timeline_interface().play()
    SM.initialize_physics()

    # ---- articulation: measured names + tensor state ----
    art = None
    try:
        art = Articulation([f"{env0}/ArmMount/Arm"],
                           reset_xform_op_properties=False)
        _ = art.num_dofs  # force property query
    except Exception as exc:  # keep failure honest
        result["blockers"].append(f"articulation bind failed: {exc}")
    if art is not None:
        result["num_dofs_measured"] = int(art.num_dofs)
        result["dof_names"] = list(art.dof_names)
        if int(art.num_dofs) != 29:
            result["blockers"].append(
                f"measured num_dofs={art.num_dofs} != 29 (gripper subtree "
                "deactivation failed)")
    result["status"] = "articulation_ready"
    atomic_json(out / "RESULT.json", result)

    # ---- checked home pose from config + measured mount residual --------
    from . import kinematics as kn
    home_q7 = [float(v) for v in (config.home_q7 or [])]
    if len(home_q7) != 7:
        result["blockers"].append("config.home_q7 missing (stage built "
                                  "without checked IK?)")
        home_q7 = [0.0] * 7
    mpos0 = tuple(config.mount_pos0)
    rot0 = tuple(config.mount_rot0_wxyz)
    mpos1 = tuple(config.mount_pos1)
    rot1 = tuple(config.mount_rot1_wxyz)
    B0 = kn.RT(kn.quat_wxyz_to_mat(rot0), mpos0)   # joint frame0 in link7
    B1 = kn.RT(kn.quat_wxyz_to_mat(rot1), mpos1)   # joint frame1 in flange
    arm_mount_col = kn.RT(kn.rpy_to_mat((0.0, 0.0, config.base_yaw)),
                          (*config.base_xy, config.table_top_z))
    # Zero-hand home prediction: the tip offset is used DIRECTLY in the
    # flange body frame (SHARPA_INDEX_TIP_BODY), not the mount frame.
    pred_ok = True
    try:
        Mm = kn.fk_mount(home_q7, mpos0, rot0)     # mount world (base frame)
        W_flange = Mm @ np.linalg.inv(B1)          # flange body world
        tip_b = W_flange[:3, 3] + W_flange[:3, :3] @ np.asarray(
            kn.SHARPA_INDEX_TIP_BODY)
        pred_tip_world = (arm_mount_col @ np.append(tip_b, 1.0))[:3]
    except Exception as exc:
        pred_ok = False
        result["blockers"].append(f"home FK prediction failed: {exc}")
        pred_tip_world = None
    if pred_ok:
        result["home_predicted_tip_world"] = [round(float(v), 5)
                                              for v in pred_tip_world]
        result["home_hover_mm"] = round(
            float(pred_tip_world[2] - table_z - pad_t) * 1000.0, 2)

    # ---- measured link poses via experimental RigidPrim (no USD xforms) --
    from isaacsim.core.experimental.prims import RigidPrim
    tip_path = f"{env0}/ArmMount/SharpaHand/right_index_fingertip"
    flange_path = f"{env0}/ArmMount/SharpaHand/right_hand_flange"
    link7_path = f"{env0}/ArmMount/Arm/panda_link7"
    rb_flange = rb_tip = rb_link7 = None
    bind_errors = []
    for label, path in (("flange", flange_path),
                        ("index_fingertip", tip_path),
                        ("link7", link7_path)):
        if not path:
            bind_errors.append(f"{label}: path not found in stage")
            continue
        try:
            rp = RigidPrim([path], reset_xform_op_properties=False)
            pos, quat = rp.get_world_poses()
            _ = np.asarray(pos).shape, np.asarray(quat).shape
            if label == "flange":
                rb_flange = rp
            elif label == "link7":
                rb_link7 = rp
            else:
                rb_tip = rp
        except Exception as exc:
            bind_errors.append(f"{label} {path}: {type(exc).__name__}: {exc}")
    if bind_errors:
        result["blockers"].append("rigid prim bind failed: "
                                  + "; ".join(bind_errors))
        result["status"] = "blocked"
        atomic_json(out / "RESULT.json", result)
        raise SystemExit(2)
    result["tip_pose_source"] = "rigid_prim_get_world_poses"

    # ---- ONE-TIME reset to the solved 29-DoF home state ------------------
    # Otherwise the articulation starts at the USD default pose (upright at
    # z~1.75 m) and drives through the scene. State write happens once,
    # before any stepping; afterwards only joint targets are used.
    if art is not None and int(art.num_dofs) == 29 and not result["blockers"]:
        q0 = np.zeros(int(art.num_dofs))
        for i, n in enumerate(art.dof_names):
            if n.startswith("panda_joint"):
                k = int(n.split("panda_joint")[-1]) - 1
                if 0 <= k < 7:
                    q0[i] = home_q7[k]
        reset_done = False
        last_err = None
        for pos_name, vel_name in (("set_dof_positions",
                                    "set_dof_velocities"),
                                   ("set_joint_positions",
                                    "set_joint_velocities")):
            try:
                getattr(art, pos_name)(q0.copy())
                getattr(art, vel_name)(np.zeros_like(q0))
                art.set_dof_position_targets(q0.copy())
                reset_done = True
                result["reset_method"] = pos_name
                break
            except AttributeError as exc:
                last_err = exc
            except Exception as exc:
                last_err = exc
                break
        if not reset_done:
            result["blockers"].append(
                f"articulation state reset failed: {last_err}")
            result["status"] = "blocked"
            atomic_json(out / "RESULT.json", result)
            raise SystemExit(2)
        result["reset_written"] = True
        atomic_json(out / "RESULT.json", result)

    gui = VirtualGUI(seed=args.seed)
    adapter = TrackpadDeviceAdapter(gui, physics_dt=dt)

    products = []

    def attach_rgb(name, path, size):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsA(UsdGeom.Camera):
            result["blockers"].append(f"missing camera {path}")
            return
        prod = rep.create.render_product(path, size)
        ann = rep.AnnotatorRegistry.get_annotator("rgb")
        ann.attach(prod)
        products.append({"name": name, "ann": ann})

    attach_rgb("overview", "/World/overview", (1280, 720))
    attach_rgb("screen_rgb", "/World/screen_rgb", (640, 480))
    attach_rgb("wrist_rgb", f"{env0}/ArmMount/Arm/panda_link7/wrist_rgb",
               (640, 480))

    def measured_tip_world():
        """Actual measured index fingertip position + flange body pose from
        RigidPrim.get_world_poses (positions + WXYZ quats, warp arrays).
        Returns (tip_pos, flange_pos, flange_quat_wxyz, source)."""
        tp, tq = rb_tip.get_world_poses()
        fp, fq = rb_flange.get_world_poses()
        tip = np.asarray(tp, dtype=float).reshape(-1)[:3]
        fpos = np.asarray(fp, dtype=float).reshape(-1)[:3]
        fquat = np.asarray(fq, dtype=float).reshape(-1)[:4]
        return tip, fpos, fquat, "rigid_prim"

    def measured_link7():
        p, q = rb_link7.get_world_poses()
        return np.asarray(p, dtype=float).reshape(-1)[:3], \
            np.asarray(q, dtype=float).reshape(-1)[:4]

    acc = 0.0
    phase_bounds = []
    for name, dur in EPISODE_PHASES:
        acc += dur
        phase_bounds.append((name, acc))

    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    logf = (out / "series.jsonl").open("w")
    steps_total = int(args.episode_seconds / dt)
    step = 0
    cam_tick = 0
    t_start = time.monotonic()
    stats = {"contact_proxy_frames": 0, "min_gap_mm": None,
             "max_proxy_force_n": 0.0}
    dof_idx = {n: i for i, n in enumerate(art.dof_names)} if art else {}

    def command_vector(phase, t):
        """29-DoF position targets: 7 arm joints stay near the checked
        home with one small bounded sway; 22 Sharpa channels mostly zero
        with a modest index curl during sway. No teleports."""
        q = list(home_q7)
        curl = 0.0
        if phase in ("sway_out", "hold"):
            q = [v + d for v, d in zip(home_q7, SWAY_DQ)]
            curl = INDEX_SWAY_RAD
        vec = [0.0] * (art.num_dofs if art else 0)
        if art:
            for name, i in dof_idx.items():
                if name.startswith("panda_joint"):
                    k = int(name.split("panda_joint")[-1]) - 1
                    if 0 <= k < 7:
                        vec[i] = q[k]
                elif "index" in name and ("MCP_FE" in name or "PIP" in name):
                    vec[i] = curl
        return vec

    result["clock_start"] = {
        "sim_time_s": round(float(SM.get_simulation_time()), 6),
        "physics_steps": int(SM.get_num_physics_steps()),
    }
    result["pid"] = os.getpid()
    result["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(out / "RESULT.json", result)

    print(f"[runner] {steps_total} steps", flush=True)
    frame_counters = {"wrist_rgb": 0, "overview": 0}  # dense per camera
    last_tip = None
    while step < steps_total:
        t = step * dt
        phase = _bounds(phase_bounds, t)
        if art is not None:
            art.set_dof_position_targets(command_vector(phase, t))
        SM.step(steps=1, update_fabric=SM.is_fabric_enabled())
        step += 1

        if art is not None:
            q_meas = art.get_dof_positions().numpy().reshape(-1).tolist()
            dq_meas = art.get_dof_velocities().numpy().reshape(-1).tolist()
        else:
            q_meas = dq_meas = []
        cmd = command_vector(phase, t)
        finite = bool(q_meas) and all(math.isfinite(v) for v in q_meas + dq_meas)

        lx = ly = 0.0
        gap_mm = None
        events = []
        tip_pos, flange_pos, flange_quat, tip_src = measured_tip_world()
        last_tip = tip_pos
        lx, ly = float(tip_pos[0] - px), float(tip_pos[1] - py)
        gap = float(tip_pos[2]) - (table_z + pad_t)
        gap_mm = gap * 1000.0
        stats["min_gap_mm"] = gap_mm if stats["min_gap_mm"] is None \
            else min(stats["min_gap_mm"], gap_mm)
        if abs(lx) <= pad_w / 2 and abs(ly) <= pad_h / 2 and gap <= 0.0:
            force = min(20.0, 850.0 * (-gap))
            events.append(ContactEvent("index", lx, ly, force, t))
            stats["contact_proxy_frames"] += 1
            stats["max_proxy_force_n"] = max(stats["max_proxy_force_n"], force)
        # Mount residual: compare FULL world joint-frame transforms on both
        # bodies from ACTUAL measured poses: J0 = W0 @ B0 (link7), J1 =
        # W1 @ B1 (flange). No FK, no column/row mixing.
        mount_resid = None
        mount_rot_resid = None
        if rb_link7 is not None:
            try:
                l7p, l7q = measured_link7()
                W0 = kn.RT(kn.quat_wxyz_to_mat(l7q), l7p)
                W1 = kn.RT(kn.quat_wxyz_to_mat(flange_quat), flange_pos)
                J0, J1 = W0 @ B0, W1 @ B1
                mount_resid = float(np.linalg.norm(J0[:3, 3] - J1[:3, 3]))
                Rrel = J0[:3, :3].T @ J1[:3, :3]
                c = min(1.0, max(-1.0, (float(np.trace(Rrel)) - 1.0) / 2.0))
                mount_rot_resid = float(math.acos(c))
                stats["max_mount_resid_m"] = max(
                    stats.get("max_mount_resid_m", 0.0), mount_resid)
                stats["max_mount_rot_resid_rad"] = max(
                    stats.get("max_mount_rot_resid_rad", 0.0),
                    mount_rot_resid)
            except Exception:
                mount_resid = None
        state = adapter.step(events, t)
        if step == 1:
            result["initial_measured_tip"] = [
                round(float(v), 5) for v in tip_pos]
            result["initial_gap_mm"] = round(gap_mm, 3)
            if pred_tip_world is not None:
                result["initial_tip_vs_prediction_mm"] = round(
                    float(np.linalg.norm(
                        tip_pos - np.asarray(pred_tip_world))) * 1000.0, 3)
            atomic_json(out / "RESULT.json", result)

        cursor_prim = stage.GetPrimAtPath(f"{env0}/Screen/cursor_disc")
        target_prim = stage.GetPrimAtPath(f"{env0}/Screen/target_disc")
        if cursor_prim:
            sw, sh = config.screen_size
            cx = (state["cursor"][0] / 640 - 0.5) * sw
            cz = -(state["cursor"][1] / 480 - 0.5) * sh
            # preserve the authored -0.014 m front-surface offset (op Set,
            # never ClearXformOpOrder)
            UsdGeom.Xformable(cursor_prim).GetTranslateOp().Set(
                Gf.Vec3d(cx, -0.014, cz))
        if target_prim:
            sw, sh = config.screen_size
            tx = (state["target"][0] / 640 - 0.5) * sw
            tz = -(state["target"][1] / 480 - 0.5) * sh
            UsdGeom.Xformable(target_prim).GetTranslateOp().Set(
                Gf.Vec3d(tx, -0.012, tz))

        if step % 2 == 0 or step == 1:
            rec = {
                "t": round(t, 4), "phase": phase,
                "t_sim": round(float(SM.get_simulation_time()), 4),
                "n_physics": int(SM.get_num_physics_steps()),
                "dof_names": list(dof_idx) if step == 2 else None,
                "q_cmd": [round(v, 5) for v in cmd],
                "q_meas": [round(v, 5) for v in q_meas],
                "dq_meas": [round(v, 4) for v in dq_meas],
                "finite_state": finite,
                "reset_q0_max_abs_err": (
                    round(float(max(abs(a - b) for a, b in
                                    zip(q_meas, cmd))), 6)
                    if step == 1 and len(q_meas) == len(cmd) else None),
                "tip_world": [round(float(v), 5) for v in tip_pos],
                "tip_pose_source": tip_src,
                "mount_residual_m": None if mount_resid is None
                else round(mount_resid, 6),
                "tip_pad_local_xy": [round(lx, 5), round(ly, 5)],
                "tip_geometry_gap_mm": None if gap_mm is None else round(gap_mm, 4),
                "gui": {k: state[k] for k in (
                    "cursor", "pressed", "cursor_over_target",
                    "click_registered", "click_on_target", "touch_count",
                    "recognition", "contact_count")},
            }
            logf.write(json.dumps(rec) + "\n")
        if step % args.cam_cadence == 0:
            cam_tick += 1
            if cam_tick % 2 == 0:
                # zero-delta render: no hidden physics step while rendering
                nphys_before = SM.get_num_physics_steps()
                rep.orchestrator.step(delta_time=0, pause_timeline=False)
                if cam_tick == 2:
                    result["render_physics_steps_observed"] = int(
                        SM.get_num_physics_steps() - nphys_before)
                    result["sim_time_at_first_render"] = round(
                        float(SM.get_simulation_time()), 6)
                    atomic_json(out / "RESULT.json", result)
                for p in products:
                    if p["name"] in ("wrist_rgb", "overview"):
                        data = p["ann"].get_data()
                        arr = data.get("data") if isinstance(data, dict) else data
                        if arr is not None and getattr(arr, "size", 0):
                            frame_counters[p["name"]] += 1
                            Image.fromarray(arr).save(
                                frames_dir /
                                f"{p['name']}_{frame_counters[p['name']]:05d}.png")
        if step % 240 == 0:
            logf.flush()
            result["status"] = "running"
            result["sim_time"] = round(t, 3)
            result["wall_seconds"] = round(time.monotonic() - t_start, 2)
            result["env_steps_per_s"] = round(
                step / max(1e-9, time.monotonic() - t_start), 1)
            atomic_json(out / "RESULT.json", result)

    logf.close()
    nphys_final = SM.get_num_physics_steps()
    rep.orchestrator.step(delta_time=0, pause_timeline=False)
    result["render_physics_steps_final"] = int(
        SM.get_num_physics_steps() - nphys_final)
    result["sim_time_final"] = round(float(SM.get_simulation_time()), 6)
    result["physics_steps_final"] = int(SM.get_num_physics_steps())
    stills = {}
    for p in products:
        data = p["ann"].get_data()
        arr = data.get("data") if isinstance(data, dict) else data
        if arr is not None and getattr(arr, "size", 0):
            path = out / f"{p['name']}.png"
            Image.fromarray(arr).save(path)
            stills[p["name"]] = str(path)
    result["clock_end"] = {
        "sim_time_s": round(float(SM.get_simulation_time()), 6),
        "physics_steps": int(SM.get_num_physics_steps()),
    }
    if result.get("clock_start"):
        result["clock_delta"] = {
            "sim_time_s": round(result["clock_end"]["sim_time_s"]
                                - result["clock_start"]["sim_time_s"], 6),
            "physics_steps": (result["clock_end"]["physics_steps"]
                              - result["clock_start"]["physics_steps"]),
        }
    result["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result["settled_measured_tip"] = ([round(float(v), 5) for v in last_tip]
                                      if last_tip is not None else None)
    result["stills"] = stills
    result["geometry_gap_stats"] = stats
    result["gui_final"] = gui.history[-1] if gui.history else {}
    result["total_physics_steps"] = step
    result["wall_seconds_total"] = round(time.monotonic() - t_start, 2)
    # actual capture cadence: one frame per 2*cam_cadence physics steps
    result["capture_fps"] = round(1.0 / (2.0 * dt * args.cam_cadence), 3)
    result["capture_frames_expected"] = int(step // (2 * args.cam_cadence))
    result["throughput_env_steps_per_s"] = round(
        step / max(1e-9, time.monotonic() - t_start), 1)
    result["status"] = "complete"
    # encode MP4s BEFORE app.close(): SimulationApp.close() tears down the
    # kit runtime and never returns to Python.
    mp4_status = {}
    for name in ("wrist_rgb", "overview"):
        frames = sorted(frames_dir.glob(f"{name}_*.png"))
        if len(frames) >= 4:
            rc = subprocess.run(
                ["ffmpeg", "-y", "-framerate",
                 str(round(1.0 / (2.0 * dt * args.cam_cadence), 3)),
                 "-i", str(frames_dir / f"{name}_%05d.png"),
                 "-pix_fmt", "yuv420p", str(out / f"{name}.mp4")],
                check=False, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL).returncode
            mp4_status[name] = {"returncode": int(rc),
                                "frames": len(frames)}
        else:
            mp4_status[name] = {"returncode": None, "frames": len(frames),
                                "error": "too few frames"}
    result["mp4_encode"] = mp4_status
    atomic_json(out / "RESULT.json", result)
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
