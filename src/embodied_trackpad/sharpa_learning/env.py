"""Sharpa learning campaign: env factory, SharpaEnv, contact diagnostic.

Implements the frozen cross-worker contract (outputs/sharpa-learning-20261118-v001
/CONTRACT.md). Owns ONLY environment-side files (env/data/expert/collect);
BC and world-model workers own their own modules.

Contact reporting uses the installed Isaac Sim 6.0.1 RigidPrim
contact-report API (``isaacsim.core.experimental.prims.RigidPrim``): one
RigidPrim view per fingertip link with ``contact_filter_paths`` pointing at
the Trackpad collision body. Raw per-contact data carries PhysX contact
points, normals and normal forces (impulse/dt). A missing sensor frame is
recorded distinctly from an observed empty frame, and the geometry-gap proxy
from the hover era is never used as a contact measurement.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np

# Fingertip semantic order for the 5-dim touch observation (contract).
FINGERS = ("index", "middle", "ring", "pinky", "thumb")
ARM_DELTA_SCALE = 0.025       # rad per normalized unit (contract)
HAND_DELTA_SCALE = 0.05       # rad per normalized unit (contract)
DECISION_HZ = 15.0
PHYSICS_HZ = 120.0
SUBSTEPS = 8                  # physics substeps per policy decision
OBS_SCREEN = (128, 128, 3)    # uint8
OBS_WRIST = (128, 128, 3)     # uint8
PROPRIO_DIM = 58              # q (29) then qd (29), canonical order
ACTION_DIM = 29
JOINT_LIMIT_MARGIN = 0.01
MAX_CMD_VEL_ARM = 2.0         # rad/s conservative controller cap
MAX_CMD_VEL_HAND = 3.0        # rad/s

CANONICAL_ARM = [f"panda_joint{i}" for i in range(1, 8)]


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _finger_from_body_path(body_path: str) -> str | None:
    """Map a Sharpa hand body path to a canonical finger name."""
    low = body_path.lower()
    for f in FINGERS:
        if f"right_{f}_" in low or f"_{f}_" in low:
            return f
    return None


def discover_finger_contact_bodies(stage, hand_root: str,
                                    pad_geom_path: str) -> dict[str, list[str]]:
    """Discover actual pad-contacting rigid bodies under the hand.

    The empty ``right_*_fingertip`` marker Xforms carry no collision geometry.
    Real contacts happen on the separate elastomer bodies (preferred pad
    toucher) and the distal-phalanx (DP) bodies.  This function walks the
    composed hand with ``Usd.PrimRange(..., Usd.TraverseInstanceProxies())``,
    finds enabled PhysicsCollisionAPI prims, and returns the unique ancestor
    body path per finger.

    A body is the nearest ancestor of a collision mesh that is a direct child
    of ``hand_root`` (e.g. ``right_index_elastomer`` or ``right_index_DP``).
    """
    from pxr import Usd, UsdPhysics

    hand_prim = stage.GetPrimAtPath(hand_root)
    if not hand_prim or not hand_prim.IsValid():
        raise RuntimeError(f"hand root missing: {hand_root}")

    bodies_by_finger: dict[str, set[str]] = {f: set() for f in FINGERS}
    for prim in Usd.PrimRange(hand_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        enabled_attr = collision.GetCollisionEnabledAttr()
        if enabled_attr and not enabled_attr.Get():
            continue
        # walk up to the body root (direct child of hand_root)
        body = prim
        while body and body.GetPath() != hand_prim.GetPath():
            parent = body.GetParent()
            if parent and parent.GetPath() == hand_prim.GetPath():
                break
            body = parent
        if not body or body.GetPath() == hand_prim.GetPath():
            continue
        finger = _finger_from_body_path(str(body.GetPath()))
        if finger is None:
            continue
        bodies_by_finger[finger].add(str(body.GetPath()))

    # prefer elastomer bodies first, then DP/other bodies; preserve attribution
    def sort_key(p: str) -> tuple[int, str]:
        low = p.lower()
        if "elastomer" in low:
            return (0, p)
        if "_dp" in low or "distal" in low:
            return (1, p)
        return (2, p)

    out = {}
    for f in FINGERS:
        paths = sorted(bodies_by_finger[f], key=sort_key)
        if not paths:
            raise RuntimeError(f"no collision body found for finger {f}")
        out[f] = paths
    return out


def canonical_action_names(dof_names: list[str]) -> list[str]:
    """panda_joint1..7 first, then the remaining measured DOF names sorted.

    Never assumes the articulation's measured order is the policy order.
    """
    rest = sorted(n for n in dof_names if n not in set(CANONICAL_ARM))
    return list(CANONICAL_ARM) + rest


def create_simulation_app(gpu_index: int = 0, width: int = 960, height: int = 720):
    """Factory: imports and constructs SimulationApp BEFORE any other Isaac
    module import (contract requirement)."""
    from isaacsim import SimulationApp
    return SimulationApp({
        "headless": True,
        "active_gpu": int(gpu_index),
        "physics_gpu": int(gpu_index),
        "multi_gpu": False,
        "max_gpu_count": 1,
        "limit_cpu_threads": 12,
        "renderer": "RaytracedLighting",
        "width": int(width),
        "height": int(height),
        "fast_shutdown": True,
    })


class _ContactBody:
    """One actual colliding rigid body (e.g. elastomer or DP) contact view."""

    def __init__(self, body_path: str, rigid_prim, physics_dt: float,
                 pad_path: str):
        self.body_path = body_path
        self.rp = rigid_prim
        self.dt = float(physics_dt)
        self.pad_path = pad_path

    def read(self) -> tuple[list[dict], bool]:
        try:
            forces, points, normals, distances, pair_counts, start_indices = \
                self.rp.get_contact_force_data(dt=self.dt)
        except Exception:
            return [], False
        forces = np.asarray(forces).reshape(-1)
        points = np.asarray(points).reshape(-1, 3)
        normals = np.asarray(normals).reshape(-1, 3)
        distances = np.asarray(distances).reshape(-1)
        pair_counts = np.asarray(pair_counts).reshape(-1)
        start_indices = np.asarray(start_indices).reshape(-1)
        contacts = []
        for i, count in enumerate(pair_counts):
            if count == 0:
                continue
            start = int(start_indices[i])
            for k in range(start, start + int(count)):
                f = float(forces[k])
                if not math.isfinite(f):
                    continue
                contacts.append({
                    "body0": self.body_path,
                    "body1": self.pad_path,
                    "position": [float(v) for v in points[k]],
                    "normal": [float(v) for v in normals[k]],
                    "distance": float(distances[k]),
                    "force_n": float(f),
                })
        return contacts, True


class _FingertipContact:
    """One fingertip's aggregated contact-report wrapper.

    Binds to the ACTUAL colliding bodies under the finger (elastomer + DP),
    not the empty kinematic ``*_fingertip`` marker.  Each body has its own
    RigidPrim view filtered to the Trackpad collision body.  Per-substep
    contacts are reset before every read so no stale force is carried over.
    """

    def __init__(self, finger: str, bodies: list[_ContactBody]):
        self.finger = finger
        self.bodies = bodies
        self.reset_window()

    def reset_window(self) -> None:
        self.max_force = 0.0
        self.frames_valid = 0
        self.frames_missing = 0
        self.any_contact = False
        self.contacts: list[dict] = []   # contacts from the latest read

    def read(self) -> tuple[float, bool]:
        """Returns (normal_force_n, valid).  force is the maximum normal
        force (N) between this fingertip's actual colliders and the pad."""
        self.contacts = []
        force = 0.0
        any_valid = False
        for b in self.bodies:
            pts, ok = b.read()
            any_valid = any_valid or ok
            for ct in pts:
                f = float(ct["force_n"])
                if math.isfinite(f):
                    force = max(force, abs(f))
                self.contacts.append(ct)
        if any_valid:
            self.frames_valid += 1
        else:
            self.frames_missing += 1
        if force > 0.0:
            self.any_contact = True
        self.max_force = max(self.max_force, force)
        return force, any_valid


class SharpaEnv:
    """Single-env Sharpa+Panda trackpad environment (contract §Actor).

    reset(seed, options) -> (obs, info); step(action) -> (obs, reward,
    terminated, truncated, info); expert_action(); close(). State writes
    happen ONLY in reset; step() sets joint position targets only.
    """

    def __init__(self, config: dict | None = None, app=None):
        cfg = dict(config or {})
        self.config = cfg
        self.out_dir = Path(cfg.get("out_dir", "run/env_out")).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gpu_index = int(cfg.get("gpu_index", 0))
        self.stage_path = cfg.get("stage")
        if not self.stage_path:
            raise ValueError("config['stage'] (built scene.usda) is required")
        self.cam_cadence = int(cfg.get("cam_cadence", 4))  # render every N decisions
        self.px, self.py = (0.0, 0.0)
        self.pad_w, self.pad_h, self.pad_t = (0.160, 0.1149, 0.0109)
        self.table_z = 0.72
        self._episode_count = 0
        self._click_registered_step = None
        self._lift_clear_after_click = False
        self._app = app if app is not None else create_simulation_app(self.gpu_index)
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        import omni.timeline
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.simulation_manager import SimulationManager as SM
        from isaacsim.core.experimental.prims import Articulation, RigidPrim

        ctx = omni.usd.get_context()
        if not ctx.open_stage(str(self.stage_path)):
            raise RuntimeError(f"open_stage failed: {self.stage_path}")
        for _ in range(30):
            self._app.update()
        deadline = time.monotonic() + 180
        while ctx.get_stage_loading_status()[2] > 0:
            if time.monotonic() > deadline:
                raise TimeoutError("stage assets did not finish loading")
            self._app.update()
        self.stage = ctx.get_stage()
        self.rep = rep
        self.SM = SM

        from embodied_trackpad.sharpa_sim.config import SharpaSceneConfig
        cfg_path = Path(str(self.stage_path)).with_suffix(".config.json")
        self.scene_config = SharpaSceneConfig.from_dict(
            json.loads(cfg_path.read_text()))
        sc = self.scene_config
        self.dt = sc.physics_dt
        self.px, self.py = sc.pad_xy
        self.pad_w, self.pad_h, self.pad_t = sc.pad_size
        self.table_z = sc.table_top_z
        self.home_q7 = [float(v) for v in sc.home_q7]
        self.env0 = "/World/envs/env_0"

        SM.setup_simulation(dt=self.dt, device="cpu")
        SM.get_physics_scenes()[0].set_enabled_gpu_dynamics(False)
        omni.timeline.get_timeline_interface().play()
        SM.initialize_physics()

        self.art = Articulation([f"{self.env0}/ArmMount/Arm"],
                                reset_xform_op_properties=False)
        _ = self.art.num_dofs
        self.dof_names = list(self.art.dof_names)
        if int(self.art.num_dofs) != ACTION_DIM:
            raise RuntimeError(f"measured num_dofs={self.art.num_dofs} != 29")
        self.action_names = canonical_action_names(self.dof_names)
        self._art_index = {n: i for i, n in enumerate(self.dof_names)}
        self._canon_to_art = [self._art_index[n] for n in self.action_names]
        self._is_arm = np.array(
            [n.startswith("panda_joint") for n in self.action_names])
        self.delta_scale = np.where(
            self._is_arm, ARM_DELTA_SCALE, HAND_DELTA_SCALE).astype(np.float32)
        self._load_joint_limits()
        self.dof_idx = self._art_index

        # rigid body views (measured poses; never USD xforms)
        # Contact-report API must be applied BEFORE the RigidPrim contact view
        # is created during __init__, otherwise create_rigid_contact_view fails.
        from pxr import PhysxSchema
        hand_root = f"{self.env0}/ArmMount/SharpaHand"
        pad_geom_path = f"{self.env0}/Trackpad/geom"

        # Kinematic markers (no collision) are kept for fingertip pose reads.
        self.tips = {}
        marker_paths = {f: f"{hand_root}/right_{f}_fingertip" for f in FINGERS}
        for f, p in marker_paths.items():
            prim = self.stage.GetPrimAtPath(p)
            if prim is None or not prim.IsValid():
                raise RuntimeError(f"fingertip marker missing: {p}")
            rp = RigidPrim([p], reset_xform_op_properties=False)
            _ = rp.get_world_poses()
            self.tips[f] = rp

        # Discover and bind the ACTUAL colliding bodies (elastomer + DP).
        contact_body_paths = discover_finger_contact_bodies(
            self.stage, hand_root, pad_geom_path)
        for paths in contact_body_paths.values():
            for p in paths:
                prim = self.stage.GetPrimAtPath(p)
                api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                api.CreateThresholdAttr().Set(0.0)
        contact_bodies: dict[str, list[_ContactBody]] = {f: [] for f in FINGERS}
        for f, paths in contact_body_paths.items():
            for p in paths:
                rp = RigidPrim(
                    [p], reset_xform_op_properties=False,
                    contact_filter_paths=[pad_geom_path],
                    max_contact_count=10)
                rp.set_enabled_contact_tracking([True])
                _ = rp.get_world_poses()
                contact_bodies[f].append(_ContactBody(
                    p, rp, self.dt, pad_geom_path))
        self.rb_flange = RigidPrim(
            [f"{self.env0}/ArmMount/SharpaHand/right_hand_flange"],
            reset_xform_op_properties=False)
        self._contact = {}
        for f in FINGERS:
            self._contact[f] = _FingertipContact(f, contact_bodies[f])

        # GUI disc prims (updated from adapter state before every render)
        self._cursor_prim = self.stage.GetPrimAtPath(
            f"{self.env0}/Screen/cursor_disc")
        self._target_prim = self.stage.GetPrimAtPath(
            f"{self.env0}/Screen/target_disc")

        # cameras: render products at contract resolution
        self.products = {}
        for name, path in (("screen_rgb", "/World/screen_rgb"),
                           ("wrist_rgb",
                            f"{self.env0}/ArmMount/Arm/panda_link7/wrist_rgb"),
                           ("overview", "/World/overview")):
            prim = self.stage.GetPrimAtPath(path)
            if prim is None or not prim.IsA(__import__("pxr").UsdGeom.Camera):
                raise RuntimeError(f"missing camera {path}")
            size = (1280, 720) if name == "overview" else (128, 128)
            prod = rep.create.render_product(path, size)
            ann = rep.AnnotatorRegistry.get_annotator("rgb")
            ann.attach(prod)
            self.products[name] = ann

        from embodied_trackpad.sharpa_sim.device import (
            TrackpadDeviceAdapter, VirtualGUI)
        self.gui = VirtualGUI(seed=0)
        self.adapter = TrackpadDeviceAdapter(self.gui, physics_dt=self.dt)

        self._prev_q_ref = None
        self._decision = 0
        self._physics_steps = 0
        self.provenance = {
            "stage": str(self.stage_path),
            "action_names": list(self.action_names),
            "delta_scale": [float(ARM_DELTA_SCALE)] * 7
            + [float(HAND_DELTA_SCALE)] * (ACTION_DIM - 7),
            "joint_limits": [
                [float(v) for v in self.joint_limits[0]],
                [float(v) for v in self.joint_limits[1]],
            ],
            "contact_backend": "PhysX isaacsim.core.experimental.prims.RigidPrim "
                               "get_contact_force_data(dt=physics_dt) "
                               "actual-elastomer/DP-bodies-vs-Trackpad/geom filtered",
            "contact_bodies_per_finger": {
                f: [str(p) for p in paths]
                for f, paths in contact_body_paths.items()},
            "fingertip_marker_paths": {
                f: str(marker_paths[f]) for f in FINGERS},
            "num_dofs_measured": int(self.art.num_dofs),
        }

    def _load_joint_limits(self) -> None:
        # Use the measured articulation limits API (radians, already converted
        # from degrees for revolute joints by the Articulation wrapper).
        from embodied_trackpad.sharpa_sim.config import PANDA_LIMITS
        lo = np.full(ACTION_DIM, -1e9, dtype=float)
        hi = np.full(ACTION_DIM, 1e9, dtype=float)
        art_lo, art_hi = self.art.get_dof_limits()
        art_lo = np.asarray(art_lo).reshape(-1)
        art_hi = np.asarray(art_hi).reshape(-1)
        for ci, name in enumerate(self.action_names):
            ai = self._canon_to_art[ci]
            if name.startswith("panda_joint"):
                k = int(name.replace("panda_joint", "")) - 1
                lo[ci], hi[ci] = PANDA_LIMITS[k]
            else:
                al = art_lo[ai]
                ah = art_hi[ai]
                if al is not None and ah is not None and math.isfinite(al) \
                        and math.isfinite(ah):
                    lo[ci], hi[ci] = float(al), float(ah)
        self.joint_limits = (lo, hi)

    def _pad_local_xy(self, p_world: np.ndarray) -> tuple[float, float]:
        """World contact point -> pad-local (x,y) in meters.

        The pad Xform is at (px, py, table_z+pad_t/2) with rotations
        pad_tilt_deg about its x-axis and pad_yaw about world z.  We remove
        the planar translation and the world-z yaw; tilt is ignored for the
        in-plane contact coordinate (the pad surface is locally flat).
        """
        from embodied_trackpad.sharpa_sim import kinematics as kn
        sc = self.scene_config
        dx = float(p_world[0]) - sc.pad_xy[0]
        dy = float(p_world[1]) - sc.pad_xy[1]
        c, s = math.cos(-sc.pad_yaw), math.sin(-sc.pad_yaw)
        return c * dx - s * dy, s * dx + c * dy

    def _update_gui_discs(self) -> None:
        """Write current GUI cursor/target to the USD disc prims."""
        from pxr import UsdGeom, Gf
        cur = self.gui.cursor
        tx, ty = self.gui.target
        sw, sh = self.scene_config.screen_size
        if self._cursor_prim:
            cx = (cur.x / 640 - 0.5) * sw
            cz = -(cur.y / 480 - 0.5) * sh
            UsdGeom.Xformable(self._cursor_prim).GetTranslateOp().Set(
                Gf.Vec3d(cx, -0.014, cz))
        if self._target_prim:
            txm = (tx / 640 - 0.5) * sw
            tzm = -(ty / 480 - 0.5) * sh
            UsdGeom.Xformable(self._target_prim).GetTranslateOp().Set(
                Gf.Vec3d(txm, -0.012, tzm))

    # ------------------------------------------------------------------
    def _write_state(self, q_canon: np.ndarray) -> None:
        q_art = np.zeros(int(self.art.num_dofs))
        q_art[self._canon_to_art] = q_canon
        done = False
        for pos_name, vel_name in (("set_dof_positions", "set_dof_velocities"),
                                   ("set_joint_positions", "set_joint_velocities")):
            try:
                getattr(self.art, pos_name)(q_art.copy())
                getattr(self.art, vel_name)(np.zeros_like(q_art))
                self.art.set_dof_position_targets(q_art.copy())
                done = True
                break
            except AttributeError:
                continue
        if not done:
            raise RuntimeError("articulation state reset failed")

    def reset(self, seed: int = 0, options: dict | None = None):
        """Reset the episode. The ONLY place actual simulator state is
        written. Returns (obs, info) per contract."""
        opts = dict(options or {})
        task_id = opts.get("task_id", "tap")
        self.gui.reset(int(seed))
        self.adapter.reset(int(seed))
        self._click_registered_step = None
        self._lift_clear_after_click = False
        q0 = np.zeros(ACTION_DIM, dtype=float)
        q0[:7] = self.home_q7
        self._write_state(q0)
        self._prev_q_ref = q0.copy()
        # settle a few steps so the state write takes hold
        for _ in range(4):
            self.SM.step(steps=1, update_fabric=self.SM.is_fabric_enabled())
        for c in self._contact.values():
            c.reset_window()
            _ = c.read()
        self._episode_count += 1
        self._decision = 0
        self._update_gui_discs()
        obs = self._observation()
        info = {
            "task_id": task_id,
            "seed": int(seed),
            "options": opts,
            "provenance": self.provenance,
            "sim_time": float(self.SM.get_simulation_time()),
        }
        info.update(self._privileged())
        return obs, info

    def _clip_limits(self, q_ref: np.ndarray) -> np.ndarray:
        lo, hi = self.joint_limits
        return np.minimum(hi - JOINT_LIMIT_MARGIN,
                          np.maximum(lo + JOINT_LIMIT_MARGIN, q_ref))

    def _clip_vel(self, q_ref: np.ndarray) -> np.ndarray:
        dt_dec = SUBSTEPS * self.dt
        vmax = np.where(self._is_arm, MAX_CMD_VEL_ARM, MAX_CMD_VEL_HAND)
        dq = np.clip(q_ref - self._prev_q_ref, -vmax * dt_dec, vmax * dt_dec)
        return self._prev_q_ref + dq

    def step(self, action):
        """action: float32 [29] normalized [-1,1]; returns the 5-tuple
        contract. Only joint position targets are written."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be finite float32 [29]")
        raw_action = action.copy()
        action = np.clip(action, -1.0, 1.0)
        q_meas_canon = self._q_canon()
        q_ref = q_meas_canon + action * self.delta_scale
        q_ref = self._clip_vel(self._clip_limits(q_ref))
        self._prev_q_ref = q_ref.copy()

        q_art = np.zeros(int(self.art.num_dofs))
        q_art[self._canon_to_art] = q_ref
        self.art.set_dof_position_targets(q_art)

        # reset per-decision contact aggregators
        for c in self._contact.values():
            c.reset_window()

        contact_events = []
        max_forces = np.zeros(5, dtype=np.float32)
        valid = np.zeros(5, dtype=np.uint8)
        sub_contacts: dict[str, list] = {f: [] for f in FINGERS}
        device_state = None
        t0 = float(self.SM.get_simulation_time())
        for k in range(SUBSTEPS):
            self.SM.step(steps=1, update_fabric=self.SM.is_fabric_enabled())
            self._physics_steps += 1
            t = float(self.SM.get_simulation_time())
            frame_missing = False
            per_finger: dict[str, list[dict]] = {f: [] for f in FINGERS}
            for i, f in enumerate(FINGERS):
                c = self._contact[f]
                force, ok = c.read()
                if not ok:
                    frame_missing = True
                max_forces[i] = max(max_forces[i], force)
                valid[i] = valid[i] or (1 if ok else 0)
                per_finger[f] = list(c.contacts)
                sub_contacts[f].extend(c.contacts)
            # Aggregate to ONE event per finger per substep: sum positive normal
            # forces and use the force-weighted centroid.  If any finger frame
            # is missing, the whole substep is missing (None), not an empty
            # release frame.
            if frame_missing:
                adapter_input = None
            else:
                frame_events: list = []
                for f in FINGERS:
                    pts = per_finger[f]
                    pos_sum = np.zeros(3)
                    force_sum = 0.0
                    for ct in pts:
                        fn = float(ct["force_n"])
                        if fn <= 0.0:
                            continue
                        pos_sum += np.asarray(ct["position"]) * fn
                        force_sum += fn
                    if force_sum > 0.0:
                        centroid = pos_sum / force_sum
                        lx, ly = self._pad_local_xy(centroid)
                        frame_events.append(_Event(f, lx, ly, force_sum, t))
                adapter_input = [e.to_event() for e in frame_events]
                contact_events.extend(frame_events)
            # drive the device adapter at 120 Hz, once per physics substep
            device_state = self.adapter.step(adapter_input, t)

        # update scene GUI discs from adapter state before any render
        self._update_gui_discs()
        self._decision += 1
        obs = self._observation()
        reward, terminated, truncated = self._reward(device_state)
        success = bool(device_state.get("click_on_target", 0) > 0) and terminated
        info = {
            "raw_action": raw_action.tolist(),
            "executed_q_ref": q_ref.astype(float).tolist(),
            "device": device_state,
            "contacts": {f: sub_contacts[f] for f in FINGERS if sub_contacts[f]},
            "measured_contact_positions": {
                f: [c["position"] for c in sub_contacts[f]] for f in FINGERS},
            "sim_time": t,
            "success": success,
        }
        info.update(self._privileged())
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _q_canon(self) -> np.ndarray:
        q = np.asarray(self.art.get_dof_positions().numpy(),
                       dtype=float).reshape(-1)
        return q[self._canon_to_art]

    def _qd_canon(self) -> np.ndarray:
        qd = np.asarray(self.art.get_dof_velocities().numpy(),
                        dtype=float).reshape(-1)
        return qd[self._canon_to_art]

    def _observation(self) -> dict:
        proprio = np.concatenate([self._q_canon(), self._qd_canon()]).astype(
            np.float32)
        touch = np.zeros(5, dtype=np.float32)
        valid = np.zeros(5, dtype=np.uint8)
        for i, f in enumerate(FINGERS):
            touch[i] = self._contact[f].max_force
            valid[i] = 1 if self._contact[f].frames_valid > 0 else 0
        # one zero-delta render for both cameras; physics clock must not advance
        nphys_before = self.SM.get_num_physics_steps()
        self.rep.orchestrator.step(delta_time=0, pause_timeline=False)
        if self.SM.get_num_physics_steps() != nphys_before:
            raise RuntimeError("render advanced physics steps")
        return {
            "screen_rgb": self._render("screen_rgb"),
            "wrist_rgb": self._render("wrist_rgb"),
            "proprio": proprio,
            "touch": touch,
            "touch_valid": valid,
        }

    def _render(self, name: str) -> np.ndarray:
        data = self.products[name].get_data()
        arr = data.get("data") if isinstance(data, dict) else data
        if arr is None or not getattr(arr, "size", 0):
            raise RuntimeError(f"camera {name} returned empty frame")
        img = np.asarray(arr, dtype=np.uint8)
        # RGBA -> RGB slice; do not reshape 4-channel into 3-channel
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.shape != OBS_SCREEN:
            raise RuntimeError(
                f"camera {name} shape {img.shape} != {OBS_SCREEN}")
        return img

    def _privileged(self) -> dict:
        cur = self.gui.cursor
        out = {
            "oracle_cursor_px": (float(cur.x), float(cur.y)),
            "oracle_target_px": tuple(float(v) for v in self.gui.target),
            "oracle_cursor_over_target": bool(self.gui.cursor_over_target),
            "oracle_click_registered": int(self.gui.click_registered),
            "oracle_click_on_target": int(self.gui.click_on_target),
        }
        for f in FINGERS:
            pos = np.asarray(self.tips[f].get_world_poses()[0]).reshape(-1)[:3]
            out[f"oracle_tip_world_{f}"] = tuple(float(v) for v in pos)
        return out

    def _reward(self, device_state: dict):
        # privileged shaping (internal only): target progress + on-target
        # click + clean release; penalties for limits/invalid contact.
        cur = self.gui.cursor
        tx, ty = self.gui.target
        dist = math.hypot(cur.x - tx, cur.y - ty)
        reward = -0.001 * dist
        if not device_state.get("events_valid", True):
            reward -= 0.1

        # click-on-target must be followed by a clean lift (no index contact)
        # before the episode terminates successfully.
        click_on_target = int(device_state.get("click_on_target", 0))
        index_force = float(self._contact["index"].max_force)
        if click_on_target > 0 and self._click_registered_step is None:
            self._click_registered_step = self._decision
        if self._click_registered_step is not None and index_force < 0.1:
            self._lift_clear_after_click = True

        terminated = bool(
            self._click_registered_step is not None
            and self._lift_clear_after_click)
        truncated = False
        return reward, terminated, truncated

    def expert_action(self) -> np.ndarray:
        """Reserved for a real feedback expert.  The placeholder must not
        produce BC-eligible zero actions."""
        raise NotImplementedError(
            "expert_action placeholder: install a feedback controller first")

    def close(self) -> None:
        # subclass/entrypoint encodes video BEFORE app.close()
        try:
            self._app.close()
        except Exception:
            pass


class _Event:
    """Adapter between PhysX raw contacts and device.ContactEvent."""

    __slots__ = ("finger", "x", "y", "force", "t", "_ev")

    def __init__(self, finger, x, y, force, t):
        from embodied_trackpad.sharpa_sim.device import ContactEvent
        self._ev = ContactEvent(finger, x, y, force, t)

    def to_event(self):
        return self._ev
