"""LEAP v1 Full + wrist-pitch MuJoCo scene, with a landscape touch surface.

The pinned Menagerie assets retain their 16 actuators, collisions, joint ranges,
masses and contact exclusions. Added engineering defaults: one wrist hinge
(actuator LAST, qpos FIRST), the P1 pad, massless tip sites and two cameras.
load_scene preserves the legacy open-pose mount for regression; load_working_scene
computes the user-directed relaxed pose and translation from collision geometry.
Positive palm inclination raises the +X finger-facing direction. The pad's local
160 mm X axis lies along world Y; contact xy must remain in the pad's local frame.
The backdrop and lights are visual only. Wrist/camera placement is a simulated
fixture, not an accepted hardware calibration or trained tap/slide controller.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path

import mujoco
import numpy as np

from embodied_trackpad.leap_v1.assets import (
    DOWNLOAD_ROOT,
    MANIFEST_JSON,
    verify_manifest,
)
from embodied_trackpad.leap_v1.model import (
    EMBODIMENT_ID as HAND_EMBODIMENT_ID,
)
from embodied_trackpad.leap_v1.model import (
    CALIBRATION_ID as HAND_CALIBRATION_ID,
)
from embodied_trackpad.leap_v1.model import (
    FINGERS,
    MOTOR_JOINTS,
    TIP_SITES,
    _tip_site_offsets,
)
from embodied_trackpad.leap_v1.visual_servo import CONTACT_PROFILE

SCENE_ID = "leap_v1_full_wrist_pitch_scene_v1"
# 17-channel identities for the COMBINED robot; the imported upstream IDs are
# retained separately as hand-only provenance (never relabelled as the new
# robot's identity).
EMBODIMENT_ID_17 = "leap_v1_full_right_wrist_pitch_menagerie_v1"
CALIBRATION_ID_17 = "uncalibrated_conservative_wrist_defaults_v1"
WRIST_JOINT = "wrist_pitch"
WRIST_BODY = "wrist_link"
PALM_BODY = "palm"
WRIST_ACTUATOR = "wrist_pitch_act"

# The 17-joint / 17-actuator named orders exposed to callers.
JOINT_NAMES17 = [WRIST_JOINT] + list(MOTOR_JOINTS)
ACTUATOR_NAMES17 = [f"{j}_act" for j in MOTOR_JOINTS] + [WRIST_ACTUATOR]

PAD_GEOM_NAME = "trackpad_160x110"
PAD_HALF_EXTENTS = (0.080, 0.055, 0.005)  # 160 x 110 mm face, 10 mm thick
DEFAULT_PAD_CENTER = (0.0, 0.0, 0.095)  # top face at z = 0.100 m

# Reviewed P1 pad contact profile, taken verbatim from the frozen
# CONTACT_PROFILE["scene_contact"] in leap_v1.visual_servo.
_SCENE_CONTACT = CONTACT_PROFILE["scene_contact"]
assert _SCENE_CONTACT["pad_geom"] == PAD_GEOM_NAME
PAD_SOLREF = tuple(float(v) for v in _SCENE_CONTACT["solref"])
PAD_SOLIMP = tuple(float(v) for v in _SCENE_CONTACT["solimp"])
PAD_FRICTION = tuple(float(v) for v in _SCENE_CONTACT["friction"])

OVERVIEW_CAMERA = "overview_cam"
WRIST_CAMERA = "wrist_cam"

# Distal tip joint per finger (child body of this joint is the pad toucher),
# same mapping as embodied_trackpad.leap_v1.pad_env.
FINGER_TIP_JOINT = {
    "index": "if_dip",
    "middle": "mf_dip",
    "ring": "rf_dip",
    "thumb": "th_ipl",
}


@dataclass(frozen=True)
class WristConfig:
    """Conservative, configurable wrist-pitch hinge + actuator parameters.

    These are engineering defaults (real inertia/limits, deliberately modest),
    NOT measured hardware values. Every value is recorded in the scene
    metadata so a later calibration can replace them without code changes.
    """

    # wrist_link mount in the world; palm hangs 0.025 m above it. At zero this
    # reproduces the accepted fixed-base v1 pad scene pose (palm-down).
    wrist_pos: tuple[float, float, float] = (0.0, 0.0, 0.2)
    wrist_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    palm_rel_pos: tuple[float, float, float] = (0.0, 0.0, 0.025)
    palm_rel_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0)  # horizontal pitch
    range_rad: tuple[float, float] = (-0.6, 0.6)
    damping: float = 0.5
    frictionloss: float = 0.02
    armature: float = 0.0005
    mass_kg: float = 0.15
    diaginertia: tuple[float, float, float] = (0.0001, 0.0001, 0.0001)
    kp: float = 12.0
    kv: float = 0.4
    # Rotary actuator torque limit, N*m (NOT newtons).
    forcerange_nm: tuple[float, float] = (-2.0, 2.0)


@dataclass(frozen=True)
class SceneConfig:
    """Full scene construction configuration (all values persisted)."""

    pad_center: tuple[float, float, float] = DEFAULT_PAD_CENTER
    pad_half_extents: tuple[float, float, float] = PAD_HALF_EXTENTS
    # Local 160 mm X axis lies across the fingers (world Y).
    pad_yaw_deg: float = 90.0
    wrist: WristConfig = field(default_factory=WristConfig)
    overview_cam_pos: tuple[float, float, float] = (0.23, -0.18, 0.25)
    overview_cam_target: tuple[float, float, float] = (-0.045, 0.0, 0.12)
    # Palm-LOCAL wrist camera frame at reset (palm frame == world orientation
    # at reset): eye 0.18 m above the pad plate, target on the plate near the
    # index/middle working region.
    wrist_cam_local_pos: tuple[float, float, float] = (0.05, 0.0, -0.045)
    wrist_cam_local_target: tuple[float, float, float] = (0.07, -0.015, -0.125)
    download_root: Path = DOWNLOAD_ROOT
    manifest_path: Path = MANIFEST_JSON


def _fmt(v: float) -> str:
    return f"{float(v):.8g}"


def _wrap_wrist(xml_path: Path, cfg: WristConfig) -> ET.Element:
    """Parse the pinned upstream MJCF and wrap ``palm`` in the wrist body."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("upstream MJCF has no worldbody")
    palm = worldbody.find(f"body[@name='{PALM_BODY}']")
    if palm is None:
        raise RuntimeError(f"upstream MJCF has no body named {PALM_BODY!r}")

    w = cfg
    wrist = ET.Element(
        "body",
        {
            "name": WRIST_BODY,
            "pos": " ".join(_fmt(v) for v in w.wrist_pos),
            "quat": " ".join(_fmt(v) for v in w.wrist_quat),
        },
    )
    ET.SubElement(
        wrist,
        "inertial",
        {
            "pos": "0 0 0",
            "mass": _fmt(w.mass_kg),
            "diaginertia": " ".join(_fmt(v) for v in w.diaginertia),
        },
    )
    ET.SubElement(
        wrist,
        "joint",
        {
            "name": WRIST_JOINT,
            "type": "hinge",
            "axis": " ".join(_fmt(v) for v in w.axis),
            "range": f"{_fmt(w.range_rad[0])} {_fmt(w.range_rad[1])}",
            "damping": _fmt(w.damping),
            "frictionloss": _fmt(w.frictionloss),
            "armature": _fmt(w.armature),
            "limited": "true",
        },
    )

    index = list(worldbody).index(palm)
    worldbody.remove(palm)
    palm.set("pos", " ".join(_fmt(v) for v in w.palm_rel_pos))
    palm.set("quat", " ".join(_fmt(v) for v in w.palm_rel_quat))
    wrist.append(palm)
    worldbody.insert(index, wrist)
    return root


def _look_at_quat(eye, target, up_hint=(0.0, 1.0, 0.0)) -> np.ndarray:
    """OpenGL-style camera orientation: looks along local -Z, +Y up."""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    forward = target - eye
    n = float(np.linalg.norm(forward))
    if n < 1e-12:
        raise ValueError("camera eye and target coincide")
    forward /= n
    z_cam = -forward
    x_cam = np.cross(np.asarray(up_hint, dtype=float), z_cam)
    xn = float(np.linalg.norm(x_cam))
    if xn < 1e-12:
        raise ValueError("degenerate camera up hint")
    x_cam /= xn
    y_cam = np.cross(z_cam, x_cam)
    rot = np.column_stack([x_cam, y_cam, z_cam])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    return quat


@dataclass
class TapSlideScene:
    """Compiled 17-DoF scene with named indexing helpers."""

    model: mujoco.MjModel
    manifest: dict
    config: SceneConfig
    adaptations: list[str]
    source_xml: str
    pad_geom_id: int
    wrist_joint_id: int
    wrist_actuator_id: int
    overview_cam_id: int
    wrist_cam_id: int
    tip_geom_ids: dict[str, list[int]]
    fingertip_site_ids: dict[str, int]

    @property
    def joint_names(self) -> list[str]:
        return [self.model.joint(i).name for i in range(self.model.njnt)]

    @property
    def actuator_names(self) -> list[str]:
        return [self.model.actuator(i).name for i in range(self.model.nu)]

    def joint_index(self, name: str) -> int:
        jid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))
        if jid < 0:
            raise KeyError(f"unknown joint: {name}")
        return jid

    def actuator_index(self, name: str) -> int:
        aid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        if aid < 0:
            raise KeyError(f"unknown actuator: {name}")
        return aid

    def open_pose_qpos(self) -> np.ndarray:
        """All-zero open pose (0 rad is inside every upstream range)."""
        return np.zeros(self.model.nq)

    def named_qpos(self, data: mujoco.MjData) -> dict[str, float]:
        m = self.model
        return {
            m.joint(i).name: float(data.qpos[m.jnt_qposadr[i]])
            for i in range(m.njnt)
        }

    def fingertip_positions(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {f: data.site(TIP_SITES[f]).xpos.copy() for f in FINGERS}

    def camera_extrinsics(self, data: mujoco.MjData) -> dict[str, dict]:
        m = self.model
        out = {}
        for label, cid in (
            (OVERVIEW_CAMERA, self.overview_cam_id),
            (WRIST_CAMERA, self.wrist_cam_id),
        ):
            out[label] = {
                "xpos_m": [float(v) for v in data.cam_xpos[cid]],
                "xmat": [float(v) for v in data.cam_xmat[cid]],
                "fovy_deg": float(m.cam_fovy[cid]),
            }
        return out


def _add_cameras(spec: mujoco.MjSpec, cfg: SceneConfig) -> None:
    # World z-up for the overview so hand/board orientation is readable.
    quat = _look_at_quat(cfg.overview_cam_pos, cfg.overview_cam_target, up_hint=(0.0, 0.0, 1.0))
    cam = spec.worldbody.add_camera(name=OVERVIEW_CAMERA)
    cam.pos = [float(v) for v in cfg.overview_cam_pos]
    cam.quat = [float(v) for v in quat]
    cam.fovy = 45.0

    # Wrist camera rigidly on the palm, explicit palm-local look-at toward the
    # pad plate (optical axis = normalize(target - eye)).
    wc = spec.body(PALM_BODY).add_camera(name=WRIST_CAMERA)
    wc.pos = [float(v) for v in cfg.wrist_cam_local_pos]
    wc.quat = [float(v) for v in _look_at_quat(cfg.wrist_cam_local_pos, cfg.wrist_cam_local_target,
                                             up_hint=(-1.0, 0.0, 0.0))]
    wc.fovy = 60.0


def load_scene(config: SceneConfig | None = None) -> TapSlideScene:
    """Compile the 17-DoF scene from the pinned verified assets."""
    cfg = config or SceneConfig()
    manifest = verify_manifest(cfg.manifest_path, cfg.download_root)
    xml_path = Path(cfg.download_root) / "right_hand" / "right_hand.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"upstream MJCF missing: {xml_path}")

    root = _wrap_wrist(xml_path, cfg.wrist)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.modelname = "leap_v1_tap_slide"
    spec.meshdir = str((xml_path.parent / "assets").resolve())

    # Fingertip sites at the compiled tip-geom frames (same method and
    # rationale as embodied_trackpad.leap_v1.model.load_hand).
    offsets = _tip_site_offsets(spec.compile())
    for finger, (body_name, local) in offsets.items():
        body = spec.body(body_name)
        site = body.add_site(name=TIP_SITES[finger], pos=[float(v) for v in local])
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.size = [0.002, 0.0, 0.0]
        site.rgba = [0.9, 0.2, 0.2, 1.0]
    light = spec.worldbody.add_light(name="key_light")
    light.pos = [0.3, -0.5, 0.8]
    light.dir = [-0.3, 0.5, -0.8]
    light.diffuse = [0.5, 0.5, 0.5]
    light.ambient = [0.15, 0.15, 0.15]
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill = spec.worldbody.add_light(name="fill_light")
    fill.pos = [-0.4, 0.3, 0.6]
    fill.dir = [0.4, -0.3, -0.6]
    fill.diffuse = [0.25, 0.25, 0.25]
    fill.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    fill.castshadow = False
    # Studio backdrop is visual only; never participates in contact.
    floor = spec.worldbody.add_geom(name="visual_table", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.5, 1.5, 0.01], pos=[0, 0, cfg.pad_center[2] - cfg.pad_half_extents[2] - 0.001])
    floor.contype = floor.conaffinity = 0
    floor.rgba = [0.77, 0.80, 0.84, 1.0]
    floor.group = 2
    sky = spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           width=256, height=1536)
    sky.rgb1 = [0.65, 0.71, 0.77]
    sky.rgb2 = [0.85, 0.88, 0.91]

    # Static pad geom (worldbody) with the reviewed P1 contact profile.
    pad = spec.worldbody.add_geom(
        name=PAD_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(cfg.pad_half_extents),
        pos=list(cfg.pad_center),
        quat=list(_axis_angle_quat((0, 0, 1), np.deg2rad(cfg.pad_yaw_deg))),
    )
    pad.solref = list(PAD_SOLREF)
    pad.solimp = list(PAD_SOLIMP)
    pad.friction = list(PAD_FRICTION)
    pad.priority = 1  # P1: pad contacts outrank hand self-contacts
    pad.rgba = [0.32, 0.39, 0.46, 1.0]

    _add_cameras(spec, cfg)

    # Wrist actuator appended LAST: upstream 16 order preserved, wrist_pitch
    # is channel 17 (index 16).
    w = cfg.wrist
    act = spec.add_actuator(name=WRIST_ACTUATOR)
    act.set_to_position(kp=w.kp, kv=w.kv)
    act.trntype = mujoco.mjtTrn.mjTRN_JOINT
    act.target = WRIST_JOINT
    act.ctrlrange = [float(w.range_rad[0]), float(w.range_rad[1])]
    act.forcerange = [float(w.forcerange_nm[0]), float(w.forcerange_nm[1])]

    model = spec.compile()
    model.vis.global_.offwidth = 800
    model.vis.global_.offheight = 600
    return _finish_scene(model, manifest, cfg, str(xml_path))


def _finish_scene(
    model: mujoco.MjModel, manifest: dict, cfg: SceneConfig, source_xml: str
) -> TapSlideScene:
    m = model
    pad_geom_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAD_GEOM_NAME))
    wrist_joint_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, WRIST_JOINT))
    wrist_actuator_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, WRIST_ACTUATOR))
    overview_cam_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, OVERVIEW_CAMERA))
    wrist_cam_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA))
    if min(pad_geom_id, wrist_joint_id, wrist_actuator_id, overview_cam_id, wrist_cam_id) < 0:
        raise RuntimeError("compiled model is missing a required named element")

    tip_geom_ids: dict[str, list[int]] = {}
    fingertip_site_ids: dict[str, int] = {}
    for finger, jname in FINGER_TIP_JOINT.items():
        jid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname))
        if jid < 0:
            raise RuntimeError(f"distal joint missing for {finger}: {jname}")
        body_id = int(m.jnt_bodyid[jid])
        gids = [
            g
            for g in range(m.ngeom)
            if int(m.geom_bodyid[g]) == body_id
            and not (int(m.geom_contype[g]) == 0 and int(m.geom_conaffinity[g]) == 0)
        ]
        if not gids:
            raise RuntimeError(f"no colliding geom on tip body of {finger}")
        tip_geom_ids[finger] = gids
        sid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TIP_SITES[finger]))
        if sid < 0 or int(m.site_bodyid[sid]) != body_id:
            raise RuntimeError(f"tip site {TIP_SITES[finger]} missing/misplaced for {finger}")
        fingertip_site_ids[finger] = sid

    adaptations = [
        "wrist_link body inserted between worldbody and upstream palm via ElementTree wrap; "
        "wrist_pitch hinge with real position actuator (conservative configurable limits, "
        "mass 0.15 kg, diaginertia 1e-4, torque limit 2 N*m); upstream 16 hand actuator "
        "order preserved, wrist actuator appended last",
        "static 160x110x10 mm trackpad geom added on worldbody with the reviewed P1 contact "
        "profile from leap_v1.visual_servo CONTACT_PROFILE['scene_contact']: "
        "solref (0.02, 1), solimp (0.9, 0.95, 0.001, 0.5, 2.0), friction (0.5, 0.005, 0.0001), "
        "geom priority 1",
        "4 massless fingertip sites (tip_index/tip_middle/tip_ring/tip_thumb) added on distal "
        "bodies at compiled frames of upstream tip mesh geoms",
        "2 cameras added: overview_cam (fixed, explicit look-at) and wrist_cam (rigid on the "
        "palm, explicit palm-local look-at toward the pad plate)",
        "studio lighting, sky and a non-colliding visual table added; upstream joint ranges, implicitfast "
        "integrator, elliptic cone, impratio=100, 2 ms timestep, 18 contact exclusions, "
        "box-primitive collision geometry, 16 upstream position actuators",
    ]
    return TapSlideScene(
        model=model,
        manifest=manifest,
        config=cfg,
        adaptations=adaptations,
        source_xml=source_xml,
        pad_geom_id=pad_geom_id,
        wrist_joint_id=wrist_joint_id,
        wrist_actuator_id=wrist_actuator_id,
        overview_cam_id=overview_cam_id,
        wrist_cam_id=wrist_cam_id,
        tip_geom_ids=tip_geom_ids,
        fingertip_site_ids=fingertip_site_ids,
    )


def scene_integrity(scene: TapSlideScene) -> dict:
    """Structural checks for the 17-DoF scene (machine-readable)."""
    m = scene.model
    w = scene.config.wrist
    return {
        "scene_id": SCENE_ID,
        "embodiment_id": EMBODIMENT_ID_17,
        "calibration_id": CALIBRATION_ID_17,
        "hand_source_embodiment_id": HAND_EMBODIMENT_ID,
        "hand_source_calibration_id": HAND_CALIBRATION_ID,
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "njnt": int(m.njnt),
        "nbody": int(m.nbody),
        "ngeom": int(m.ngeom),
        "nmesh": int(m.nmesh),
        "neq": int(m.neq),
        "nexclude": int(m.nexclude),
        "timestep_s": float(m.opt.timestep),
        "integrator": str(m.opt.integrator).split(".")[-1],
        "cone": str(m.opt.cone).split(".")[-1],
        "impratio": float(m.opt.impratio),
        "gravity": [float(v) for v in m.opt.gravity],
        "joint_names": scene.joint_names,
        "actuator_names": scene.actuator_names,
        "wrist_first_hand_order_preserved": scene.joint_names == JOINT_NAMES17,
        "hand_actuator_order_preserved": scene.actuator_names[:16] == ACTUATOR_NAMES17[:16],
        "wrist_actuator_appended_last": scene.actuator_names[16] == WRIST_ACTUATOR,
        "wrist_joint": {
            "name": WRIST_JOINT,
            "axis": [float(v) for v in m.jnt_axis[scene.wrist_joint_id]],
            "range": [float(v) for v in m.jnt_range[scene.wrist_joint_id]],
            "damping": float(m.dof_damping[int(m.jnt_dofadr[scene.wrist_joint_id])]),
            "frictionloss": float(m.dof_frictionloss[int(m.jnt_dofadr[scene.wrist_joint_id])]),
            "armature": float(m.dof_armature[int(m.jnt_dofadr[scene.wrist_joint_id])]),
        },
        "wrist_actuator": {
            "name": WRIST_ACTUATOR,
            "ctrlrange": [float(v) for v in m.actuator_ctrlrange[scene.wrist_actuator_id]],
            "kp": float(m.actuator_gainprm[scene.wrist_actuator_id, 0]),
            "kv": float(-m.actuator_biasprm[scene.wrist_actuator_id, 2]),
            "forcerange": [float(v) for v in m.actuator_forcerange[scene.wrist_actuator_id]],
            "force_unit": "N*m (rotary torque limit)",
        },
        "wrist_config": {
            "wrist_pos": list(w.wrist_pos),
            "wrist_quat": list(w.wrist_quat),
            "palm_rel_pos": list(w.palm_rel_pos),
            "palm_rel_quat": list(w.palm_rel_quat),
            "mass_kg": w.mass_kg,
            "diaginertia": list(w.diaginertia),
            "note": "conservative simulation defaults, not XC330 calibration",
        },
        "pad": {
            "geom_name": PAD_GEOM_NAME,
            "contact_profile_source": "embodied_trackpad.leap_v1.visual_servo CONTACT_PROFILE['scene_contact'] (P1)",
            "body_is_world": int(m.geom_bodyid[scene.pad_geom_id]) == 0,
            "yaw_deg": scene.config.pad_yaw_deg,
            "quaternion_wxyz": [float(v) for v in m.geom_quat[scene.pad_geom_id]],
            "contype": int(m.geom_contype[scene.pad_geom_id]),
            "conaffinity": int(m.geom_conaffinity[scene.pad_geom_id]),
            "priority": int(m.geom_priority[scene.pad_geom_id]),
            "solref": [float(v) for v in m.geom_solref[scene.pad_geom_id]],
            "solimp": [float(v) for v in m.geom_solimp[scene.pad_geom_id]],
            "friction": [float(v) for v in m.geom_friction[scene.pad_geom_id]],
        },
        "cameras": {
            "overview_cam_id": scene.overview_cam_id,
            "wrist_cam_id": scene.wrist_cam_id,
            "wrist_cam_body": m.body(int(m.cam_bodyid[scene.wrist_cam_id])).name,
        },
        "tip_geom_ids": {k: [int(g) for g in v] for k, v in scene.tip_geom_ids.items()},
        "fingertip_sites": dict(scene.fingertip_site_ids),
        "source_xml": scene.source_xml,
    }


def scene_metadata(scene: TapSlideScene) -> dict:
    """Full machine-readable scene metadata (for scene-metadata.json)."""
    m = scene.model
    meta = scene_integrity(scene)
    meta.update(
        {
            "source": {
                "repository": scene.manifest["upstream"]["repository"],
                "pinned_commit": scene.manifest["upstream"]["pinned_commit"],
                "license": "MIT (leap_hand/LICENSE, retained verbatim)",
            },
            "joints": [
                {
                    "id": i,
                    "name": m.joint(i).name,
                    "qposadr": int(m.jnt_qposadr[i]),
                    "axis": [float(v) for v in m.jnt_axis[i]],
                    "range": [float(v) for v in m.jnt_range[i]],
                }
                for i in range(m.njnt)
            ],
            "actuators": [
                {
                    "id": i,
                    "name": m.actuator(i).name,
                    "joint": m.joint(int(m.actuator_trnid[i, 0])).name,
                    "ctrlrange": [float(v) for v in m.actuator_ctrlrange[i]],
                    "forcerange": [float(v) for v in m.actuator_forcerange[i]],
                }
                for i in range(m.nu)
            ],
            "mass_kg": {
                "wrist_link": float(scene.config.wrist.mass_kg),
                "total": float(m.body_mass[1:].sum()),
                "note": "upstream MJCF inertials for the hand; wrist value is a configured default",
            },
            "adaptations": scene.adaptations,
            "pending": [
                "contact characterization through dynamics",
                "physical pad device + GUI task integration",
                "teacher trajectories and machine-readable episodes",
            ],
        }
    )
    return meta


# --- working mount + initial pose (half-curled, centred, near the pad) -----
#
# The legacy reset fixture above reproduces the old fixed-base pose (palm at
# z = 0.225 m, tips ~104 mm above the pad) and is retained for regression.
# The WORKING mount below is the user-directed posture: index/middle gently
# curled and centred at equal height, ring/thumb clear, wrist
# lower than knuckles with a modest upward palm inclination (~10 deg), and
# the index lowest COLLISION surface ~6 mm above the pad at reset. The
# translation is computed from named FK of the curl pose (one analytic pass;
# FK is equivariant in the wrist translation), never hand-tuned.


@dataclass(frozen=True)
class InitialPoseConfig:
    """Modest initial finger curl per finger (rad, upstream joint order)."""

    index_mcp: float = 0.12
    index_pip: float = 0.16
    index_dip: float = 0.10
    middle_mcp: float = 0.12
    middle_pip: float = 0.16
    middle_dip: float = 0.10
    ring_mcp: float = 0.1
    thumb_cmc: float = -0.25

    def qpos17(self) -> dict[str, float]:
        return {
            "if_mcp": self.index_mcp,
            "if_rot": 0.0,
            "if_pip": self.index_pip,
            "if_dip": self.index_dip,
            "mf_mcp": self.middle_mcp,
            "mf_rot": 0.0,
            "mf_pip": self.middle_pip,
            "mf_dip": self.middle_dip,
            "rf_mcp": self.ring_mcp,
            "rf_rot": 0.0,
            "rf_pip": 0.1,
            "rf_dip": 0.08,
            "th_cmc": self.thumb_cmc,
            "th_axl": 0.0,
            "th_mcp": 0.1,
            "th_ipl": 0.08,
            WRIST_JOINT: 0.0,
        }


@dataclass(frozen=True)
class WorkingMountConfig:
    """Parameters for the computed working mount."""

    palm_inclination_deg: float = 10.0
    index_clearance_m: float = 0.006
    # User-requested vertical translation of the entire hand/camera assembly.
    height_offset_m: float = 0.02
    contact_zone_center: tuple[float, float] = (0.0, 0.0)
    nominal_wrist_pos: tuple[float, float, float] = (0.0, 0.0, 0.2)
    # Wrist camera placement in WORLD frame at the working reset (converted to
    # palm-local before compile): above the contact zone, tilted to include
    # board margin.
    cam_eye_offset: tuple[float, float, float] = (-0.08, 0.0, 0.18)
    cam_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _axis_angle_quat(axis, angle_rad: float) -> tuple[float, float, float, float]:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = float(np.sin(angle_rad / 2.0))
    return (
        float(np.cos(angle_rad / 2.0)),
        float(axis[0] * s),
        float(axis[1] * s),
        float(axis[2] * s),
    )


def _lowest_collision_z(scene: TapSlideScene, data: mujoco.MjData, finger: str) -> float:
    """Lowest collision surface z, using mesh vertices rather than an AABB.

    A rotated mesh AABB can extend below its convex hull. The minimum of
    transformed mesh vertices equals the support of the collision convex hull.
    """
    m = scene.model
    lowest = float("inf")
    for g in scene.tip_geom_ids[finger]:
        rot = data.geom_xmat[g].reshape(3, 3)
        pos = data.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = int(m.geom_dataid[g])
            start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            vertices = m.mesh_vert[start:start + count]
            lowest = min(lowest, float(np.min(vertices @ rot[2]) + pos[2]))
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("clearance helper requires LEAP mesh/box collision geoms")
        center = m.geom_aabb[g, 0:3]
        half = m.geom_aabb[g, 3:6]
        rot = data.geom_xmat[g].reshape(3, 3)
        pos = data.geom_xpos[g]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    corner_local = center + half * np.array([sx, sy, sz])
                    corner_world = pos + rot @ corner_local
                    lowest = min(lowest, float(corner_world[2]))
    return lowest


def measure_working_pose(
    scene: TapSlideScene, pose: InitialPoseConfig
) -> dict:
    """Named FK measurement of the curl pose at wrist=0 (no dynamics)."""
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    q = scene.open_pose_qpos()
    wanted = pose.qpos17()
    for i in range(m.njnt):
        name = m.joint(i).name
        if name in wanted:
            q[m.jnt_qposadr[i]] = wanted[name]
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    tips = scene.fingertip_positions(d)
    knuckle = d.body("if_bs").xpos.copy()
    wrist = d.body(WRIST_BODY).xpos.copy()
    return {
        "qpos": scene.named_qpos(d),
        "tips": {k: [float(v) for v in p] for k, p in tips.items()},
        "index_lowest_collision_z_m": _lowest_collision_z(scene, d, "index"),
        "middle_lowest_collision_z_m": _lowest_collision_z(scene, d, "middle"),
        "wrist_xpos_m": [float(v) for v in wrist],
        "knuckle_xpos_m": [float(v) for v in knuckle],
        "palm_xpos_m": [float(v) for v in d.body(PALM_BODY).xpos],
        "palm_xmat": [float(v) for v in d.body(PALM_BODY).xmat],
    }


def load_working_scene(
    pose: InitialPoseConfig | None = None,
    mount: WorkingMountConfig | None = None,
    base: SceneConfig | None = None,
) -> tuple[TapSlideScene, dict]:
    """Build the WORKING scene: inclined palm + FK-computed mount translation.

    One analytic pass: build with the nominal wrist position and palm
    inclination, measure the curled pose via named FK, then translate the
    wrist so that (a) the midpoint of the index/middle curled tips sits at
    the configured contact-zone centre and (b) the index lowest collision
    surface is ``index_clearance_m`` above the pad top. FK is exactly
    equivariant in the wrist translation, so a single pass converges; no
    pose scan.
    """
    pose = pose or InitialPoseConfig()
    mount = mount or WorkingMountConfig()
    base = base or SceneConfig()
    incl = -np.deg2rad(mount.palm_inclination_deg)  # +X toward fingers rises
    wrist_cfg = replace(base.wrist,
        wrist_pos=mount.nominal_wrist_pos,
        palm_rel_quat=_axis_angle_quat((0.0, 1.0, 0.0), incl),
    )
    cfg_a = replace(base, wrist=wrist_cfg)
    scene_a = load_scene(cfg_a)
    meas = measure_working_pose(scene_a, pose)

    pad_top = base.pad_center[2] + base.pad_half_extents[2]
    idx = np.asarray(meas["tips"]["index"])
    mid = np.asarray(meas["tips"]["middle"])
    mid_xy = 0.5 * (idx[:2] + mid[:2])
    target_xy = np.asarray(mount.contact_zone_center, dtype=float)
    shift_xy = target_xy - mid_xy
    shift_z = (pad_top + mount.index_clearance_m) - meas["index_lowest_collision_z_m"]
    shift = np.array([shift_xy[0], shift_xy[1], shift_z])

    # Wrist camera: desired world eye/target at the working reset, converted
    # to palm-local frames for the spec.
    cam_eye_world = np.asarray(target_xy.tolist() + [pad_top]) + np.asarray(mount.cam_eye_offset)
    cam_target_world = np.asarray(target_xy.tolist() + [pad_top]) + np.asarray(mount.cam_target_offset)
    palm_xpos = np.asarray(meas["palm_xpos_m"]) + shift
    # Use measured world orientation, including any configured wrist rotation.
    rot_palm = np.asarray(meas["palm_xmat"]).reshape(3, 3)
    cam_eye_local = rot_palm.T @ (cam_eye_world - palm_xpos)
    cam_target_local = rot_palm.T @ (cam_target_world - palm_xpos)

    new_wrist_pos = tuple(
        float(v) for v in (np.asarray(mount.nominal_wrist_pos) + shift
                          + np.array([0.0, 0.0, mount.height_offset_m]))
    )
    cfg_b = replace(base,
        wrist=replace(wrist_cfg,
            wrist_pos=new_wrist_pos,
            palm_rel_quat=_axis_angle_quat((0.0, 1.0, 0.0), incl),
        ),
        wrist_cam_local_pos=tuple(float(v) for v in cam_eye_local),
        wrist_cam_local_target=tuple(float(v) for v in cam_target_local),
    )
    scene_b = load_scene(cfg_b)
    meas_b = measure_working_pose(scene_b, pose)
    info = {
        "pose": {k: float(v) for k, v in pose.qpos17().items()},
        "mount": {
            "palm_inclination_deg": mount.palm_inclination_deg,
            "index_clearance_target_m": mount.index_clearance_m + mount.height_offset_m,
            "base_contact_clearance_m": mount.index_clearance_m,
            "height_offset_m": mount.height_offset_m,
            "contact_zone_center": list(mount.contact_zone_center),
            "nominal_wrist_pos": list(mount.nominal_wrist_pos),
            "computed_wrist_pos": list(new_wrist_pos),
            "translation": [float(v) for v in shift],
            "pad_top_m": float(pad_top),
        },
        "measured_nominal": meas,
        "measured_working": meas_b,
        "wrist_cam": {
            "local_pos": [float(v) for v in cam_eye_local],
            "local_target": [float(v) for v in cam_target_local],
            "world_eye_at_reset": [float(v) for v in cam_eye_world + [0, 0, mount.height_offset_m]],
            "world_target_at_reset": [float(v) for v in cam_target_world + [0, 0, mount.height_offset_m]],
        },
    }
    return scene_b, info
