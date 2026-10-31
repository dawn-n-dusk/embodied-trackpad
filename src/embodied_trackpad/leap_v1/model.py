"""LEAP Hand v1 Full MuJoCo import (pinned MuJoCo Menagerie ``leap_hand``).

Source: https://github.com/google-deepmind/mujoco_menagerie @ a662de6
(``leap_hand/right_hand.xml`` + 10 OBJ meshes, MIT licensed). The upstream
files under ``assets/downloads/leap_v1_full`` are NEVER modified. The upstream
MJCF is used verbatim except for the following disclosed adaptations:

1. Four massless, non-colliding fingertip sites (``tip_index``, ``tip_middle``,
   ``tip_ring``, ``tip_thumb``) are added on the four distal phalanx bodies at
   the compiled frame of the upstream fingertip mesh geoms
   (``if_tip``/``mf_tip``/``rf_tip``/``th_tip``) so fingertip motion can be
   traced without touching collision geometry.
2. One render light is added. Nothing else is changed: upstream joint ranges,
   solver options (implicitfast, elliptic cone, impratio=100, 2 ms timestep),
   the 18 upstream contact exclusions, box-primitive collision geometry and
   the 16 upstream position actuators are all retained.

Calibration status (explicit, per ADR 0007): actuator gains kp=3.0 / kv=0.01,
joint damping 0.03, frictionloss 0.001, all link masses/inertias and the
box-primitive collision geometry are UPSTREAM SIMULATION DEFAULTS. They are
NOT XC330-M288-T motor calibrations and NOT measured hardware inertias. This
model is a fixed-palm, right-hand, four-finger, 16-independent-channel
physical baseline; no tendon coupling exists in v1 (unlike v2's four rigid
dip=pip equalities, which are NOT present here and must not be reintroduced
from v2 results).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from embodied_trackpad.leap_v1 import assets as _assets
from embodied_trackpad.leap_v1.assets import LeapAssetError

REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_ROOT = _assets.DOWNLOAD_ROOT
MANIFEST_JSON = _assets.MANIFEST_JSON
RIGHT_HAND_XML = DOWNLOAD_ROOT / "right_hand" / "right_hand.xml"

EMBODIMENT_ID = "leap_v1_full_right_menagerie_v1"
CALIBRATION_ID = "uncalibrated_upstream_defaults_v1"

FINGERS = ("index", "middle", "ring", "thumb")
FINGER_JOINTS = {
    "index": ["if_mcp", "if_rot", "if_pip", "if_dip"],
    "middle": ["mf_mcp", "mf_rot", "mf_pip", "mf_dip"],
    "ring": ["rf_mcp", "rf_rot", "rf_pip", "rf_dip"],
    "thumb": ["th_cmc", "th_axl", "th_mcp", "th_ipl"],
}
# Upstream 16-motor command order (right_hand.xml actuator block).
MOTOR_JOINTS = [j for f in FINGERS for j in FINGER_JOINTS[f]]

TIP_GEOMS = {"index": "if_tip", "middle": "mf_tip", "ring": "rf_tip", "thumb": "th_tip"}
TIP_SITES = {f: f"tip_{f}" for f in FINGERS}
PALM_BODY = "palm"

# The open pose is the all-zero configuration: 0 rad lies strictly inside
# every upstream joint range and corresponds to the flat open hand. Applied
# once at reset; all subsequent motion goes through ctrl/mj_step only.
OPEN_POSE = {j: 0.0 for j in MOTOR_JOINTS}


def open_pose_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = np.zeros(model.nq)
    for j in range(model.njnt):
        name = model.joint(j).name
        if name in OPEN_POSE:
            q[model.jnt_qposadr[j]] = OPEN_POSE[name]
    return q


def verify_manifest(
    manifest_path: Path = MANIFEST_JSON, download_root: Path = DOWNLOAD_ROOT
) -> dict:
    """Verify downloaded v1 assets against the committed pinned manifest."""
    return _assets.verify_manifest(manifest_path, download_root)


def _tip_site_offsets(model: mujoco.MjModel) -> dict[str, tuple[str, np.ndarray]]:
    """Local (body-frame) offsets of the four upstream tip mesh geoms.

    Mesh geom frames are re-centred at the mesh centre of mass during
    compilation, so the offsets are measured from a compiled throwaway model
    instead of being guessed from the XML."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    out: dict[str, tuple[str, np.ndarray]] = {}
    for finger, gname in TIP_GEOMS.items():
        gid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname))
        if gid < 0:
            raise LeapAssetError(f"tip geom missing in compiled model: {gname}")
        bid = int(model.geom_bodyid[gid])
        rel = np.asarray(d.geom_xpos[gid]) - np.asarray(d.xpos[bid])
        local = np.asarray(d.xmat[bid]).reshape(3, 3).T @ rel
        out[finger] = (model.body(bid).name, local)
    return out


@dataclass
class LeapV1Hand:
    model: mujoco.MjModel
    manifest: dict
    adaptations: list[str] = field(default_factory=list)

    @property
    def joint_names(self) -> list[str]:
        return [self.model.joint(i).name for i in range(self.model.njnt)]

    def fingertip_positions(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {f: data.site(TIP_SITES[f]).xpos.copy() for f in FINGERS}

    def tip_distance(self, data: mujoco.MjData, a: str = "index", b: str = "middle") -> float:
        return float(
            np.linalg.norm(data.site(TIP_SITES[a]).xpos - data.site(TIP_SITES[b]).xpos)
        )


def load_hand(
    download_root: Path = DOWNLOAD_ROOT,
    manifest_path: Path = MANIFEST_JSON,
    spec_hook: "callable | None" = None,
) -> LeapV1Hand:
    manifest = verify_manifest(manifest_path, download_root)
    xml_path = Path(download_root) / "right_hand" / "right_hand.xml"
    if not xml_path.exists():
        raise LeapAssetError(
            f"upstream MJCF missing: {xml_path}; run tools/fetch_leap_v1_assets.py"
        )
    spec = mujoco.MjSpec.from_file(str(xml_path))

    # Adaptation 1: fingertip sites at the compiled tip-geom frames.
    offsets = _tip_site_offsets(spec.compile())
    for finger, (body_name, local) in offsets.items():
        body = spec.body(body_name)
        site = body.add_site(name=TIP_SITES[finger], pos=[float(v) for v in local])
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.size = [0.002, 0.0, 0.0]
        site.rgba = [0.9, 0.2, 0.2, 1.0]

    # Adaptation 2: render light.
    light = spec.worldbody.add_light(name="key_light")
    light.pos = [0.3, -0.5, 0.8]
    light.dir = [-0.3, 0.5, -0.8]
    light.diffuse = [0.9, 0.9, 0.9]

    if spec_hook is not None:
        spec_hook(spec)
    model = spec.compile()

    # Structural contract: exactly 16 actuators, each driving exactly one
    # distinct joint, in the upstream command order; no coupling equalities.
    if model.nu != 16:
        raise LeapAssetError(f"expected 16 actuators, got {model.nu}")
    act_joints = [model.joint(model.actuator_trnid[i, 0]).name for i in range(model.nu)]
    if act_joints != MOTOR_JOINTS:
        raise LeapAssetError(f"actuator order mismatch: {act_joints} != {MOTOR_JOINTS}")
    if len(set(act_joints)) != 16:
        raise LeapAssetError("actuator->joint mapping is not one-to-one")
    if model.neq != 0:
        raise LeapAssetError(f"v1 model must have no joint equalities, got {model.neq}")
    for finger, sname in TIP_SITES.items():
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname) < 0:
            raise LeapAssetError(f"tip site missing after compile: {sname}")

    return LeapV1Hand(
        model=model,
        manifest=manifest,
        adaptations=[
            "4 massless fingertip sites (tip_index/tip_middle/tip_ring/tip_thumb) added on the distal bodies at the compiled frames of the upstream tip mesh geoms; no collision geometry touched",
            "1 render light added",
            "everything else verbatim upstream: joint ranges, implicitfast integrator, elliptic cone, impratio=100, 2 ms timestep, 18 contact exclusions, box-primitive collision geometry, 16 position actuators",
        ],
    )


def model_inventory(hand: LeapV1Hand) -> dict:
    m = hand.model
    joints = []
    for i in range(m.njnt):
        name = m.joint(i).name
        finger = next((f for f, js in FINGER_JOINTS.items() if name in js), None)
        dof = int(m.jnt_dofadr[i])
        joints.append(
            {
                "id": i,
                "name": name,
                "finger": finger,
                "qposadr": int(m.jnt_qposadr[i]),
                "axis": [float(v) for v in m.jnt_axis[i]],
                "range": [float(v) for v in m.jnt_range[i]],
                "damping": float(m.dof_damping[dof]),
                "frictionloss": float(m.dof_frictionloss[dof]),
            }
        )
    actuators = []
    for i in range(m.nu):
        trg = int(m.actuator_trnid[i, 0])
        actuators.append(
            {
                "id": i,
                "motor_index": i,
                "name": m.actuator(i).name,
                "joint": m.joint(trg).name,
                "ctrlrange": [float(v) for v in m.actuator_ctrlrange[i]],
                "kp": float(m.actuator_gainprm[i, 0]),
                "kv": float(-m.actuator_biasprm[i, 2]),
                "forcerange": [float(v) for v in m.actuator_forcerange[i]],
                "gain_note": "upstream simulation default, NOT XC330-M288-T calibration",
            }
        )
    exclusions = []
    for i in range(m.nexclude):
        b1 = m.body(int(m.exclude_signature[i] & 0xFFFF)).name
        b2 = m.body(int(m.exclude_signature[i] >> 16)).name
        exclusions.append([b1, b2])
    geoms_visual = sum(1 for g in range(m.ngeom) if int(m.geom_group[g]) == 1)
    mesh_files = sorted(
        {entry["path"] for entry in hand.manifest["files"] if entry["path"].endswith(".obj")}
    )
    return {
        "model": "LEAP Hand v1 Full, right hand, fixed palm (hand-only scene)",
        "embodiment_id": EMBODIMENT_ID,
        "calibration_id": CALIBRATION_ID,
        "source": {
            "repository": hand.manifest["upstream"]["repository"],
            "subdirectory": hand.manifest["upstream"].get("subdirectory"),
            "pinned_commit": hand.manifest["upstream"]["pinned_commit"],
            "license": "MIT (leap_hand/LICENSE, retained verbatim)",
        },
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "njnt": int(m.njnt),
        "ngeom": int(m.ngeom),
        "ngeom_visual": int(geoms_visual),
        "ngeom_collision": int(m.ngeom - geoms_visual),
        "nmesh": int(m.nmesh),
        "neq": int(m.neq),
        "nexclude": int(m.nexclude),
        "timestep": float(m.opt.timestep),
        "integrator": str(m.opt.integrator).split(".")[-1],
        "cone": str(m.opt.cone).split(".")[-1],
        "impratio": float(m.opt.impratio),
        "gravity": [float(v) for v in m.opt.gravity],
        "dof_split": {
            "hardware_motors": 16,
            "model_revolute_joints": int(m.njnt),
            "model_actuators": int(m.nu),
            "passive_coupled_joints": 0,
            "fingers": 4,
        },
        "mass_kg": {
            "total": float(m.body_mass[1:].sum()),
            "note": "upstream MJCF inertials; not measured hardware values",
        },
        "joints": joints,
        "actuators": actuators,
        "contact_exclusions": exclusions,
        "mesh_files": mesh_files,
        "sensors": [
            {"name": m.sensor(i).name, "type": str(m.sensor_type[i]).split(".")[-1]}
            for i in range(m.nsensor)
        ],
        "sites": [m.site(i).name for i in range(m.nsite)],
        "calibration": (
            "Actuator gains kp=3.0/kv=0.01, joint damping 0.03, frictionloss 0.001, "
            "link masses/inertias and box-primitive collision geometry are UPSTREAM "
            "SIMULATION DEFAULTS, uncalibrated for the XC330-M288-T v1 Full hardware."
        ),
        "adaptations": hand.adaptations,
    }
