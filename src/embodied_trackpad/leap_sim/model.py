"""LEAP Hand v2 Advanced MuJoCo import (pinned upstream, declared local adaptations).

Source: https://github.com/leap-hand/LEAP_Hand_V2_Adv_API @ a0936196 (right hand
URDF + STL meshes). The upstream files under assets/downloads are NEVER modified;
this module generates a derived URDF in the ignored download root with the
following disclosed adaptations:

1. ``package:///name.stl`` mesh URIs are rewritten to local relative paths.
2. Meshes above MuJoCo's STL decoder cap (200000 faces) are converted losslessly
   (same triangles, normals dropped) to OBJ cached under ``derived/``.
3. The 10 auxiliary IK marker links (``realtip*``/``realdip*``, 0.1 kg each,
   1.0 kg total that does not exist on the hardware) are removed and their
   frames are preserved as massless, non-colliding MuJoCo sites.
4. The right URDF ships no ``<mimic>`` elements. The left URDF at the same
   commit carries ``*_pip -> *_dip`` mimics with multiplier 1.0 (its
   ``middle_pip -> index_dip`` entry is an upstream typo; every other finger is
   self-coupled and the hardware has one curl motor per non-thumb finger).
   The coupling is reconstructed here as 4 per-finger MuJoCo joint equalities
   ``<finger>_dip = <finger>_pip`` (rigid 1:1, reviewed approximation; soft
   tendon system identification is future work).
5. 17 position-servo actuators follow the upstream motor order
   (index/middle/ring/pinky [side, forward, curl->dip], thumb
   [side, forward, ip], palm_thumb, palm_4_finger). Actuator gains and joint
   damping are PLACEHOLDER simulation values, not hardware calibration.

URDF inertials are upstream Onshape CAD exports; whether they describe the Metal
or Plastic v2 Advanced build is unknown, so Metal dynamics stay UNCALIBRATED.
"""
from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from embodied_trackpad.leap_assets import LeapAssetError
from embodied_trackpad.leap_assets import verify_manifest as _verify_pinned_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
DOWNLOAD_ROOT = REPO_ROOT / "assets" / "downloads" / "leap_hand_v2_adv"
MANIFEST_JSON = REPO_ROOT / "assets" / "manifests" / "leap-selected.json"
URDF = DOWNLOAD_ROOT / "leap_v2_right" / "robot.urdf"
DERIVED_DIR = DOWNLOAD_ROOT / "leap_v2_right" / "derived"
DERIVED_URDF = DOWNLOAD_ROOT / "leap_v2_right" / "robot_mujoco_derived.urdf"

STL_FACE_CAP = 200000  # MuJoCo binary-STL decoder limit

FINGERS = ("index", "middle", "ring", "pinky")

# Upstream 17-motor command order (README "Direct Motor Commands"). The curl
# channel drives the finger DIP joint; the coupled PIP follows via equality.
MOTOR_JOINTS = []
for _f in FINGERS:
    MOTOR_JOINTS += [f"{_f}_mcps", f"{_f}_mcpf", f"{_f}_dip"]
MOTOR_JOINTS += ["thumb_mcps", "thumb_mcpf", "thumb_ip", "palm_thumb", "palm_4_finger"]

PASSIVE_COUPLED_JOINTS = [f"{_f}_pip" for _f in FINGERS]

MARKER_LINK_RE = re.compile(
    r'<link name="((?:realtip|realdip)[^"]*)">.*?</link>', re.DOTALL
)
MARKER_JOINT_RE = re.compile(
    r'<joint name="((?:realtip|realdip)_joint[^"]*)" type="fixed">.*?</joint>', re.DOTALL
)

# Placeholder servo gains (NOT hardware calibration). Tuned only for stable,
# bounded CPU motion of the light CAD links at TIMESTEP.
FINGER_KP = 1.0
PALM_KP = {"palm_thumb": 10.0, "palm_4_finger": 12.0}
FINGER_DAMPING = 0.02
PALM_DAMPING = 0.3
TIMESTEP = 5e-4

# Open pose sits slightly inside the one-sided joint ranges so the position
# servos (not the soft limit constraints) carry gravity. Applied once at reset;
# all subsequent motion goes through ctrl/mj_step. It satisfies dip==pip.
OPEN_POSE = {
    **{f"{_f}_pip": 0.05 for _f in FINGERS},
    **{f"{_f}_dip": 0.05 for _f in FINGERS},
    **{f"{_f}_mcpf": 0.05 for _f in FINGERS},
    "thumb_ip": 0.05,
    "thumb_mcpf": 0.05,
}


def open_pose_qpos(model: mujoco.MjModel) -> np.ndarray:
    q = np.zeros(model.nq)
    for j in range(model.njnt):
        name = model.joint(j).name
        if name in OPEN_POSE:
            q[model.jnt_qposadr[j]] = OPEN_POSE[name]
    return q

TIP_SITES = ("realtip", "realtip_2", "realtip_3", "realtip_4", "realtip_5")

# Mechanical-neighbor contact exclusion table (documented adaptation, NOT a
# blanket self-collision disable). These exclusions are a TEMPORARY
# convex-hull approximation of the CAD: imported meshes are convex hulls with
# holes/cavities filled, so they are not accurate flexible-finger collision
# geometry. Diagnosis: .local/leap-correction/initial-contacts.json. All pairs
# below are rigidly assembled neighbors whose hulls interpenetrate by 6-30 mm
# at the open reset pose:
#   - ROOT_BODY (no_finger_frame_mirror) is the fixed root link; it is kept as
#     a named static body (compiler fusestatic=False) instead of being welded
#     into world, so the exclusions target exactly the physical base and a
#     future world-mounted pad can still collide with the palm. Its hull fills
#     the palm cavity where the two articulated palm frames sit (-30/-25 mm).
#   - 4_finger_frame_mirror vs each finger_v2_with_motor: the palm-frame hull
#     fills the motor cavity; these are MCP grandparent links (-7..-13 mm).
#   - 4_finger_frame_mirror vs thumb_frame_mirror: assembled palm seam
#     (-7.3 mm).
#   - thumb_frame_mirror vs thumb_with_motor: thumb MCP grandparent (-12.7 mm).
# Cross-finger, fingertip-palm (dip_tip/thumb_tip vs frames) and all future
# hand-pad contacts REMAIN ACTIVE.
ROOT_BODY = "no_finger_frame_mirror"
EXCLUDE_PAIRS = [
    (ROOT_BODY, "4_finger_frame_mirror"),
    (ROOT_BODY, "thumb_frame_mirror"),
    ("4_finger_frame_mirror", "thumb_frame_mirror"),
    ("4_finger_frame_mirror", "finger_v2_with_motor"),
    ("4_finger_frame_mirror", "finger_v2_with_motor_2"),
    ("4_finger_frame_mirror", "finger_v2_with_motor_3"),
    ("4_finger_frame_mirror", "finger_v2_with_motor_4"),
    ("thumb_frame_mirror", "thumb_with_motor"),
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(
    manifest_path: Path = MANIFEST_JSON, download_root: Path = DOWNLOAD_ROOT
) -> dict:
    """Verify downloaded assets against the committed pinned manifest.

    Delegates to the stdlib-only shared verifier
    :func:`embodied_trackpad.leap_assets.verify_manifest`; LeapAssetError is
    the shared class re-exported here for caller compatibility."""
    return _verify_pinned_manifest(manifest_path, download_root)


def _stl_face_count(path: Path) -> int:
    with path.open("rb") as fh:
        fh.seek(80)
        return struct.unpack("<I", fh.read(4))[0]


def _stl_to_obj(stl_path: Path, obj_path: Path) -> float:
    """Convert binary STL to OBJ with float32 roundtrip-exact coordinates.

    Returns the max absolute vertex deviation vs the raw float32 source values
    (expected 0.0 with 9-significant-digit formatting). Same triangle topology;
    normals are dropped (MuJoCo recomputes them).
    """
    n = _stl_face_count(stl_path)
    rec = np.frombuffer(
        stl_path.read_bytes(),
        dtype=np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")]),
        offset=84,
        count=n,
    )
    verts = rec["v"].reshape(-1, 3)
    max_err = 0.0
    with obj_path.open("w") as fh:
        for p in verts:
            fields = [f"{np.float32(v):.9g}" for v in p]
            for v, s in zip(p, fields):
                max_err = max(max_err, abs(float(v) - float(s)))
            fh.write(f"v {fields[0]} {fields[1]} {fields[2]}\n")
        for i in range(0, len(verts), 3):
            fh.write(f"f {i+1} {i+2} {i+3}\n")
    return max_err


def _derived_sidecar_check(sidecar: Path, source: Path, derived: Path, converter_id: str) -> bool:
    if not (sidecar.exists() and derived.exists()):
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return False
    return (
        meta.get("source_sha256") == _sha256(source)
        and meta.get("converter") == converter_id
        and meta.get("derived_sha256") == _sha256(derived)
    )


def _extract_markers(urdf_text: str) -> tuple[str, list[dict], float]:
    """Remove realtip/realdip marker links+joints; return (text, frames, removed_mass)."""
    frames = []
    removed_mass = 0.0
    for m in MARKER_LINK_RE.finditer(urdf_text):
        mass_m = re.search(r'<mass value="([^"]+)"', m.group(0))
        removed_mass += float(mass_m.group(1)) if mass_m else 0.0
    for m in MARKER_JOINT_RE.finditer(urdf_text):
        block = m.group(0)
        parent = re.search(r'<parent link="([^"]+)"', block).group(1)
        child = re.search(r'<child link="([^"]+)"', block).group(1)
        xyz = re.search(r'<origin xyz="([^"]+)"', block).group(1)
        frames.append(
            {"site": child, "parent_body": parent, "pos": [float(v) for v in xyz.split()]}
        )
    text = MARKER_LINK_RE.sub("", urdf_text)
    text = MARKER_JOINT_RE.sub("", text)
    return text, frames, removed_mass


def prepare_derived_urdf(download_root: Path = DOWNLOAD_ROOT) -> tuple[Path, list[dict], float]:
    """Generate the derived URDF deterministically; returns (path, marker_frames, removed_mass_kg).

    Derived artifacts are cache-validated by a sidecar binding
    source-sha256 + converter-id + derived-sha256; any mismatch forces a
    deterministic rebuild, so stale or tampered derived content cannot evade
    the raw-asset manifest checks.
    """
    urdf_path = download_root / "leap_v2_right" / "robot.urdf"
    if not urdf_path.exists():
        raise LeapAssetError(f"upstream URDF missing: {urdf_path}; run tools/fetch_leap_assets.py")
    derived_dir = download_root / "leap_v2_right" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    converter_id = f"leap_sim.model/{_sha256(Path(__file__).resolve())}"

    text = urdf_path.read_text()
    text = re.sub(r"package:///+", "", text)
    obj_max_err = {}
    for stl in sorted((download_root / "leap_v2_right").glob("*.stl")):
        if _stl_face_count(stl) > STL_FACE_CAP:
            obj = derived_dir / (stl.stem + ".obj")
            sidecar = derived_dir / (stl.stem + ".obj.json")
            if not _derived_sidecar_check(sidecar, stl, obj, converter_id):
                max_err = _stl_to_obj(stl, obj)
                sidecar.write_text(
                    json.dumps(
                        {
                            "source_sha256": _sha256(stl),
                            "converter": converter_id,
                            "derived_sha256": _sha256(obj),
                            "max_vertex_error_m": max_err,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            else:
                meta = json.loads(sidecar.read_text())
                max_err = float(meta.get("max_vertex_error_m", 0.0))
            obj_max_err[obj.name] = max_err
            text = text.replace(stl.name, f"derived/{obj.name}")
    text, frames, removed_mass = _extract_markers(text)
    out = download_root / "leap_v2_right" / "robot_mujoco_derived.urdf"
    sidecar = download_root / "leap_v2_right" / "robot_mujoco_derived.urdf.json"
    if not _derived_sidecar_check(sidecar, urdf_path, out, converter_id + "/urdf"):
        out.write_text(text)
        sidecar.write_text(
            json.dumps(
                {
                    "source_sha256": _sha256(urdf_path),
                    "converter": converter_id + "/urdf",
                    "derived_sha256": _sha256(out),
                    "obj_max_vertex_error_m": obj_max_err,
                    "removed_marker_mass_kg": removed_mass,
                },
                indent=2,
            )
            + "\n"
        )
    return out, frames, removed_mass


@dataclass
class LeapHand:
    model: mujoco.MjModel
    marker_frames: list[dict]
    removed_marker_mass_kg: float
    manifest: dict
    adaptations: list[str] = field(default_factory=list)

    @property
    def joint_names(self) -> list[str]:
        return [self.model.joint(i).name for i in range(self.model.njnt)]

    def fingertip_positions(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        return {name: data.site(name).xpos.copy() for name in TIP_SITES}

    def tip_distance(self, data: mujoco.MjData, a: str = "realtip", b: str = "realtip_5") -> float:
        return float(np.linalg.norm(data.site(a).xpos - data.site(b).xpos))


def load_hand(
    download_root: Path = DOWNLOAD_ROOT,
    manifest_path: Path = MANIFEST_JSON,
    spec_hook: "callable | None" = None,
) -> LeapHand:
    manifest = verify_manifest(manifest_path, download_root)
    urdf_path, frames, removed_mass = prepare_derived_urdf(download_root)
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.option.timestep = TIMESTEP
    # Keep the fixed root link as a named static body so contact exclusions
    # target the physical base instead of the whole world body (which would
    # also shield any future world-mounted pad geom from palm collisions).
    # Static bodies carry no joints; 21 joints / 17 motors are unaffected.
    spec.compiler.fusestatic = False

    for jnt in spec.joints:
        damp = np.zeros(3)
        damp[0] = PALM_DAMPING if jnt.name in PALM_KP else FINGER_DAMPING
        jnt.damping = damp

    for f in FINGERS:
        eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name=f"tendon_{f}_dip_pip")
        eq.name1 = f"{f}_dip"
        eq.name2 = f"{f}_pip"
        eq.data[0] = 0.0
        eq.data[1] = 1.0
        eq.solref = [0.001, 1.0]

    for b1, b2 in EXCLUDE_PAIRS:
        spec.add_exclude(bodyname1=b1, bodyname2=b2)

    for geom in spec.geoms:
        geom.solref = [0.005, 1.0]

    light = spec.worldbody.add_light(name="key_light")
    light.pos = [0.3, -0.5, 0.8]
    light.dir = [-0.3, 0.5, -0.8]
    light.diffuse = [0.9, 0.9, 0.9]

    joint_range = {}
    effort_limit = {}
    for jnt in spec.joints:
        joint_range[jnt.name] = list(jnt.range)
    # effort limits from the URDF limits (MuJoCo stores them on the joint actuator
    # force range only when an actuator exists), so read them from the raw text.
    raw = (download_root / "leap_v2_right" / "robot.urdf").read_text()
    for jm in re.finditer(
        r'<joint name="([^"]+)" type="revolute">(.*?)</joint>', raw, re.DOTALL
    ):
        lim = re.search(r'<limit effort="([^"]+)"', jm.group(2))
        if lim:
            effort_limit[jm.group(1)] = float(lim.group(1))

    for jname in MOTOR_JOINTS:
        act = spec.add_actuator(name=f"motor_{jname}")
        kp = PALM_KP.get(jname, FINGER_KP)
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.gainprm[0] = kp
        act.biasprm[0] = 0.0
        act.biasprm[1] = -kp
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        act.target = jname
        act.ctrllimited = True
        act.ctrlrange = joint_range[jname]
        act.forcelimited = True
        lo, hi = -effort_limit[jname], effort_limit[jname]
        act.forcerange = [lo, hi]

    for frame in frames:
        body = spec.body(frame["parent_body"])
        site = body.add_site(name=frame["site"], pos=frame["pos"])
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.size = [0.002, 0.0, 0.0]

    if spec_hook is not None:
        spec_hook(spec)
    model = spec.compile()
    return LeapHand(
        model=model,
        marker_frames=frames,
        removed_marker_mass_kg=removed_mass,
        manifest=manifest,
        adaptations=[
            "package:/// mesh URIs rewritten to local relative paths",
            "STL meshes above MuJoCo's 200000-face cap converted losslessly to OBJ (same triangles)",
            f"10 IK marker links removed ({removed_mass:.3f} kg of non-hardware mass); frames preserved as massless sites",
            "4 per-finger dip=pip joint equalities added (right URDF has no mimic; left-URDF 1:1 mimics as evidence, middle_pip->index_dip upstream typo corrected)",
            f"{len(EXCLUDE_PAIRS)} mechanical-neighbor contact exclusions for assembled pairs whose convex-hull CAD meshes interpenetrate (diagnosed in initial-contacts.json); TEMPORARY hull approximation, not accurate flexible-finger collision geometry; cross-finger/tip-palm/pad contacts stay active",
            "17 position servos in upstream motor order; gains/damping are placeholders, not hardware calibration",
            f"fixed root link {ROOT_BODY} (0.140 kg) kept as a named static body via compiler fusestatic=False so base exclusions do not shield future world-mounted pad geoms; CAD total 1.505 kg splits into 0.140 kg static + 1.365 kg dynamic mass",
        ],
    )


def model_inventory(hand: LeapHand) -> dict:
    m = hand.model
    joints = []
    for i in range(m.njnt):
        name = m.joint(i).name
        cls = "actuated_motor" if name in MOTOR_JOINTS else "passive_coupled"
        joints.append(
            {
                "id": i,
                "name": name,
                "qposadr": int(m.jnt_qposadr[i]),
                "range": [float(v) for v in m.jnt_range[i]],
                "damping": float(m.dof_damping[m.jnt_dofadr[i]]),
                "class": cls,
            }
        )
    actuators = []
    for i in range(m.nu):
        trg = m.actuator_trnid[i, 0]
        actuators.append(
            {
                "id": i,
                "motor_index": i,
                "name": m.actuator(i).name,
                "joint": m.joint(trg).name,
                "ctrlrange": [float(v) for v in m.actuator_ctrlrange[i]],
                "forcerange": [float(v) for v in m.actuator_forcerange[i]],
                "kp": float(m.actuator_gainprm[i, 0]),
            }
        )
    equalities = []
    for i in range(m.neq):
        eq = m.eq(i)
        equalities.append(
            {
                "name": eq.name,
                "type": "joint",
                "joint1": m.joint(m.eq_obj1id[i]).name,
                "joint2": m.joint(m.eq_obj2id[i]).name,
                "polycoef": [float(v) for v in m.eq_data[i, :5]],
                "note": "1:1 rigid approximation of the hardware PIP-DIP tendon; ratio UNCALIBRATED",
            }
        )
    root_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    total_cad_mass = float(m.body_mass[1:].sum())  # all bodies except worldbody
    fixed_root_mass = float(m.body_mass[root_id])
    return {
        "model": "LEAP Hand v2 Advanced, right hand, fixed palm (hand-only scene)",
        "nq": int(m.nq),
        "nv": int(m.nv),
        "nu": int(m.nu),
        "njnt": int(m.njnt),
        "ngeom": int(m.ngeom),
        "timestep": float(m.opt.timestep),
        "dof_split": {
            "hardware_motors": 17,
            "model_revolute_joints": int(m.njnt),
            "model_actuators": int(m.nu),
            "passive_coupled_joints": len(PASSIVE_COUPLED_JOINTS),
            "upstream_pose_interface_dofs": 20,
        },
        "mass_kg": {
            "total_cad": total_cad_mass,
            "fixed_root": fixed_root_mass,
            "moving_links": total_cad_mass - fixed_root_mass,
        },
        "removed_marker_mass_kg": hand.removed_marker_mass_kg,
        "joints": joints,
        "actuators": actuators,
        "equalities": equalities,
        "sensors": [],
        "sensor_note": "No sensor channels added; none are invented beyond what a future accepted task declares.",
        "sites": [m.site(i).name for i in range(m.nsite)],
        "calibration": "URDF inertials are upstream CAD exports; Metal/Plastic identity UNCALIBRATED. Actuator gains/damping are placeholders.",
        "adaptations": hand.adaptations,
    }
