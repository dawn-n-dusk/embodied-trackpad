"""Static touch-pad contact environment for the LEAP v1 Full hand.

Extends :func:`embodied_trackpad.leap_v1.model.load_hand` through its public
``spec_hook`` with ONE static world-mounted box geom (a 160 x 110 mm trackpad,
10 mm thick) and ONE fixed palm re-mount. The accepted v1 hand is otherwise
untouched: same upstream MJCF, 16 joints / 16 independent position actuators,
no coupling equalities, the 18 upstream contact exclusions, upstream joint
ranges, gains (kp=3.0/kv=0.01), damping, frictionloss, masses/inertias and
box-primitive collision geometry. No contact pair is excluded and no collision
is disabled; the pad is an ordinary colliding geom welded to the worldbody, so
it is static by construction and never moves during an episode.

Fixed mounting transform (scene-level, documented, NOT a hardware
calibration): the upstream model is palm-up with fingers pointing +x and
flexion curling +z, so a horizontal top-face pad cannot be reached without
colliding with the palm (measured: palm collision boxes occupy the space below
the open fingers). The palm body is therefore re-mounted palm-down with the
scene transform ``PALM_POS``/``PALM_QUAT`` below (world-frame 180-degree
rotation about x relative to the upstream root pose, then raised). This adds
no wrist DOF: the palm still carries no joint and the 16-DOF structural
contract is unchanged. All numbers were measured on the compiled model
(.local/v1-pad/explore2.py, exploration only, not acceptance evidence):

* open-pose fingertip sites: index (0.1158, 0.0081, 0.2042), middle
  (0.1157, -0.0373, 0.2042), ring (0.1157, -0.0827, 0.2042), thumb
  (-0.0810, 0.1223, 0.2036) [m]
* lowest hand collision geometry at the open pose: z ~= 0.178 m, i.e.
  ~78 mm above the pad top face (z = 0.100 m); the open reset is contact-free
* index press pose if_mcp=0.60/if_pip=0.30/if_dip=0.30 rad gives a steady
  ~1.9 N top-face contact on the if_tip mesh geom (cos(normal, +z) = 1.0000,
  penetration ~1 um, contact centroid near (+0.072, +0.007) pad-frame metres)
* pad contact parameters match the upstream geom defaults and tip class:
  solref (0.0001, 1), solimp (0.999, 0.999, 0.001, 0.0001, 1),
  friction (0.5, 0.005, 0.0001).

Contact truth comes only from actual MuJoCo geom pairs: every step the active
``d.contact`` array is read together with ``mujoco.mj_contactForce`` and the
normal component is taken as the compressive force (positive = compressive, as
produced by the solver; NEVER passed through ``abs``; a negative normal force
is recorded as an anomaly). The full 6-axis force and the 9-element contact
frame are preserved in the raw evidence. Non-finite raw contact values (force,
position, frame, distance) are rejected BEFORE any aggregation. Only the four
named distal fingertip bodies (child bodies of the distal joints ``if_dip``,
``mf_dip``, ``rf_dip``, ``th_ipl``: ``if_ds``/``mf_ds``/``rf_ds``/``th_ds``) are
legitimate pad touchers, and only against the pad TOP face (pad->hand contact
normal aligned with the pad up axis +z). Side/back tip contacts, any
non-tip/pad contact and any hand self-contact are recorded separately as
unintended contacts. The 5 N safety gate uses the total positive normal force
over ALL pad contacts (top, side, back and non-tip alike).

Per finger and step, multiple positive top-face contact points aggregate into
one entry with the summed normal force and a force-weighted centroid in
measured pad-frame coordinates (metres, pad-centered) — exactly the touch.v1
contact shape ``{finger_id, x_m, y_m, normal_force_n}``. FSR is absent: the
normal force is device/evaluator truth, never an actor observation. The GUI is
NOT launched here; this is physics only.

This module is a REUSABLE scene/contact-extraction library: it loads the
scene, classifies and aggregates raw contacts, and builds touch.v1 frames. It
does NOT record or run episodes — recording, pacing and per-step safety stops
are the caller's responsibility (see
:mod:`embodied_trackpad.leap_v1.visual_physics` for the per-step validated
physics core that consumes this library). The historical scripted-episode
diagnostic evaluator that once lived here was removed from the public library;
its exact original source is preserved in private revision history and the
historical contact worktrees/receipts, and it is not an acceptance path.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from embodied_trackpad.leap_sim.pad_env import (  # generic helpers, reused unmodified
    validate_raw_contacts,
)
from embodied_trackpad.leap_v1.model import (
    FINGERS,
    MOTOR_JOINTS,
    TIP_SITES,
    LeapV1Hand,
    load_hand,
)

# --- scene geometry (all values measured on the compiled model; see docstring)
PAD_GEOM_NAME = "trackpad_160x110"
PAD_HALF_EXTENTS = (0.080, 0.055, 0.005)  # 160 x 110 mm face, 10 mm thick
DEFAULT_PAD_CENTER = (0.0, 0.0, 0.095)  # top face at z = 0.100 m
PALM_POS = (0.0, 0.0, 0.225)  # fixed palm-down re-mount (scene transform)
PALM_QUAT = (1.0, 0.0, 0.0, 0.0)  # identity: upstream (0,1,0,0) pre-rotated by Rx(180)

# Distal tip body = child body of the finger's last articulation (v1 body map).
FINGER_TIP_JOINT = {
    "index": "if_dip",
    "middle": "mf_dip",
    "ring": "rf_dip",
    "thumb": "th_ipl",
}
# Flexion joint triple per finger (mcp/pip/dip order; thumb uses cmc/mcp/ipl).
FINGER_FLEX_JOINTS = {
    "index": ("if_mcp", "if_pip", "if_dip"),
    "middle": ("mf_mcp", "mf_pip", "mf_dip"),
    "ring": ("rf_mcp", "rf_pip", "rf_dip"),
    "thumb": ("th_cmc", "th_mcp", "th_ipl"),
}
FINGER_ROT_JOINT = {"index": "if_rot", "middle": "mf_rot", "ring": "rf_rot"}

TOP_FACE_MIN_COS = 0.99  # cos(pad up axis +z, pad->hand contact normal)
TOUCH_HALF_X = 0.080
TOUCH_HALF_Y = 0.055
TOUCH_MAX_FRAME_INTERVAL_S = 0.05  # bounded-rate continuous touch.v1 stream

# Acceptance bounds (frozen diagnostic gates fixed by the bounded task).
LIMIT_EXCURSION_TOL = 0.01  # rad
PENETRATION_TOL = 0.001  # m
PAD_FORCE_MAX_N = 5.0
MIN_CLEARANCE_S = 0.3  # initial and final contact-free windows
MIN_CONTACT_FORCE_N = 0.05
MIN_CONTACT_SUSTAIN_S = 0.3
MIN_SLIDE_DISPLACEMENT_M = 0.005
PRESS_FORCE_N = 1.0  # existing software device press threshold (not an Apple measurement)
RELEASE_FORCE_N = 0.6  # existing software device release threshold

PAD_CONTACT_KINDS = ("tip_pad_top", "tip_pad_side", "tip_pad_back", "nontip_pad")


def pad_spec_hook(
    pad_center: tuple[float, float, float] = DEFAULT_PAD_CENTER,
    half_extents: tuple[float, float, float] = PAD_HALF_EXTENTS,
    palm_pos: tuple[float, float, float] = PALM_POS,
    palm_quat: tuple[float, float, float, float] = PALM_QUAT,
):
    """Return a spec_hook applying the fixed palm re-mount + static pad geom."""

    def hook(spec: mujoco.MjSpec) -> None:
        palm = spec.body("palm")
        palm.pos = list(palm_pos)
        palm.quat = list(palm_quat)
        geom = spec.worldbody.add_geom(
            name=PAD_GEOM_NAME,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(half_extents),
            pos=list(pad_center),
        )
        # Match the upstream geom defaults exactly (see module docstring).
        geom.solref = [0.0001, 1.0]
        geom.solimp = [0.999, 0.999, 0.001, 0.0001, 1.0]
        geom.friction = [0.5, 0.005, 0.0001]
        geom.rgba = [0.25, 0.28, 0.32, 1.0]

    return hook


@dataclass
class PadScene:
    hand: LeapV1Hand
    pad_geom_id: int
    pad_body_id: int
    pad_center: tuple[float, float, float]
    pad_half_extents: tuple[float, float, float]
    palm_pos: tuple[float, float, float]
    palm_quat: tuple[float, float, float, float]
    tip_geom_ids: dict[str, list[int]]  # finger -> colliding geom ids of the distal body
    tip_site: dict[str, str]  # finger -> tip_* site name on that body

    @property
    def model(self) -> mujoco.MjModel:
        return self.hand.model


def load_pad_scene(
    pad_center: tuple[float, float, float] = DEFAULT_PAD_CENTER,
    half_extents: tuple[float, float, float] = PAD_HALF_EXTENTS,
    palm_pos: tuple[float, float, float] = PALM_POS,
    palm_quat: tuple[float, float, float, float] = PALM_QUAT,
) -> PadScene:
    """Load the accepted v1 hand plus fixed palm transform + static pad."""
    hand = load_hand(spec_hook=pad_spec_hook(pad_center, half_extents, palm_pos, palm_quat))
    m = hand.model
    pad_geom_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAD_GEOM_NAME))
    if pad_geom_id < 0:
        raise RuntimeError("pad geom missing after spec_hook")
    tip_geom_ids: dict[str, list[int]] = {}
    tip_site: dict[str, str] = {}
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
        sname = TIP_SITES[finger]
        sid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname))
        if sid < 0 or int(m.site_bodyid[sid]) != body_id:
            raise RuntimeError(f"tip site {sname} missing or not on distal body of {finger}")
        tip_site[finger] = sname
    palm_body = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm"))
    if palm_body < 0 or int(m.body_jntnum[palm_body]) != 0:
        raise RuntimeError("palm must exist and carry no joint (no wrist DOF added)")
    return PadScene(
        hand=hand,
        pad_geom_id=pad_geom_id,
        pad_body_id=int(m.geom_bodyid[pad_geom_id]),
        pad_center=tuple(float(v) for v in pad_center),
        pad_half_extents=tuple(float(v) for v in half_extents),
        palm_pos=tuple(float(v) for v in palm_pos),
        palm_quat=tuple(float(v) for v in palm_quat),
        tip_geom_ids=tip_geom_ids,
        tip_site=tip_site,
    )


def scene_integrity(scene: PadScene) -> dict:
    """Structural checks proving the accepted v1 hand model is preserved."""
    m = scene.model
    act_joints = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    palm_body = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm"))
    return {
        "nq": int(m.nq),
        "njnt": int(m.njnt),
        "nu": int(m.nu),
        "neq": int(m.neq),
        "nexclude": int(m.nexclude),
        "motor_joint_order": act_joints,
        "motor_order_preserved": act_joints == MOTOR_JOINTS,
        "pad_body_is_world": scene.pad_body_id == 0,
        "pad_contype": int(m.geom_contype[scene.pad_geom_id]),
        "pad_conaffinity": int(m.geom_conaffinity[scene.pad_geom_id]),
        "pad_solref": [float(v) for v in m.geom_solref[scene.pad_geom_id]],
        "pad_friction": [float(v) for v in m.geom_friction[scene.pad_geom_id]],
        "palm_body_id": palm_body,
        "palm_joint_num": int(m.body_jntnum[palm_body]),
        "palm_pos": list(scene.palm_pos),
        "palm_quat": list(scene.palm_quat),
        "tip_geom_ids": {k: [int(g) for g in v] for k, v in scene.tip_geom_ids.items()},
        "tip_sites": dict(scene.tip_site),
    }


# --- per-step contact extraction --------------------------------------------

@dataclass
class RawContact:
    geom1: int
    geom2: int
    body1: str
    body2: str
    pos_world: list[float]
    frame: list[float]  # actual 9-element contact frame (normal, tangent1, tangent2)
    force6: list[float]  # actual 6-axis mj_contactForce wrench in the contact frame
    normal_force_n: float  # == force6[0], compressive positive, never abs
    dist_m: float
    pad_local: list[float] | None  # (x_m, y_m, z_m) in the pad frame, if pad involved
    top_face_cos: float | None  # cos(pad up axis, pad->hand normal), if pad involved
    kind: str  # tip_pad_top | tip_pad_side | tip_pad_back | nontip_pad | hand_self


def _body_name(m: mujoco.MjModel, geom_id: int) -> str:
    return m.body(int(m.geom_bodyid[int(geom_id)])).name or "world"


def classify_pad_contact(top_face_cos: float) -> str:
    """Top vs side vs back from the pad->hand normal alignment with +z."""
    if top_face_cos >= TOP_FACE_MIN_COS:
        return "tip_pad_top"
    if top_face_cos <= -TOP_FACE_MIN_COS:
        return "tip_pad_back"
    return "tip_pad_side"


def read_step_contacts(scene: PadScene, d: mujoco.MjData) -> list[RawContact]:
    """Read every active contact with its actual force; nothing is invented."""
    m = scene.model
    pad_id = scene.pad_geom_id
    tip_owner: dict[int, str] = {}
    for finger, gids in scene.tip_geom_ids.items():
        for g in gids:
            tip_owner[g] = finger
    pad_xpos = d.geom_xpos[pad_id]
    pad_rot = d.geom_xmat[pad_id].reshape(3, 3)
    pad_up = pad_rot[:, 2]
    out: list[RawContact] = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        f6 = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f6)
        fn = float(f6[0])  # compressive positive; never abs()
        normal = np.asarray(c.frame[0:3], dtype=float)
        pad_local = None
        top_cos = None
        if g1 == pad_id or g2 == pad_id:
            # Orient the normal so it points from the pad toward the hand.
            pad_to_hand = normal if g1 == pad_id else -normal
            top_cos = float(np.dot(pad_to_hand, pad_up))
            rel = np.asarray(c.pos, dtype=float) - pad_xpos
            loc = pad_rot.T @ rel
            pad_local = [float(v) for v in loc]
            other = g2 if g1 == pad_id else g1
            if other in tip_owner:
                kind = classify_pad_contact(top_cos)
            else:
                kind = "nontip_pad"
        else:
            kind = "hand_self"
        out.append(
            RawContact(
                geom1=g1,
                geom2=g2,
                body1=_body_name(m, g1),
                body2=_body_name(m, g2),
                pos_world=[float(v) for v in c.pos],
                frame=[float(v) for v in c.frame],
                force6=[float(v) for v in f6],
                normal_force_n=fn,
                dist_m=float(c.dist),
                pad_local=pad_local,
                top_face_cos=top_cos,
                kind=kind,
            )
        )
    return out


def aggregate_fingers(
    scene: PadScene, raw: list[RawContact]
) -> tuple[dict[str, dict], list[RawContact]]:
    """Aggregate top-face tip contacts per finger (force-weighted centroid).

    Only contacts classified ``tip_pad_top`` with strictly positive normal
    force contribute; callers MUST run
    :func:`embodied_trackpad.leap_sim.pad_env.validate_raw_contacts` first so
    no NaN can hide behind the ``> 0`` filter. Returns (per_finger,
    unintended) where per_finger has one entry per finger, never duplicated.
    """
    geom_to_finger: dict[int, str] = {}
    for finger, gids in scene.tip_geom_ids.items():
        for g in gids:
            geom_to_finger[g] = finger
    per_finger: dict[str, dict] = {}
    unintended: list[RawContact] = []
    for c in raw:
        if c.kind != "tip_pad_top":
            unintended.append(c)
            continue
        finger = geom_to_finger[c.geom1 if c.geom1 != scene.pad_geom_id else c.geom2]
        entry = per_finger.setdefault(finger, {"force_n": 0.0, "wx": 0.0, "wy": 0.0, "n": 0})
        if c.normal_force_n > 0.0:
            entry["force_n"] += c.normal_force_n
            entry["wx"] += c.normal_force_n * c.pad_local[0]
            entry["wy"] += c.normal_force_n * c.pad_local[1]
            entry["n"] += 1
    aggregated: dict[str, dict] = {}
    for finger, e in per_finger.items():
        if e["force_n"] > 0.0:
            aggregated[finger] = {
                "finger_id": finger,
                "x_m": e["wx"] / e["force_n"],
                "y_m": e["wy"] / e["force_n"],
                "normal_force_n": e["force_n"],
                "contact_points": e["n"],
            }
    return aggregated, unintended


def touch_v1_message(sequence: int, time_s: float, aggregated: dict[str, dict]) -> dict:
    """Build one touch.v1 frame (physics-side only; no app launch)."""
    contacts = []
    for finger in FINGERS:  # deterministic order, one entry per finger
        if finger in aggregated:
            a = aggregated[finger]
            contacts.append(
                {
                    "finger_id": finger,
                    "x_m": float(a["x_m"]),
                    "y_m": float(a["y_m"]),
                    "normal_force_n": float(a["normal_force_n"]),
                }
            )
    return {
        "schema_version": "touch.v1",
        "sequence": int(sequence),
        "simulation_time_s": float(time_s),
        "contacts": contacts,
    }

