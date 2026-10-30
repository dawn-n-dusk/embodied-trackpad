"""Static touch-pad contact environment for the LEAP v2 Advanced adapter.

Extends :func:`embodied_trackpad.leap_sim.load_hand` through its public
``spec_hook`` with ONE static world-mounted box geom (a 160 x 110 mm trackpad).
The accepted hand model is untouched: same URDF, same 21 joints / 17 motors /
4 coupling equalities, same eight mechanical-neighbor exclusions, same servo
constants. No contact pair is added to the exclusion table and no collision is
disabled; the pad is an ordinary colliding geom attached to the worldbody, so
it is static by construction and never moves during an episode.

Contact truth comes only from actual MuJoCo geom pairs: every step the active
``d.contact`` array is read together with ``mujoco.mj_contactForce`` and the
normal component is taken as the compressive force (positive = compressive, as
produced by the contact solver; the value is NEVER passed through ``abs`` and a
negative normal force is recorded as an anomaly). The full 6-axis contact
force and the 9-element contact frame are preserved in the raw evidence for
audit; only the positive normal component drives the device message. Non-finite
raw contact values (force, position, frame, distance) are rejected BEFORE any
aggregation, so a NaN force can never vanish through the ``> 0`` filter. Only
the five named distal fingertip bodies (the ``*_dip``/``thumb_ip`` child
bodies) are legitimate pad touchers, and only against the pad TOP face (contact
normal aligned with the pad up axis). Edge/backside tip contacts and any
non-tip/pad or hand self contact are recorded separately as unintended
contacts. The 5 N safety gate uses the total positive normal force over ALL
pad contacts (eligible tip, edge and non-tip alike); eligible tip force and
unintended force are additionally reported as separate series.

Per finger and step, multiple positive contact points are aggregated into one
entry with the summed normal force and a force-weighted centroid expressed in
measured pad-frame coordinates (metres, pad-centered), which is exactly the
``touch.v1`` contact shape ``{finger_id, x_m, y_m, normal_force_n}`` consumed
by the confined Mac GUI (one entry per finger, never duplicated). A CONTINUOUS
bounded-rate touch.v1 stream (maximum frame interval 0.05 s, monotonic
sequence and simulation timestamps) is recorded for the whole episode —
including empty clear/lift frames and every contact-set transition — so a
future consumer always sees a lift and never stays pressed. No synthetic
contacts are ever added. The GUI is NOT launched here; this is physics only.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from embodied_trackpad.leap_sim import OPEN_POSE, LeapHand, load_hand, open_pose_qpos
from embodied_trackpad.leap_sim.model import MOTOR_JOINTS

# --- pad geometry (task-fixed dimensions) -----------------------------------
PAD_GEOM_NAME = "trackpad_160x110"
PAD_HALF_EXTENTS = (0.080, 0.055, 0.005)  # 160 x 110 mm face, 10 mm thick
# Default pad pose: top face at z = -0.172 m. Measured open-pose surface
# minima (.local/explore5.py): lowest hand bodies (thumb/pip undersides) reach
# -0.1546 m, so the reset clearance is ~17 mm; the index distal tip is the
# only body that can reach the pad (mcpf>=0.10 with dip 0.45 puts the tip
# ~5 mm below the top face while the index middle phalanx stays ~10 mm clear).
DEFAULT_PAD_CENTER = (0.0, -0.05, -0.177)

FINGERS = ("index", "middle", "ring", "pinky", "thumb")
# Distal tip body = child body of the finger's last articulation.
FINGER_TIP_JOINT = {
    "index": "index_dip",
    "middle": "middle_dip",
    "ring": "ring_dip",
    "pinky": "pinky_dip",
    "thumb": "thumb_ip",
}

TOP_FACE_MIN_COS = 0.99  # |cos| between pad up axis and pad->hand contact normal
TOUCH_HALF_X = 0.080
TOUCH_HALF_Y = 0.055
TOUCH_MAX_FRAME_INTERVAL_S = 0.05  # bounded-rate continuous touch.v1 stream

# Acceptance bounds (fixed by the bounded task before execution).
LIMIT_EXCURSION_TOL = 0.01  # rad
COUPLING_TOL = 0.01  # rad
PENETRATION_TOL = 0.001  # m
PAD_FORCE_MAX_N = 5.0
MIN_CLEARANCE_S = 0.3  # initial and final contact-free windows
MIN_CONTACT_FORCE_N = 0.05
MIN_CONTACT_SUSTAIN_S = 0.3
MIN_SLIDE_DISPLACEMENT_M = 0.005
HELD_TRACKING_TOL = 0.10  # rad, unblocked joints during hold windows
PRESS_FORCE_N = 1.0  # existing device press threshold (DeviceModel.pressForceN)
RELEASE_FORCE_N = 0.6  # existing device release threshold (DeviceModel.releaseForceN)

FINGER_JOINTS = {
    f: [j for j in MOTOR_JOINTS if j == f"{f}_mcps" or j == f"{f}_mcpf" or j == f"{f}_dip"]
    for f in ("index", "middle", "ring", "pinky")
}
FINGER_JOINTS["thumb"] = ["thumb_mcps", "thumb_mcpf", "thumb_ip"]

PAD_CONTACT_KINDS = ("tip_pad_top", "tip_pad_edge", "nontip_pad")


def pad_spec_hook(
    pad_center: tuple[float, float, float] = DEFAULT_PAD_CENTER,
    half_extents: tuple[float, float, float] = PAD_HALF_EXTENTS,
):
    """Return a spec_hook that adds the static pad geom to the worldbody."""

    def hook(spec: mujoco.MjSpec) -> None:
        geom = spec.worldbody.add_geom(
            name=PAD_GEOM_NAME,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(half_extents),
            pos=list(pad_center),
        )
        # Match the hand geoms' contact stiffness (model.py sets that loop
        # BEFORE this hook runs, so the pad must set its own explicitly).
        geom.solref = [0.005, 1.0]
        geom.friction = [1.0, 0.005, 0.0001]
        geom.rgba = [0.25, 0.28, 0.32, 1.0]

    return hook


@dataclass
class PadScene:
    hand: LeapHand
    pad_geom_id: int
    pad_body_id: int
    pad_center: tuple[float, float, float]
    pad_half_extents: tuple[float, float, float]
    tip_geom_ids: dict[str, list[int]]  # finger -> colliding geom ids of the tip body
    tip_site: dict[str, str]  # finger -> realtip* site name on that body

    @property
    def model(self) -> mujoco.MjModel:
        return self.hand.model


def load_pad_scene(
    pad_center: tuple[float, float, float] = DEFAULT_PAD_CENTER,
    half_extents: tuple[float, float, float] = PAD_HALF_EXTENTS,
) -> PadScene:
    """Load the accepted hand plus the static pad via the public spec hook."""
    hand = load_hand(spec_hook=pad_spec_hook(pad_center, half_extents))
    m = hand.model
    pad_geom_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, PAD_GEOM_NAME))
    if pad_geom_id < 0:
        raise RuntimeError("pad geom missing after spec_hook")
    tip_geom_ids: dict[str, list[int]] = {}
    tip_site: dict[str, str] = {}
    for finger, jname in FINGER_TIP_JOINT.items():
        jid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname))
        body_id = int(m.jnt_bodyid[jid])
        gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body_id]
        if not gids:
            raise RuntimeError(f"no geom on tip body of {finger}")
        tip_geom_ids[finger] = gids
        candidates = [
            m.site(s).name
            for s in range(m.nsite)
            if int(m.site_bodyid[s]) == body_id and m.site(s).name.startswith("realtip")
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"expected exactly one realtip site on {finger} tip body")
        tip_site[finger] = candidates[0]
    return PadScene(
        hand=hand,
        pad_geom_id=pad_geom_id,
        pad_body_id=int(m.geom_bodyid[pad_geom_id]),
        pad_center=tuple(float(v) for v in pad_center),
        pad_half_extents=tuple(float(v) for v in half_extents),
        tip_geom_ids=tip_geom_ids,
        tip_site=tip_site,
    )


def scene_integrity(scene: PadScene) -> dict:
    """Structural checks proving the accepted hand model is preserved."""
    m = scene.model
    eq_names = [m.eq(i).name for i in range(m.neq)]
    act_joints = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    return {
        "nq": int(m.nq),
        "njnt": int(m.njnt),
        "nu": int(m.nu),
        "neq": int(m.neq),
        "neq_names": eq_names,
        "nexclude": int(m.nexclude),
        "motor_joint_order": act_joints,
        "motor_order_preserved": act_joints == MOTOR_JOINTS,
        "pad_body_is_world": scene.pad_body_id == 0,
        "pad_contype": int(m.geom_contype[scene.pad_geom_id]),
        "pad_conaffinity": int(m.geom_conaffinity[scene.pad_geom_id]),
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
    kind: str  # tip_pad_top | tip_pad_edge | nontip_pad | hand_self


def _body_name(m: mujoco.MjModel, geom_id: int) -> str:
    return m.body(int(m.geom_bodyid[int(geom_id)])).name or "world"


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
                kind = "tip_pad_top" if top_cos >= TOP_FACE_MIN_COS else "tip_pad_edge"
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


def validate_raw_contacts(raw: list[RawContact]) -> None:
    """Reject any non-finite raw contact value BEFORE aggregation.

    A NaN or infinite force/position/frame/distance must never reach the
    positive-force filter (where NaN would silently compare False and vanish).
    Raises ValueError naming every offending contact and field.
    """
    bad: list[str] = []
    for i, c in enumerate(raw):
        problems = []
        if not np.isfinite(c.normal_force_n):
            problems.append(f"normal_force_n={c.normal_force_n}")
        if not all(np.isfinite(v) for v in c.force6):
            problems.append("force6 non-finite")
        if not all(np.isfinite(v) for v in c.pos_world):
            problems.append("pos_world non-finite")
        if not all(np.isfinite(v) for v in c.frame):
            problems.append("frame non-finite")
        if not np.isfinite(c.dist_m):
            problems.append(f"dist_m={c.dist_m}")
        if c.pad_local is not None and not all(np.isfinite(v) for v in c.pad_local):
            problems.append("pad_local non-finite")
        if c.top_face_cos is not None and not np.isfinite(c.top_face_cos):
            problems.append("top_face_cos non-finite")
        if problems:
            bad.append(
                f"contact[{i}] {c.body1}|{c.body2} ({c.kind}): {', '.join(problems)}"
            )
    if bad:
        raise ValueError("non-finite raw contact values: " + "; ".join(bad))


def aggregate_fingers(
    scene: PadScene, raw: list[RawContact]
) -> tuple[dict[str, dict], list[RawContact]]:
    """Aggregate top-face tip contacts per finger (force-weighted centroid).

    Returns (per_finger, unintended). per_finger[finger] has summed positive
    normal force and the force-weighted pad-frame centroid — one entry per
    finger, never duplicated. Only contacts with strictly positive normal
    force contribute to the aggregate; callers MUST run
    :func:`validate_raw_contacts` first so no NaN can hide behind that filter.
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


# --- scripted episode ---------------------------------------------------------

@dataclass
class PadEpisodeConfig:
    """One clear -> press -> hold -> slide -> lift candidate parameterisation.

    All 17 motor channels are commanded every step; channels not listed in a
    phase inherit the previous phase target (open pose at t=0). ``smoke``
    marks non-acceptance diagnostic runs: their verdict can never be PASS.
    """

    candidate_id: str
    pad_center: tuple[float, float, float] = DEFAULT_PAD_CENTER
    # phase end times (s); phase targets are full joint->value overlays
    clear1_end: float = 0.6
    press_ramp_end: float = 1.2
    press_hold_end: float = 1.9
    slide_end: float = 2.7
    lift_ramp_end: float = 3.1
    t_end: float = 3.8
    press_finger: str = "index"
    press_mcpf: float = 0.12
    press_dip: float = 0.45
    slide_mcps: float = -0.22
    open_value: float = 0.05
    smoke: bool = False

    def phases(self) -> list[tuple[float, dict[str, float]]]:
        f = self.press_finger
        return [
            (self.clear1_end, {}),
            (self.press_ramp_end, {f"{f}_mcpf": self.press_mcpf, f"{f}_dip": self.press_dip}),
            (self.press_hold_end, {}),
            (self.slide_end, {f"{f}_mcps": self.slide_mcps}),
            (self.lift_ramp_end, {f"{f}_mcpf": self.open_value, f"{f}_dip": self.open_value, f"{f}_mcps": 0.0}),
            (self.t_end, {}),
        ]


def _open_ctrl(m: mujoco.MjModel, open_value: float = 0.05) -> np.ndarray:
    ctrl = np.zeros(m.nu)
    for i in range(m.nu):
        jname = m.joint(m.actuator_trnid[i, 0]).name
        ctrl[i] = OPEN_POSE.get(jname, 0.0) * (open_value / 0.05)
    return ctrl


def ctrl_schedule(m: mujoco.MjModel, cfg: PadEpisodeConfig) -> "callable":
    """Piecewise-linear full-vector schedule; every channel commanded always."""
    phases = cfg.phases()
    vecs: list[tuple[float, np.ndarray]] = [(0.0, _open_ctrl(m, cfg.open_value))]
    prev = vecs[0][1]
    t_prev = 0.0
    for t_end, overlay in phases:
        vec = prev.copy()
        for i in range(m.nu):
            jname = m.joint(m.actuator_trnid[i, 0]).name
            if jname in overlay:
                vec[i] = overlay[jname]
        if t_end > t_prev:
            vecs.append((t_end, vec))
            prev = vec
            t_prev = t_end
    times = [t for t, _ in vecs]

    def ctrl_at(t: float) -> np.ndarray:
        if t <= times[0]:
            return vecs[0][1]
        for (t0, v0), (t1, v1) in zip(vecs, vecs[1:]):
            if t0 <= t <= t1:
                if t1 == t0:
                    return v1
                a = (t - t0) / (t1 - t0)
                return (1 - a) * v0 + a * v1
        return vecs[-1][1]

    return ctrl_at


def apply_ctrl_checked(m: mujoco.MjModel, d: mujoco.MjData, desired: np.ndarray) -> None:
    """Validate shape/finiteness/ctrlrange BEFORE any write; never clip/patch."""
    desired = np.asarray(desired, dtype=float)
    if desired.shape != (m.nu,):
        raise ValueError(f"ctrl shape {desired.shape} != ({m.nu},)")
    if not np.all(np.isfinite(desired)):
        raise ValueError("non-finite ctrl command")
    lo = m.actuator_ctrlrange[:, 0]
    hi = m.actuator_ctrlrange[:, 1]
    bad = np.nonzero((desired < lo - 1e-9) | (desired > hi + 1e-9))[0]
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"ctrl[{i}]={float(desired[i])} outside ctrlrange "
            f"[{float(lo[i])}, {float(hi[i])}] for actuator {m.actuator(i).name}"
        )
    d.ctrl[:] = desired


def coupling_error(m: mujoco.MjModel, d: mujoco.MjData) -> float:
    err = 0.0
    for f in ("index", "middle", "ring", "pinky"):
        jd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_dip")
        jp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f}_pip")
        err = max(err, abs(float(d.qpos[m.jnt_qposadr[jd]] - d.qpos[m.jnt_qposadr[jp]])))
    return err


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal stdlib PNG writer (8-bit RGB).

    Byte-identical chunk assembly to the accepted writer in
    tools/leap_model_probe.py: length, tag, DATA, CRC(tag+data).
    """
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def decode_png_rgb(path: Path) -> np.ndarray:
    """Stdlib decoder for the exact 8-bit RGB/filter-0 PNGs write_png emits.

    Exists so tests and the probe verify an ACTUAL decodable image instead of
    trusting file bytes. Raises on any structural mismatch or a degenerate
    single-colour image.
    """
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("bad PNG signature")
    pos = 8
    width = height = None
    idat = b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        if zlib.crc32(tag + payload) & 0xFFFFFFFF != crc:
            raise ValueError(f"bad CRC in {tag!r}")
        if tag == b"IHDR":
            width, height, depth, color, comp, filt, inter = struct.unpack(">IIBBBBB", payload)
            if (depth, color, comp, filt, inter) != (8, 2, 0, 0, 0):
                raise ValueError("unsupported PNG encoding")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
        pos += 12 + length
    if width is None or not idat:
        raise ValueError("missing IHDR/IDAT")
    raw = zlib.decompress(idat)
    stride = width * 3
    rows = []
    for y in range(height):
        line = raw[y * (stride + 1) : (y + 1) * (stride + 1)]
        if line[0] != 0:
            raise ValueError("unexpected PNG filter type")
        rows.append(np.frombuffer(line[1:], dtype=np.uint8))
    img = np.stack(rows).reshape(height, width, 3)
    if int(img.max()) == int(img.min()):
        raise ValueError("degenerate single-colour image")
    return img


def _jsonable(obj, path: str = "$"):
    """Convert numpy scalars for JSON; reject non-finite floats outright."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, f"{path}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        raise ValueError(f"non-finite float at {path}")
    return obj


def dump_json(path: Path, payload) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class EpisodeResult:
    metrics: dict
    physics_seconds: float


def _clearance_window(
    ts: np.ndarray, pad_contact: np.ndarray, t_lo: float, t_hi: float, dt: float
) -> dict:
    """Measured clearance evidence for [t_lo, t_hi]; never vacuous.

    The window is only valid when measured samples actually cover it: at least
    99% of the expected per-step samples must be present and the first/last
    sample must lie within one timestep of the window edges. ``pad_contact``
    counts ALL pad-involved contacts per step (tip, edge and non-tip alike).
    """
    mask = (ts >= t_lo - 1e-9) & (ts <= t_hi + 1e-9)
    n = int(mask.sum())
    expected = int(round((t_hi - t_lo) / dt)) + 1
    covers = False
    if n:
        covers = (
            n >= max(2, int(np.floor(0.99 * expected)))
            and ts[mask][0] <= t_lo + 1.5 * dt
            and ts[mask][-1] >= t_hi - 1.5 * dt
        )
    contact_free = bool(n) and not bool(np.any(pad_contact[mask] > 0))
    return {
        "window_s": [t_lo, t_hi],
        "samples": n,
        "expected_samples": expected,
        "coverage_complete": bool(covers),
        "contact_free": contact_free,
        "ok": bool(covers and contact_free),
    }


def run_pad_episode(
    scene: PadScene,
    cfg: PadEpisodeConfig,
    out_dir: Path | None = None,
    render: bool = True,
    sample_every: int = 10,
) -> EpisodeResult:
    """Execute one candidate episode through ctrl + mj_step only.

    The only live qpos write is the reset to the open pose; all motion is
    servo references. After every mj_step, mj_forward refreshes the forward
    state so recorded contacts/forces/sites match the recorded qpos and the
    actual data.time; the control-start time of each step is stored as t_ctrl.
    """
    m = scene.model
    if out_dir is not None:
        if out_dir.exists():
            raise FileExistsError(f"output directory already exists: {out_dir}")
        out_dir.mkdir(parents=True)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    q0 = open_pose_qpos(m)
    d.qpos[:] = q0  # the only live qpos write of the whole episode
    apply_ctrl_checked(m, d, _open_ctrl(m, cfg.open_value))
    mujoco.mj_forward(m, d)

    ctrl_at = ctrl_schedule(m, cfg)
    steps = int(round(cfg.t_end / m.opt.timestep))
    trnid = m.actuator_trnid[:, 0]
    act_qposadr = m.jnt_qposadr[trnid]
    act_jnames = [m.joint(m.actuator_trnid[i, 0]).name for i in range(m.nu)]

    render_steps: dict[int, str] = {}
    if render and out_dir is not None:
        render_steps = {
            0: "initial",
            int(round(cfg.press_hold_end / m.opt.timestep)) - int(round(0.15 / m.opt.timestep)): "press",
            int(round(cfg.slide_end / m.opt.timestep)) - 1: "slide",
            steps - 1: "lift",
        }

    renderer = None
    cams: dict[str, mujoco.MjvCamera] = {}
    if render_steps:
        renderer = mujoco.Renderer(m, height=480, width=640)
        # Framing covers the whole hand (root z=0 down to the flexed tip
        # z=-0.22) plus the full pad top; images are always from actual state.
        side = mujoco.MjvCamera()
        side.lookat = [0.0, -0.02, -0.095]
        side.distance = 0.42
        side.azimuth = 135.0
        side.elevation = -15.0
        top = mujoco.MjvCamera()
        top.lookat = [0.0, -0.05, -0.12]
        top.distance = 0.42
        top.azimuth = 90.0
        top.elevation = -40.0
        cams = {"side": side, "top": top}

    # every-step scalar series (gate maxima evidence)
    series: dict[str, list] = {
        "t": [],
        "t_ctrl": [],
        "pad_force_tip_n": [],  # eligible distal-tip top-face aggregate
        "pad_force_all_n": [],  # ALL pad contacts (tip+edge+nontip), 5 N safety gate
        "unintended_force_n": [],
        "press_finger_force_n": [],
        "n_pad_contacts": [],
        "max_penetration_m": [],
        "coupling_error_rad": [],
        "limit_excursion_rad": [],
        "ncon": [],
        "n_unintended": [],
        "press_finger_x_m": [],
        "press_finger_y_m": [],
    }
    # sampled full-state trace (every `sample_every` steps + final)
    trace: dict[str, list] = {"t": [], "t_ctrl": [], "qpos": [], "qvel": [], "qacc": [],
                              "ctrl": [], "actfrc": [], "tips": []}
    raw_contact_log: list[dict] = []  # every step that has any contact
    touch_log: list[dict] = []
    failures: list[str] = []
    max_limit_excursion = 0.0
    max_coupling = 0.0
    max_penetration = 0.0
    max_pad_force_tip = 0.0
    max_pad_force_all = 0.0
    max_unintended_force = 0.0
    max_negative_normal = 0.0
    pad_pos_ref = d.geom_xpos[scene.pad_geom_id].copy()
    pad_max_dev = 0.0
    warn_start = np.array([w.number for w in d.warning], dtype=int)
    hold_tracking_free: dict[str, float] = {}
    hold_tracking_blocked: dict[str, float] = {}
    hold_windows = ((0.0, cfg.clear1_end), (cfg.press_ramp_end, cfg.press_hold_end),
                    (cfg.lift_ramp_end, cfg.t_end))
    touch_seq = 0
    last_touch_t = -np.inf
    prev_finger_set: tuple = ()
    steps_executed = 0
    physics_t0 = time.perf_counter()

    for k in range(steps):
        t_ctrl = k * m.opt.timestep
        desired = ctrl_at(t_ctrl)
        apply_ctrl_checked(m, d, desired)  # raises before mj_step on invalid input
        mujoco.mj_step(m, d)
        # Refresh forward state at the observation instant: contacts, forces
        # and site positions below belong to this recorded qpos / d.time.
        mujoco.mj_forward(m, d)
        t = float(d.time)
        steps_executed = k + 1

        if not (
            np.all(np.isfinite(d.qpos))
            and np.all(np.isfinite(d.qvel))
            and np.all(np.isfinite(d.qacc))
            and np.all(np.isfinite(d.ctrl))
            and np.all(np.isfinite(d.actuator_force))
        ):
            failures.append(f"non-finite state at step {k} (t_ctrl={t_ctrl:.3f})")
            break

        raw = read_step_contacts(scene, d)
        try:
            validate_raw_contacts(raw)  # NaN must never reach the >0 filter
        except ValueError as exc:
            failures.append(f"step {k} (t_ctrl={t_ctrl:.3f}): {exc}")
            break
        aggregated, unintended = aggregate_fingers(scene, raw)
        pad_force_tip = float(sum(a["normal_force_n"] for a in aggregated.values()))
        pad_force_all = float(
            sum(c.normal_force_n for c in raw
                if c.kind in PAD_CONTACT_KINDS and c.normal_force_n > 0.0)
        )
        unintended_force = float(
            sum(c.normal_force_n for c in unintended if c.normal_force_n > 0.0)
        )
        n_pad_contacts = sum(1 for c in raw if c.kind in PAD_CONTACT_KINDS)
        pen = 0.0
        for c in raw:
            if c.dist_m < 0:
                pen = max(pen, -c.dist_m)
            if c.normal_force_n < 0:
                max_negative_normal = max(max_negative_normal, -c.normal_force_n)
        coup = coupling_error(m, d)
        lim_exc = 0.0
        for j in range(m.njnt):
            lo = float(m.jnt_range[j][0])
            hi = float(m.jnt_range[j][1])
            q = float(d.qpos[m.jnt_qposadr[j]])
            lim_exc = float(max(lim_exc, max(lo - q, q - hi, 0.0)))
        pad_max_dev = float(max(
            pad_max_dev, float(np.abs(d.geom_xpos[scene.pad_geom_id] - pad_pos_ref).max())
        ))

        pf = cfg.press_finger
        pf_force = float(aggregated.get(pf, {}).get("normal_force_n", 0.0))
        pf_x = float(aggregated.get(pf, {}).get("x_m", np.nan))
        pf_y = float(aggregated.get(pf, {}).get("y_m", np.nan))
        series["t"].append(t)
        series["t_ctrl"].append(t_ctrl)
        series["pad_force_tip_n"].append(pad_force_tip)
        series["pad_force_all_n"].append(pad_force_all)
        series["unintended_force_n"].append(unintended_force)
        series["press_finger_force_n"].append(pf_force)
        series["n_pad_contacts"].append(int(n_pad_contacts))
        series["max_penetration_m"].append(pen)
        series["coupling_error_rad"].append(coup)
        series["limit_excursion_rad"].append(lim_exc)
        series["ncon"].append(int(d.ncon))
        series["n_unintended"].append(len(unintended))
        series["press_finger_x_m"].append(pf_x)
        series["press_finger_y_m"].append(pf_y)
        max_limit_excursion = max(max_limit_excursion, lim_exc)
        max_coupling = max(max_coupling, coup)
        max_penetration = max(max_penetration, pen)
        max_pad_force_tip = max(max_pad_force_tip, pad_force_tip)
        max_pad_force_all = max(max_pad_force_all, pad_force_all)
        max_unintended_force = max(max_unintended_force, unintended_force)

        if raw:
            raw_contact_log.append(
                {
                    "t": t,
                    "t_ctrl": t_ctrl,
                    "contacts": [
                        {
                            "geom1": c.geom1,
                            "geom2": c.geom2,
                            "body1": c.body1,
                            "body2": c.body2,
                            "kind": c.kind,
                            "normal_force_n": c.normal_force_n,
                            "force6": c.force6,
                            "frame": c.frame,
                            "dist_m": c.dist_m,
                            "pos_world": c.pos_world,
                            "pad_local": c.pad_local,
                            "top_face_cos": c.top_face_cos,
                        }
                        for c in raw
                    ],
                }
            )

        # Continuous bounded-rate touch.v1 stream: first frame, every contact-
        # set transition (including the empty<->nonempty lift/press edges), at
        # least one frame every TOUCH_MAX_FRAME_INTERVAL_S, and the final
        # step. Empty frames carry zero contacts; none are ever invented.
        finger_set = tuple(sorted(aggregated))
        due = (t - last_touch_t) >= (TOUCH_MAX_FRAME_INTERVAL_S - 1e-9)
        if touch_seq == 0 or due or finger_set != prev_finger_set or k == steps - 1:
            touch_log.append(touch_v1_message(touch_seq, t, aggregated))
            touch_seq += 1
            last_touch_t = t
            prev_finger_set = finger_set

        # hold-window tracking: contact-constrained joints of any finger with
        # an active pad contact are reported separately (NOT free-space).
        in_hold = any(a - 1e-9 <= t <= b + 1e-9 for a, b in hold_windows)
        if in_hold:
            constrained = set()
            for finger in aggregated:
                constrained.update(FINGER_JOINTS.get(finger, []))
            err = np.abs(d.ctrl - d.qpos[act_qposadr])
            for i, jname in enumerate(act_jnames):
                e = float(err[i])
                if jname in constrained:
                    hold_tracking_blocked[jname] = max(hold_tracking_blocked.get(jname, 0.0), e)
                else:
                    hold_tracking_free[jname] = max(hold_tracking_free.get(jname, 0.0), e)

        if k % sample_every == 0 or k == steps - 1:
            trace["t"].append(t)
            trace["t_ctrl"].append(t_ctrl)
            trace["qpos"].append(d.qpos.copy())
            trace["qvel"].append(d.qvel.copy())
            trace["qacc"].append(d.qacc.copy())
            trace["ctrl"].append(d.ctrl.copy())
            trace["actfrc"].append(d.actuator_force.copy())
            trace["tips"].append(
                np.stack([d.site(scene.tip_site[f]).xpos for f in FINGERS])
            )
        if k in render_steps and renderer is not None:
            for cam_name, cam in cams.items():
                renderer.update_scene(d, cam)
                png = out_dir / f"{render_steps[k]}_{cam_name}.png"
                write_png(png, renderer.render())
                decode_png_rgb(png)  # fail loudly on a broken/blank render

    physics_seconds = time.perf_counter() - physics_t0
    if renderer is not None:
        renderer.close()

    warn_end = np.array([w.number for w in d.warning], dtype=int)
    warn_delta = (warn_end - warn_start).tolist()
    warn_names = [str(w).split(".")[-1] for w in mujoco.mjtWarning.__members__.values()]
    warnings = {warn_names[i]: int(warn_delta[i]) for i in range(len(warn_delta)) if warn_delta[i]}

    # --- derived contact/phase metrics -------------------------------------
    dt = float(m.opt.timestep)
    episode_complete = steps_executed == steps and not failures
    ts = np.asarray(series["t"])
    pad_contact = np.asarray(series["n_pad_contacts"], dtype=float)
    pf_force = np.asarray(series["press_finger_force_n"])
    # sustained-contact windows: consecutive steps with press-finger force >= MIN_CONTACT_FORCE_N
    sustained = pf_force >= MIN_CONTACT_FORCE_N
    sustain_windows: list[tuple[int, int]] = []
    start = None
    for i, ok in enumerate(sustained):
        if ok and start is None:
            start = i
        if start is not None and (not ok or i == len(sustained) - 1):
            end = i if ok else i - 1
            sustain_windows.append((int(start), int(end)))
            start = None
    sustain_durations = [float(ts[b] - ts[a] + dt) for a, b in sustain_windows]
    best_window = None
    if sustain_windows:
        best_window = max(sustain_windows, key=lambda ab: ab[1] - ab[0])
    max_sustain_s = float(max(sustain_durations)) if sustain_durations else 0.0
    # measured tangential displacement of the force-weighted pad-frame centroid
    slide_disp_m = 0.0
    slide_path_m = 0.0
    if best_window is not None:
        a, b = best_window
        xs = np.asarray(series["press_finger_x_m"][a : b + 1], dtype=float)
        ys = np.asarray(series["press_finger_y_m"][a : b + 1], dtype=float)
        if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)) and len(xs) > 1:
            slide_disp_m = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
            slide_path_m = float(np.hypot(np.diff(xs), np.diff(ys)).sum())
    init_clear = _clearance_window(ts, pad_contact, 0.0, MIN_CLEARANCE_S, dt)
    final_clear = _clearance_window(ts, pad_contact, cfg.t_end - MIN_CLEARANCE_S, cfg.t_end, dt)
    press_peak_n = float(pf_force.max()) if len(pf_force) else 0.0
    press_threshold_crossed = bool(press_peak_n >= PRESS_FORCE_N)
    # release: after the last sustained contact step, force stays <= RELEASE_FORCE_N
    if best_window is not None:
        post = pf_force[best_window[1] + 1 :]
        release_ok = bool(len(post)) and bool(np.all(post <= RELEASE_FORCE_N))
        force_at_lift_end = float(post[-1]) if len(post) else 0.0
    else:
        release_ok = False
        force_at_lift_end = float(pf_force[-1]) if len(pf_force) else 0.0
    # unique STEPS with >=1 unintended contact, separately from the EVENT sum
    unintended_steps = int(sum(1 for v in series["n_unintended"] if v > 0))
    unintended_events = int(sum(series["n_unintended"]))
    unintended_kinds: dict[str, int] = {}
    for entry in raw_contact_log:
        for c in entry["contacts"]:
            if c["kind"] != "tip_pad_top":
                unintended_kinds[c["kind"]] = unintended_kinds.get(c["kind"], 0) + 1

    # touch stream sanity: monotonic time/sequence, bounded frame interval
    touch_times = [f["simulation_time_s"] for f in touch_log]
    touch_seqs = [f["sequence"] for f in touch_log]
    touch_monotonic = touch_seqs == list(range(len(touch_log))) and all(
        b >= a for a, b in zip(touch_times, touch_times[1:])
    )
    touch_max_interval = float(
        max((b - a for a, b in zip(touch_times, touch_times[1:])), default=0.0)
    )
    touch_bounded = touch_max_interval <= TOUCH_MAX_FRAME_INTERVAL_S + 1e-9
    touch_covers_episode = bool(touch_log) and touch_times[0] <= dt and (
        not episode_complete or touch_times[-1] >= cfg.t_end - dt
    )

    free_max = max(hold_tracking_free.values()) if hold_tracking_free else 0.0
    gates = {
        "finite": not failures,
        "episode_complete": episode_complete,
        "no_warnings": not warnings,
        "limit_excursion<=0.01": max_limit_excursion <= LIMIT_EXCURSION_TOL,
        "coupling<=0.01": max_coupling <= COUPLING_TOL,
        "ctrl_in_bounds": True,  # apply_ctrl_checked raises before any invalid write
        "penetration<=0.001": max_penetration <= PENETRATION_TOL,
        "pad_force<=5N_all_pad_contacts": max_pad_force_all <= PAD_FORCE_MAX_N,
        "no_negative_normal_force": max_negative_normal == 0.0,
        "no_unintended_or_edge_contact": unintended_steps == 0,
        "initial_clearance>=0.3s": init_clear["ok"],
        "final_clearance>=0.3s": final_clear["ok"],
        "sustained_contact>=0.3s@>=0.05N": max_sustain_s >= MIN_CONTACT_SUSTAIN_S,
        "slide_displacement>=5mm": slide_disp_m >= MIN_SLIDE_DISPLACEMENT_M,
        "hold_tracking<=0.10_unblocked": free_max <= HELD_TRACKING_TOL,
        "pad_static": pad_max_dev == 0.0,
        "touch_stream_continuous": touch_monotonic and touch_bounded and touch_covers_episode,
    }
    # Device press-threshold result is reported alongside the physics verdict
    # (the task requires reporting, not silently redefining, this gate).
    device_press = {
        "press_threshold_1N_crossed": press_threshold_crossed,
        "press_peak_n": press_peak_n,
        "release<=0.6N_after_contact": release_ok,
        "force_at_trace_end_n": force_at_lift_end,
    }
    if cfg.smoke:
        verdict = "SMOKE"  # diagnostic runs are never acceptance candidates
    else:
        verdict = "PASS" if all(gates.values()) else "FAIL"

    metrics = {
        "candidate_id": cfg.candidate_id,
        "smoke": bool(cfg.smoke),
        "verdict": verdict,
        "failure_reasons": [name for name, ok in gates.items() if not ok] + failures,
        "config": {
            "candidate_id": cfg.candidate_id,
            "pad_center": list(cfg.pad_center),
            "pad_half_extents": list(scene.pad_half_extents),
            "press_finger": cfg.press_finger,
            "press_mcpf": cfg.press_mcpf,
            "press_dip": cfg.press_dip,
            "slide_mcps": cfg.slide_mcps,
            "phase_times_s": {
                "clear1_end": cfg.clear1_end,
                "press_ramp_end": cfg.press_ramp_end,
                "press_hold_end": cfg.press_hold_end,
                "slide_end": cfg.slide_end,
                "lift_ramp_end": cfg.lift_ramp_end,
                "t_end": cfg.t_end,
            },
        },
        "steps_planned": steps,
        "steps_executed": steps_executed,
        "episode_complete": episode_complete,
        "timestep": dt,
        "physics_seconds_wall": physics_seconds,
        "gates": gates,
        "device_press_thresholds": device_press,
        "checks": {
            "max_limit_excursion_rad": max_limit_excursion,
            "max_coupling_error_rad": max_coupling,
            "max_penetration_m": max_penetration,
            "max_pad_normal_force_all_n": max_pad_force_all,
            "max_pad_normal_force_tip_eligible_n": max_pad_force_tip,
            "max_unintended_force_n": max_unintended_force,
            "max_negative_normal_force_n": max_negative_normal,
            "warnings": warnings,
            "unintended_contact_steps": unintended_steps,
            "unintended_contact_events": unintended_events,
            "unintended_contact_kinds": unintended_kinds,
            "pad_geom_max_deviation_m": pad_max_dev,
            "hold_tracking_free_rad": hold_tracking_free,
            "hold_tracking_blocked_rad": hold_tracking_blocked,
            "initial_clearance": init_clear,
            "final_clearance": final_clear,
            "touch_stream": {
                "frames": len(touch_log),
                "monotonic": touch_monotonic,
                "max_frame_interval_s": touch_max_interval,
                "bounded_interval_ok": touch_bounded,
                "covers_episode": touch_covers_episode,
                "empty_frames": sum(1 for f in touch_log if not f["contacts"]),
            },
        },
        "contact": {
            "press_finger": cfg.press_finger,
            "max_sustained_s_ge_0.05N": max_sustain_s,
            "sustain_windows_steps": [list(w) for w in sustain_windows],
            "slide_net_displacement_m": slide_disp_m,
            "slide_path_length_m": slide_path_m,
            "press_peak_n": press_peak_n,
            "contact_steps": int(sum(1 for f in pf_force if f > 0.0)),
            "touch_frames": len(touch_log),
        },
    }

    if out_dir is not None:
        np.savez_compressed(
            out_dir / "trace.npz",
            t=np.asarray(trace["t"]),
            t_ctrl=np.asarray(trace["t_ctrl"]),
            qpos=np.asarray(trace["qpos"]),
            qvel=np.asarray(trace["qvel"]),
            qacc=np.asarray(trace["qacc"]),
            ctrl=np.asarray(trace["ctrl"]),
            actfrc=np.asarray(trace["actfrc"]),
            tips=np.asarray(trace["tips"]),
            tip_site_order=np.asarray(FINGERS),
            **{f"series_{k}": np.asarray(v) for k, v in series.items()},
        )
        dump_json(out_dir / "metrics.json", metrics)
        dump_json(out_dir / "raw-contacts.json", raw_contact_log)
        dump_json(out_dir / "touch-v1.json", touch_log)
    return EpisodeResult(metrics=metrics, physics_seconds=physics_seconds)
