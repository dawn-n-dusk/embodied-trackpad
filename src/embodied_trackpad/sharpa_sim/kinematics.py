"""Panda 7-DoF kinematics + Sharpa mount math for the trackpad home pose.

CPU NumPy/SciPy (no isaacsim import). Internal convention is the standard
COLUMN-vector robotics convention (p' = M @ p). Conversion to pxr/USD
ROW-vector matrices is provided by :func:`col_to_row` (transpose) — pxr
composes world transforms as ``local * parent`` with row vectors, so a
USD local matrix is ``M_row = W_row * inverse(P_row)``, never
``inverse(P) * W``. Verified numerically on the Isaac host (mount
alignment residual 2.2e-16) before the GPU run.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy.spatial.transform import Rotation as _SciRotation

from .config import PANDA_JOINT_ORIGINS, PANDA_LIMITS

# Home design: palm facing DOWN toward the pad, fingers spread forward.
# Official Sharpa geometry: fingers spread in body-Y, extend body-Z, palm
# normal is body-X. With base yaw pi, rpy (pi, pi/2-0.2, -pi/2) sends body
# +X mostly DOWN and body +Z forward/slightly down (coordinator-reviewed).
HOME_BODY_ROT_RPY_BASE = (math.pi, math.pi / 2 - 0.2, -math.pi / 2)

# Default Sharpa index-tip position (metres) in the Sharpa
# right_hand_flange BODY frame at the hand's all-zero joint state,
# measured from the pinned right_sharpa_wave_with_flange.usda
# (rev d323b371d02f4fc04a0873c19c5234cae6d803dd) by the v001 scene audit.
SHARPA_INDEX_TIP_BODY = (0.0010001, 0.0303005, 0.229909)
# Palm height (m) of the C_MC board in the same body frame (audit value).
SHARPA_PALM_CMC_Z_BODY = 0.0295


# ---------------------------------------------------------------------------
# rotation helpers (column convention)
# ---------------------------------------------------------------------------
def quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mat_to_quat_wxyz(R) -> tuple[float, float, float, float]:
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return ((0.25 * s, (R[2, 1] - R[1, 2]) / s,
                 (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s))
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                0.25 * s, (R[1, 2] + R[2, 1]) / s)
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s, 0.25 * s)


def rpy_to_mat(rpy) -> np.ndarray:
    rx, ry, rz = (float(v) for v in rpy)
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat_to_rpy(R) -> tuple[float, float, float]:
    return (math.atan2(R[2, 1], R[2, 2]), math.asin(-R[2, 0]),
            math.atan2(R[1, 0], R[0, 0]))


def T_mat(t) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = np.asarray(t, dtype=float)
    return M


def RT(R, t) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(t, dtype=float)
    return M


def col_to_row(M: np.ndarray) -> np.ndarray:
    """Column-convention 4x4 -> pxr row-vector matrix (transpose)."""
    return np.ascontiguousarray(np.asarray(M, dtype=np.float64).T)


# ---------------------------------------------------------------------------
# Panda FK (column convention; arm base frame)
# ---------------------------------------------------------------------------
def panda_fk(q) -> np.ndarray:
    """World-from-flange-link (link7) 4x4 in the ARM BASE frame.

    Compose parent->child per URDF: M = M @ (T(origin) @ R(rpy) @ Rz(q)).
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (7,):
        raise ValueError(f"expected 7 joint angles, got {q.shape}")
    M = np.eye(4)
    for i, ((xyz, rpy), qi) in enumerate(zip(PANDA_JOINT_ORIGINS, q)):
        A = T_mat(xyz) @ RT(rpy_to_mat(rpy), (0, 0, 0))
        A = A @ RT(rpy_to_mat((0, 0, qi)), (0, 0, 0))
        M = M @ A
        lo, hi = PANDA_LIMITS[i]
        if not (lo - 1e-9 <= qi <= hi + 1e-9):
            raise ValueError(f"q[{i}]={qi} outside limits {PANDA_LIMITS[i]}")
    return M


def mount_frame_in_link7(mount_pos0=(0.0, 0.0, 0.107),
                         mount_rot0_wxyz=(0.70710677, 0.0, 0.70710677, 0.0)):
    """FixedJoint frame0 of the Panda hand mount expressed in link7.

    Values default to the official NVIDIA franka.usd
    /panda/panda_link7/panda_hand_joint anchors; the scene builder reads
    them live from the asset and passes them here.
    """
    return RT(quat_wxyz_to_mat(mount_rot0_wxyz), mount_pos0)


def body_in_mount(mount_rot1_wxyz):
    """Rotation of the Sharpa body frame relative to the mount frame.

    The weld uses the Panda hand-joint frame1 rotation (asset value) as the
    frame1 rotation in the Sharpa right_hand_flange body, i.e. the Sharpa
    flange body frame is taken to coincide with the Panda hand body mount
    frame. mount->body rotation is therefore frame1's inverse... expressed
    here as: p_mount = R1^T @ p_body, so body-in-mount = R1^T.
    """
    R1 = quat_wxyz_to_mat(mount_rot1_wxyz)
    return R1.T


def fk_mount(q, mount_pos0=(0.0, 0.0, 0.107),
             mount_rot0_wxyz=(0.70710677, 0.0, 0.70710677, 0.0)) -> np.ndarray:
    """Mount frame (where the hand is welded) world-from, arm base frame."""
    return panda_fk(q) @ mount_frame_in_link7(mount_pos0, mount_rot0_wxyz)


def fk_index_tip(q, mount_pos0, mount_rot0_wxyz, mount_rot1_wxyz,
                 tip_body=SHARPA_INDEX_TIP_BODY) -> np.ndarray:
    """Index fingertip world position (arm base frame) at joint state q with
    the Sharpa hand at its all-zero pose."""
    Mm = fk_mount(q, mount_pos0, mount_rot0_wxyz)
    R1T = body_in_mount(mount_rot1_wxyz)
    t_tip_mount = R1T @ np.asarray(tip_body, dtype=float)
    return (Mm @ np.append(t_tip_mount, 1.0))[:3]


# ---------------------------------------------------------------------------
# IK: damped least squares, finite-difference Jacobian, deterministic seeds
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class HomeSolution:
    q: np.ndarray
    success: bool
    position_error_m: float
    rotation_error: float
    iterations: int
    tip_world: np.ndarray          # arm base frame, predicted
    mount_world_R: np.ndarray
    mount_world_t: np.ndarray

    def to_dict(self) -> dict:
        return {
            "q": [round(float(v), 6) for v in self.q],
            "success": bool(self.success),
            "position_error_m": round(self.position_error_m, 8),
            "rotation_error": round(self.rotation_error, 8),
            "iterations": int(self.iterations),
            "tip_world": [round(float(v), 6) for v in self.tip_world],
        }


def _orient_error(R_cur, R_des) -> np.ndarray:
    """True SO(3) log error (rad): rotation vector taking R_cur to R_des.
    Robust to 180-degree ambiguities (the sin(theta) approximation is
    not). Verified against Rotation.from_matrix trace/acos."""
    return _SciRotation.from_matrix(R_des.T @ R_cur).as_rotvec()


def solve_home(tip_target_world, mount_pos0, mount_rot0_wxyz,
               mount_rot1_wxyz, body_rot_world_des=None,
               tip_body=SHARPA_INDEX_TIP_BODY,
               seeds=(), max_iter=400, tol_pos=1e-4, tol_rot=1e-3):
    """Solve 7 arm joint angles placing the Sharpa index tip at
    ``tip_target_world`` (arm base frame) with the palm near-horizontal,
    tilted slightly downward toward the pad (HOME_BODY_ROT_RPY_BASE in the
    arm base frame).

    Returns the best HomeSolution over all seeds (success if within tol).
    """
    if body_rot_world_des is None:
        body_rot_world_des = rpy_to_mat(HOME_BODY_ROT_RPY_BASE)
    R1T = body_in_mount(mount_rot1_wxyz)
    t_tip_mount = R1T @ np.asarray(tip_body, dtype=float)
    R_mount_des = body_rot_world_des @ quat_wxyz_to_mat(mount_rot1_wxyz)
    t_mount_des = np.asarray(tip_target_world, dtype=float) \
        - R_mount_des @ t_tip_mount

    if not seeds:
        # First seed: coordinator-reviewed CPU-checked solution for the
        # palm-down home (real rotation error 2.0e-7 rad, position 7e-8 m).
        seeds = [
            (0.089940, 0.608528, -0.120211, -2.272798, 1.527936,
             1.695347, 1.050544),
            (0.367308, 0.512398, -0.380658, -2.405812, 1.596629,
             1.594995, -0.525176),
            (0.050361, 0.226926, 0.615755, -2.269741, -0.204123,
             2.445749, 1.596885),
            (0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785),
            (0.0, -1.2, 0.0, -2.0, 0.0, 1.9, 0.785),
            (0.3, -0.9, 0.3, -2.4, 0.0, 1.65, 0.785),
            (-0.3, -0.9, -0.3, -2.4, 0.0, 1.65, 0.785),
            (0.0, -0.6, 0.0, -2.8, 0.0, 1.2, 0.785),
        ]

    def errors(q):
        Mm = fk_mount(q, mount_pos0, mount_rot0_wxyz)
        Rm, tm = Mm[:3, :3], Mm[:3, 3]
        e_pos = tm - t_mount_des
        e_rot = _orient_error(Rm, R_mount_des)
        return np.concatenate([e_pos, e_rot]), Mm

    def jacobian(q, e0):
        J = np.zeros((6, 7))
        for j in range(7):
            dq = np.zeros(7)
            h = 1e-6
            dq[j] = h
            ej, _ = errors(q + dq)
            J[:, j] = (ej - e0) / h
        return J

    best = None
    for seed in seeds:
        q = np.clip(np.asarray(seed, dtype=float),
                    [lo + 0.02 for lo, _ in PANDA_LIMITS],
                    [hi - 0.02 for _, hi in PANDA_LIMITS])
        it = 0
        for it in range(1, max_iter + 1):
            e, Mm = errors(q)
            n = float(np.linalg.norm(e[:3]))
            nr = float(np.linalg.norm(e[3:]))
            if n < tol_pos and nr < tol_rot:
                break
            J = jacobian(q, e)
            lam = 1e-3 if it < 100 else 3e-3
            dq = -J.T @ np.linalg.solve(J @ J.T + lam * lam * np.eye(6), e)
            step = float(np.linalg.norm(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = q + dq
            q = np.clip(q, [lo + 0.01 for lo, _ in PANDA_LIMITS],
                        [hi - 0.01 for _, hi in PANDA_LIMITS])
        e, Mm = errors(q)
        pos_err = float(np.linalg.norm(e[:3]))
        rot_err = float(np.linalg.norm(e[3:]))
        Rm, tm = Mm[:3, :3], Mm[:3, 3]
        tip = (Mm @ np.append(t_tip_mount, 1.0))[:3]
        cand = HomeSolution(q=q, success=pos_err < tol_pos and rot_err < tol_rot,
                            position_error_m=pos_err, rotation_error=rot_err,
                            iterations=it, tip_world=tip,
                            mount_world_R=Rm, mount_world_t=tm)
        if best is None or (cand.position_error_m + cand.rotation_error) < \
                (best.position_error_m + best.rotation_error):
            best = cand
    return best


def tip_target_in_base(config, hover_m: float) -> np.ndarray:
    """Index tip target (ARM BASE frame): above pad centre by hover_m.

    Base yaw pi means the arm base frame is rotated 180 deg about z
    relative to the world; convert world (px, py, z) into base frame.
    """
    import math as _m
    ax, ay = config.base_xy
    px, py = config.pad_xy
    z = config.table_top_z + config.pad_size[2] + hover_m
    dx, dy = px - ax, py - ay
    c, s = _m.cos(-config.base_yaw), _m.sin(-config.base_yaw)
    return np.array([c * dx - s * dy, s * dx + c * dy, z - config.table_top_z])
