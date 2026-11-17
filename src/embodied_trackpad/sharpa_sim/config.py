"""Scene configuration for the Sharpa right-hand arm + trackpad environment.

Pure Python (no isaacsim import) so contracts can be validated on any host.
All poses are in meters / radians in the world frame: +X right, +Y away from
the operator side, +Z up. The operator sits at -Y.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Named action schema: 7 arm joint position targets + 22 Sharpa hand
# actuation channels. Order is part of the contract.
# ---------------------------------------------------------------------------
ARM_PANDA_7DOF = "franka_panda_7dof_gripper_removed__TEMPORARY_SIM_ARM"
# NOTE: the user has not chosen a procurement arm yet. We use the official
# documented 7-DoF Franka Panda URDF with the gripper removed as the
# temporary simulation arm, clearly labelled as such (not an FR3).

HAND_22_CHANNELS = "sharpawave_right_22ch_with_flange"

ACTION_SCHEMA = {
    "schema": "sharpa-trackpad-action-v1",
    "arm": {
        "id": ARM_PANDA_7DOF,
        "joints": [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
        "units": "rad",
        "control": "position_target",
    },
    "hand": {
        "id": HAND_22_CHANNELS,
        "note": "22 independent actuation channels enumerated from the pinned "
        "right_sharpa_wave_with_flange.usda rev d323b371d0 (verified at runtime)",
        "units": "rad",
        "control": "position_target",
    },
    "dim": 29,
}

# Panda joint limits (rad) from the official franka_description
# robots/panda/joint_limits.yaml (frankaemika/franka_ros, noetic-devel,
# fetched by tools/sharpa_fetch_assets.py; sha256 recorded in the manifest).
PANDA_LIMITS = [
    (-2.8973, 2.8973),
    (-1.7628, 1.7628),
    (-2.8973, 2.8973),
    (-3.0718, -0.0698),
    (-2.8973, 2.8973),
    (-0.0175, 3.7525),
    (-2.8973, 2.8973),
]
# Official kinematic chain (franka_description robots/common/franka_arm.xacro,
# noetic-devel): per-joint origin xyz + rpy relative to the parent link, all
# joints revolute about their local z. Flange (link8) at z=0.107 after joint7.
PANDA_JOINT_ORIGINS = [
    ((0.0, 0.0, 0.333), (0.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0)),
    ((0.0, -0.316, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((0.0825, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((-0.0825, 0.384, 0.0), (-math.pi / 2, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
    ((0.088, 0.0, 0.0), (math.pi / 2, 0.0, 0.0)),
]
PANDA_FLANGE_Z = 0.107


@dataclass
class CameraConfig:
    name: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    width: int = 640
    height: int = 480
    focal_length: float = 24.0
    role: str = "actor"  # actor | teacher | overview


@dataclass
class SharpaSceneConfig:
    """Everything needed to build/rebuild the stage."""

    seed: int = 0
    # World layout -----------------------------------------------------
    base_xy: tuple[float, float] = (0.42, 0.30)  # arm base, right-rear of pad
    base_yaw: float = math.radians(180.0)  # base rotated 180 degrees about world Z
    pad_xy: tuple[float, float] = (0.0, 0.0)
    pad_yaw: float = 0.0
    pad_tilt_deg: float = 0.0  # rotation about pad x-axis
    # Magic Trackpad-sized plate: Apple publishes Magic Trackpad 2 as
    # 160.0 x 114.9 x 10.9 mm (landscape). Assumption recorded here.
    pad_size: tuple[float, float, float] = (0.160, 0.1149, 0.0109)
    pad_source: str = (
        "Assumed Apple Magic Trackpad 2 dimensions 160.0x114.9x10.9 mm "
        "(landscape), public manufacturer spec; not measured."
    )
    table_size: tuple[float, float, float] = (1.20, 0.80, 0.04)
    table_top_z: float = 0.72
    # Physical screen behind the pad (faces the operator at -Y).
    screen_size: tuple[float, float] = (0.30, 0.19)  # width x height m
    screen_center: tuple[float, float, float] = (-0.38, 0.34, 1.15)
    screen_tilt_deg: float = -8.0
    # Physics ----------------------------------------------------------
    physics_dt: float = 1.0 / 120.0
    render_dt: float = 1.0 / 30.0
    # Cameras ----------------------------------------------------------
    cameras: tuple[CameraConfig, ...] = field(
        default_factory=lambda: (
            CameraConfig(
                name="overview",
                position=(1.4, -1.5, 1.8),
                target=(0.0, 0.15, 1.05),
                focal_length=18.0,
                role="overview",
            ),
            CameraConfig(
                name="screen_rgb",
                position=(-0.38, -0.28, 1.22),
                target=(-0.38, 0.34, 1.15),
                role="actor",
            ),
            # wrist camera pose is relative to the wrist link; set by builder.
            CameraConfig(
                name="wrist_rgb",
                position=(0.0, 0.0, 0.0),
                target=(0.0, 0.0, 0.0),
                role="actor",
            ),
        )
    )
    wrist_cam_offset_xyz: tuple[float, float, float] = (0.03, -0.04, 0.06)
    wrist_cam_pitch_deg: float = -55.0  # look down at index-pad area
    # Solved home configuration (set by the scene builder after checked IK;
    # None until solved). home_q7 are arm joint targets in rad.
    home_q7: tuple[float, ...] | None = None
    home_tip_world: tuple[float, ...] | None = None  # world frame, metres
    home_hover_m: float = 0.035  # index tip 25-50 mm above pad top
    # Panda hand-mount anchors read live from the franka.usd asset
    # (/panda/panda_link7/panda_hand_joint). Stored here so the runner can
    # FK-check the measured flange pose without re-opening the asset.
    mount_pos0: tuple[float, ...] = (0.0, 0.0, 0.107)
    mount_rot0_wxyz: tuple[float, ...] = (0.70710677, 0.0, 0.70710677, 0.0)
    mount_pos1: tuple[float, ...] = (0.0, 0.0, 0.0)
    mount_rot1_wxyz: tuple[float, ...] = (
        0.6532815, -0.2705981, 0.6532814, 0.27059814)

    def validate(self) -> None:
        if self.physics_dt <= 0 or self.render_dt <= 0:
            raise ValueError("dt must be positive")
        if min(self.pad_size) <= 0 or min(self.table_size) <= 0:
            raise ValueError("sizes must be positive")
        if abs(self.pad_tilt_deg) > 45 or abs(self.screen_tilt_deg) > 45:
            raise ValueError("tilt suspiciously large")
        seen = set()
        for cam in self.cameras:
            if cam.name in seen:
                raise ValueError(f"duplicate camera {cam.name}")
            seen.add(cam.name)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["action_schema"] = ACTION_SCHEMA
        d["arm_joint_limits_rad"] = PANDA_LIMITS
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SharpaSceneConfig":
        d = dict(d)
        d.pop("action_schema", None)
        d.pop("arm_joint_limits_rad", None)
        cams = d.get("cameras")
        if cams is not None:
            d["cameras"] = tuple(CameraConfig(**c) for c in cams)
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
