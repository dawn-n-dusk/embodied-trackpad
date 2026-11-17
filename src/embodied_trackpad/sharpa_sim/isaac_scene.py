"""Offline USD stage builder for the Sharpa right-hand trackpad scene.

Runs with pxr only (no GPU). Envs under ``/World/envs/env_i`` are offset
along +X by the env root transform ONLY (children use env-local coords).

Audit-driven structure:
- Table / Trackpad / Screen are STATIC collision geometry: no RigidBodyAPI,
  no inter-object joints. Their Xforms carry explicit translate+orient ops
  (translate op, then rotateXYZ op about the prim's own centre).
- Arm = official NVIDIA FrankaPanda USD referenced at ``ArmMount/Arm``
  (Gripper=None variant, non-instanceable). Panda articulation root stays.
- Sharpa hand is a SIBLING reference at ``ArmMount/SharpaHand`` (never nested
  under a rigid link). The hand USD's own ``root_joint`` is a world anchor;
  it is disabled here and the hand's local ArticulationRootAPI is removed in
  the composed stage. A single FixedJoint welds ``panda_link7`` to the hand
  base link ``right_hand_flange`` at the official flange frame, so the arm +
  hand form one 29-DoF articulation with a single root at the arm base.
- Cameras use explicit USD look-at; a dome light illuminates the scene.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from . import kinematics as kn
from .config import SharpaSceneConfig

ENV_SPACING_X = 1.6


def _M_from_row_np(M: np.ndarray) -> Gf.Matrix4d:
    """numpy COLUMN-convention 4x4 -> pxr row-vector Gf.Matrix4d."""
    return Gf.Matrix4d(*[float(v) for v in kn.col_to_row(M).flatten()])


def _paint(stage, prim, rgb):
    material = UsdShade.Material.Define(stage, prim.GetPath().AppendChild("mtl"))
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _box_local(stage, path, size, color):
    """Box mesh with LOCAL-space vertices centred at the origin."""
    prim = UsdGeom.Mesh.Define(stage, path)
    sx, sy, sz = (s / 2 for s in size)
    pts = [
        (-sx, -sy, -sz), (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz),
        (-sx, -sy, sz), (sx, -sy, sz), (sx, sy, sz), (-sx, sy, sz),
    ]
    prim.CreatePointsAttr(pts)
    prim.CreateFaceVertexCountsAttr([4] * 6)
    prim.CreateFaceVertexIndicesAttr([
        0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 7, 3, 1, 5, 6, 2, 0, 1, 5, 4, 3, 2, 6, 7,
    ])
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    _paint(stage, prim.GetPrim(), color)
    return prim


def _disc_xz(stage, path, radius, y_offset, color, segments=48):
    """Disc in the local XZ plane (faces -Y), offset along +Y from surface."""
    prim = UsdGeom.Mesh.Define(stage, path)
    pts = [(0.0, y_offset, 0.0)]
    for i in range(segments):
        a = 2 * math.pi * i / segments
        pts.append((radius * math.cos(a), y_offset, radius * math.sin(a)))
    counts, idx = [], []
    for i in range(segments):
        counts.append(3)
        idx += [0, 1 + i, 1 + (i + 1) % segments]
    prim.CreatePointsAttr(pts)
    prim.CreateFaceVertexCountsAttr(counts)
    prim.CreateFaceVertexIndicesAttr(idx)
    _paint(stage, prim.GetPrim(), color)
    return prim


def _place(xf: UsdGeom.Xformable, translate, rot_x_deg=0.0, rot_z_deg=0.0):
    """Explicit translate + rotateXYZ ops (rotate applied about own centre)."""
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(rot_x_deg, 0.0, rot_z_deg))


def _look_at_quat(position, target):
    px, py, pz = position
    tx, ty, tz = target
    f = Gf.Vec3d(tx - px, ty - py, tz - pz).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    z = -f
    x = Gf.Cross(up, z).GetNormalized()
    y = Gf.Cross(z, x).GetNormalized()
    m4 = Gf.Matrix4d(1.0)
    m4.SetRotate(Gf.Matrix3d(x[0], x[1], x[2],
                             y[0], y[1], y[2],
                             z[0], z[1], z[2]))
    return m4.ExtractRotation().GetQuaternion()


def _look_at_rowM(position, target) -> Gf.Matrix4d:
    """Full row-vector look-at matrix (camera convention: looks down -Z)."""
    px, py, pz = position
    tx, ty, tz = target
    f = Gf.Vec3d(tx - px, ty - py, tz - pz).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    z = -f
    x = Gf.Cross(up, z).GetNormalized()
    y = Gf.Cross(z, x).GetNormalized()
    m4 = Gf.Matrix4d(1.0)
    m4.SetRotate(Gf.Matrix3d(x[0], x[1], x[2],
                             y[0], y[1], y[2],
                             z[0], z[1], z[2]))
    m4.SetTranslateOnly(Gf.Vec3d(*position))
    return m4


def _camera(stage, path, position, target, focal=24.0):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(20.955)
    # stage units are METRES: near/far must not clip sub-metre targets
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50.0))
    q = _look_at_quat(position, target)
    qi = q.GetImaginary()
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddOrientOp().Set(Gf.Quatf(float(q.GetReal()), Gf.Vec3f(qi[0], qi[1], qi[2])))
    return cam


def _flange_pose(franka_usd: Path):
    """Official hand-mount joint anchors from the NVIDIA FrankaPanda USD.

    The real prim is /panda/panda_link7/panda_hand_joint (a fixed joint).
    Returns (pos0, rot0, pos1, rot1): joint frame in panda_link7 and in
    panda_hand respectively. Raises if the joint or its anchors are missing;
    there is intentionally no silent fallback while calling this official.
    """
    stage = Usd.Stage.Open(str(franka_usd))
    joint = stage.GetPrimAtPath("/panda/panda_link7/panda_hand_joint")
    if joint is None:
        raise RuntimeError("official /panda/panda_link7/panda_hand_joint missing")
    j = UsdPhysics.FixedJoint(joint)
    pos0, rot0 = j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()
    pos1, rot1 = j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()
    if None in (pos0, rot0, pos1, rot1):
        raise RuntimeError("official panda_hand_joint anchors incomplete")
    return Gf.Vec3f(pos0), Gf.Quatf(rot0), Gf.Vec3f(pos1), Gf.Quatf(rot1)


def _rest_link7_world(franka_usd: Path):
    """Rest-pose world matrix of /panda/panda_link7 in the franka stage."""
    stage = Usd.Stage.Open(str(franka_usd))
    prim = stage.GetPrimAtPath("/panda/panda_link7")
    if prim is None:
        raise RuntimeError("official /panda/panda_link7 missing")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default())


def _M_from_quat_trans(q: Gf.Quatf, t: Gf.Vec3f):
    qi = q.GetImaginary()
    m = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Quatd(float(q.GetReal()), Gf.Vec3d(float(qi[0]),
                                                          float(qi[1]),
                                                          float(qi[2])))))
    m.SetTranslateOnly(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))
    return m


def _disable_hand_world_anchor(stage, hand_root: str):
    """Disable the hand's root_joint world anchor + drop its articulation root.

    Authored as local overrides in the composed stage; payloads untouched.
    """
    root = UsdPhysics.ArticulationRootAPI.Get(stage, hand_root)
    if root:
        root.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
    for name in ("root_joint",):
        prim = stage.GetPrimAtPath(f"{hand_root}/{name}")
        if prim is not None:
            prim.CreateAttribute("physics:jointEnabled",
                                 Sdf.ValueTypeNames.Bool).Set(False)


def _disable_gripper_subtree(stage, arm_path: str):
    """Physically remove the old Franka gripper from the composed stage.

    Gripper=None only hides geometry; the prims remain. Deactivating the
    panda_hand subtree removes panda_finger_joint1/2 (they live under it),
    leaving exactly 7 arm + 22 Sharpa DOFs.
    """
    for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
        prim = stage.GetPrimAtPath(f"{arm_path}/{name}")
        if prim:
            prim.SetActive(False)
    joint = stage.GetPrimAtPath(f"{arm_path}/panda_link7/panda_hand_joint")
    if joint is not None:
        joint.CreateAttribute("physics:jointEnabled",
                              Sdf.ValueTypeNames.Bool).Set(False)


def build_stage(config: SharpaSceneConfig, num_envs: int, hand_usd: Path,
                franka_usd: Path, out_path: Path) -> dict:
    config.validate()
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.Scene.Define(stage, "/World/physicsScene")

    ground = UsdGeom.Xform.Define(stage, "/World/ground")
    gmesh = UsdGeom.Mesh.Define(stage, "/World/ground/mesh")
    gmesh.CreatePointsAttr([(-20, -20, 0), (20, -20, 0), (20, 20, 0), (-20, 20, 0)])
    gmesh.CreateFaceVertexCountsAttr([4])
    gmesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    UsdPhysics.CollisionAPI.Apply(gmesh.GetPrim())

    dome = UsdLux.DomeLight.Define(stage, "/World/domeLight")
    dome.CreateIntensityAttr(1000.0)

    pad_w, pad_h, pad_t = config.pad_size
    table_z = config.table_top_z
    px, py = config.pad_xy
    sw, sh = config.screen_size
    sc = config.screen_center
    fpos, frot, fpos1, frot1 = _flange_pose(franka_usd)
    M7 = _rest_link7_world(franka_usd)  # rest world of link7 (franka frame)
    M_weld = _M_from_quat_trans(frot, fpos)    # joint frame0 in panda_link7
    M_weld1 = _M_from_quat_trans(frot1, fpos1)  # joint frame1 in hand body

    # ---- checked home pose (IK, verified against FK below) -------------
    rot0_wxyz = (float(frot.GetReal()), float(frot.GetImaginary()[0]),
                 float(frot.GetImaginary()[1]), float(frot.GetImaginary()[2]))
    rot1_wxyz = (float(frot1.GetReal()), float(frot1.GetImaginary()[0]),
                 float(frot1.GetImaginary()[1]), float(frot1.GetImaginary()[2]))
    mount_pos0 = (float(fpos[0]), float(fpos[1]), float(fpos[2]))
    mount_pos1 = (float(fpos1[0]), float(fpos1[1]), float(fpos1[2]))
    config.mount_pos0 = mount_pos0
    config.mount_rot0_wxyz = rot0_wxyz
    config.mount_pos1 = mount_pos1
    config.mount_rot1_wxyz = rot1_wxyz
    tip_target_base = kn.tip_target_in_base(config, config.home_hover_m)
    home = kn.solve_home(tip_target_base, mount_pos0, rot0_wxyz, rot1_wxyz)
    if not home.success:
        raise RuntimeError(
            f"home IK failed to converge: {home.to_dict()} (target "
            f"{tip_target_base.tolist()} m above pad top "
            f"{config.home_hover_m * 1000:.0f} mm)")
    config.home_q7 = tuple(round(float(v), 6) for v in home.q)
    # FK self-check of the solved configuration (independent recompute).
    # tip_check is in the ARM BASE frame; the base sits at table_top_z, so
    # hover above the pad TOP is base-frame z minus pad thickness only.
    tip_check = kn.fk_index_tip(home.q, mount_pos0, rot0_wxyz, rot1_wxyz)
    if not np.allclose(tip_check, home.tip_world, atol=1e-9):
        raise RuntimeError("FK self-check of solved home pose disagrees")
    hover_check = tip_check[2] - config.pad_size[2]
    if not (0.02 <= hover_check <= 0.06):
        raise RuntimeError(f"solved hover {hover_check * 1000:.1f} mm outside "
                           "20-60 mm sanity band (FK convention bug?)")

    ax, ay = config.base_xy
    arm_mount_col = kn.RT(kn.rpy_to_mat((0.0, 0.0, config.base_yaw)),
                          (ax, ay, config.table_top_z))
    q7 = np.asarray(config.home_q7, dtype=float)
    tip_home_world = (arm_mount_col @ np.append(home.tip_world, 1.0))[:3]
    config.home_tip_world = tuple(round(float(v), 6) for v in tip_home_world)

    def _hand_local_row() -> Gf.Matrix4d:
        """pxr ROW-matrix local for the Sharpa reference under ArmMount.

        Independent Gf check on the installed runtime established pxr
        composition: a child's world is local*parent with the local matrix
        applied FIRST (row vectors). Hence W0(link7) = M7*W_mount and
            local = W1*inv(W_mount) = inv(B1)*B0*M7 ,
        which is identical for every env (the env offset sits above
        ArmMount and cancels). Weld and placement share this frame.
        """
        return M_weld1.GetInverse() * M_weld * M7

    # wrist camera world extrinsics at the nominal (home) pose: behind the
    # hand toward the operator, looking at the index tip. The local
    # extrinsic under panda_link7 is identical for every env (env offset
    # cancels in local = W_cam*inv(W_link7)), so it is computed once.
    tip = tip_home_world
    # wrist camera: thumb/index side of the tip (body +Y -> world +X),
    # looking back at it; focal 18
    cam_pos = (float(tip[0]) + 0.10, float(tip[1]) - 0.16,
               float(tip[2]) + 0.17)

    def _wrist_local_row() -> Gf.Matrix4d:
        W_cam = _look_at_rowM(cam_pos, tuple(tip))
        L7_row = _M_from_row_np(kn.panda_fk(q7))       # link7 @ home
        W_mount_row = _M_from_row_np(arm_mount_col)    # ArmMount world
        W_link7 = L7_row * W_mount_row                 # row: local first
        return W_cam * W_link7.GetInverse()

    info = {"envs": [], "static_furniture": True}
    for i in range(num_envs):
        root = f"/World/envs/env_{i}"
        env = UsdGeom.Xform.Define(stage, root)
        _place(UsdGeom.Xformable(env.GetPrim()), (i * ENV_SPACING_X, 0, 0))

        # ---- static furniture (no rigid bodies, no joints) ----
        table = UsdGeom.Xform.Define(stage, f"{root}/Table")
        _place(UsdGeom.Xformable(table.GetPrim()), (0, 0, table_z))
        _box_local(stage, f"{root}/Table/geom", config.table_size, (0.45, 0.36, 0.28))
        # geom sits so its top face is at local z=0 (tabletop == table_z)
        UsdGeom.Xformable(stage.GetPrimAtPath(f"{root}/Table/geom")).AddTranslateOp().Set(
            Gf.Vec3d(0, 0, -config.table_size[2] / 2))

        pad = UsdGeom.Xform.Define(stage, f"{root}/Trackpad")
        _place(UsdGeom.Xformable(pad.GetPrim()), (px, py, table_z + pad_t / 2),
               rot_x_deg=config.pad_tilt_deg, rot_z_deg=math.degrees(config.pad_yaw))
        _box_local(stage, f"{root}/Trackpad/geom", config.pad_size, (0.75, 0.78, 0.82))

        screen = UsdGeom.Xform.Define(stage, f"{root}/Screen")
        _place(UsdGeom.Xformable(screen.GetPrim()), sc,
               rot_x_deg=config.screen_tilt_deg)
        _box_local(stage, f"{root}/Screen/geom", (sw, 0.02, sh), (0.08, 0.09, 0.11))
        # discs face the operator (-Y); geometry in the panel plane, with the
        # front-surface offset carried by an explicit translate op (y) so the
        # runner can update x/z from GUI truth while preserving it.
        _disc_xz(stage, f"{root}/Screen/target_disc", sw * 26.0 / 640.0, 0.0,
                 (0.1, 0.8, 0.2))
        _disc_xz(stage, f"{root}/Screen/cursor_disc", sw * 5.0 / 640.0, 0.0,
                 (0.95, 0.9, 0.1))
        for disc, yoff in (("target_disc", -0.012), ("cursor_disc", -0.014)):
            UsdGeom.Xformable(stage.GetPrimAtPath(
                f"{root}/Screen/{disc}")).AddTranslateOp().Set(
                Gf.Vec3d(0.0, yoff, 0.0))

        # ---- official NVIDIA Franka Panda arm (temporary simulation arm) ----
        arm_mount = UsdGeom.Xform.Define(stage, f"{root}/ArmMount")
        ax, ay = config.base_xy
        _place(UsdGeom.Xformable(arm_mount.GetPrim()), (ax, ay, table_z),
               rot_z_deg=math.degrees(config.base_yaw))
        arm_prim = stage.DefinePrim(f"{root}/ArmMount/Arm")
        arm_prim.GetReferences().AddReference(str(franka_usd))
        arm_prim.SetInstanceable(False)
        vset = arm_prim.GetVariantSet("Gripper")
        if vset and vset.GetVariantSelection() is not None:
            vset.SetVariantSelection("None")
        _disable_gripper_subtree(stage, f"{root}/ArmMount/Arm")
        for name in ("panda_hand", "panda_leftfinger", "panda_rightfinger"):
            gp = stage.GetPrimAtPath(f"{root}/ArmMount/Arm/{name}")
            if gp is not None and gp.IsActive():
                raise RuntimeError(
                    f"gripper subtree still active after deactivation: {name}")

        # ---- Sharpa hand as SIBLING reference, welded at the flange ----
        href = stage.DefinePrim(f"{root}/ArmMount/SharpaHand")
        href.GetReferences().AddReference(str(hand_usd))
        href.SetInstanceable(False)
        _disable_hand_world_anchor(stage, f"{root}/ArmMount/SharpaHand")
        # Pre-physics alignment: SAME adapter frame as the weld below, so the
        # fixed joint starts satisfied instead of snapping at startup.
        UsdGeom.Xformable(href).AddTransformOp().Set(_hand_local_row())
        hfix = UsdPhysics.FixedJoint.Define(stage, f"{root}/HandFlangeWeld")
        hfix.CreateBody0Rel().SetTargets(
            [f"{root}/ArmMount/Arm/panda_link7"])
        hfix.CreateBody1Rel().SetTargets(
            [f"{root}/ArmMount/SharpaHand/right_hand_flange"])
        hfix.CreateLocalPos0Attr().Set(fpos)
        hfix.CreateLocalRot0Attr().Set(frot)
        # Adapter frame in the Sharpa flange body: documented assumption is
        # that the Sharpa right_hand_flange body frame coincides with the
        # Panda hand body mount frame, so the asset's own frame1 rotation
        # (and tiny translation) is reused verbatim. Placement + weld share
        # this frame.
        hfix.CreateLocalPos1Attr().Set(fpos1)
        hfix.CreateLocalRot1Attr().Set(frot1)

        # wrist RGB camera: child of panda_link7, local extrinsic fixed at
        # the nominal home pose (world position/target computed above).
        wc = UsdGeom.Camera.Define(
            stage, f"{root}/ArmMount/Arm/panda_link7/wrist_rgb")
        wc.CreateFocalLengthAttr(18.0)
        wc.CreateHorizontalApertureAttr(20.955)
        wc.CreateClippingRangeAttr(Gf.Vec2f(0.005, 50.0))
        UsdGeom.Xformable(wc.GetPrim()).AddTransformOp().Set(
            _wrist_local_row())
        wc.GetPrim().CreateAttribute("role", Sdf.ValueTypeNames.String).Set(
            "actor")
        info["envs"].append({"root": root, "home_q7": list(config.home_q7)})

    for cam_cfg in config.cameras:
        if cam_cfg.name == "wrist_rgb":
            continue
        cam = _camera(stage, f"/World/{cam_cfg.name}", cam_cfg.position,
                      cam_cfg.target, cam_cfg.focal_length)
        cam.GetPrim().CreateAttribute("role", Sdf.ValueTypeNames.String).Set(cam_cfg.role)
        cam.GetPrim().CreateAttribute("resolution", Sdf.ValueTypeNames.Int2).Set(
            Gf.Vec2i(cam_cfg.width, cam_cfg.height))

    stage.GetRootLayer().Save()
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--num-envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hand-usd", type=Path, required=True)
    ap.add_argument("--franka-usd", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cfg = SharpaSceneConfig(seed=args.seed)
    if args.config:
        cfg = SharpaSceneConfig.from_dict(json.loads(args.config.read_text()))
    info = build_stage(cfg, args.num_envs, args.hand_usd, args.franka_usd, args.out)
    args.out.with_suffix(".config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2) + "\n")
    print("wrote", args.out, "envs:", args.num_envs, "info:", info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
