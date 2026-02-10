"""
最小化碰撞测试 v2：
- 修复：物理材质必须用 UsdShade.Material.Define() 创建
- 简化：球的 RigidBody + Collision 放在同一个 prim 上（扁平结构）

用法: C:\IsaacSim\python.bat collision_test.py
"""

import time

# Isaac Sim 启动
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False, "width": 1280, "height": 720})

import numpy as np
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
import omni.usd


def create_physics_material(stage, path, static_friction, dynamic_friction, restitution):
    """
    创建物理材质 — ★ 必须用 UsdShade.Material.Define() 而非 stage.DefinePrim()
    否则 MaterialBindingAPI.Bind() 不会生效
    """
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    mat_api.CreateStaticFrictionAttr(static_friction)
    mat_api.CreateDynamicFrictionAttr(dynamic_friction)
    mat_api.CreateRestitutionAttr(restitution)
    return material


def main():
    stage = omni.usd.get_context().get_stage()

    # =========================================
    # 1. 物理场景
    # =========================================
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableGPUDynamicsAttr(False)
    physx_scene.CreateBounceThresholdAttr(0.01)

    # =========================================
    # 2. 物理材质（★ 正确方式：UsdShade.Material.Define）
    # =========================================
    ground_mat = create_physics_material(
        stage, "/Materials/GroundPhysMat", 0.8, 0.6, 0.75
    )
    ball_mat = create_physics_material(
        stage, "/Materials/BallPhysMat", 0.6, 0.4, 0.75
    )
    net_mat = create_physics_material(
        stage, "/Materials/NetPhysMat", 0.3, 0.2, 0.3
    )

    # =========================================
    # 3. 地面
    # =========================================
    PhysicsSchemaTools.addGroundPlane(
        stage, "/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.5)
    )
    # 绑定物理材质到碰撞面
    ground_geom = stage.GetPrimAtPath("/groundPlane/geom")
    if ground_geom.IsValid():
        binding = UsdShade.MaterialBindingAPI.Apply(ground_geom)
        binding.Bind(ground_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        print("[OK] 地面物理材质已绑定到 /groundPlane/geom")
    else:
        # 查找实际的碰撞子 prim
        print("[WARN] /groundPlane/geom 不存在，查找子 prim...")
        for child in stage.GetPrimAtPath("/groundPlane").GetChildren():
            print(f"  子 prim: {child.GetPath()} type={child.GetTypeName()}")
            if child.GetTypeName() in ("Plane", "Mesh", "Cube"):
                binding = UsdShade.MaterialBindingAPI.Apply(child)
                binding.Bind(ground_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
                print(f"  [OK] 绑定到 {child.GetPath()}")

    # =========================================
    # 4. 球网 — 静态碰撞盒子（x=0, 厚 20cm）
    # =========================================
    net = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Net"))
    net.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(net)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1.215))
    xf.AddScaleOp().Set(Gf.Vec3f(0.2, 10.0, 2.43))
    UsdPhysics.CollisionAPI.Apply(net.GetPrim())
    # 绑定物理材质
    net_binding = UsdShade.MaterialBindingAPI.Apply(net.GetPrim())
    net_binding.Bind(net_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    print("[OK] 球网碰撞体已创建")

    # =========================================
    # 5. 球 — 扁平结构（RigidBody + Collision 同层）
    # =========================================
    ball_path = "/World/Ball"
    ball = UsdGeom.Sphere.Define(stage, Sdf.Path(ball_path))
    ball.CreateRadiusAttr(0.105)

    xf = UsdGeom.Xformable(ball)
    xf.AddTranslateOp().Set(Gf.Vec3d(-5.0, 0.0, 3.0))

    # Collision
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())

    # RigidBody
    UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())

    # Mass
    mass_api = UsdPhysics.MassAPI.Apply(ball.GetPrim())
    mass_api.CreateMassAttr(0.27)

    # 绑定物理材质
    ball_binding = UsdShade.MaterialBindingAPI.Apply(ball.GetPrim())
    ball_binding.Bind(ball_mat, UsdShade.Tokens.weakerThanDescendants, "physics")

    # PhysX 扩展
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(ball.GetPrim())
    physx_rb.CreateLinearDampingAttr(0.0)
    physx_rb.CreateAngularDampingAttr(0.0)
    physx_rb.CreateEnableCCDAttr(True)

    # 视觉材质
    vis_mat = UsdShade.Material.Define(stage, Sdf.Path("/Materials/BallVisMat"))
    shader = UsdShade.Shader.Define(stage, Sdf.Path("/Materials/BallVisMat/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.85, 0.3))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    vis_mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(ball).Bind(vis_mat)

    print(f"[OK] 球已创建: RigidBody + Collision + PhysMat")

    # 灯光
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    dome.CreateIntensityAttr(300)

    # =========================================
    # 6. 仿真
    # =========================================
    simulation_app.update()

    sim_context = SimulationContext(physics_dt=1.0/240.0, rendering_dt=1.0/60.0)
    set_camera_view(
        eye=[0.0, -12.0, 6.0],
        target=[0.0, 0.0, 1.0],
        camera_prim_path="/OmniverseKit_Persp"
    )

    sim_context.initialize_physics()
    ball_rigid = RigidPrim(prim_paths_expr=ball_path, name="ball_view")

    sim_context.play()

    # 初始速度：水平 15 m/s + 向上 2 m/s
    vel_6 = np.zeros((1, 6), dtype=np.float32)
    vel_6[0, 0] = 15.0
    vel_6[0, 2] = 2.0
    ball_rigid.set_velocities(vel_6)

    print("=" * 60)
    print("  碰撞测试 v2 — 球 (-5,0,3) → 15m/s 飞向球网")
    print("  修复：物理材质用 UsdShade.Material.Define() 创建")
    print("  期望：弹跳 + 被球网挡住 + 落地摩擦减速")
    print("=" * 60)

    step = 0
    last_print = time.time()
    prev_z = 3.0
    bounce_count = 0

    while simulation_app.is_running():
        sim_context.step(render=True)
        step += 1

        pos, _ = ball_rigid.get_world_poses()
        vel = ball_rigid.get_linear_velocities()
        p = pos[0]
        v = vel[0]
        speed = np.linalg.norm(v)
        z = float(p[2])

        # 弹跳检测
        if z > prev_z + 0.01 and prev_z < 0.3:
            bounce_count += 1
            print(f"  🏀 弹跳 #{bounce_count}! z={z:.3f} vz={v[2]:+.2f}")
        prev_z = z

        # 打印
        now = time.time()
        if now - last_print >= 0.3:
            print(
                f"  pos=({p[0]:+7.2f}, {p[1]:+5.2f}, {p[2]:+6.3f}) "
                f"vel=({v[0]:+6.2f}, {v[1]:+5.2f}, {v[2]:+6.2f}) "
                f"|v|={speed:5.1f}m/s"
            )
            last_print = now

        if step > 240 * 15:
            print("\n  ⏱ 15秒超时")
            break
        if speed < 0.01 and z < 0.2 and step > 240:
            print(f"\n  ✅ 球停下了 (step={step}, t={step/240:.1f}s, 弹跳{bounce_count}次)")
            break
        if abs(p[0]) > 30 or z < -1:
            print(f"\n  ❌ 球出界 x={p[0]:.1f} z={z:.1f}")
            break

    print(f"\n  总结: 弹跳 {bounce_count} 次")
    if bounce_count > 0:
        print("  ✅ 碰撞 + 弹跳 + 恢复系数正常！")
    else:
        print("  ❌ 没有弹跳 — 碰撞有问题")

    sim_context.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
