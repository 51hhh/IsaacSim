# SPDX-FileCopyrightText: Copyright (c) 2025 Your Name
# SPDX-License-Identifier: MIT
"""
简单倒立摆示例 - 使用 Isaac Sim 原生 API

本示例演示如何：
1. 从 URDF 文件导入 cartpole 模型
2. 设置物理仿真环境
3. 使用简单的 PD 控制器保持杆的平衡

运行方法：
    C:\IsaacSim\python.bat cartpole_simple.py
"""

from isaacsim import SimulationApp

# 启动仿真应用（带渲染）
simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import math

import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, UsdLux, UsdPhysics


def setup_physics_scene(stage):
    """设置物理场景参数"""
    # 创建物理场景
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    
    # 配置 PhysX 求解器
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    physx_scene_api.CreateEnableCCDAttr(True)
    physx_scene_api.CreateEnableStabilizationAttr(True)
    physx_scene_api.CreateEnableGPUDynamicsAttr(True)  # 启用 GPU 物理加速
    physx_scene_api.CreateSolverTypeAttr("TGS")


def setup_environment(stage):
    """设置环境（地面和灯光）"""
    # 添加地面平面
    PhysicsSchemaTools.addGroundPlane(
        stage, "/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.3)
    )
    
    # 添加灯光
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distant_light.CreateIntensityAttr(500)
    
    # 添加环境光
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    dome_light.CreateIntensityAttr(100)


def import_cartpole():
    """从 URDF 导入 cartpole 模型"""
    # 设置 URDF 导入配置
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True  # 固定底座（滑轨）
    import_config.distance_scale = 1.0
    
    # 获取 URDF 文件路径
    extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
    urdf_path = extension_path + "/data/urdf/robots/cartpole/cartpole.urdf"
    
    # 导入 URDF
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=import_config,
        get_articulation_root=True,
    )
    
    print(f"[INFO] Cartpole 已导入到: {prim_path}")
    return prim_path


class CartpolePDController:
    """
    简单的 PD 控制器实现倒立摆平衡
    
    控制策略：
    - 使用杆的角度和角速度计算平衡力
    - 同时控制小车位置保持在中心
    """
    
    def __init__(self):
        # PD 控制参数（针对杆的角度）- 调优后的参数
        self.kp_pole = 150.0   # 杆角度的比例增益
        self.kd_pole = 20.0    # 杆角速度的微分增益
        
        # PD 控制参数（针对小车位置）
        self.kp_cart = 5.0     # 小车位置的比例增益
        self.kd_cart = 10.0    # 小车速度的微分增益
        
        # 目标状态
        self.target_pole_angle = 0.0  # 目标杆角度（直立）
        self.target_cart_pos = 0.0    # 目标小车位置（中心）
    
    def compute_force(self, cart_pos, cart_vel, pole_angle, pole_vel):
        """
        计算施加在小车上的力
        
        Args:
            cart_pos: 小车位置
            cart_vel: 小车速度
            pole_angle: 杆的角度（弧度）
            pole_vel: 杆的角速度
        
        Returns:
            施加在小车上的力
        """
        # 杆平衡控制（核心部分）
        pole_error = self.target_pole_angle - pole_angle
        force_pole = self.kp_pole * pole_error - self.kd_pole * pole_vel
        
        # 小车位置控制（辅助部分）
        cart_error = self.target_cart_pos - cart_pos
        force_cart = self.kp_cart * cart_error - self.kd_cart * cart_vel
        
        # 总力 = 杆平衡力 + 小车位置控制力
        total_force = force_pole + force_cart
        
        # 限制最大力（模拟电机限制）
        max_force = 500.0
        total_force = np.clip(total_force, -max_force, max_force)
        
        return total_force


def main():
    """主函数"""
    # 获取 Stage
    stage = omni.usd.get_context().get_stage()
    
    # 设置物理场景
    setup_physics_scene(stage)
    
    # 设置环境
    setup_environment(stage)
    
    # 导入 cartpole
    prim_path = import_cartpole()
    
    # 更新一次让资源加载完成
    simulation_app.update()
    
    # 创建仿真上下文
    sim_context = SimulationContext(physics_dt=1.0/120.0, rendering_dt=1.0/60.0)
    
    # 设置相机视角
    set_camera_view(
        eye=[5.0, 0.0, 2.0],
        target=[0.0, 0.0, 1.0],
        camera_prim_path="/OmniverseKit_Persp"
    )
    
    # 初始化物理
    sim_context.initialize_physics()
    
    # 创建 Articulation 对象
    cartpole = Articulation(prim_path)
    cartpole.initialize()
    
    # 获取关节索引
    # slider_to_cart: 小车位置（prismatic）
    # cart_to_pole: 杆角度（continuous）
    cart_joint_idx = cartpole.get_dof_index("slider_to_cart")
    pole_joint_idx = cartpole.get_dof_index("cart_to_pole")
    
    print(f"[INFO] 关节索引 - 小车: {cart_joint_idx}, 杆: {pole_joint_idx}")
    print(f"[INFO] 总关节数: {cartpole.num_dof}")
    
    # 创建控制器
    controller = CartpolePDController()
    
    # 开始仿真
    sim_context.play()
    
    # 给杆一个初始扰动（不然它会完美平衡，没有挑战性）
    initial_pole_angle = 0.05  # 弧度，约 2.9 度
    cartpole.set_joint_positions([[0.0, initial_pole_angle]])
    
    print("[INFO] 仿真开始...")
    print("[INFO] 按 Ctrl+C 或关闭窗口退出")
    print(f"[INFO] 初始杆角度: {math.degrees(initial_pole_angle):.1f} 度")
    
    step_count = 0
    
    # 仿真主循环
    while simulation_app.is_running():
        # 获取当前状态
        joint_positions = cartpole.get_joint_positions()
        joint_velocities = cartpole.get_joint_velocities()
        
        if joint_positions is not None and joint_velocities is not None:
            cart_pos = joint_positions[0, cart_joint_idx]
            cart_vel = joint_velocities[0, cart_joint_idx]
            pole_angle = joint_positions[0, pole_joint_idx]
            pole_vel = joint_velocities[0, pole_joint_idx]
            
            # 计算控制力
            force = controller.compute_force(cart_pos, cart_vel, pole_angle, pole_vel)
            
            # 施加力到小车关节
            # 注意：对于 prismatic 关节，我们施加力（force）
            efforts = np.zeros((1, cartpole.num_dof))
            efforts[0, cart_joint_idx] = force
            cartpole.set_joint_efforts(efforts)
            
            # 定期打印状态
            if step_count % 120 == 0:  # 每秒打印一次（120 Hz 物理）
                print(f"[状态] 小车位置: {cart_pos:6.2f} m | "
                      f"杆角度: {math.degrees(pole_angle):6.1f}° | "
                      f"控制力: {force:7.1f} N")
        
        # 推进仿真
        sim_context.step(render=True)
        step_count += 1
    
    # 清理
    sim_context.stop()
    simulation_app.close()
    print("[INFO] 仿真结束")


if __name__ == "__main__":
    main()
