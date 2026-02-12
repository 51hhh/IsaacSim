# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
舵轮底盘 URDF 机器人模块

包含:
- URDF 加载与配置
- 舵轮运动学控制
- 全向移动指令转换

机器人结构 (N4 四轮舵轮底盘):
- 4个舵向关节: FL, FB, RB, RL (6020电机)
- 4个驱动轮关节: FLW, FBW, RBW, RLW (3508电机)
"""

import math
import os
import numpy as np

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, PhysxSchema


# =====================================================================
# 默认路径配置
# =====================================================================

# 默认 URDF 路径（使用修复版，link名称兼容USD）
DEFAULT_URDF_PATH = r"C:\Users\Rick\Desktop\isaacsim\URDF\N4\urdf\robot_fixed.urdf"


# =====================================================================
# 机器人参数
# =====================================================================

# 几何参数
ROBOT_WHEEL_BASE_X = 0.64      # 前后轮距 (m)
ROBOT_WHEEL_BASE_Y = 0.63      # 左右轮距 (m)
ROBOT_WHEEL_RADIUS = 0.05      # 轮子半径 (m)
ROBOT_BODY_HEIGHT = 0.10       # 底盘高度 (m)

# 控制参数
ROBOT_MAX_SPEED = 3.0          # 最大平移速度 (m/s)
ROBOT_MAX_OMEGA = 5.0          # 最大角速度 (rad/s)
ROBOT_MAX_WHEEL_SPEED = 50.0   # 最大轮子转速 (rad/s)
ROBOT_MAX_STEER_SPEED = 30.0   # 最大舵向转速 (rad/s)

# 关节名称
STEER_JOINTS = ["FL", "FB", "RB", "RL"]       # 舵向关节
WHEEL_JOINTS = ["FLW", "FBW", "RBW", "RLW"]   # 驱动轮关节

# 轮子位置 (相对于底盘中心, 从URDF提取)
WHEEL_POSITIONS = {
    "FL": (-0.32, 0.315),   # 前左
    "FB": (-0.32, -0.315),  # 后左 (URDF中FB实际是后左)
    "RB": (0.32, 0.315),    # 前右 (URDF中RB实际是前右)
    "RL": (0.32, -0.315),   # 后右
}


# =====================================================================
# 舵轮运动学
# =====================================================================

class SwerveKinematics:
    """
    四轮舵轮底盘运动学
    
    将底盘速度指令 (vx, vy, ω) 转换为各轮的舵向角度和轮速
    """
    
    def __init__(self, wheel_positions: dict = None, wheel_radius: float = ROBOT_WHEEL_RADIUS):
        self.wheel_positions = wheel_positions or WHEEL_POSITIONS
        self.wheel_radius = wheel_radius
    
    def inverse_kinematics(self, vx: float, vy: float, omega: float) -> dict:
        """
        逆运动学：底盘速度 -> 各轮状态
        
        Args:
            vx: 底盘X方向速度 (m/s)
            vy: 底盘Y方向速度 (m/s)
            omega: 底盘角速度 (rad/s)
        
        Returns:
            dict: {轮名: (舵向角度rad, 轮速rad/s)}
        """
        result = {}
        
        for wheel_name, (wx, wy) in self.wheel_positions.items():
            # 轮子由于底盘旋转产生的速度分量
            v_rot_x = -omega * wy
            v_rot_y = omega * wx
            
            # 轮子总速度
            wheel_vx = vx + v_rot_x
            wheel_vy = vy + v_rot_y
            
            # 轮速大小
            wheel_speed = math.sqrt(wheel_vx**2 + wheel_vy**2)
            
            # 舵向角度 (轮子朝向)
            if wheel_speed > 1e-4:
                steer_angle = math.atan2(wheel_vy, wheel_vx)
            else:
                steer_angle = 0.0
            
            # 轮子角速度
            wheel_omega = wheel_speed / self.wheel_radius
            
            result[wheel_name] = (steer_angle, wheel_omega)
        
        return result
    
    def optimize_steering(self, current_angles: dict, target_states: dict) -> dict:
        """
        优化舵向：选择最短旋转路径（可反转轮速）
        
        Args:
            current_angles: 当前舵向角度
            target_states: 目标状态 {轮名: (角度, 轮速)}
        
        Returns:
            优化后的状态
        """
        optimized = {}
        
        for wheel_name, (target_angle, wheel_omega) in target_states.items():
            current = current_angles.get(wheel_name, 0.0)
            
            # 计算到目标角度的差值
            diff = target_angle - current
            # 归一化到 [-π, π]
            diff = math.atan2(math.sin(diff), math.cos(diff))
            
            # 如果差值大于90度，反转轮速，舵向取反方向
            if abs(diff) > math.pi / 2:
                # 反转
                target_angle = target_angle + math.pi
                target_angle = math.atan2(math.sin(target_angle), math.cos(target_angle))
                wheel_omega = -wheel_omega
            
            optimized[wheel_name] = (target_angle, wheel_omega)
        
        return optimized


# =====================================================================
# 舵轮机器人类
# =====================================================================

class SwerveRobot:
    """
    舵轮底盘 URDF 机器人
    
    功能:
    - 从 URDF 加载机器人
    - 舵轮运动学控制
    - 速度指令接口
    """
    
    def __init__(self, stage, urdf_path: str = None, 
                 name: str = "SwerveRobot", position: tuple = (0, 0, 0.1)):
        """
        初始化机器人
        
        Args:
            stage: USD stage
            urdf_path: URDF文件路径
            name: 机器人名称
            position: 初始位置 (x, y, z)
        """
        self.stage = stage
        self.urdf_path = urdf_path or DEFAULT_URDF_PATH
        self.name = name
        self.prim_path = f"/World/{name}"
        self.initial_position = position
        
        self.kinematics = SwerveKinematics()
        
        # 关节控制器（延迟初始化）
        self._articulation = None
        self._joint_indices = {}
        
        # 当前舵向角度（用于优化）
        self._current_steer_angles = {j: 0.0 for j in STEER_JOINTS}
    
    def load_urdf(self):
        """
        从 URDF 加载机器人
        
        需要在 Isaac Sim 环境中调用
        使用 omni.kit.commands 方式导入 URDF
        """
        import omni.kit.commands
        from isaacsim.core.utils.extensions import enable_extension
        
        # 启用 URDF 导入扩展
        enable_extension("isaacsim.asset.importer.urdf")
        
        # 导入 URDF 枚举类型
        from isaacsim.asset.importer.urdf import _urdf as urdf_module
        
        # 创建 URDF 导入配置
        status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not status:
            raise RuntimeError("Failed to create URDF import config")
        
        # 配置导入参数
        import_config.merge_fixed_joints = False
        import_config.fix_base = False  # 底盘可移动
        import_config.import_inertia_tensor = True
        import_config.distance_scale = 1.0
        import_config.density = 0.0
        import_config.default_drive_type = urdf_module.UrdfJointTargetType.JOINT_DRIVE_VELOCITY
        import_config.default_drive_strength = 1000.0
        import_config.default_position_drive_damping = 100.0
        import_config.make_default_prim = False
        import_config.create_physics_scene = False  # 不创建新的物理场景
        
        # 导入 URDF (不使用 dest_path，让导入器自动分配路径)
        status, robot_prim_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self.urdf_path,
            import_config=import_config,
            get_articulation_root=True,
        )
        
        if not status or not robot_prim_path:
            raise RuntimeError(f"Failed to import URDF: {self.urdf_path}")
        
        # 更新实际的 prim 路径
        self.prim_path = robot_prim_path
        
        # 设置初始位置
        self._set_initial_pose()
        
        print(f"[SwerveRobot] Loaded URDF from {self.urdf_path}")
        print(f"[SwerveRobot] Robot prim path: {robot_prim_path}")
        
        return robot_prim_path
    
    def _set_initial_pose(self):
        """设置初始位姿"""
        robot_prim = self.stage.GetPrimAtPath(self.prim_path)
        if robot_prim.IsValid():
            xformable = UsdGeom.Xformable(robot_prim)
            # 清除现有变换
            xformable.ClearXformOpOrder()
            # 设置新位置
            xformable.AddTranslateOp().Set(Gf.Vec3d(*self.initial_position))
    
    def initialize_articulation(self):
        """
        初始化关节控制（需在仿真初始化后调用）
        """
        from isaacsim.core.prims import Articulation
        
        self._articulation = Articulation(self.prim_path)
        self._articulation.initialize()
        
        # 获取关节索引
        dof_names = self._articulation.dof_names
        print(f"[SwerveRobot] DOF names: {dof_names}")
        
        for joint_name in STEER_JOINTS + WHEEL_JOINTS:
            if joint_name in dof_names:
                self._joint_indices[joint_name] = dof_names.index(joint_name)
            else:
                print(f"[SwerveRobot] Warning: Joint '{joint_name}' not found")
        
        print(f"[SwerveRobot] Joint indices: {self._joint_indices}")
    
    def get_state(self) -> tuple:
        """
        获取机器人状态
        
        Returns:
            (position, orientation, linear_vel, angular_vel)
        """
        pos, ori = self._articulation.get_world_poses()
        lin_vel = self._articulation.get_linear_velocities()
        ang_vel = self._articulation.get_angular_velocities()
        
        return (
            np.array(pos[0]),
            np.array(ori[0]),
            np.array(lin_vel[0]),
            np.array(ang_vel[0])
        )
    
    def get_position_2d(self) -> np.ndarray:
        """获取2D位置 (x, y)"""
        pos, _, _, _ = self.get_state()
        return pos[:2]
    
    def get_velocity_2d(self) -> np.ndarray:
        """获取2D速度 (vx, vy)"""
        _, _, lin_vel, _ = self.get_state()
        return lin_vel[:2]
    
    def set_velocity_command(self, vx: float, vy: float, omega: float = 0.0):
        """
        设置速度指令（全向移动）
        
        Args:
            vx: X方向速度 (m/s)
            vy: Y方向速度 (m/s)
            omega: 角速度 (rad/s)
        """
        # 限制速度
        speed = math.sqrt(vx**2 + vy**2)
        if speed > ROBOT_MAX_SPEED:
            scale = ROBOT_MAX_SPEED / speed
            vx *= scale
            vy *= scale
        omega = np.clip(omega, -ROBOT_MAX_OMEGA, ROBOT_MAX_OMEGA)
        
        # 逆运动学
        wheel_states = self.kinematics.inverse_kinematics(vx, vy, omega)
        
        # 舵向优化（最短路径）
        wheel_states = self.kinematics.optimize_steering(self._current_steer_angles, wheel_states)
        
        # 设置关节目标
        self._apply_wheel_commands(wheel_states)
    
    def _apply_wheel_commands(self, wheel_states: dict):
        """应用轮子指令"""
        if self._articulation is None:
            return
        
        num_dofs = self._articulation.num_dof
        
        # 获取当前位置/速度目标
        pos_targets = np.zeros(num_dofs)
        vel_targets = np.zeros(num_dofs)
        
        for wheel_name, (steer_angle, wheel_omega) in wheel_states.items():
            # 舵向关节（位置控制）
            steer_joint = wheel_name.replace("W", "") if wheel_name.endswith("W") else wheel_name
            if steer_joint not in STEER_JOINTS:
                steer_joint = wheel_name[:2]  # FL, FB, RB, RL
            
            if steer_joint in self._joint_indices:
                idx = self._joint_indices[steer_joint]
                pos_targets[idx] = steer_angle
                self._current_steer_angles[steer_joint] = steer_angle
            
            # 驱动轮关节（速度控制）
            wheel_joint = wheel_name + "W" if not wheel_name.endswith("W") else wheel_name
            if wheel_joint not in WHEEL_JOINTS:
                wheel_joint = wheel_name[:2] + "W"
            
            if wheel_joint in self._joint_indices:
                idx = self._joint_indices[wheel_joint]
                vel_targets[idx] = wheel_omega
        
        # 应用目标
        self._articulation.set_joint_position_targets(pos_targets)
        self._articulation.set_joint_velocity_targets(vel_targets)
    
    def reset(self, position: tuple = None, orientation: tuple = None):
        """
        重置机器人位姿
        
        Args:
            position: 新位置 (x, y, z)
            orientation: 新朝向四元数 (w, x, y, z)
        """
        if position is None:
            position = self.initial_position
        if orientation is None:
            orientation = (1.0, 0.0, 0.0, 0.0)
        
        self._articulation.set_world_poses(
            positions=np.array([position], dtype=np.float32),
            orientations=np.array([orientation], dtype=np.float32)
        )
        
        # 重置速度
        self._articulation.set_velocities(np.zeros((1, 6), dtype=np.float32))
        
        # 重置关节
        num_dofs = self._articulation.num_dof
        self._articulation.set_joint_positions(np.zeros(num_dofs))
        self._articulation.set_joint_velocities(np.zeros(num_dofs))
        
        # 重置舵向状态
        self._current_steer_angles = {j: 0.0 for j in STEER_JOINTS}
    
    def stop(self):
        """停止机器人"""
        self.set_velocity_command(0.0, 0.0, 0.0)


# =====================================================================
# 简化控制器（直接速度控制，不涉及舵轮运动学）
# =====================================================================

class SimpleVelocityController:
    """
    简化速度控制器
    
    直接对机器人底盘施加力，绕过舵轮运动学
    用于 RL 训练的初步实现
    """
    
    def __init__(self, articulation, max_speed: float = ROBOT_MAX_SPEED):
        self.articulation = articulation
        self.max_speed = max_speed
        self.kp = 200.0  # 比例增益
        self.kd = 50.0   # 阻尼
    
    def set_target_velocity(self, vx: float, vy: float):
        """
        设置目标速度（通过力控制）
        
        Args:
            vx: X方向速度 (m/s), 范围 [-1, 1] 会乘以 max_speed
            vy: Y方向速度 (m/s), 范围 [-1, 1] 会乘以 max_speed
        """
        target_vx = np.clip(vx, -1.0, 1.0) * self.max_speed
        target_vy = np.clip(vy, -1.0, 1.0) * self.max_speed
        
        # 获取当前速度
        lin_vel = self.articulation.get_linear_velocities()[0]
        current_vx, current_vy = lin_vel[0], lin_vel[1]
        
        # PD 控制
        force_x = self.kp * (target_vx - current_vx) - self.kd * current_vx
        force_y = self.kp * (target_vy - current_vy) - self.kd * current_vy
        
        # 施加力到基座
        # 注意：对于 Articulation，需要使用不同的方法
        forces = np.array([[force_x, force_y, 0.0]], dtype=np.float32)
        self.articulation.apply_forces(forces, is_global=True)
