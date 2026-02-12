# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球物理模型模块

包含:
- 排球空气动力学计算 (阻力、马格努斯力、飘球效应)
- 排球刚体创建与状态管理

基于 FIVB 规格和学术风洞数据
"""

import math
import random
import numpy as np

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, PhysxSchema


# =====================================================================
# 排球物理常量 (FIVB 规格 + 学术文献)
# =====================================================================

# --- 排球基本属性 ---
BALL_DIAMETER = 0.210         # 直径 (m) — FIVB标准: 20.5–21.5cm
BALL_RADIUS = BALL_DIAMETER / 2
BALL_MASS = 0.270             # 质量 (kg) — FIVB标准: 260–280g
BALL_CROSS_SECTION = math.pi * BALL_RADIUS**2  # 截面积 = 0.03464 m²
BALL_INERTIA = (2/3) * BALL_MASS * BALL_RADIUS**2  # 薄壳球转动惯量

# --- 空气属性 ---
AIR_DENSITY = 1.225           # 空气密度 (kg/m³) @ 海平面 20°C
AIR_KINEMATIC_VISCOSITY = 1.516e-5  # 运动黏度 (m²/s) @ 20°C

# --- 阻力模型参数 (风洞测试数据) ---
CD_SUBCRITICAL = 0.50         # 亚临界 Cd (Re < 1.0e5, v < ~7 m/s)
CD_SUPERCRITICAL = 0.12       # 超临界 Cd (Re > 2.8e5, v > ~20 m/s)
RE_TRANSITION_START = 1.0e5   # 过渡区起始 (v ≈ 7 m/s)
RE_TRANSITION_MID = 2.0e5     # 阻力危机中点 (v ≈ 14 m/s)
RE_TRANSITION_END = 2.8e5     # 过渡区结束 (v ≈ 20 m/s)

# --- 马格努斯力参数 ---
# 基于学术文献数据 (Watts & Ferrer 1987, Asai et al. 2010)
# 排球典型 Cl ≈ 0.20-0.25, 自旋比 S=0.3 时 Cl ≈ 0.20
CL_SLOPE = 0.35               # Cl 与自旋比的线性系数 (降低自 0.60)
CL_MAX = 0.25                 # Cl 上限 (降低自 0.40)

# --- 飘球效应参数 ---
FLOAT_SPIN_THRESHOLD = 3.0    # 飘球效应的旋转阈值 (rad/s)
FLOAT_AMPLITUDE = 0.15        # 飘球扰动强度系数 (降低自 0.5)
FLOAT_MAX_FORCE_RATIO = 0.25  # 飘球扰动力最大为重力的比例

# --- 旋转衰减 ---
SPIN_DECAY_COEFF = 0.003      # 角速度衰减系数

# --- 碰撞参数 ---
FLOOR_COR = 0.75              # 硬木地板恢复系数
FLOOR_STATIC_FRICTION = 0.60
FLOOR_DYNAMIC_FRICTION = 0.50


# =====================================================================
# 发球类型参数 (基于生物力学研究)
# =====================================================================

SERVE_TYPES = {
    "power_jump": {
        "name": "力量跳发 (Power Jump)",
        "speed_range": (20.0, 28.0),
        "spin_range": (40.0, 90.0),
        "height_range": (3.0, 3.5),
        "angle_range": (-10, 5),
        "spin_type": "topspin",
    },
    "float": {
        "name": "飘球发球 (Float)",
        "speed_range": (12.0, 18.0),
        "spin_range": (0.0, 1.5),
        "height_range": (2.3, 2.8),
        "angle_range": (5, 15),
        "spin_type": "none",
    },
    "overhand": {
        "name": "上手发球 (Overhand)",
        "speed_range": (14.0, 19.0),
        "spin_range": (5.0, 25.0),
        "height_range": (2.3, 2.8),
        "angle_range": (5, 20),
        "spin_type": "topspin",
    },
}


# =====================================================================
# 排球空气动力学模型
# =====================================================================

class VolleyballAerodynamics:
    """
    排球空气动力学计算模型
    
    实现:
    - 动态阻力系数 Cd(Re)（含阻力危机）
    - 马格努斯力（旋转升力）
    - 飘球效应（低旋转随机扰动）
    - 旋转衰减
    """

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self._float_phase_y = self.rng.uniform(0, 2 * math.pi)
        self._float_phase_z = self.rng.uniform(0, 2 * math.pi)
        self._float_freq = self.rng.uniform(2.0, 5.0)

    def reynolds_number(self, speed: float) -> float:
        """计算雷诺数 Re = d * v / ν"""
        return BALL_DIAMETER * speed / AIR_KINEMATIC_VISCOSITY

    def drag_coefficient(self, speed: float) -> float:
        """
        动态阻力系数 Cd(Re)
        排球在临界雷诺数附近急剧下降（阻力危机）
        """
        Re = self.reynolds_number(speed)
        
        if Re < RE_TRANSITION_START:
            return CD_SUBCRITICAL
        elif Re < RE_TRANSITION_MID:
            t = (Re - RE_TRANSITION_START) / (RE_TRANSITION_MID - RE_TRANSITION_START)
            t_smooth = 0.5 * (1 - math.cos(math.pi * t))
            return CD_SUBCRITICAL - (CD_SUBCRITICAL - 0.20) * t_smooth
        elif Re < RE_TRANSITION_END:
            t = (Re - RE_TRANSITION_MID) / (RE_TRANSITION_END - RE_TRANSITION_MID)
            t_smooth = 0.5 * (1 - math.cos(math.pi * t))
            return 0.20 - (0.20 - CD_SUPERCRITICAL) * t_smooth
        else:
            return CD_SUPERCRITICAL

    def drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """计算空气阻力: F = -½ρv²ACd · v̂"""
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)
        
        Cd = self.drag_coefficient(speed)
        F_mag = 0.5 * AIR_DENSITY * speed**2 * BALL_CROSS_SECTION * Cd
        return -F_mag * (velocity / speed)

    def lift_coefficient(self, speed: float, spin_rate: float) -> float:
        """马格努斯升力系数 Cl = 0.60 × S (S = ωr/v)"""
        if speed < 1e-6:
            return 0.0
        spin_ratio = (spin_rate * BALL_RADIUS) / speed
        return min(CL_SLOPE * spin_ratio, CL_MAX)

    def magnus_force(self, velocity: np.ndarray, angular_velocity: np.ndarray) -> np.ndarray:
        """
        计算马格努斯力
        
        基于 Kutta-Joukowski 定理的球体近似:
        F_magnus = (1/2) * ρ * A * Cl * v² * (ω̂ × v̂)
        
        其中:
        - ω̂ 是角速度的单位向量
        - v̂ 是速度的单位向量
        - Cl 是基于自旋比的升力系数
        - 力的方向由 ω̂ × v̂ 确定（垂直于旋转轴和速度）
        
        物理原理:
        - 上旋 (topspin): 绕 -Y 轴旋转 → 产生向下的力 (增强下落)
        - 下旋 (backspin): 绕 +Y 轴旋转 → 产生向上的力 (延长飞行)
        - 侧旋: 绕 Z 轴旋转 → 产生水平侧向力
        """
        speed = np.linalg.norm(velocity)
        spin_rate = np.linalg.norm(angular_velocity)
        
        if speed < 1e-6 or spin_rate < 0.1:
            return np.zeros(3)
        
        # 计算升力系数
        Cl = self.lift_coefficient(speed, spin_rate)
        
        # 角速度单位向量
        omega_hat = angular_velocity / spin_rate
        # 速度单位向量
        v_hat = velocity / speed
        
        # 马格努斯力方向: ω̂ × v̂ (垂直于旋转轴和速度)
        # 这个叉积的大小为 sin(θ)，其中 θ 是 ω 和 v 的夹角
        magnus_direction = np.cross(omega_hat, v_hat)
        direction_mag = np.linalg.norm(magnus_direction)
        
        if direction_mag < 1e-6:
            return np.zeros(3)
        
        # 归一化方向向量
        magnus_unit = magnus_direction / direction_mag
        
        # 力的大小: (1/2) * ρ * A * v² * Cl * sin(θ)
        # sin(θ) = direction_mag (因为 |ω̂ × v̂| = sin(θ))
        sin_theta = direction_mag
        F_magnitude = 0.5 * AIR_DENSITY * BALL_CROSS_SECTION * Cl * speed**2 * sin_theta
        
        return F_magnitude * magnus_unit

    def float_perturbation(self, velocity: np.ndarray, angular_velocity: np.ndarray, 
                           sim_time: float) -> np.ndarray:
        """
        飘球效应：低旋转时的随机气动扰动

        当排球几乎不旋转时（ω < 3 rad/s），气流在球表面的分离点不稳定，
        导致随机的侧向扰动力。这使得飘球发球难以预测和接住。

        约束: 扰动力的大小被限制在重力的一定比例内，防止产生不合理的运动。
        """
        speed = np.linalg.norm(velocity)
        spin_rate = np.linalg.norm(angular_velocity)

        if spin_rate > FLOAT_SPIN_THRESHOLD or speed < 3.0:
            return np.zeros(3)

        # 基础强度计算
        intensity = FLOAT_AMPLITUDE * speed * (1.0 - spin_rate / FLOAT_SPIN_THRESHOLD)

        # 在阻力危机区域（临界雷诺数附近）扰动更强
        Re = self.reynolds_number(speed)
        if RE_TRANSITION_START < Re < RE_TRANSITION_END:
            intensity *= 1.3  # 降低自 1.5
        else:
            intensity *= 0.6  # 降低自 0.7

        # 施加强度上限，防止扰动力过大
        # 最大扰动力 = 重力 * FLOAT_MAX_FORCE_RATIO
        max_intensity = BALL_MASS * 9.81 * FLOAT_MAX_FORCE_RATIO
        intensity = min(intensity, max_intensity)

        # 低频正弦波 + 高斯噪声，模拟气流不稳定性
        fy = (intensity * math.sin(self._float_freq * sim_time + self._float_phase_y)
              + self.rng.gauss(0, intensity * 0.2))  # 降低高斯噪声方差
        fz = (intensity * 0.5 * math.sin(self._float_freq * 0.7 * sim_time + self._float_phase_z)
              + self.rng.gauss(0, intensity * 0.15))  # 降低垂直扰动
        fx = self.rng.gauss(0, intensity * 0.1)  # 降低纵向扰动

        return np.array([fx, fy, fz])

    def total_force(self, velocity: np.ndarray, angular_velocity: np.ndarray, 
                    sim_time: float) -> np.ndarray:
        """
        计算所有空气动力学力的总和

        包含物理合理性约束，防止产生违反物理的运动（如向上加速超过重力）。
        """
        F_d = self.drag_force(velocity)
        F_m = self.magnus_force(velocity, angular_velocity)
        F_f = self.float_perturbation(velocity, angular_velocity, sim_time)

        # 物理合理性约束:
        # 马格努斯力和飘球力的向上分量之和不应超过重力的 80%
        # 这确保球不会因空气动力学效应而向上加速到超过合理水平
        gravity_magnitude = BALL_MASS * 9.81
        upward_aero_force = F_m[2] + F_f[2]

        max_upward_force = 0.8 * gravity_magnitude
        if upward_aero_force > max_upward_force:
            # 按比例缩放向上分量
            scale = max_upward_force / upward_aero_force
            F_m_scaled = F_m.copy()
            F_f_scaled = F_f.copy()
            F_m_scaled[2] *= scale
            F_f_scaled[2] *= scale
            total = F_d + F_m_scaled + F_f_scaled
        else:
            total = F_d + F_m + F_f

        return total


# =====================================================================
# 排球刚体封装
# =====================================================================

class VolleyballRigidBody:
    """
    排球刚体管理类
    
    封装排球的创建、状态获取/设置、力的施加
    """
    
    def __init__(self, stage, name: str = "Volleyball", position: tuple = (0, 0, 2)):
        self.stage = stage
        self.name = name
        self.prim_path = f"/World/{name}"
        
        self._create_ball(position)
        self._rigid_prim = None  # 延迟初始化
        
        # 空气动力学模型
        self.aero = VolleyballAerodynamics()
    
    def _create_material(self, path: str, color: tuple, roughness=0.5, metallic=0.0):
        """创建视觉材质"""
        material = UsdShade.Material.Define(self.stage, Sdf.Path(path))
        shader = UsdShade.Shader.Define(self.stage, Sdf.Path(f"{path}/Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material
    
    def _create_physics_material(self, path: str, static_f, dynamic_f, restitution):
        """创建物理材质"""
        material = UsdShade.Material.Define(self.stage, Sdf.Path(path))
        physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_mat.CreateStaticFrictionAttr(static_f)
        physics_mat.CreateDynamicFrictionAttr(dynamic_f)
        physics_mat.CreateRestitutionAttr(restitution)
        return material
    
    def _create_ball(self, position: tuple):
        """创建排球几何体和物理属性"""
        ball = UsdGeom.Sphere.Define(self.stage, Sdf.Path(self.prim_path))
        ball.CreateRadiusAttr(BALL_RADIUS)
        
        xform = UsdGeom.Xformable(ball)
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        
        # 视觉材质 (黄色排球)
        ball_mat = self._create_material(
            f"/Materials/{self.name}Mat", (0.95, 0.85, 0.30), roughness=0.65
        )
        UsdShade.MaterialBindingAPI(ball).Bind(ball_mat)
        
        # 碰撞
        UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
        
        # 刚体
        UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
        
        # 质量
        mass_api = UsdPhysics.MassAPI.Apply(ball.GetPrim())
        mass_api.CreateMassAttr(BALL_MASS)
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(BALL_INERTIA, BALL_INERTIA, BALL_INERTIA))
        
        # 物理材质
        ball_phys_mat = self._create_physics_material(
            f"/Materials/{self.name}PhysMat",
            FLOOR_STATIC_FRICTION, FLOOR_DYNAMIC_FRICTION, FLOOR_COR
        )
        phys_binding = UsdShade.MaterialBindingAPI.Apply(ball.GetPrim())
        phys_binding.Bind(ball_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        
        # PhysX 扩展
        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(ball.GetPrim())
        physx_rb.CreateLinearDampingAttr(0.3)
        physx_rb.CreateAngularDampingAttr(0.05)
        physx_rb.CreateEnableCCDAttr(True)
    
    def initialize_rigid_prim(self):
        """初始化 RigidPrim（需在仿真初始化后调用）"""
        from isaacsim.core.prims import RigidPrim
        self._rigid_prim = RigidPrim(prim_paths_expr=self.prim_path, name=f"{self.name}_view")
    
    def get_state(self) -> tuple:
        """获取球状态: (position, linear_vel, angular_vel)"""
        positions, _ = self._rigid_prim.get_world_poses()
        linear_vels = self._rigid_prim.get_linear_velocities()
        angular_vels = self._rigid_prim.get_angular_velocities()
        return (
            np.array(positions[0]),
            np.array(linear_vels[0]),
            np.array(angular_vels[0])
        )
    
    def set_state(self, position: tuple, linear_vel: tuple, angular_vel: tuple = (0, 0, 0)):
        """设置球状态"""
        self._rigid_prim.set_world_poses(
            positions=np.array([position], dtype=np.float32),
            orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        vel_6 = np.zeros((1, 6), dtype=np.float32)
        vel_6[0, 0:3] = linear_vel
        vel_6[0, 3:6] = angular_vel
        self._rigid_prim.set_velocities(vel_6)
    
    def apply_force(self, force: np.ndarray):
        """施加外力"""
        self._rigid_prim.apply_forces(np.array([force], dtype=np.float32), is_global=True)
    
    def apply_aerodynamic_forces(self, sim_time: float):
        """计算并施加空气动力学力"""
        pos, vel, ang_vel = self.get_state()
        speed = np.linalg.norm(vel)
        
        if speed > 0.5:
            force = self.aero.total_force(vel, ang_vel, sim_time)
            self.apply_force(force)
    
    def reset_aerodynamics(self, seed=None):
        """重置空气动力学模型（新的飘球相位）"""
        self.aero = VolleyballAerodynamics(seed)


# =====================================================================
# 发球生成器
# =====================================================================

def generate_serve(serve_type: str = "float", from_x: float = -8.0, 
                   court_width: float = 9.0) -> tuple:
    """
    生成发球参数
    
    Returns:
        (position, linear_velocity, angular_velocity, info_string)
    """
    params = SERVE_TYPES.get(serve_type, SERVE_TYPES["float"])
    
    # 位置
    y_pos = random.uniform(-court_width / 4, court_width / 4)
    z_pos = random.uniform(*params["height_range"])
    position = (from_x, y_pos, z_pos)
    
    # 速度
    speed = random.uniform(*params["speed_range"])
    angle_deg = random.uniform(*params["angle_range"])
    angle = math.radians(angle_deg)
    
    vx = speed * math.cos(angle)  # 朝正X方向
    vy = random.uniform(-1.5, 1.5)
    vz = speed * math.sin(angle)
    linear_vel = (vx, vy, vz)
    
    # 旋转
    spin_rate = random.uniform(*params["spin_range"])
    
    if params["spin_type"] == "topspin":
        wx, wy, wz = 0, -spin_rate, random.uniform(-2, 2)
    elif params["spin_type"] == "none":
        wx = random.uniform(-0.5, 0.5)
        wy = random.uniform(-0.5, 0.5)
        wz = random.uniform(-0.5, 0.5)
    else:
        wx = random.uniform(-spin_rate, spin_rate)
        wy = random.uniform(-spin_rate, spin_rate)
        wz = random.uniform(-spin_rate * 0.3, spin_rate * 0.3)
    
    angular_vel = (wx, wy, wz)
    
    info = (f"{params['name']} | {speed:.1f}m/s ({speed*3.6:.0f}km/h) "
            f"| θ={angle_deg:+.1f}° | ω={spin_rate:.1f}rad/s")
    
    return position, linear_vel, angular_vel, info


def generate_fixed_serve(
    speed: float = 15.0,
    angle_deg: float = 20.0,
    y_offset: float = 0.0,
    from_x: float = -10.0,
    height: float = 2.5,
    spin_rate: float = 0.0
) -> tuple:
    """
    生成固定参数发球
    
    Args:
        speed: 发球速度 (m/s)
        angle_deg: 发球仰角 (度)
        y_offset: Y方向偏移 (m)
        from_x: 发球X位置 (m)
        height: 发球高度 (m)
        spin_rate: 旋转速率 (rad/s), 0=无旋转
    
    Returns:
        (position, linear_velocity, angular_velocity, info_string)
    """
    position = (from_x, y_offset, height)
    
    angle = math.radians(angle_deg)
    vx = speed * math.cos(angle)
    vy = 0.0
    vz = speed * math.sin(angle)
    linear_vel = (vx, vy, vz)
    
    # 无旋转或上旋
    if spin_rate > 0:
        angular_vel = (0, -spin_rate, 0)  # 上旋
    else:
        angular_vel = (0, 0, 0)
    
    info = (f"固定发球 | {speed:.1f}m/s ({speed*3.6:.0f}km/h) "
            f"| θ={angle_deg:+.1f}° | ω={spin_rate:.1f}rad/s")
    
    return position, linear_vel, angular_vel, info


def predict_landing_point(ball_pos: np.ndarray, ball_vel: np.ndarray, 
                          target_height: float = 0.1, max_iter: int = 300) -> tuple:
    """
    预测排球在目标高度处的落点和到达时间
    使用欧拉积分 + 简化空气阻力
    
    Returns:
        (x, y, time) 或 None
    """
    CD_AVG = 0.35
    dt = 0.01
    pos = np.array(ball_pos, dtype=float)
    vel = np.array(ball_vel, dtype=float)
    g = np.array([0, 0, -9.81])
    
    t = 0.0
    for _ in range(max_iter):
        speed = np.linalg.norm(vel)
        if speed > 0.1:
            drag_mag = 0.5 * AIR_DENSITY * speed**2 * BALL_CROSS_SECTION * CD_AVG
            drag = -drag_mag * (vel / speed) / BALL_MASS
        else:
            drag = np.zeros(3)
        
        acc = g + drag
        vel = vel + acc * dt
        pos = pos + vel * dt
        t += dt
        
        if pos[2] <= target_height:
            return pos[0], pos[1], t
    
    return None
