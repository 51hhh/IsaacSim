# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球物理模拟 — RL 训练环境

设计目标：
  - 真实的空中排球物理（阻力、马格努斯力、飘球效应）
  - GPU dynamics 支持（多 agent 并行训练）
  - 快速重置（落地/碰网/出界 → 立即重置发球）

物理模型:
  1. 空气阻力（动态 Cd，含阻力危机 Drag Crisis）
  2. 马格努斯力（旋转产生的侧向/升降力）
  3. 飘球效应（低旋转时的随机气动扰动）
  4. 重力 + PhysX 碰撞（地面）

参数来源: 见 VOLLEYBALL_PHYSICS_MODEL.md

运行方法：
    C:\\IsaacSim\\python.bat volleyball_simple.py
"""

from isaacsim import SimulationApp

# 启动仿真应用
simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import math
import random
import time

import numpy as np
import omni.usd
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Sdf,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdShade,
)


# =====================================================================
# 物理常量 & 排球参数（基于 FIVB 规格 + 学术文献）
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

# --- 阻力模型参数 ---
# 基于风洞测试数据: Cd 随雷诺数 Re 变化
# Re = d * v / ν
CD_SUBCRITICAL = 0.50         # 亚临界 Cd (Re < 1.0e5, v < ~7 m/s)
CD_SUPERCRITICAL = 0.12       # 超临界 Cd (Re > 2.8e5, v > ~20 m/s)
RE_TRANSITION_START = 1.0e5   # 过渡区起始 (v ≈ 7 m/s)
RE_TRANSITION_MID = 2.0e5     # 阻力危机中点 (v ≈ 14 m/s)
RE_TRANSITION_END = 2.8e5     # 过渡区结束 (v ≈ 20 m/s)

# --- 马格努斯力参数 ---
CL_SLOPE = 0.60               # Cl 与自旋比的线性系数
CL_MAX = 0.40                 # Cl 上限

# --- 飘球效应参数 ---
FLOAT_SPIN_THRESHOLD = 3.0    # 飘球效应的旋转阈值 (rad/s)
FLOAT_AMPLITUDE = 0.5         # 飘球扰动强度系数

# --- 旋转衰减 ---
SPIN_DECAY_COEFF = 0.003      # 角速度衰减系数

# --- 碰撞参数（地板） ---
FLOOR_COR = 0.75              # 硬木地板恢复系数
FLOOR_STATIC_FRICTION = 0.60
FLOOR_DYNAMIC_FRICTION = 0.50
NET_COR = 0.20                # 球网恢复系数
NET_FRICTION = 0.75

# --- 排球场尺寸 ---
COURT_LENGTH = 18.0           # 全场长度 (m)
COURT_WIDTH = 9.0             # 场地宽度 (m)
NET_HEIGHT = 2.43             # 男子球网高度 (m)
NET_THICKNESS = 0.05

# --- RL 重置阈值 ---
# GROUND_RESET_Z = BALL_RADIUS + 0.02  # ★ 取消落地重置
NET_HALF_THICKNESS = 0.15              # 碰网几何检测半厚
OOB_X = COURT_LENGTH * 1.2             # X 出界阈值
OOB_Y = COURT_WIDTH * 1.5              # Y 出界阈值
OOB_Z_HIGH = 15.0                      # Z 过高阈值

# --- 发球参数（基于生物力学研究数据） ---
SERVE_TYPES = {
    "power_jump": {   # 力量跳发
        "name": "力量跳发 (Power Jump)",
        "speed_range": (20.0, 28.0),    # m/s (73-100 km/h)
        "spin_range": (40.0, 90.0),     # rad/s (6-14 rev/s)
        "height_range": (3.0, 3.5),     # m
        "angle_range": (-10, 5),        # 度（上旋所以角度较平甚至下压）
        "spin_type": "topspin",         # 上旋
    },
    "float": {        # 飘球发球
        "name": "飘球发球 (Float)",
        "speed_range": (12.0, 18.0),    # m/s (43-65 km/h)
        "spin_range": (0.0, 1.5),       # rad/s (几乎不旋转)
        "height_range": (2.3, 2.8),     # m
        "angle_range": (5, 15),         # 度
        "spin_type": "none",            # 无旋转
    },
    "overhand": {     # 普通上手发球
        "name": "上手发球 (Overhand)",
        "speed_range": (14.0, 19.0),    # m/s (50-68 km/h)
        "spin_range": (5.0, 25.0),      # rad/s
        "height_range": (2.3, 2.8),     # m
        "angle_range": (5, 20),         # 度
        "spin_type": "topspin",
    },
}

# --- 仿真参数 ---
PHYSICS_DT = 1.0 / 240.0     # 240Hz 物理（更精确的气动力计算）
RENDER_DT = 1.0 / 60.0       # 60Hz 渲染


# =====================================================================
# 排球空气动力学模型
# =====================================================================

class VolleyballAerodynamics:
    """
    排球空气动力学计算模型
    
    基于学术研究数据实现：
    - 动态阻力系数（含阻力危机）
    - 马格努斯力（旋转升力）
    - 飘球效应（低旋转随机扰动）
    - 旋转衰减
    """

    def __init__(self):
        self.rng = random.Random(42)
        # 飘球扰动的 Perlin-like 低频噪声状态
        self._float_phase_y = self.rng.uniform(0, 2 * math.pi)
        self._float_phase_z = self.rng.uniform(0, 2 * math.pi)
        self._float_freq = self.rng.uniform(2.0, 5.0)  # Hz

    def reynolds_number(self, speed):
        """计算雷诺数 Re = d * v / ν"""
        return BALL_DIAMETER * speed / AIR_KINEMATIC_VISCOSITY

    def drag_coefficient(self, speed):
        """
        动态阻力系数 Cd(Re)
        
        排球的阻力系数在临界雷诺数附近急剧下降（阻力危机）。
        这对飘球的飞行轨迹有决定性影响。
        """
        Re = self.reynolds_number(speed)
        
        if Re < RE_TRANSITION_START:
            # 亚临界：层流边界层，高阻力
            return CD_SUBCRITICAL
        elif Re < RE_TRANSITION_MID:
            # 过渡区前段：Cd 开始下降
            t = (Re - RE_TRANSITION_START) / (RE_TRANSITION_MID - RE_TRANSITION_START)
            # 使用平滑 S 曲线过渡（而非线性）
            t_smooth = 0.5 * (1 - math.cos(math.pi * t))
            return CD_SUBCRITICAL - (CD_SUBCRITICAL - 0.20) * t_smooth
        elif Re < RE_TRANSITION_END:
            # 过渡区后段：继续下降到超临界值
            t = (Re - RE_TRANSITION_MID) / (RE_TRANSITION_END - RE_TRANSITION_MID)
            t_smooth = 0.5 * (1 - math.cos(math.pi * t))
            return 0.20 - (0.20 - CD_SUPERCRITICAL) * t_smooth
        else:
            # 超临界：湍流边界层，低阻力
            return CD_SUPERCRITICAL

    def drag_force(self, velocity):
        """
        计算空气阻力向量
        
        F_drag = -½ρv²ACd · v̂
        方向与速度方向相反
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)
        
        Cd = self.drag_coefficient(speed)
        F_mag = 0.5 * AIR_DENSITY * speed**2 * BALL_CROSS_SECTION * Cd
        # 方向与速度方向相反
        F_drag = -F_mag * (velocity / speed)
        return F_drag

    def lift_coefficient(self, speed, spin_rate):
        """
        马格努斯升力系数 Cl(S)
        
        S = (ω × r) / v (自旋比)
        Cl ≈ 0.60 × S (线性近似，S < 0.7 时精度较好)
        """
        if speed < 1e-6:
            return 0.0
        spin_ratio = (spin_rate * BALL_RADIUS) / speed
        Cl = CL_SLOPE * spin_ratio
        return min(Cl, CL_MAX)

    def magnus_force(self, velocity, angular_velocity):
        """
        计算马格努斯力向量
        
        F_magnus = ½ρv²ACl · (ω̂ × v̂)
        力的方向垂直于旋转轴和速度方向
        """
        speed = np.linalg.norm(velocity)
        spin_rate = np.linalg.norm(angular_velocity)
        
        if speed < 1e-6 or spin_rate < 0.1:
            return np.zeros(3)
        
        Cl = self.lift_coefficient(speed, spin_rate)
        F_mag = 0.5 * AIR_DENSITY * speed**2 * BALL_CROSS_SECTION * Cl
        
        # 力的方向: ω × v（叉乘）
        omega_cross_v = np.cross(angular_velocity, velocity)
        cross_mag = np.linalg.norm(omega_cross_v)
        
        if cross_mag < 1e-6:
            return np.zeros(3)
        
        direction = omega_cross_v / cross_mag
        return F_mag * direction

    def float_perturbation(self, velocity, angular_velocity, sim_time):
        """
        飘球效应：低旋转时的随机气动扰动
        
        当排球几乎不旋转时，气流分离点不稳定，
        导致球的轨迹产生不可预测的偏移（类似棒球Knuckleball）。
        
        使用低频正弦波 + 随机扰动模拟（比纯白噪声更真实）。
        """
        speed = np.linalg.norm(velocity)
        spin_rate = np.linalg.norm(angular_velocity)
        
        if spin_rate > FLOAT_SPIN_THRESHOLD or speed < 3.0:
            return np.zeros(3)
        
        # 扰动强度: 随旋转增加而减弱，随速度增加而增强
        intensity = FLOAT_AMPLITUDE * speed * (1.0 - spin_rate / FLOAT_SPIN_THRESHOLD)
        # 阻力危机区域内扰动更强
        Re = self.reynolds_number(speed)
        if RE_TRANSITION_START < Re < RE_TRANSITION_END:
            crisis_factor = 1.5  # 阻力危机区扰动加强
        else:
            crisis_factor = 0.7
        
        intensity *= crisis_factor
        
        # 低频正弦（模拟大尺度涡脱落）+ 高频随机（模拟小尺度湍流）
        fy = (
            intensity * math.sin(self._float_freq * sim_time + self._float_phase_y)
            + self.rng.gauss(0, intensity * 0.3)
        )
        fz = (
            intensity * 0.6 * math.sin(
                self._float_freq * 0.7 * sim_time + self._float_phase_z
            )
            + self.rng.gauss(0, intensity * 0.2)
        )
        fx = self.rng.gauss(0, intensity * 0.15)
        
        return np.array([fx, fy, fz])

    def spin_decay(self, angular_velocity, velocity, dt):
        """
        旋转衰减：空气对旋转球的阻力矩
        
        dω/dt = -C_decay · |v| · ω
        """
        speed = np.linalg.norm(velocity)
        decay_rate = SPIN_DECAY_COEFF * speed
        return angular_velocity * math.exp(-decay_rate * dt)

    def total_aerodynamic_force(self, velocity, angular_velocity, sim_time):
        """
        计算所有空气动力学力的总和
        
        F_total = F_drag + F_magnus + F_float
        """
        F_d = self.drag_force(velocity)
        F_m = self.magnus_force(velocity, angular_velocity)
        F_f = self.float_perturbation(velocity, angular_velocity, sim_time)
        return F_d + F_m + F_f


# =====================================================================
# 场景创建辅助函数
# =====================================================================

def create_material(stage, path, color, roughness=0.5, metallic=0.0):
    """创建带颜色的 USD 材质"""
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_physics_material(stage, path, static_friction, dynamic_friction, restitution):
    """
    创建物理材质
    ★ 必须用 UsdShade.Material.Define() 而非 stage.DefinePrim()
      否则 MaterialBindingAPI.Bind() 不会被 PhysX 识别
    """
    material = UsdShade.Material.Define(stage, Sdf.Path(path))
    physics_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_mat.CreateStaticFrictionAttr(static_friction)
    physics_mat.CreateDynamicFrictionAttr(dynamic_friction)
    physics_mat.CreateRestitutionAttr(restitution)
    return material  # 返回 UsdShade.Material 对象


def setup_physics_scene(stage):
    """
    设置物理场景
    ★ 启用 GPU dynamics 以支持多 agent 并行 RL 训练
    """
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)

    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene"))
    physx_scene_api = PhysxSchema.PhysxSceneAPI.Get(stage, "/physicsScene")
    physx_scene_api.CreateEnableCCDAttr(True)
    physx_scene_api.CreateEnableStabilizationAttr(True)
    # ★ GPU dynamics：支持多环境并行计算
    physx_scene_api.CreateEnableGPUDynamicsAttr(True)
    physx_scene_api.CreateSolverTypeAttr("TGS")


def setup_environment(stage):
    """设置灯光环境"""
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distant_light.CreateIntensityAttr(800)
    distant_light.CreateAngleAttr(1.0)
    xform = UsdGeom.Xformable(distant_light)
    xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/DomeLight"))
    dome_light.CreateIntensityAttr(200)


def create_court(stage):
    """
    创建排球场（地面仅用于安全碰撞兜底，RL 依靠几何检测重置）
    """
    # 地面（PhysX 碰撞兜底）
    PhysicsSchemaTools.addGroundPlane(
        stage, "/groundPlane", "Z", 1500, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.15, 0.15, 0.15)
    )
    # ★ 恢复地面物理材质（否则落地无摩擦、无弹跳）
    ground_phys_mat = create_physics_material(
        stage, "/Materials/GroundPhysMat",
        FLOOR_STATIC_FRICTION, FLOOR_DYNAMIC_FRICTION, FLOOR_COR
    )
    ground_geom = stage.GetPrimAtPath("/groundPlane/geom")
    if ground_geom.IsValid():
        phys_binding = UsdShade.MaterialBindingAPI.Apply(ground_geom)
        phys_binding.Bind(
            ground_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    # 球场地面（纯视觉，无碰撞——RL 用几何检测落地）
    court = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Court"))
    court.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(court)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.005))
    xf.AddScaleOp().Set(Gf.Vec3f(COURT_LENGTH, COURT_WIDTH, 0.01))
    court_mat = create_material(stage, "/Materials/CourtMat", (0.82, 0.68, 0.46), roughness=0.7)
    UsdShade.MaterialBindingAPI(court).Bind(court_mat)
    # 绑定物理材质到球场视觉层（因为球场覆盖了地面）
    UsdPhysics.CollisionAPI.Apply(court.GetPrim())
    phys_binding = UsdShade.MaterialBindingAPI.Apply(court.GetPrim())
    phys_binding.Bind(
        ground_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    # 球场边线 + 中线（纯视觉）
    line_mat = create_material(stage, "/Materials/LineMat", (1.0, 1.0, 1.0), roughness=0.9)
    lw = 0.05
    lines = [
        ("LineN", (0, COURT_WIDTH/2, 0.015), (COURT_LENGTH, lw, 0.005)),
        ("LineS", (0, -COURT_WIDTH/2, 0.015), (COURT_LENGTH, lw, 0.005)),
        ("LineE", (COURT_LENGTH/2, 0, 0.015), (lw, COURT_WIDTH, 0.005)),
        ("LineW", (-COURT_LENGTH/2, 0, 0.015), (lw, COURT_WIDTH, 0.005)),
        ("LineMid", (0, 0, 0.015), (lw, COURT_WIDTH, 0.005)),
    ]
    for name, pos, scale in lines:
        line = UsdGeom.Cube.Define(stage, Sdf.Path(f"/World/{name}"))
        line.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(line)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdShade.MaterialBindingAPI(line).Bind(line_mat)


def create_net(stage):
    """
    创建球网（纯视觉，无 PhysX 碰撞）
    RL 使用几何检测碰网，不需要物理碰撞
    """
    # 球网主体（纯视觉参考）
    net = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Net/Mesh"))
    net.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(net)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, NET_HEIGHT / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(NET_THICKNESS, COURT_WIDTH + 1.0, NET_HEIGHT))
    net_mat = create_material(stage, "/Materials/NetMat", (0.15, 0.15, 0.15), roughness=0.9)
    UsdShade.MaterialBindingAPI(net).Bind(net_mat)

    # 顶带
    top = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Net/TopBand"))
    top.CreateSizeAttr(1.0)
    xf2 = UsdGeom.Xformable(top)
    xf2.AddTranslateOp().Set(Gf.Vec3d(0, 0, NET_HEIGHT + 0.035))
    xf2.AddScaleOp().Set(Gf.Vec3f(NET_THICKNESS + 0.02, COURT_WIDTH + 1.0, 0.07))
    top_mat = create_material(stage, "/Materials/TopBandMat", (0.95, 0.95, 0.95), roughness=0.8)
    UsdShade.MaterialBindingAPI(top).Bind(top_mat)

    # 柱子
    pole_mat = create_material(stage, "/Materials/PoleMat", (0.6, 0.6, 0.65), roughness=0.3, metallic=0.8)
    pole_h = NET_HEIGHT + 0.3
    for i, y in enumerate([-(COURT_WIDTH/2 + 0.5), (COURT_WIDTH/2 + 0.5)]):
        pole = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"/World/Net/Pole{i}"))
        pole.CreateRadiusAttr(0.04)
        pole.CreateHeightAttr(pole_h)
        pole.CreateAxisAttr("Z")
        xf = UsdGeom.Xformable(pole)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, y, pole_h/2))
        UsdShade.MaterialBindingAPI(pole).Bind(pole_mat)


def create_volleyball(stage, name="Volleyball", position=(0, 0, 1)):
    """
    创建排球刚体（RL 训练用）
    无阻尼—空中物理由自定义气动力模型 + PhysX 重力/碰撞处理
    """
    ball_path = f"/World/{name}"
    ball = UsdGeom.Sphere.Define(stage, Sdf.Path(ball_path))
    ball.CreateRadiusAttr(BALL_RADIUS)

    xform = UsdGeom.Xformable(ball)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    # 视觉材质
    ball_mat = create_material(stage, f"/Materials/{name}Mat", (0.95, 0.85, 0.30), roughness=0.65)
    UsdShade.MaterialBindingAPI(ball).Bind(ball_mat)

    # 碰撞（兜底：地面碰撞防止穿透）
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())

    # 刚体
    UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())

    # 质量与转动惯量
    mass_api = UsdPhysics.MassAPI.Apply(ball.GetPrim())
    mass_api.CreateMassAttr(BALL_MASS)
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(BALL_INERTIA, BALL_INERTIA, BALL_INERTIA))

    # 物理材质
    ball_phys_mat = create_physics_material(
        stage, f"/Materials/{name}PhysMat",
        FLOOR_STATIC_FRICTION, FLOOR_DYNAMIC_FRICTION, FLOOR_COR
    )
    phys_binding = UsdShade.MaterialBindingAPI.Apply(ball.GetPrim())
    phys_binding.Bind(
        ball_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )

    # PhysX 刚体扩展
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(ball.GetPrim())
    physx_rb.CreateLinearDampingAttr(0.3)   # ★虽然是RL，但为了落地后不永远滑行，仍需阻力
    physx_rb.CreateAngularDampingAttr(0.05)
    physx_rb.CreateEnableCCDAttr(True)

    return ball_path


# =====================================================================
# 发球生成器
# =====================================================================

def generate_serve(serve_type="float", from_side="left"):
    """
    生成发球参数（基于真实运动学数据）

    Returns:
        (position, linear_velocity, angular_velocity, serve_info_string)
    """
    params = SERVE_TYPES[serve_type]

    # --- 位置 ---
    x_sign = -1 if from_side == "left" else 1
    x_pos = x_sign * (COURT_LENGTH / 2 - 0.5)
    y_pos = random.uniform(-COURT_WIDTH / 4, COURT_WIDTH / 4)
    z_pos = random.uniform(*params["height_range"])
    position = (x_pos, y_pos, z_pos)

    # --- 速度 ---
    speed = random.uniform(*params["speed_range"])
    angle_deg = random.uniform(*params["angle_range"])
    angle = math.radians(angle_deg)

    vx = -x_sign * speed * math.cos(angle)
    vy = random.uniform(-1.5, 1.5)  # 轻微侧向
    vz = speed * math.sin(angle)
    linear_vel = (vx, vy, vz)

    # --- 旋转 ---
    spin_rate = random.uniform(*params["spin_range"])

    if params["spin_type"] == "topspin":
        # 上旋: 绕Y轴（使球加速下坠）
        wx = 0
        wy = -x_sign * spin_rate  # 上旋方向
        wz = random.uniform(-2, 2)
    elif params["spin_type"] == "none":
        # 飘球: 几乎无旋转
        wx = random.uniform(-0.5, 0.5)
        wy = random.uniform(-0.5, 0.5)
        wz = random.uniform(-0.5, 0.5)
    else:
        wx = random.uniform(-spin_rate, spin_rate)
        wy = random.uniform(-spin_rate, spin_rate)
        wz = random.uniform(-spin_rate * 0.3, spin_rate * 0.3)

    angular_vel = (wx, wy, wz)

    info = (
        f"{params['name']} | 速度={speed:.1f}m/s ({speed*3.6:.0f}km/h) "
        f"| 角度={angle_deg:+.1f}° | 旋转={spin_rate:.1f}rad/s"
    )

    return position, linear_vel, angular_vel, info


# =====================================================================
# RigidPrim 辅助函数 — 通过 Isaac Sim 封装类操作刚体
# =====================================================================
# ★★★ 核心设计变更 ★★★
# 之前：每帧通过 set_velocities 覆盖速度 → 破坏了 PhysX 碰撞/弹跳
# 现在：通过 apply_forces 施加力 → PhysX 自己处理碰撞+重力+我们的力

def get_ball_state(ball_rigid: RigidPrim):
    """通过 RigidPrim 获取球的状态（解包批量维度）"""
    positions, orientations = ball_rigid.get_world_poses()
    linear_vels = ball_rigid.get_linear_velocities()
    angular_vels = ball_rigid.get_angular_velocities()
    return np.array(positions[0]), np.array(linear_vels[0]), np.array(angular_vels[0])


def apply_aero_force(ball_rigid: RigidPrim, force):
    """
    通过 RigidPrim.apply_forces() 施加空气动力学力
    
    ★ 关键改进：不再覆盖速度！
    apply_forces 只是给 PhysX 添加一个外力，
    PhysX 在下一步 step() 时会把：
      重力 + 碰撞力 + 我们的外力 = 合力
    统一计算加速度和速度变化。
    
    这样碰撞弹跳不会被破坏。
    """
    # apply_forces 需要 (N, 3) 形状
    ball_rigid.apply_forces(np.array([force], dtype=np.float32), is_global=True)


def reset_ball(ball_rigid: RigidPrim, position, linear_vel, angular_vel):
    """
    重置球的位置和速度（使用 RigidPrim 封装类）
    在仿真运行中也能正确工作
    """
    # 位置 (1,3)，姿态 (1,4)
    ball_rigid.set_world_poses(
        positions=np.array([position], dtype=np.float32),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    # 速度 (1,6) = [linear(3), angular(3)]
    vel_6 = np.zeros((1, 6), dtype=np.float32)
    vel_6[0, 0:3] = linear_vel
    vel_6[0, 3:6] = angular_vel
    ball_rigid.set_velocities(vel_6)


# =====================================================================
# 主函数
# =====================================================================

# =====================================================================
# RL 重置检测
# =====================================================================

def check_episode_end(ball_pos, step_count, max_steps):
    """
    检测 episode 是否结束
    ★ 落地不重置，只有违规（碰网/出界）或超时才重置
    """
    x, y, z = float(ball_pos[0]), float(ball_pos[1]), float(ball_pos[2])

    # 1. 落地检测（仅用于统计，不重置）
    is_grounded = z < BALL_RADIUS + 0.05

    # 2. 碰网（几何包围盒检测）
    if (abs(x) < NET_HALF_THICKNESS + BALL_RADIUS
            and abs(y) < (COURT_WIDTH / 2 + 1.0)
            and z < NET_HEIGHT + BALL_RADIUS):
        return True, "net"

    # 3. 出界
    if abs(x) > OOB_X or abs(y) > OOB_Y or z > OOB_Z_HIGH:
        return True, "oob"

    # 4. 超时
    if step_count > max_steps:
        return True, "timeout"

    return False, None


# =====================================================================
# 主函数
# =====================================================================

def main():
    stage = omni.usd.get_context().get_stage()

    # 场景搭建
    setup_physics_scene(stage)
    setup_environment(stage)
    create_court(stage)
    create_net(stage)
    ball_path = create_volleyball(stage, "Volleyball", (-7, 0, 2.5))

    simulation_app.update()

    sim_context = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT)

    set_camera_view(
        eye=[0.0, -18.0, 10.0],
        target=[0.0, 0.0, 1.5],
        camera_prim_path="/OmniverseKit_Persp"
    )

    sim_context.initialize_physics()
    ball_rigid = RigidPrim(prim_paths_expr=ball_path, name="volleyball_view")
    sim_context.play()

    # --- 发球设置 ---
    serve_types_list = list(SERVE_TYPES.keys())
    serve_idx = 0
    serve_side = "left"
    serve_type = serve_types_list[serve_idx]

    pos, vel, ang, info = generate_serve(serve_type, serve_side)
    reset_ball(ball_rigid, pos, vel, ang)

    # 空气动力学模型
    aero = VolleyballAerodynamics()

    # --- Episode 追踪 ---
    max_episode_steps = int(8.0 / PHYSICS_DT)  # 8秒超时
    episode_stats = {"ground": 0, "net": 0, "oob": 0, "timeout": 0}

    print("=" * 70)
    print("  🏐 排球物理模拟 — RL 训练环境")
    print("=" * 70)
    print(f"  球场: {COURT_LENGTH}×{COURT_WIDTH}m | 球网: {NET_HEIGHT}m")
    print(f"  排球: d={BALL_DIAMETER*100:.0f}cm  m={BALL_MASS*1000:.0f}g")
    print(f"  物理: {1/PHYSICS_DT:.0f}Hz | 渲染: {1/RENDER_DT:.0f}Hz | GPU dynamics")
    print(f"  重置: 碰网 |x|<{NET_HALF_THICKNESS+BALL_RADIUS:.2f} | 出界 | 超时")
    print("=" * 70)
    print(f"  ▶ 首发: {info}")
    print("-" * 70)

    step_count = 0
    sim_time = 0.0
    serve_count = 1
    last_print_time = time.time()
    max_speed_display = 0.0

    while simulation_app.is_running():
        sim_context.step(render=True)
        step_count += 1
        sim_time += PHYSICS_DT

        # 读取球状态
        ball_pos, ball_vel, ball_ang = get_ball_state(ball_rigid)
        speed = np.linalg.norm(ball_vel)
        spin_rate = np.linalg.norm(ball_ang)

        # --- 空气动力学力（仅空中、速度 > 0.5 m/s）---
        if speed > 0.5:
            aero_force = aero.total_aerodynamic_force(
                ball_vel, ball_ang, sim_time
            )
            apply_aero_force(ball_rigid, aero_force)

        # --- 状态追踪 ---
        if speed > max_speed_display:
            max_speed_display = speed

        # 打印（0.5秒间隔）
        now = time.time()
        if now - last_print_time >= 0.5:
            Cd = aero.drag_coefficient(speed) if speed > 0.1 else 0
            Re = aero.reynolds_number(speed) if speed > 0.1 else 0
            Cl = aero.lift_coefficient(speed, spin_rate) if speed > 0.1 else 0
            print(
                f"  🏐 pos=({ball_pos[0]:+6.2f}, {ball_pos[1]:+5.2f}, {ball_pos[2]:+5.2f}) "
                f"| v={speed:5.1f}m/s "
                f"| Cd={Cd:.3f} | Re={Re:.1e} "
                f"| ω={spin_rate:5.1f}rad/s | Cl={Cl:.3f}"
            )
            last_print_time = now

        # --- Episode 结束检测 ---
        is_done, reason = check_episode_end(ball_pos, step_count, max_episode_steps)
        if is_done:
            episode_stats[reason] = episode_stats.get(reason, 0) + 1
            serve_count += 1

            # 轮换发球类型和方向
            serve_idx = (serve_idx + 1) % len(serve_types_list)
            serve_type = serve_types_list[serve_idx]
            serve_side = "right" if serve_side == "left" else "left"

            pos, vel, ang, info = generate_serve(serve_type, serve_side)
            reset_ball(ball_rigid, pos, vel, ang)
            aero = VolleyballAerodynamics()

            print(f"\n  {'='*60}")
            print(f"  ⏹ Episode #{serve_count-1} 结束: {reason} "
                  f"(最高速 {max_speed_display:.1f}m/s)")
            print(f"  ▶ 第{serve_count}次发球: {info}")
            print(f"  📊 统计: {episode_stats}")
            print(f"  {'='*60}")

            step_count = 0
            max_speed_display = 0.0

    sim_context.stop()
    simulation_app.close()
    print(f"[INFO] 仿真结束 | 总 episode: {serve_count-1} | 统计: {episode_stats}")


if __name__ == "__main__":
    main()
