# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球 RL 环境

任务: 舵轮底盘移动到排球落点位置接球

观察空间 (8D):
    [0:2] - 预测落点相对于机器人 (rel_x, rel_y)
    [2]   - 球到达时间
    [3:5] - 机器人位置 (x, y)
    [5:7] - 机器人速度 (vx, vy)
    [7]   - 球高度

动作空间 (2D):
    [0:2] - 底盘速度指令 (vx, vy), 范围 [-1, 1]

奖励:
    - 接住球: +100
    - 触底未接住: -10 - distance * 2
    - 靠近落点: +0.1 * velocity_towards_target

终止条件:
    - 排球触底（接住或未接住）
    - 超时 (6秒)
"""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# =====================================================================
# 环境参数
# =====================================================================

# 机器人参数
ROBOT_HEIGHT = 0.15           # 机器人顶部高度 (m)
ROBOT_CATCH_RADIUS = 0.40     # 接球有效半径 (m)
ROBOT_MAX_SPEED = 3.0         # 最大速度 (m/s)
CATCH_HEIGHT_MARGIN = 0.20    # 接球高度容差 (m)，球下落到 ROBOT_HEIGHT + 此值 可以接

# RL 参数
PHYSICS_DT = 1.0 / 120.0
RENDER_DT = 1.0 / 30.0
CONTROL_DT = 1.0 / 20.0   # 20Hz 控制
MAX_EPISODE_TIME = 6.0

# 标准排球场参数
COURT_LENGTH = 18.0       # 全场长度 (m)
COURT_WIDTH = 9.0         # 场地宽度 (m)
NET_HEIGHT = 2.43         # 男子球网高度 (m)

# 发球参数（固定参数，用于调试）
SERVE_X = -10.0
SERVE_SPEED = 15.0            # 固定发球速度 (m/s)
SERVE_ANGLE = 20.0            # 固定发球仰角 (度)
SERVE_HEIGHT = 3.5         # 固定发球高度 (m)
SERVE_SPIN = 0.0              # 固定旋转率 (rad/s), 0=无旋转


# =====================================================================
# 接球 RL 环境
# =====================================================================

class VolleyballCatchEnv(gym.Env):
    """
    排球接球 Gymnasium 环境
    
    使用模块化组件:
    - VolleyballRigidBody: 完整排球物理（空气动力学）
    - VolleyballCourt: 完整排球场地（含球网、边线）
    - SwerveRobot: URDF 舵轮底盘机器人
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, simulation_app=None, render_mode=None, verbose=True):
        """
        初始化环境
        
        Args:
            simulation_app: Isaac Sim SimulationApp 实例
            render_mode: 渲染模式
            verbose: 是否打印详细信息 (训练时设为 False)
        """
        super().__init__()
        
        self.simulation_app = simulation_app
        self.render_mode = render_mode
        self.verbose = verbose
        
        # 观察和动作空间
        self.observation_space = spaces.Box(
            low=np.array([-20, -20, 0, -10, -10, -5, -5, 0], dtype=np.float32),
            high=np.array([20, 20, 10, 10, 10, 5, 5, 20], dtype=np.float32),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Isaac Sim 组件（延迟初始化）
        self._initialized = False
        self.stage = None
        self.sim_context = None
        self.ball = None
        self.robot = None
        self.court = None
        
        # Episode 状态
        self.step_count = 0
        self.max_steps = int(MAX_EPISODE_TIME / CONTROL_DT)
        self.physics_steps_per_control = int(CONTROL_DT / PHYSICS_DT)
        self.sim_time = 0.0
        
        # 预测
        self.predicted_landing = None
        self.time_to_landing = None
        
        # 统计
        self.episode_count = 0
        self.catch_count = 0
        self.miss_count = 0
    
    def _initialize_sim(self):
        """初始化 Isaac Sim (首次调用时)"""
        if self._initialized:
            return
        
        import omni.usd
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.utils.viewports import set_camera_view
        
        # 导入模块
        from .volleyball_physics import VolleyballRigidBody, BALL_RADIUS
        from .volleyball_court import VolleyballCourt
        from .swerve_robot import SwerveRobot
        
        self.BALL_RADIUS = BALL_RADIUS
        
        self.stage = omni.usd.get_context().get_stage()
        
        # 创建完整排球场地（含球网、边线）
        self.court = VolleyballCourt(
            self.stage, 
            court_length=18.0,   # 标准排球场长度
            court_width=9.0,     # 标准排球场宽度
            net_height=2.43,     # 男子球网高度
            enable_net_collision=True  # 启用球网碰撞
        )
        self.court.setup_all()
        if self.verbose:
            print("[VolleyballCatchEnv] Full volleyball court created")
        
        # 创建排球（完整空气动力学）
        self.ball = VolleyballRigidBody(self.stage, "Volleyball", (SERVE_X, 0, 2.5))
        if self.verbose:
            print("[VolleyballCatchEnv] Volleyball created with full aerodynamics")
        
        # 创建 URDF 舵轮底盘机器人
        self.robot = SwerveRobot(self.stage, position=(0, 0, 0.1))
        self.robot.load_urdf()
        if self.verbose:
            print("[VolleyballCatchEnv] URDF swerve robot loaded")
        
        # 更新一次让资源加载完成
        if self.simulation_app is not None:
            self.simulation_app.update()
        
        self.sim_context = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT)
        
        set_camera_view(
            eye=[0.0, -10.0, 6.0],
            target=[0.0, 0.0, 1.0],
            camera_prim_path="/OmniverseKit_Persp"
        )
        
        self.sim_context.initialize_physics()
        self.sim_context.play()
        
        # 初始化刚体控制
        self.ball.initialize_rigid_prim()
        self.robot.initialize_articulation()
        
        # 创建预测落点标记
        self._create_landing_marker()
        
        self._initialized = True
        if self.verbose:
            print("[VolleyballCatchEnv] Environment initialized")
    
    def _create_landing_marker(self):
        """创建预测落点可视化标记"""
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        
        # 创建圆柱体作为落点标记（红色圆圈）
        marker_path = "/World/LandingMarker"
        marker = UsdGeom.Cylinder.Define(self.stage, Sdf.Path(marker_path))
        marker.CreateRadiusAttr(0.3)  # 30cm 半径
        marker.CreateHeightAttr(0.02)  # 2cm 高度（扁平）
        marker.CreateAxisAttr("Z")
        
        xform = UsdGeom.Xformable(marker)
        self._marker_translate_op = xform.AddTranslateOp()
        self._marker_translate_op.Set(Gf.Vec3d(0, 0, 0.01))
        
        # 红色材质
        mat = UsdShade.Material.Define(self.stage, Sdf.Path("/Materials/LandingMarkerMat"))
        shader = UsdShade.Shader.Define(self.stage, Sdf.Path("/Materials/LandingMarkerMat/Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.2, 0.2))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(marker).Bind(mat)
        
        self._landing_marker_path = marker_path
        if self.verbose:
            print("[VolleyballCatchEnv] Landing marker created")
    
    def _update_landing_marker(self, x: float, y: float):
        """更新预测落点标记位置"""
        from pxr import Gf
        if hasattr(self, '_marker_translate_op'):
            self._marker_translate_op.Set(Gf.Vec3d(x, y, 0.01))
    
    def _get_ball_state(self):
        """获取球状态"""
        return self.ball.get_state()
    
    def _get_robot_state(self):
        """获取机器人状态"""
        pos, _, lin_vel, _ = self.robot.get_state()
        return pos, lin_vel
    
    def _set_robot_velocity(self, vx: float, vy: float):
        """设置机器人速度"""
        self.robot.set_velocity_command(vx * ROBOT_MAX_SPEED, vy * ROBOT_MAX_SPEED, 0.0)
    
    def _reset_robot(self, position: tuple = None):
        """重置机器人到接发球区域"""
        if position is None:
            # 机器人在对面半场接发球区（X: 3-8m, Y: -3到3m）
            position = (random.uniform(3.0, 8.0), random.uniform(-3.0, 3.0), ROBOT_HEIGHT)
        
        self.robot.reset(position)
    
    def _update_prediction(self):
        """更新落点预测并更新可视化标记"""
        from .volleyball_physics import predict_landing_point
        
        ball_pos, ball_vel, _ = self._get_ball_state()
        result = predict_landing_point(ball_pos, ball_vel, ROBOT_HEIGHT)
        
        if result is not None:
            self.predicted_landing = np.array([result[0], result[1]])
            self.time_to_landing = result[2]
            # 更新可视化标记
            self._update_landing_marker(result[0], result[1])
        else:
            self.predicted_landing = ball_pos[:2]
            self.time_to_landing = 0.0
            self._update_landing_marker(ball_pos[0], ball_pos[1])
    
    def _get_observation(self) -> np.ndarray:
        """获取观察值"""
        ball_pos, _, _ = self._get_ball_state()
        robot_pos, robot_vel = self._get_robot_state()
        
        rel_landing = self.predicted_landing - robot_pos[:2] if self.predicted_landing is not None else np.zeros(2)
        
        obs = np.array([
            rel_landing[0],
            rel_landing[1],
            self.time_to_landing if self.time_to_landing else 0.0,
            robot_pos[0],
            robot_pos[1],
            robot_vel[0],
            robot_vel[1],
            ball_pos[2],
        ], dtype=np.float32)
        
        return obs
    
    def _generate_serve(self) -> tuple:
        """生成固定参数发球"""
        from .volleyball_physics import generate_fixed_serve
        pos, lin_vel, ang_vel, info = generate_fixed_serve(
            speed=SERVE_SPEED,
            angle_deg=SERVE_ANGLE,
            y_offset=0.0,
            from_x=SERVE_X,
            height=SERVE_HEIGHT,
            spin_rate=SERVE_SPIN
        )
        return pos, lin_vel, ang_vel, info
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 首次调用时初始化
        if not self._initialized:
            self._initialize_sim()
        
        self.step_count = 0
        self.sim_time = 0.0
        self.episode_count += 1
        
        # 生成新发球
        ball_pos, ball_vel, ball_ang, info = self._generate_serve()
        
        # 重置排球
        self.ball.set_state(ball_pos, ball_vel, ball_ang)
        self.ball.reset_aerodynamics()
        
        # 重置机器人
        self._reset_robot()
        
        # 更新预测
        self._update_prediction()
        
        obs = self._get_observation()
        
        # 打印发球信息和预测落点
        if self.verbose:
            print(f"[Env] Episode #{self.episode_count}: {info}")
            print(f"      预测落点: ({self.predicted_landing[0]:.2f}, {self.predicted_landing[1]:.2f}) | "
                  f"到达时间: {self.time_to_landing:.2f}s")
        
        return obs, {}
    
    def step(self, action):
        """执行动作"""
        self.step_count += 1
        
        # 保存上一步状态（用于计算进度奖励）
        prev_robot_pos, _ = self._get_robot_state()
        prev_dist_to_landing = np.linalg.norm(prev_robot_pos[:2] - self.predicted_landing) if self.predicted_landing is not None else 0.0
        
        # 解析动作
        vx = float(np.clip(action[0], -1.0, 1.0))
        vy = float(np.clip(action[1], -1.0, 1.0))
        
        # 执行物理步骤
        for _ in range(self.physics_steps_per_control):
            self._set_robot_velocity(vx, vy)
            
            # 施加空气动力学力
            self.ball.apply_aerodynamic_forces(self.sim_time)
            
            self.sim_context.step(render=False)
            self.sim_time += PHYSICS_DT
        
        # 渲染
        if self.render_mode == "human":
            self.sim_context.render()
        
        # 更新预测
        self._update_prediction()
        
        # 获取状态
        ball_pos, ball_vel, _ = self._get_ball_state()
        robot_pos, robot_vel = self._get_robot_state()
        
        # 计算指标
        ball_robot_xy_dist = np.linalg.norm(ball_pos[:2] - robot_pos[:2])
        dist_to_landing = np.linalg.norm(robot_pos[:2] - self.predicted_landing) if self.predicted_landing is not None else 0.0
        
        # ==================== 奖励计算 ====================
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        
        # 接球检测参数
        CATCH_HEIGHT_THRESHOLD = ROBOT_HEIGHT + CATCH_HEIGHT_MARGIN  # 球下落到此高度可以接
        GROUND_THRESHOLD = self.BALL_RADIUS + 0.02    # 球触地判定
        
        # 接球检测：球进入接球区域（考虑球的高度和水平距离）
        ball_in_catch_height = ball_pos[2] <= CATCH_HEIGHT_THRESHOLD
        ball_above_ground = ball_pos[2] > GROUND_THRESHOLD
        ball_in_catch_range = ball_robot_xy_dist < ROBOT_CATCH_RADIUS
        
        if ball_in_catch_height and ball_above_ground and ball_in_catch_range:
            # ===== 接住球 =====
            reward = 100.0
            terminated = True
            info["result"] = "catch"
            self.catch_count += 1
            if self.verbose:
                print(f"  CATCH! Episode #{self.episode_count}: dist={ball_robot_xy_dist:.2f}m, ball_z={ball_pos[2]:.2f}m")
        
        elif ball_pos[2] <= self.BALL_RADIUS + 0.02:
            # ===== 球触地（未接住）=====
            # 惩罚 = 基础惩罚 + 距离惩罚
            reward = -10.0 - ball_robot_xy_dist * 5.0
            terminated = True
            info["result"] = "miss"
            info["miss_distance"] = ball_robot_xy_dist
            self.miss_count += 1
            if self.verbose:
                print(f"  MISS  Episode #{self.episode_count}: dist={ball_robot_xy_dist:.2f}m")
        
        elif self.step_count >= self.max_steps:
            # ===== 超时 =====
            reward = -20.0
            truncated = True
            info["result"] = "timeout"
        
        else:
            # ===== 正常步骤奖励（Dense Reward）=====
            
            # 1. 【核心】到落点距离奖励（越近越好）
            # 使用指数衰减，在落点附近奖励更高
            if dist_to_landing < 0.5:
                r_position = 2.0  # 在目标区域内
            elif dist_to_landing < 1.0:
                r_position = 1.0
            elif dist_to_landing < 2.0:
                r_position = 0.5
            else:
                r_position = -dist_to_landing * 0.1  # 距离惩罚
            
            # 2. 【进度】接近落点的速度奖励
            progress = prev_dist_to_landing - dist_to_landing
            r_approach = progress * 2.0  # 每接近1m奖励2分
            
            # 3. 【效率】已到达目标区域的等待奖励
            if dist_to_landing < 0.5:
                r_wait = 0.5  # 提前到位奖励
            else:
                r_wait = 0.0
            
            # 4. 【时间】步数惩罚（鼓励快速）
            r_time = -0.02
            
            # 5. 【安全】出界惩罚
            if abs(robot_pos[0]) > COURT_LENGTH or abs(robot_pos[1]) > COURT_WIDTH:
                r_boundary = -5.0
            else:
                r_boundary = 0.0
            
            reward = r_position + r_approach + r_wait + r_time + r_boundary
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染"""
        if self.render_mode == "human" and self.sim_context:
            self.sim_context.render()
    
    def close(self):
        """关闭环境"""
        if self.sim_context:
            self.sim_context.stop()
    
    def get_stats(self) -> dict:
        """获取统计"""
        total = self.catch_count + self.miss_count
        catch_rate = self.catch_count / total if total > 0 else 0.0
        return {
            "episodes": self.episode_count,
            "catches": self.catch_count,
            "misses": self.miss_count,
            "catch_rate": catch_rate,
        }
