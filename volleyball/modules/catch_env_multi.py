# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球 RL 环境 - 多环境并行版本

基于 Isaac Sim 的 GPU 并行仿真，支持同时运行多个环境实例。

观察空间 (8D) - 每个环境:
    [0:2] - 预测落点相对于机器人 (rel_x, rel_y)
    [2]   - 球到达时间
    [3:5] - 机器人位置 (x, y)
    [5:7] - 机器人速度 (vx, vy)
    [7]   - 球高度

动作空间 (2D) - 每个环境:
    [0:2] - 底盘速度指令 (vx, vy), 范围 [-1, 1]
"""

import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces


# =====================================================================
# 环境参数
# =====================================================================

# 机器人参数
ROBOT_HEIGHT = 0.15
ROBOT_CATCH_RADIUS = 0.40
ROBOT_MAX_SPEED = 3.0
CATCH_HEIGHT_MARGIN = 0.20

# RL 参数
PHYSICS_DT = 1.0 / 120.0
RENDER_DT = 1.0 / 30.0
CONTROL_DT = 1.0 / 20.0
MAX_EPISODE_TIME = 6.0

# 标准排球场参数
COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
NET_HEIGHT = 2.43

# 发球参数
SERVE_X = -10.0
SERVE_SPEED = 15.0
SERVE_ANGLE = 20.0
SERVE_HEIGHT = 3.5
SERVE_SPIN = 0.0

# 多环境布局
ENV_SPACING = 25.0  # 环境间距 (m)


# =====================================================================
# 多环境接球 RL 环境
# =====================================================================

class VolleyballCatchMultiEnv(gym.Env):
    """
    多环境并行排球接球 Gymnasium 环境
    
    同时运行 num_envs 个独立的环境实例，共享同一个物理仿真。
    所有输入/输出都是张量形式 (num_envs, dim)。
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, simulation_app=None, render_mode=None, 
                 num_envs: int = 64, device: str = "cuda:0", verbose=False):
        """
        初始化多环境
        
        Args:
            simulation_app: Isaac Sim SimulationApp 实例
            render_mode: 渲染模式
            num_envs: 并行环境数量
            device: 计算设备
            verbose: 是否打印详细信息
        """
        super().__init__()
        
        self.simulation_app = simulation_app
        self.render_mode = render_mode
        self.num_envs = num_envs
        self.device = device
        self.verbose = verbose
        
        # 观察和动作空间 (单个环境)
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
        
        # 多环境实例
        self.balls = []        # List[VolleyballRigidBody]
        self.robots = []       # List[SwerveRobot]
        self.courts = []       # List[VolleyballCourt] (可选，共享一个也行)
        self.env_origins = []  # 每个环境的原点偏移
        
        # Episode 状态 (张量形式)
        self.step_count = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.max_steps = int(MAX_EPISODE_TIME / CONTROL_DT)
        self.physics_steps_per_control = int(CONTROL_DT / PHYSICS_DT)
        self.sim_time = 0.0
        
        # 预测落点 (张量形式)
        self.predicted_landings = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
        self.time_to_landings = torch.zeros(num_envs, dtype=torch.float32, device=device)
        
        # 统计
        self.episode_count = 0
        self.catch_count = 0
        self.miss_count = 0
        
        # 用于 RSL-RL 的属性
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32, device=device)
        self.max_episode_length = self.max_steps
    
    def _compute_env_origins(self):
        """计算每个环境的原点位置 (网格布局)"""
        num_per_row = int(np.ceil(np.sqrt(self.num_envs)))
        
        self.env_origins = []
        for i in range(self.num_envs):
            row = i // num_per_row
            col = i % num_per_row
            x = col * ENV_SPACING
            y = row * ENV_SPACING
            self.env_origins.append((x, y, 0.0))
        
        self.env_origins_tensor = torch.tensor(
            self.env_origins, dtype=torch.float32, device=self.device
        )
    
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
        
        # 计算环境原点
        self._compute_env_origins()
        
        if self.verbose:
            print(f"[MultiEnv] Creating {self.num_envs} parallel environments...")
        
        # 为每个环境创建资产
        for env_id in range(self.num_envs):
            origin = self.env_origins[env_id]
            env_prefix = f"Env_{env_id}"
            
            # 创建场地 (只有第一个环境创建物理场景和地面)
            is_first = (env_id == 0)
            court = VolleyballCourt(
                self.stage,
                court_length=18.0,
                court_width=9.0,
                net_height=2.43,
                enable_net_collision=True,
                prim_prefix=f"/World/{env_prefix}",
                origin=origin
            )
            court.setup_all(create_physics_scene=is_first, create_ground=is_first)
            self.courts.append(court)
            
            # 创建排球
            ball_pos = (origin[0] + SERVE_X, origin[1], origin[2] + 2.5)
            ball = VolleyballRigidBody(
                self.stage, 
                name=f"{env_prefix}_Ball",
                position=ball_pos
            )
            self.balls.append(ball)
            
            # 创建机器人
            robot_pos = (origin[0] + 5.0, origin[1], origin[2] + ROBOT_HEIGHT)
            robot = SwerveRobot(
                self.stage,
                name=f"{env_prefix}_Robot",
                position=robot_pos
            )
            robot.load_urdf()
            self.robots.append(robot)
            
            if self.verbose and (env_id + 1) % 10 == 0:
                print(f"[MultiEnv] Created {env_id + 1}/{self.num_envs} environments")
        
        # 更新一次让资源加载完成
        if self.simulation_app is not None:
            self.simulation_app.update()
        
        self.sim_context = SimulationContext(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT)
        
        # 设置相机观察全局
        total_span = int(np.ceil(np.sqrt(self.num_envs))) * ENV_SPACING
        set_camera_view(
            eye=[total_span / 2, -total_span, total_span / 2],
            target=[total_span / 2, total_span / 2, 0.0],
            camera_prim_path="/OmniverseKit_Persp"
        )
        
        self.sim_context.initialize_physics()
        self.sim_context.play()
        
        # 初始化所有刚体控制
        for ball in self.balls:
            ball.initialize_rigid_prim()
        for robot in self.robots:
            robot.initialize_articulation()
        
        self._initialized = True
        if self.verbose:
            print(f"[MultiEnv] All {self.num_envs} environments initialized")
    
    def _get_ball_states(self) -> tuple:
        """获取所有球的状态 (向量化)"""
        positions = []
        velocities = []
        angular_vels = []
        
        for ball in self.balls:
            pos, vel, ang = ball.get_state()
            positions.append(pos)
            velocities.append(vel)
            angular_vels.append(ang)
        
        return (
            torch.tensor(np.array(positions), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(velocities), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(angular_vels), dtype=torch.float32, device=self.device)
        )
    
    def _get_robot_states(self) -> tuple:
        """获取所有机器人的状态 (向量化)"""
        positions = []
        velocities = []
        
        for robot in self.robots:
            pos, _, lin_vel, _ = robot.get_state()
            positions.append(pos)
            velocities.append(lin_vel)
        
        return (
            torch.tensor(np.array(positions), dtype=torch.float32, device=self.device),
            torch.tensor(np.array(velocities), dtype=torch.float32, device=self.device)
        )
    
    def _set_robot_velocities(self, actions: torch.Tensor):
        """设置所有机器人的速度"""
        actions_np = actions.cpu().numpy()
        for i, robot in enumerate(self.robots):
            vx = float(np.clip(actions_np[i, 0], -1.0, 1.0)) * ROBOT_MAX_SPEED
            vy = float(np.clip(actions_np[i, 1], -1.0, 1.0)) * ROBOT_MAX_SPEED
            robot.set_velocity_command(vx, vy, 0.0)
    
    def _reset_env(self, env_id: int):
        """重置单个环境"""
        from .volleyball_physics import generate_fixed_serve
        
        origin = self.env_origins[env_id]
        
        # 生成新发球
        pos, lin_vel, ang_vel, info = generate_fixed_serve(
            speed=SERVE_SPEED,
            angle_deg=SERVE_ANGLE,
            y_offset=0.0,
            from_x=SERVE_X,
            height=SERVE_HEIGHT,
            spin_rate=SERVE_SPIN
        )
        
        # 重置排球 (加上环境偏移)
        ball_pos = (pos[0] + origin[0], pos[1] + origin[1], pos[2] + origin[2])
        self.balls[env_id].set_state(ball_pos, lin_vel, ang_vel)
        self.balls[env_id].reset_aerodynamics()
        
        # 重置机器人到接发球区
        robot_x = origin[0] + random.uniform(3.0, 8.0)
        robot_y = origin[1] + random.uniform(-3.0, 3.0)
        robot_pos = (robot_x, robot_y, origin[2] + ROBOT_HEIGHT)
        self.robots[env_id].reset(robot_pos)
        
        # 重置步数
        self.step_count[env_id] = 0
        self.episode_length_buf[env_id] = 0
    
    def _update_predictions(self):
        """更新所有环境的落点预测"""
        from .volleyball_physics import predict_landing_point
        
        ball_positions, ball_velocities, _ = self._get_ball_states()
        
        for i in range(self.num_envs):
            ball_pos = ball_positions[i].cpu().numpy()
            ball_vel = ball_velocities[i].cpu().numpy()
            origin = self.env_origins[i]
            
            result = predict_landing_point(ball_pos, ball_vel, ROBOT_HEIGHT)
            
            if result is not None:
                # 落点是绝对坐标
                self.predicted_landings[i, 0] = result[0]
                self.predicted_landings[i, 1] = result[1]
                self.time_to_landings[i] = result[2]
            else:
                self.predicted_landings[i, 0] = ball_pos[0]
                self.predicted_landings[i, 1] = ball_pos[1]
                self.time_to_landings[i] = 0.0
    
    def _get_observations(self) -> torch.Tensor:
        """获取所有环境的观察值"""
        ball_positions, _, _ = self._get_ball_states()
        robot_positions, robot_velocities = self._get_robot_states()
        
        # 相对落点 = 预测落点 - 机器人位置
        rel_landings = self.predicted_landings - robot_positions[:, :2]
        
        obs = torch.zeros((self.num_envs, 8), dtype=torch.float32, device=self.device)
        obs[:, 0:2] = rel_landings
        obs[:, 2] = self.time_to_landings
        obs[:, 3:5] = robot_positions[:, :2] - self.env_origins_tensor[:, :2]  # 相对于环境原点
        obs[:, 5:7] = robot_velocities[:, :2]
        obs[:, 7] = ball_positions[:, 2]
        
        return obs
    
    def reset(self, seed=None, options=None):
        """重置所有环境"""
        super().reset(seed=seed)
        
        # 首次调用时初始化
        if not self._initialized:
            self._initialize_sim()
        
        self.sim_time = 0.0
        self.episode_count += 1
        
        # 重置所有环境
        for i in range(self.num_envs):
            self._reset_env(i)
        
        # 更新预测
        self._update_predictions()
        
        obs = self._get_observations()
        
        if self.verbose:
            print(f"[MultiEnv] Episode #{self.episode_count}: Reset {self.num_envs} environments")
        
        return obs, {}
    
    def step(self, actions: torch.Tensor):
        """执行动作 (向量化)"""
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        
        if actions.dim() == 1:
            actions = actions.unsqueeze(0).expand(self.num_envs, -1)
        
        self.step_count += 1
        self.episode_length_buf += 1
        
        # 保存上一步状态
        robot_positions_prev, _ = self._get_robot_states()
        prev_dist_to_landing = torch.norm(
            robot_positions_prev[:, :2] - self.predicted_landings, dim=1
        )
        
        # 执行物理步骤
        for _ in range(self.physics_steps_per_control):
            self._set_robot_velocities(actions)
            
            # 施加空气动力学力
            for ball in self.balls:
                ball.apply_aerodynamic_forces(self.sim_time)
            
            self.sim_context.step(render=False)
            self.sim_time += PHYSICS_DT
        
        # 渲染
        if self.render_mode == "human":
            self.sim_context.render()
        
        # 更新预测
        self._update_predictions()
        
        # 获取状态
        ball_positions, _, _ = self._get_ball_states()
        robot_positions, robot_velocities = self._get_robot_states()
        
        # 计算指标
        ball_robot_xy_dist = torch.norm(
            ball_positions[:, :2] - robot_positions[:, :2], dim=1
        )
        dist_to_landing = torch.norm(
            robot_positions[:, :2] - self.predicted_landings, dim=1
        )
        
        # ==================== 奖励计算 ====================
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 接球检测参数
        CATCH_HEIGHT_THRESHOLD = ROBOT_HEIGHT + CATCH_HEIGHT_MARGIN
        GROUND_THRESHOLD = self.BALL_RADIUS + 0.02
        
        # 接球检测
        ball_in_catch_height = ball_positions[:, 2] <= CATCH_HEIGHT_THRESHOLD
        ball_above_ground = ball_positions[:, 2] > GROUND_THRESHOLD
        ball_in_catch_range = ball_robot_xy_dist < ROBOT_CATCH_RADIUS
        
        # 接住球
        catch_mask = ball_in_catch_height & ball_above_ground & ball_in_catch_range
        rewards[catch_mask] = 100.0
        terminated[catch_mask] = True
        
        # 球触地（未接住）
        miss_mask = (ball_positions[:, 2] <= GROUND_THRESHOLD) & ~catch_mask
        rewards[miss_mask] = -10.0 - ball_robot_xy_dist[miss_mask] * 5.0
        terminated[miss_mask] = True
        
        # 超时
        timeout_mask = (self.step_count >= self.max_steps) & ~terminated
        rewards[timeout_mask] = -20.0
        truncated[timeout_mask] = True
        
        # 正常步骤奖励 (Dense Reward)
        ongoing_mask = ~terminated & ~truncated
        
        # 位置奖励
        r_position = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        r_position[ongoing_mask & (dist_to_landing < 0.5)] = 2.0
        r_position[ongoing_mask & (dist_to_landing >= 0.5) & (dist_to_landing < 1.0)] = 1.0
        r_position[ongoing_mask & (dist_to_landing >= 1.0) & (dist_to_landing < 2.0)] = 0.5
        r_position[ongoing_mask & (dist_to_landing >= 2.0)] = -dist_to_landing[ongoing_mask & (dist_to_landing >= 2.0)] * 0.1
        
        # 进度奖励
        progress = prev_dist_to_landing - dist_to_landing
        r_approach = progress * 2.0
        
        # 等待奖励
        r_wait = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        r_wait[ongoing_mask & (dist_to_landing < 0.5)] = 0.5
        
        # 时间惩罚
        r_time = torch.full((self.num_envs,), -0.02, dtype=torch.float32, device=self.device)
        
        # 合计
        rewards[ongoing_mask] += (r_position + r_approach + r_wait + r_time)[ongoing_mask]
        
        # 自动重置终止的环境
        reset_mask = terminated | truncated
        if reset_mask.any():
            for i in torch.where(reset_mask)[0].tolist():
                self._reset_env(i)
            self.catch_count += catch_mask.sum().item()
            self.miss_count += miss_mask.sum().item()
        
        obs = self._get_observations()
        
        info = {
            "catches": catch_mask.sum().item(),
            "misses": miss_mask.sum().item(),
            "timeouts": timeout_mask.sum().item(),
        }
        
        return obs, rewards, terminated, truncated, info
    
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


# =====================================================================
# RSL-RL VecEnv 接口
# =====================================================================

class MultiEnvVecEnvWrapper:
    """
    将 VolleyballCatchMultiEnv 包装为 RSL-RL 的 VecEnv 接口
    """
    
    def __init__(self, env: VolleyballCatchMultiEnv):
        # 导入 TensorDict
        from tensordict import TensorDict
        self._TensorDict = TensorDict
        
        self.env = env
        self.num_envs = env.num_envs
        self.num_obs = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        
        # RSL-RL 需要的属性
        self.episode_length_buf = env.episode_length_buf
        
        # 配置对象
        self.cfg = type('Config', (), {
            'env': type('EnvCfg', (), {
                'num_envs': env.num_envs,
            })(),
        })()
        
        # 属性
        self.step_dt = CONTROL_DT
        self._obs = None
    
    @property
    def unwrapped(self):
        return self
    
    def get_observations(self):
        """获取观察值 (TensorDict 格式)"""
        return self._TensorDict({"policy": self._obs}, batch_size=[self.num_envs])
    
    def reset(self):
        """重置环境"""
        obs, _ = self.env.reset()
        self._obs = obs
        return self._TensorDict({"policy": obs}, batch_size=[self.num_envs]), {}
    
    def step(self, actions: torch.Tensor) -> tuple:
        """执行动作"""
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        self._obs = obs
        dones = terminated | truncated
        return self._TensorDict({"policy": obs}, batch_size=[self.num_envs]), rewards, dones, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
