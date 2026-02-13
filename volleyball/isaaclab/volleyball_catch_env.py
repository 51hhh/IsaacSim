# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球 Isaac Lab 环境
"""

from __future__ import annotations

import math
import torch
from typing import Dict

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject
import isaaclab.sim as sim_utils

from volleyball_catch_cfg import VolleyballCatchEnvCfg


class VolleyballCatchEnv(DirectRLEnv):
    """排球接球强化学习环境 (Isaac Lab GPU 并行版本)"""
    
    cfg: VolleyballCatchEnvCfg

    def __init__(self, cfg: VolleyballCatchEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)
        self._sim_time = 0.0
        
        self.predicted_landing = torch.zeros(self.num_envs, 2, device=self.device)
        self.time_to_landing = torch.zeros(self.num_envs, device=self.device)
        
        print(f"[VolleyballCatchEnv] 初始化完成，{self.num_envs} 个并行环境")

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        
        self.ball = RigidObject(self.cfg.ball_cfg)
        self.scene.rigid_objects["ball"] = self.ball
        
        # 创建完整排球场（与原始 Sim 版本一致）
        self._create_court_scene()
        
        self.scene.clone_environments(copy_from_source=False)
        
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _create_court_scene(self):
        """创建完整排球场场景（与原始 Sim 版本一致）"""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade, UsdPhysics
        
        stage = omni.usd.get_context().get_stage()
        
        court_length = self.cfg.court_length
        court_width = self.cfg.court_width
        net_height = self.cfg.net_height
        
        # ==================== 创建材质 ====================
        # 场地材质（木地板色）
        court_mat = UsdShade.Material.Define(stage, Sdf.Path("/Materials/CourtMat"))
        court_shader = UsdShade.Shader.Define(stage, Sdf.Path("/Materials/CourtMat/Shader"))
        court_shader.CreateIdAttr("UsdPreviewSurface")
        court_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.82, 0.68, 0.46))
        court_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
        court_mat.CreateSurfaceOutput().ConnectToSource(court_shader.ConnectableAPI(), "surface")
        
        # 边线材质（白色）
        line_mat = UsdShade.Material.Define(stage, Sdf.Path("/Materials/LineMat"))
        line_shader = UsdShade.Shader.Define(stage, Sdf.Path("/Materials/LineMat/Shader"))
        line_shader.CreateIdAttr("UsdPreviewSurface")
        line_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        line_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        line_mat.CreateSurfaceOutput().ConnectToSource(line_shader.ConnectableAPI(), "surface")
        
        # 球网材质
        net_mat = UsdShade.Material.Define(stage, Sdf.Path("/Materials/NetMat"))
        net_shader = UsdShade.Shader.Define(stage, Sdf.Path("/Materials/NetMat/Shader"))
        net_shader.CreateIdAttr("UsdPreviewSurface")
        net_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.15, 0.15, 0.15))
        net_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        net_mat.CreateSurfaceOutput().ConnectToSource(net_shader.ConnectableAPI(), "surface")
        
        # 顶带材质（白色）
        top_band_mat = UsdShade.Material.Define(stage, Sdf.Path("/Materials/TopBandMat"))
        top_band_shader = UsdShade.Shader.Define(stage, Sdf.Path("/Materials/TopBandMat/Shader"))
        top_band_shader.CreateIdAttr("UsdPreviewSurface")
        top_band_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.95, 0.95, 0.95))
        top_band_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        top_band_mat.CreateSurfaceOutput().ConnectToSource(top_band_shader.ConnectableAPI(), "surface")
        
        # 网柱材质（金属色）
        pole_mat = UsdShade.Material.Define(stage, Sdf.Path("/Materials/PoleMat"))
        pole_shader = UsdShade.Shader.Define(stage, Sdf.Path("/Materials/PoleMat/Shader"))
        pole_shader.CreateIdAttr("UsdPreviewSurface")
        pole_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.6, 0.65))
        pole_shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        pole_shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.8)
        pole_mat.CreateSurfaceOutput().ConnectToSource(pole_shader.ConnectableAPI(), "surface")
        
        # 地板物理材质（与 Sim 版本一致）
        ground_phys_mat = UsdShade.Material.Define(stage, Sdf.Path("/Materials/GroundPhysMat"))
        ground_phys = UsdPhysics.MaterialAPI.Apply(ground_phys_mat.GetPrim())
        ground_phys.CreateStaticFrictionAttr(0.6)
        ground_phys.CreateDynamicFrictionAttr(0.5)
        ground_phys.CreateRestitutionAttr(0.75)
        
        # ==================== 创建地面平面（物理）====================
        # 使用 PhysicsSchemaTools 创建地面（与 Sim 版本一致）
        from pxr import PhysicsSchemaTools
        PhysicsSchemaTools.addGroundPlane(
            stage, "/groundPlane", 
            "Z", 
            1500,  # 大小
            Gf.Vec3f(0, 0, 0),  # 位置
            Gf.Vec3f(0.15, 0.15, 0.15)  # 颜色
        )
        
        # 绑定物理材质到地面
        ground_geom = stage.GetPrimAtPath("/groundPlane/geom")
        if ground_geom.IsValid():
            phys_binding = UsdShade.MaterialBindingAPI.Apply(ground_geom)
            phys_binding.Bind(ground_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        
        # ==================== 创建排球场地面（视觉）====================
        court = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Court"))
        court.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(court)
        xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.005))
        xf.AddScaleOp().Set(Gf.Vec3f(court_length, court_width, 0.01))
        UsdShade.MaterialBindingAPI(court).Bind(court_mat)
        
        # 添加碰撞
        UsdPhysics.CollisionAPI.Apply(court.GetPrim())
        phys_binding = UsdShade.MaterialBindingAPI.Apply(court.GetPrim())
        phys_binding.Bind(ground_phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        
        # ==================== 创建边线（与 Sim 版本一致）====================
        line_width = 0.05
        lw = line_width
        ox, oy, oz = 0.0, 0.0, 0.0
        
        lines = [
            ("LineN", (ox, oy + court_width/2, oz + 0.015), (court_length, lw, 0.005)),
            ("LineS", (ox, oy - court_width/2, oz + 0.015), (court_length, lw, 0.005)),
            ("LineE", (ox + court_length/2, oy, oz + 0.015), (lw, court_width, 0.005)),
            ("LineW", (ox - court_length/2, oy, oz + 0.015), (lw, court_width, 0.005)),
            ("LineMid", (ox, oy, oz + 0.015), (lw, court_width, 0.005)),
        ]
        
        for name, pos, scale in lines:
            line = UsdGeom.Cube.Define(stage, Sdf.Path(f"/World/{name}"))
            line.CreateSizeAttr(1.0)
            xf = UsdGeom.Xformable(line)
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
            xf.AddScaleOp().Set(Gf.Vec3f(*scale))
            UsdShade.MaterialBindingAPI(line).Bind(line_mat)
        
        # ==================== 创建球网（与 Sim 版本一致）====================
        net_thickness = 0.05
        
        # 球网主体
        net = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Net/Mesh"))
        net.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(net)
        xf.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz + net_height / 2))
        xf.AddScaleOp().Set(Gf.Vec3f(net_thickness, court_width + 1.0, net_height))
        UsdShade.MaterialBindingAPI(net).Bind(net_mat)
        
        # 启用球网碰撞（与配置一致）
        if hasattr(self.cfg, 'enable_net_collision') and self.cfg.enable_net_collision:
            UsdPhysics.CollisionAPI.Apply(net.GetPrim())
        
        # 顶带（白色边）
        top_band = UsdGeom.Cube.Define(stage, Sdf.Path("/World/Net/TopBand"))
        top_band.CreateSizeAttr(1.0)
        xf2 = UsdGeom.Xformable(top_band)
        xf2.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz + net_height + 0.035))
        xf2.AddScaleOp().Set(Gf.Vec3f(net_thickness + 0.02, court_width + 1.0, 0.07))
        UsdShade.MaterialBindingAPI(top_band).Bind(top_band_mat)
        
        # 网柱
        pole_radius = 0.04
        pole_height = net_height + 0.3
        
        for i, y_offset in enumerate([-(court_width/2 + 0.5), (court_width/2 + 0.5)]):
            pole = UsdGeom.Cylinder.Define(stage, Sdf.Path(f"/World/Net/Pole{i}"))
            pole.CreateRadiusAttr(pole_radius)
            pole.CreateHeightAttr(pole_height)
            pole.CreateAxisAttr("Z")
            xf = UsdGeom.Xformable(pole)
            xf.AddTranslateOp().Set(Gf.Vec3d(ox, oy + y_offset, oz + pole_height/2))
            UsdShade.MaterialBindingAPI(pole).Bind(pole_mat)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._apply_aerodynamics()

    def _apply_action(self):
        target_vel = self._actions * self.cfg.action_scale
        
        robot_pos = self.robot.data.root_pos_w
        robot_quat = self.robot.data.root_quat_w
        robot_vel = self.robot.data.root_lin_vel_w
        robot_ang_vel = self.robot.data.root_ang_vel_w
        
        new_vel = torch.zeros_like(robot_vel)
        new_vel[:, 0] = target_vel[:, 0]
        new_vel[:, 1] = target_vel[:, 1]
        new_vel[:, 2] = robot_vel[:, 2]
        
        self.robot.write_root_velocity_to_sim(
            torch.cat([new_vel, robot_ang_vel], dim=-1)
        )

    def _apply_aerodynamics(self):
        """空气动力学力 (简化版)"""
        ball_vel = self.ball.data.root_lin_vel_w
        ball_ang_vel = self.ball.data.root_ang_vel_w
        
        speed = torch.norm(ball_vel, dim=-1, keepdim=True)
        speed_safe = torch.clamp(speed, min=1e-6)
        vel_dir = ball_vel / speed_safe
        
        Cd = 0.35
        q = 0.5 * 1.225 * speed ** 2 * 0.03464
        F_drag = -Cd * q * vel_dir
        
        spin_rate = torch.norm(ball_ang_vel, dim=-1, keepdim=True)
        Cl = 0.2 * ((spin_rate * 0.105) / speed_safe).clamp(max=0.25)
        
        spin_safe = torch.clamp(spin_rate, min=1e-6)
        omega_hat = ball_ang_vel / spin_safe
        magnus_dir = torch.cross(omega_hat, vel_dir, dim=-1)
        F_magnus = 0.5 * 1.225 * 0.03464 * Cl * speed ** 2 * magnus_dir
        
        F_total = F_drag + F_magnus
        mask = (speed > 0.1).squeeze(-1)
        F_total = torch.where(mask.unsqueeze(-1), F_total, torch.zeros_like(F_total))
        
        self.ball.set_external_force_and_torque(F_total.unsqueeze(1), torch.zeros_like(F_total.unsqueeze(1)))

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        ball_pos = self.ball.data.root_pos_w
        ball_vel = self.ball.data.root_lin_vel_w
        robot_pos = self.robot.data.root_pos_w
        robot_vel = self.robot.data.root_lin_vel_w
        robot_ang_vel = self.robot.data.root_ang_vel_w
        robot_quat = self.robot.data.root_quat_w
        
        pred_land = self._predict_landing_point(ball_pos, ball_vel)
        self.predicted_landing = pred_land
        
        t_land = self._calc_time_to_landing(ball_pos, ball_vel)
        self.time_to_landing = t_land
        
        # 计算机器人偏航角 Wz (从四元数)
        robot_yaw = torch.atan2(
            2.0 * (robot_quat[:, 3] * robot_quat[:, 2] + robot_quat[:, 0] * robot_quat[:, 1]),
            1.0 - 2.0 * (robot_quat[:, 1] ** 2 + robot_quat[:, 2] ** 2)
        )
        
        # 计算偏航角速度 Vyaw (Z轴角速度)
        robot_Vyaw = robot_ang_vel[:, 2]
        
        # 转换为场地坐标系 (左后角为原点)
        # 世界坐标: 中心在 (0,0)，场地坐标: 左后角在 (0,0)
        robot_x_court = robot_pos[:, 0] + self.cfg.court_length / 2
        robot_y_court = robot_pos[:, 1] + self.cfg.court_width / 2
        
        # 9维观察空间
        obs = torch.cat([
            pred_land[:, 0:1] - robot_pos[:, 0:1],  # [0] rel_x
            pred_land[:, 1:2] - robot_pos[:, 1:2],  # [1] rel_y
            t_land.unsqueeze(-1),                    # [2] time_to_land
            robot_x_court.unsqueeze(-1),             # [3] robot_x (场地坐标)
            robot_y_court.unsqueeze(-1),             # [4] robot_y (场地坐标)
            robot_yaw.unsqueeze(-1),                 # [5] robot_yaw (Wz)
            robot_vel[:, 0:1],                       # [6] robot_vx
            robot_vel[:, 1:2],                       # [7] robot_vy
            robot_Vyaw.unsqueeze(-1),                # [8] robot_Vyaw
        ], dim=-1)
        
        self._prev_dist = torch.norm(
            pred_land[:, :2] - robot_pos[:, :2], dim=-1
        )
        
        return {"policy": obs}

    def _predict_landing_point(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        g = 9.81
        z0 = pos[:, 2]
        vz = vel[:, 2]
        
        discriminant = vz ** 2 + 2 * g * z0
        discriminant = torch.clamp(discriminant, min=0.0)
        t_land = (vz + torch.sqrt(discriminant)) / g
        t_land = torch.clamp(t_land, min=0.0, max=10.0)
        
        land_x = pos[:, 0] + vel[:, 0] * t_land
        land_y = pos[:, 1] + vel[:, 1] * t_land
        
        return torch.stack([land_x, land_y], dim=-1)

    def _calc_time_to_landing(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        g = 9.81
        z0 = pos[:, 2]
        vz = vel[:, 2]
        
        discriminant = vz ** 2 + 2 * g * z0
        t_land = torch.where(
            discriminant >= 0,
            (vz + torch.sqrt(torch.clamp(discriminant, min=0.0))) / g,
            torch.zeros_like(z0)
        )
        return torch.clamp(t_land, min=0.0, max=10.0)

    def _get_rewards(self) -> torch.Tensor:
        ball_pos = self.ball.data.root_pos_w
        ball_vel = self.ball.data.root_lin_vel_w
        robot_pos = self.robot.data.root_pos_w
        robot_vel = self.robot.data.root_lin_vel_w
        
        pred_land = self.predicted_landing
        dist_to_land = torch.norm(pred_land - robot_pos[:, :2], dim=-1)
        dist_to_ball = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=-1)
        ball_z = ball_pos[:, 2]
        robot_z = robot_pos[:, 2]
        
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # ========== 终止条件检测 ==========
        catch_height_thresh = robot_z + self.cfg.catch_height_margin
        grounded = ball_z < self.cfg.ground_threshold
        
        # 接球成功：球在接球高度范围内且在接球半径内
        caught = (ball_z <= catch_height_thresh) & (ball_z > self.cfg.ground_threshold) & (dist_to_ball < self.cfg.catch_radius)
        
        # 接球失败：球触地但未接住
        missed = grounded & ~caught
        
        # ========== 非线性指数距离奖励 ==========
        # 公式: r = r_max * exp(-scale * distance)
        # 越接近落点，奖励增长越快（非线性）
        dist_reward = self.cfg.rew_dist_max * torch.exp(
            -self.cfg.rew_dist_exp_scale * dist_to_land
        )
        
        # ========== 接近奖励 ==========
        # 计算向落点移动的进度
        approach_reward = (self._prev_dist - dist_to_land) * self.cfg.rew_approach_scale
        approach_reward = torch.clamp(approach_reward, -1.0, 1.0)
        
        # ========== 时间惩罚 ==========
        time_penalty = torch.full_like(reward, self.cfg.rew_time_penalty)
        
        # ========== 出界惩罚 ==========
        robot_out = (
            (robot_pos[:, 0] < self.cfg.robot_bounds_x[0]) |
            (robot_pos[:, 0] > self.cfg.robot_bounds_x[1]) |
            (robot_pos[:, 1] < self.cfg.robot_bounds_y[0]) |
            (robot_pos[:, 1] > self.cfg.robot_bounds_y[1])
        )
        boundary_penalty = torch.where(
            robot_out, 
            torch.full_like(reward, self.cfg.rew_boundary), 
            torch.zeros_like(reward)
        )
        
        # ========== 碰网惩罚 ==========
        net_collision = self._check_net_collision(robot_pos)
        net_penalty = torch.where(
            net_collision,
            torch.full_like(reward, self.cfg.rew_net_collision),
            torch.zeros_like(reward)
        )
        
        # ========== 车球碰撞惩罚（非接球情况）==========
        robot_ball_collision = self._check_robot_ball_collision(
            robot_pos, ball_pos, dist_to_ball, caught
        )
        collision_penalty = torch.where(
            robot_ball_collision,
            torch.full_like(reward, self.cfg.rew_robot_ball_collision),
            torch.zeros_like(reward)
        )
        
        # ========== 汇总持续奖励 ==========
        ongoing_reward = (
            dist_reward + 
            approach_reward + 
            time_penalty + 
            boundary_penalty + 
            net_penalty + 
            collision_penalty
        )
        
        # ========== 终止奖励/惩罚 ==========
        # 成功接球
        reward = torch.where(caught, torch.full_like(reward, self.cfg.rew_catch), reward)
        
        # 未接住（球触地）
        miss_penalty = self.cfg.rew_miss_base + self.cfg.rew_miss_dist_scale * dist_to_ball
        reward = torch.where(missed, miss_penalty, reward)
        
        # 持续回合奖励
        ongoing_mask = ~caught & ~missed
        reward = torch.where(ongoing_mask, ongoing_reward, reward)
        
        return reward

    def _check_net_collision(self, robot_pos: torch.Tensor) -> torch.Tensor:
        """检测机器人是否与球网碰撞"""
        # 球网在 x=0，宽度为 court_width + 1.0，高度 net_height
        net_x_range = 0.3  # 网厚度 + 机器人半径
        net_y_range = (self.cfg.court_width / 2) + 0.5
        net_height = self.cfg.net_height
        
        # 检查机器人在网区域内
        net_collision = (
            (robot_pos[:, 0].abs() < net_x_range) &
            (robot_pos[:, 1].abs() < net_y_range) &
            (robot_pos[:, 2] < net_height)
        )
        return net_collision

    def _check_robot_ball_collision(
        self, 
        robot_pos: torch.Tensor, 
        ball_pos: torch.Tensor,
        dist_to_ball: torch.Tensor,
        caught: torch.Tensor
    ) -> torch.Tensor:
        """检测机器人与球的非接球碰撞"""
        # 机器人半径约 0.3m，球半径 0.105m
        robot_radius = 0.3
        ball_radius = 0.105
        collision_dist = robot_radius + ball_radius
        
        # 碰撞检测：距离小于碰撞距离且不是成功接球
        collision = (dist_to_ball < collision_dist) & ~caught
        return collision

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ball_pos = self.ball.data.root_pos_w
        robot_pos = self.robot.data.root_pos_w
        
        ball_z = ball_pos[:, 2]
        robot_z = robot_pos[:, 2]
        
        dist_to_ball = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=-1)
        
        caught = (ball_z <= robot_z + self.cfg.catch_height_margin) & (ball_z > self.cfg.ground_threshold) & (dist_to_ball < self.cfg.catch_radius)
        grounded = ball_z < self.cfg.ground_threshold
        
        ball_out = (
            (ball_pos[:, 0].abs() > self.cfg.court_length * 1.2) |
            (ball_pos[:, 1].abs() > self.cfg.court_width * 1.5) |
            (ball_z > 15.0)
        )
        
        robot_out = (
            (robot_pos[:, 0] < self.cfg.robot_bounds_x[0]) |
            (robot_pos[:, 0] > self.cfg.robot_bounds_x[1]) |
            (robot_pos[:, 1] < self.cfg.robot_bounds_y[0]) |
            (robot_pos[:, 1] > self.cfg.robot_bounds_y[1])
        )
        
        # 添加碰网检测作为终止条件
        net_collision = self._check_net_collision(robot_pos)
        
        terminated = caught | grounded | ball_out | robot_out | net_collision
        truncated = self.episode_length_buf >= self.max_episode_length
        
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)
        
        num_reset = len(env_ids)
        
        # ========== 固定发球参数（诱导学习）==========
        # 使用配置中的固定发球参数
        serve_x = self.cfg.ball_cfg.init_state.pos[0]  # SERVE_X = -10.0
        serve_y = 0.0  # 固定 Y=0（场地中心）
        serve_z = self.cfg.ball_cfg.init_state.pos[2]  # SERVE_HEIGHT = 3.5
        
        # 从配置读取发球速度（与原来一致）
        serve_speed_x = self.cfg.ball_cfg.init_state.lin_vel[0]  # SERVE_SPEED = 15.0
        serve_speed_z = self.cfg.ball_cfg.init_state.lin_vel[2]  # SERVE_SPEED * sin(20°)
        
        # 固定发球位置
        ball_pos = torch.zeros(num_reset, 3, device=self.device)
        ball_pos[:, 0] = serve_x
        ball_pos[:, 1] = serve_y
        ball_pos[:, 2] = serve_z
        
        # 固定发球速度（与配置一致）
        ball_vel = torch.zeros(num_reset, 3, device=self.device)
        ball_vel[:, 0] = serve_speed_x  # x方向速度
        ball_vel[:, 2] = serve_speed_z  # z方向速度
        
        # 添加随机扰动（模拟深度相机噪声，小幅度）
        # 通过配置控制是否添加扰动
        if hasattr(self.cfg, 'enable_ball_perturbation') and self.cfg.enable_ball_perturbation:
            vel_noise_std = 0.1  # 速度扰动标准差
            ball_vel[:, 0] += torch.empty(num_reset, device=self.device).normal_(0, vel_noise_std)
            ball_vel[:, 2] += torch.empty(num_reset, device=self.device).normal_(0, vel_noise_std)
        
        # 无旋转
        ball_ang_vel = torch.zeros(num_reset, 3, device=self.device)
        
        # 应用排球初始状态
        default_ball_state = self.ball.data.default_root_state[env_ids]
        default_ball_state[:, :3] = ball_pos
        default_ball_state[:, 7:10] = ball_vel
        default_ball_state[:, 10:13] = ball_ang_vel
        
        self.ball.write_root_state_to_sim(default_ball_state, env_ids)
        
        # ========== 固定机器人初始位置（接发球区域）==========
        # 所有环境从相同位置开始（世界坐标：X=5m, Y=0m）
        robot_x = torch.full((num_reset,), 5.0, device=self.device)
        robot_y = torch.full((num_reset,), 0.0, device=self.device)  # 修正：原来是4.5
        
        default_robot_state = self.robot.data.default_root_state[env_ids]
        default_robot_state[:, 0] = robot_x
        default_robot_state[:, 1] = robot_y
        default_robot_state[:, 2] = self.cfg.robot_cfg.init_state.pos[2]
        # 设置朝向：正对球网（X轴正方向）
        default_robot_state[:, 3] = 1.0   # w
        default_robot_state[:, 4] = 0.0   # x
        default_robot_state[:, 5] = 0.0   # y
        default_robot_state[:, 6] = 0.0   # z
        default_robot_state[:, 7:13] = 0.0  # 速度为0
        
        self.robot.write_root_state_to_sim(default_robot_state, env_ids)
        
        # 重置关节状态
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        # 重置距离跟踪
        self._prev_dist[env_ids] = 0.0


import gymnasium as gym

gym.register(
    id="Volleyball-Catch-Direct-v0",
    entry_point="volleyball_catch_env:VolleyballCatchEnv",
    kwargs={"cfg": VolleyballCatchEnvCfg()},
    disable_env_checker=True,
)
