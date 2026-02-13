# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球 Isaac Lab 环境配置
"""

from __future__ import annotations

import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.sim.converters import UrdfConverterCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


# =====================================================================
# 物理常量
# =====================================================================

BALL_DIAMETER = 0.210
BALL_RADIUS = BALL_DIAMETER / 2
BALL_MASS = 0.270

ROBOT_HEIGHT = 0.15
ROBOT_CATCH_RADIUS = 0.40
ROBOT_MAX_SPEED = 3.0

COURT_LENGTH = 18.0
COURT_WIDTH = 9.0
NET_HEIGHT = 2.43

SERVE_X = -10.0
SERVE_SPEED = 15.0
SERVE_ANGLE_DEG = 20.0
SERVE_HEIGHT = 3.5

ROBOT_BOUNDS_X = (-COURT_LENGTH / 2 + 1.0, COURT_LENGTH / 2 - 1.0)
ROBOT_BOUNDS_Y = (-COURT_WIDTH / 2 + 1.0, COURT_WIDTH / 2 - 1.0)


# =====================================================================
# URDF 机器人配置
# =====================================================================

from isaaclab.actuators import ImplicitActuatorCfg

SWERVE_ROBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=r"C:\Users\Rick\Desktop\isaacsim\URDF\N4\urdf\robot_fixed.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            target_type="velocity",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=10.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(5.0, 0.0, ROBOT_HEIGHT),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "steer": ImplicitActuatorCfg(
            joint_names_expr=["FL", "FB", "RB", "RL"],
            effort_limit=20.0,
            velocity_limit=30.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=["FLW", "FBW", "RBW", "RLW"],
            effort_limit=10.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=5.0,
        ),
    },
)


# =====================================================================
# 排球配置
# =====================================================================

VOLLEYBALL_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Ball",
    spawn=sim_utils.SphereCfg(
        radius=BALL_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
            linear_damping=0.3,
            angular_damping=0.05,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.95, 0.85, 0.30),
            roughness=0.65,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(SERVE_X, 0.0, SERVE_HEIGHT),
        lin_vel=(SERVE_SPEED, 0.0, SERVE_SPEED * math.sin(math.radians(SERVE_ANGLE_DEG))),
    ),
)


# =====================================================================
# 环境配置
# =====================================================================

@configclass
class VolleyballCatchEnvCfg(DirectRLEnvCfg):
    decimation = 6
    episode_length_s = 6.0
    action_space = 2
    observation_space = 9
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=6,
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.6,
            dynamic_friction=0.5,
            restitution=0.75,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.5,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=25.0,
        replicate_physics=True,
    )

    robot_cfg: ArticulationCfg = SWERVE_ROBOT_CFG
    ball_cfg: RigidObjectCfg = VOLLEYBALL_CFG

    action_scale = ROBOT_MAX_SPEED

    rew_catch = 100.0
    rew_miss_base = -10.0
    rew_miss_dist_scale = -5.0
    rew_timeout = -20.0
    rew_dist_exp_scale = 2.0      # 指数距离奖励系数
    rew_dist_max = 3.0            # 最大距离奖励
    rew_approach_scale = 2.0
    rew_time_penalty = -0.02
    rew_boundary = -10.0          # 出界惩罚（增加）
    rew_net_collision = -5.0
    rew_robot_ball_collision = -3.0  # 车球非接球碰撞惩罚

    catch_height_margin = 0.20
    catch_radius = ROBOT_CATCH_RADIUS
    ground_threshold = BALL_RADIUS + 0.02

    court_length = COURT_LENGTH
    court_width = COURT_WIDTH
    net_height = NET_HEIGHT
    robot_bounds_x = ROBOT_BOUNDS_X
    robot_bounds_y = ROBOT_BOUNDS_Y
    
    # 场地碰撞设置
    enable_net_collision = True  # 是否启用球网碰撞（与 Sim 版本一致）
