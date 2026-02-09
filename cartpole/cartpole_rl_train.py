# Copyright (c) 2025 Your Name
# SPDX-License-Identifier: MIT
"""
强化学习倒立摆示例 - 使用 Isaac Lab + SKRL PPO

本示例演示如何：
1. 使用 Isaac Lab 的 Cartpole 环境（GPU 并行仿真）
2. 使用 SKRL 库的 PPO 算法进行训练
3. 无需 PD 控制器，策略网络直接学习平衡

运行方法：
    C:\IsaacSim\python.bat cartpole_rl_train.py --num_envs 1024
    
    # 使用更少环境（调试用）
    C:\IsaacSim\python.bat cartpole_rl_train.py --num_envs 64 --headless

参数说明：
    --num_envs      并行环境数量（默认 1024，GPU 加速）
    --headless      无头模式（不渲染，训练更快）
    --max_iterations 最大训练迭代次数（默认 500）
"""

import argparse
import os

# -- 首先启动 Isaac Sim --
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="使用 RL 训练倒立摆")
parser.add_argument("--num_envs", type=int, default=1024, help="并行环境数量")
parser.add_argument("--max_iterations", type=int, default=500, help="最大训练迭代次数")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动仿真应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- 接下来导入其他模块 --
import os

import torch
import torch.nn as nn

# Isaac Lab 环境
import gymnasium as gym

# 注册 Isaac Lab 的 Gym 环境
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Isaac Lab 的 SKRL 包装器
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# SKRL 强化学习
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# =====================
# 定义策略网络（Actor）
# =====================
class Policy(GaussianMixin, Model):
    """高斯策略网络 - 输出连续动作的均值和标准差"""
    
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        
        # 简单的 MLP 网络
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, self.num_actions),
        )
        # 可学习的对数标准差
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
    
    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


# =====================
# 定义价值网络（Critic）
# =====================
class Value(DeterministicMixin, Model):
    """价值网络 - 估计状态价值 V(s)"""
    
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        
        # 简单的 MLP 网络
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 1),
        )
    
    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # =====================
    # 1. 创建环境
    # =====================
    task_name = "Isaac-Cartpole-Direct-v0"
    print(f"[INFO] 创建 Cartpole 环境，并行数量: {args_cli.num_envs}")
    
    # 使用 Isaac Lab 的配置加载工具
    env_cfg = parse_env_cfg(
        task_name=task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    
    # 创建环境（必须传入 cfg 参数）
    env = gym.make(task_name, cfg=env_cfg)
    
    # 使用 Isaac Lab 的 SKRL 包装器
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    
    device = env.device
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 观测空间: {env.observation_space}")
    print(f"[INFO] 动作空间: {env.action_space}")
    
    # =====================
    # 2. 创建模型
    # =====================
    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, device)
    models["value"] = Value(env.observation_space, env.action_space, device)
    
    # =====================
    # 3. 配置 PPO
    # =====================
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 16                    # 每次收集的 rollout 步数
    cfg["learning_epochs"] = 8              # 每次更新的 epoch 数
    cfg["mini_batches"] = 4                 # mini-batch 数量
    cfg["discount_factor"] = 0.99           # 折扣因子
    cfg["lambda"] = 0.95                    # GAE lambda
    cfg["learning_rate"] = 3e-4             # 学习率
    cfg["grad_norm_clip"] = 1.0             # 梯度裁剪
    cfg["ratio_clip"] = 0.2                 # PPO clip 参数
    cfg["value_clip"] = 0.2                 # 价值函数 clip
    cfg["state_preprocessor"] = None
    cfg["value_preprocessor"] = None
    cfg["experiment"]["write_interval"] = 50
    cfg["experiment"]["checkpoint_interval"] = 500
    
    # 设置日志目录
    log_dir = os.path.join(os.path.dirname(__file__), "logs", "cartpole_ppo")
    cfg["experiment"]["directory"] = log_dir
    
    # 创建 Memory
    memory = RandomMemory(memory_size=cfg["rollouts"], num_envs=env.num_envs, device=device)
    
    # 创建 Agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    
    # =====================
    # 4. 训练
    # =====================
    print(f"[INFO] 开始训练，最大迭代次数: {args_cli.max_iterations}")
    print(f"[INFO] 日志目录: {log_dir}")
    print("[INFO] 训练中...")
    print("-" * 50)
    
    # 创建 Trainer
    trainer = SequentialTrainer(
        env=env,
        agents=agent,
        cfg={
            "timesteps": args_cli.max_iterations * cfg["rollouts"] * env.num_envs,
            "headless": args_cli.headless,
        }
    )
    
    # 开始训练
    trainer.train()
    
    print("-" * 50)
    print("[INFO] 训练完成!")
    print(f"[INFO] 模型已保存到: {log_dir}")
    
    # 清理
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
