# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球 Isaac Lab 训练脚本

使用 RSL-RL 训练 GPU 并行环境
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Volleyball Catch RL Training (Isaac Lab)")
parser.add_argument("--num_envs", type=int, default=64, help="并行环境数量")
parser.add_argument("--max_iterations", type=int, default=3000, help="最大训练迭代数")
parser.add_argument("--checkpoint", type=str, default=None, help="加载检查点继续训练")
parser.add_argument("--log_dir", type=str, default="logs/volleyball_catch", help="日志目录")
parser.add_argument("--experiment_name", type=str, default=None, help="实验名称")
parser.add_argument("--config", type=str, default=None, help="RSL-RL 配置文件路径")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import yaml
import gymnasium as gym
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

from volleyball_catch_cfg import VolleyballCatchEnvCfg
import volleyball_catch_env


DEFAULT_CONFIG = {
    "num_steps_per_env": 24,
    "max_iterations": 3000,
    "seed": 42,
    "obs_groups": {
        "actor": ["policy"],
        "critic": ["policy"],
    },
    "save_interval": 100,
    "experiment_name": "volleyball_catch_lab",
    "run_name": "",
    "logger": "tensorboard",
    "wandb_project": "volleyball_catch",
    
    "actor": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "stochastic": True,
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
        "state_dependent_std": False,
    },
    
    "critic": {
        "class_name": "MLPModel",
        "hidden_dims": [256, 256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "stochastic": False,
    },
    
    "algorithm": {
        "class_name": "PPO",
        "optimizer": "adam",
        "learning_rate": 0.001,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "schedule": "adaptive",
        "value_loss_coef": 1.0,
        "clip_param": 0.2,
        "use_clipped_value_loss": True,
        "desired_kl": 0.01,
        "entropy_coef": 0.01,
        "gamma": 0.99,
        "lam": 0.95,
        "max_grad_norm": 1.0,
        "normalize_advantage_per_mini_batch": False,
        "rnd_cfg": None,
    },
}


class IsaacLabVecEnv:
    def __init__(self, env, cfg: dict, device: str = "cuda:0", num_envs: int = 64):
        self.env = env
        self._env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self.cfg = cfg
        self.device = device
        self._num_envs = num_envs
        
        self.num_obs = self._env.observation_space.shape[1]
        self.num_actions = self._env.action_space.shape[1]
        
        self.action_shape = (self.num_actions,)
        
        self.max_episode_length = getattr(self._env, 'max_episode_length', 
                                          getattr(self._env, '_max_episode_length', 1000))
        self.episode_length_buf = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        
        print(f"[IsaacLabVecEnv] num_envs={self._num_envs}, obs={self.num_obs}, act={self.num_actions}")
    
    @property
    def num_envs(self):
        return self._num_envs

    def get_observations(self) -> TensorDict:
        obs = self._env._get_observations()
        if isinstance(obs, dict):
            return TensorDict(obs, batch_size=self._num_envs)
        return TensorDict({"policy": obs}, batch_size=self._num_envs)

    def reset(self):
        obs_dict, _ = self._env.reset()
        self.episode_length_buf[:] = 0
        if isinstance(obs_dict, dict):
            return TensorDict(obs_dict, batch_size=self._num_envs)
        return obs_dict

    def step(self, actions: torch.Tensor):
        obs_dict, rewards, terminated, truncated, info = self._env.step(actions)
        dones = terminated | truncated
        
        self.episode_length_buf += 1
        self.episode_length_buf[dones] = 0
        
        extras = {}
        if "episode" in info:
            extras["episode"] = info["episode"]
        
        if isinstance(obs_dict, dict):
            obs_dict = TensorDict(obs_dict, batch_size=self._num_envs)
        
        return obs_dict, rewards, dones, extras


def main():
    env_cfg = VolleyballCatchEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    env = gym.make("Volleyball-Catch-Direct-v0", cfg=env_cfg)
    
    print(f"[Train] 创建 {args.num_envs} 个并行环境")
    print(f"[Train] 观察空间: {env.observation_space}")
    print(f"[Train] 动作空间: {env.action_space}")
    
    if args.config and os.path.exists(args.config):
        with open(args.config, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
        print(f"[Train] 从 {args.config} 加载配置")
    else:
        train_cfg = DEFAULT_CONFIG.copy()
        print("[Train] 使用默认配置")
    
    train_cfg["max_iterations"] = args.max_iterations
    if args.experiment_name:
        train_cfg["experiment_name"] = args.experiment_name
    
    vec_env = IsaacLabVecEnv(env, train_cfg, device=args.device, num_envs=args.num_envs)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"[Train] 日志目录: {log_dir}")
    
    runner = OnPolicyRunner(
        env=vec_env,
        train_cfg=train_cfg,
        log_dir=log_dir,
        device=args.device,
    )
    
    if args.checkpoint:
        print(f"[Train] 加载检查点: {args.checkpoint}")
        runner.load(args.checkpoint)
    
    print(f"[Train] 开始训练，最大迭代: {args.max_iterations}")
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    print("[Train] 训练完成")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
