# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球环境评估/播放脚本

加载训练好的模型并可视化运行
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Volleyball Catch with trained model")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")
parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
parser.add_argument("--num_steps", type=int, default=1000, help="运行步数")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.headless = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

from volleyball_catch_cfg import VolleyballCatchEnvCfg
import volleyball_catch_env


class IsaacLabVecEnvWrapper:
    def __init__(self, gym_env, num_envs: int, device: str):
        self._env = gym_env
        self._num_envs = num_envs
        self._device = device
        
        self.num_envs = num_envs
        self.num_obs = gym_env.observation_space.shape[1]
        self.num_actions = gym_env.action_space.shape[1]
        self.max_episode_length = getattr(gym_env, 'max_episode_length', getattr(gym_env, '_max_episode_length', 120))
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {
            "env_name": "VolleyballCatch",
            "num_envs": num_envs,
            "num_actions": self.num_actions,
            "num_obs": self.num_obs,
        }
    
    def get_observations(self) -> TensorDict:
        obs = self._env._get_observations()
        if isinstance(obs, TensorDict):
            return obs
        if isinstance(obs, dict):
            return TensorDict(obs, batch_size=self._num_envs)
        return TensorDict({"policy": obs}, batch_size=self._num_envs)
    
    def reset(self):
        obs, _ = self._env.reset()
        self.episode_length_buf[:] = 0
        if isinstance(obs, dict):
            return TensorDict(obs, batch_size=self._num_envs)
        return TensorDict({"policy": obs}, batch_size=self._num_envs)
    
    def step(self, actions: torch.Tensor):
        obs, rewards, terminated, truncated, info = self._env.step(actions)
        dones = terminated | truncated
        self.episode_length_buf += 1
        self.episode_length_buf[dones] = 0
        extras = {}
        if isinstance(info, dict) and "episode" in info:
            extras["episode"] = info["episode"]
        if isinstance(obs, TensorDict):
            return obs, rewards, dones, extras
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=self._num_envs)
        return TensorDict({"policy": obs}, batch_size=self._num_envs), rewards, dones, extras


def main():
    print("=" * 60)
    print("排球接球环境评估")
    print("=" * 60)
    
    env_cfg = VolleyballCatchEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    print(f"\n创建 {args.num_envs} 个并行环境...")
    
    env = gym.make("Volleyball-Catch-Direct-v0", cfg=env_cfg)
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    device = "cuda:0"
    
    vec_env = IsaacLabVecEnvWrapper(env, args.num_envs, device)
    
    train_cfg = {
        "num_steps_per_env": 24,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
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
        "algorithm": {"class_name": "PPO"},
    }
    
    print("\n加载模型...")
    runner = OnPolicyRunner(
        env=vec_env,
        train_cfg=train_cfg,
        log_dir=os.path.dirname(args.checkpoint),
        device=device,
    )
    runner.load(args.checkpoint, strict=False)
    
    policy = runner.get_inference_policy(device=device)
    policy.eval()
    
    print("\n开始评估...")
    total_rewards = torch.zeros(args.num_envs, device=device)
    episode_count = torch.zeros(args.num_envs, device=device)
    
    obs = vec_env.reset()
    
    for step in range(args.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        
        obs, rewards, dones, extras = vec_env.step(actions)
        
        total_rewards += rewards
        
        if torch.any(dones):
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            episode_count[done_indices] += 1
            
            for idx in done_indices:
                avg = float(total_rewards[idx] / max(1, int(episode_count[idx])))
                print(f"  环境 {int(idx)}: Episode 完成, 奖励 = {float(rewards[idx]):.2f}, 平均 = {avg:.2f}")
        
        if step > 0 and step % 100 == 0:
            completed = int(episode_count.sum())
            print(f"步 {step}/{args.num_steps}: 完成 {completed} 个 episodes")
    
    print("\n" + "=" * 60)
    print("评估结果:")
    print(f"  总步数: {args.num_steps}")
    print(f"  完成 episodes: {int(episode_count.sum())}")
    print(f"  平均奖励: {float(total_rewards.mean()):.2f}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
