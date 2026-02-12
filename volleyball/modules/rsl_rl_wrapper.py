# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
RSL-RL VecEnv 包装器

将 Gymnasium 环境包装成 RSL-RL 兼容的 VecEnv 接口
"""

import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv


class GymToRslRlWrapper(VecEnv):
    """
    将单个 Gymnasium 环境包装成 RSL-RL 的 VecEnv 接口
    
    RSL-RL 期望的接口:
    - num_envs: 环境数量 (这里是 1)
    - num_actions: 动作维度
    - max_episode_length: 最大 episode 长度
    - episode_length_buf: 当前 episode 步数张量
    - device: PyTorch 设备
    - cfg: 配置对象
    - get_observations(): 返回 TensorDict
    - step(actions): 返回 (TensorDict, rewards, dones, extras)
    """
    
    def __init__(
        self, 
        gym_env, 
        device: str = "cuda:0",
        max_episode_length: int = 120,  # 6s at 20Hz
        clip_actions: bool = True,
    ):
        """
        Args:
            gym_env: Gymnasium 环境实例
            device: PyTorch 设备
            max_episode_length: 最大 episode 步数
            clip_actions: 是否裁剪动作
        """
        self._env = gym_env
        self._device = device
        self._max_episode_length = max_episode_length
        self._clip_actions = clip_actions
        
        # VecEnv 必需属性
        self.num_envs = 1
        self.num_actions = gym_env.action_space.shape[0]
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=device)
        self.device = device
        
        # 配置对象 (RSL-RL logger 需要)
        self.cfg = {
            "env_name": "VolleyballCatch",
            "num_envs": 1,
            "num_actions": self.num_actions,
            "num_obs": gym_env.observation_space.shape[0],
        }
        
        # 观察维度
        self._num_obs = gym_env.observation_space.shape[0]
        
        # 当前观察缓存
        self._obs_buf = torch.zeros(1, self._num_obs, dtype=torch.float32, device=device)
        
        # Episode 统计
        self._episode_reward = 0.0
        self._episode_length = 0
        
        # RSL-RL 需要的额外属性
        self.step_dt = 1.0 / 20.0  # 控制频率 20Hz
        
    @property
    def unwrapped(self):
        """返回自身，RSL-RL 需要此属性"""
        return self
        
    @property
    def obs_dim(self) -> int:
        return self._num_obs
    
    def get_observations(self) -> TensorDict:
        """返回当前观察的 TensorDict"""
        return TensorDict({
            "policy": self._obs_buf.clone(),
        }, batch_size=[self.num_envs], device=self.device)
    
    def reset(self) -> TensorDict:
        """重置环境"""
        obs, info = self._env.reset()
        self._obs_buf[0] = torch.from_numpy(obs).float().to(self.device)
        self.episode_length_buf[0] = 0
        self._episode_reward = 0.0
        self._episode_length = 0
        return self.get_observations()
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """
        执行一步环境交互
        
        Args:
            actions: 动作张量 (num_envs, num_actions)
            
        Returns:
            observations: TensorDict
            rewards: 奖励张量 (num_envs,)
            dones: 终止标志 (num_envs,)
            extras: 额外信息字典
        """
        # 裁剪动作
        if self._clip_actions:
            actions = torch.clamp(actions, -1.0, 1.0)
        
        # 转换为 numpy
        action_np = actions[0].cpu().numpy()
        
        # 执行一步
        obs, reward, terminated, truncated, info = self._env.step(action_np)
        done = terminated or truncated
        
        # 更新观察缓存
        self._obs_buf[0] = torch.from_numpy(obs).float().to(self.device)
        
        # 更新 episode 计数
        self.episode_length_buf[0] += 1
        self._episode_reward += reward
        self._episode_length += 1
        
        # 创建返回张量
        rewards = torch.tensor([reward], dtype=torch.float32, device=self.device)
        dones = torch.tensor([done], dtype=torch.bool, device=self.device)
        
        # 判断是否是 timeout
        is_timeout = truncated and not terminated
        
        # 构建 extras
        extras = {
            "time_outs": torch.tensor([is_timeout], dtype=torch.bool, device=self.device),
            "log": {},
        }
        
        # 如果 episode 结束，添加日志
        if done:
            extras["log"]["/episode/reward"] = self._episode_reward
            extras["log"]["/episode/length"] = float(self._episode_length)
            result = info.get("result", "unknown")
            extras["log"]["/episode/catch"] = 1.0 if result == "catch" else 0.0
            
            # 自动重置
            self.reset()
        
        return self.get_observations(), rewards, dones, extras
    
    def close(self):
        """关闭环境"""
        self._env.close()
    
    def seed(self, seed: int):
        """设置随机种子"""
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


class MultiEnvWrapper(VecEnv):
    """
    多环境包装器 - 用于并行训练
    
    注意: 当前 Isaac Sim 单进程限制，实际只能运行 1 个环境
    未来可以扩展为多进程并行
    """
    
    def __init__(
        self,
        env_fn,
        num_envs: int = 1,
        device: str = "cuda:0",
        max_episode_length: int = 120,
    ):
        """
        Args:
            env_fn: 创建 Gymnasium 环境的函数
            num_envs: 环境数量
            device: PyTorch 设备
            max_episode_length: 最大 episode 长度
        """
        # 创建环境
        self._envs = [GymToRslRlWrapper(env_fn(), device, max_episode_length) for _ in range(num_envs)]
        self._main_env = self._envs[0]
        
        # VecEnv 属性
        self.num_envs = num_envs
        self.num_actions = self._main_env.num_actions
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.device = device
        self.cfg = self._main_env.cfg
        self.cfg["num_envs"] = num_envs
        
        self._num_obs = self._main_env._num_obs
        self._obs_buf = torch.zeros(num_envs, self._num_obs, dtype=torch.float32, device=device)
        
    def get_observations(self) -> TensorDict:
        for i, env in enumerate(self._envs):
            self._obs_buf[i] = env._obs_buf[0]
        return TensorDict({
            "policy": self._obs_buf.clone(),
        }, batch_size=[self.num_envs], device=self.device)
    
    def reset(self) -> TensorDict:
        for i, env in enumerate(self._envs):
            env.reset()
            self._obs_buf[i] = env._obs_buf[0]
            self.episode_length_buf[i] = 0
        return self.get_observations()
    
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_outs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        log_dict = {}
        
        for i, env in enumerate(self._envs):
            obs, r, d, extras = env.step(actions[i:i+1])
            self._obs_buf[i] = env._obs_buf[0]
            self.episode_length_buf[i] = env.episode_length_buf[0]
            rewards[i] = r[0]
            dones[i] = d[0]
            time_outs[i] = extras["time_outs"][0]
            
            # 合并日志
            for k, v in extras["log"].items():
                if k not in log_dict:
                    log_dict[k] = []
                log_dict[k].append(v)
        
        # 对日志取平均
        for k in list(log_dict.keys()):
            if len(log_dict[k]) > 0:
                log_dict[k] = sum(log_dict[k]) / len(log_dict[k])
        
        extras = {
            "time_outs": time_outs,
            "log": log_dict,
        }
        
        return self.get_observations(), rewards, dones, extras
    
    def close(self):
        for env in self._envs:
            env.close()
    
    def seed(self, seed: int):
        for i, env in enumerate(self._envs):
            env.seed(seed + i)
