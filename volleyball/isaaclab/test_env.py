# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球环境测试脚本

验证环境是否可以正确创建和运行
"""

from __future__ import annotations

import argparse

# =====================================================================
# Isaac Lab / Isaac Sim 启动
# =====================================================================

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Volleyball Catch Environment")
parser.add_argument("--num_envs", type=int, default=4, help="并行环境数量")

# AppLauncher 参数 (包括 --headless)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 启动 Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# =====================================================================
# 导入依赖 (必须在 AppLauncher 后)
# =====================================================================

import torch
import gymnasium as gym

# 导入环境
from volleyball_catch_cfg import VolleyballCatchEnvCfg
import volleyball_catch_env  # 触发 gym.register


def main():
    """测试环境"""
    
    print("=" * 60)
    print("排球接球环境测试")
    print("=" * 60)
    
    # 创建环境配置
    env_cfg = VolleyballCatchEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    
    print(f"\n创建 {args.num_envs} 个并行环境...")
    
    # 创建环境
    env = gym.make("Volleyball-Catch-Direct-v0", cfg=env_cfg)
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 获取底层环境
    unwrapped_env = env.unwrapped
    device = unwrapped_env.device
    print(f"设备: {device}")
    
    # 重置环境
    print("\n重置环境...")
    obs, info = env.reset()
    print(f"观察形状: {obs['policy'].shape}")
    
    # 运行几步
    print("\n运行 100 步测试...")
    for i in range(100):
        # 随机动作
        actions = torch.rand(args.num_envs, 2, device=device) * 2 - 1
        
        # 执行动作
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if i % 20 == 0:
            print(f"  步 {i}: 平均奖励 = {rewards.mean().item():.3f}")
        
        # 检查是否有环境需要重置
        if terminated.any() or truncated.any():
            done_count = (terminated | truncated).sum().item()
            print(f"  步 {i}: {done_count} 个环境完成")
    
    print("\n测试完成!")
    
    # 关闭
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
