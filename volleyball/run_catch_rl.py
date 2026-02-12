# SPDX-FileCopyrightText: Copyright (c) 2025 Rick
# SPDX-License-Identifier: MIT
"""
排球接球 RL 运行脚本

运行方法:
    # 启发式测试 (有头模式)
    C:\\IsaacSim\\python.bat run_catch_rl.py --test
    
    # RSL-RL PPO 训练 (有头模式，查看训练过程)
    C:\\IsaacSim\\python.bat run_catch_rl.py --train
    
    # RSL-RL PPO 训练 (无头模式，更快)
    C:\\IsaacSim\\python.bat run_catch_rl.py --train --headless
    
    # 清理日志和检查点
    python run_catch_rl.py --clean

日志和模型保存位置:
    volleyball/logs/rsl_rl/volleyball_catch/<timestamp>/

包含:
    - 完整排球场地（球网、边线）
    - URDF 舵轮底盘机器人
    - 真实排球空气动力学
    - RSL-RL PPO 训练框架 (ETH RSL)
"""

import argparse
import math
import os
import sys
from datetime import datetime

# 添加模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Volleyball Catch RL")
    parser.add_argument("--test", action="store_true", help="Run heuristic test")
    parser.add_argument("--train", action="store_true", help="Train with RSL-RL PPO")
    parser.add_argument("--clean", action="store_true", help="Clean logs and checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--headless", action="store_true", help="Run without rendering")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes for testing")
    parser.add_argument("--max_iterations", type=int, default=5000, help="Max training iterations")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch device")
    return parser.parse_args()


def heuristic_policy(obs):
    """
    启发式策略：朝着预测落点移动
    """
    rel_x, rel_y = obs[0], obs[1]
    dist = math.sqrt(rel_x**2 + rel_y**2)
    
    if dist < 0.1:
        return [0.0, 0.0]
    
    # 归一化方向
    vx = rel_x / dist
    vy = rel_y / dist
    
    # 根据距离调整速度
    speed_scale = min(1.0, dist / 2.0)
    
    return [vx * speed_scale, vy * speed_scale]


def run_test(args):
    """运行启发式测试"""
    from isaacsim import SimulationApp
    
    # 启动仿真
    simulation_app = SimulationApp({
        "headless": args.headless, 
        "renderer": "RaytracedLighting"
    })
    
    # 导入环境
    from modules.catch_env import VolleyballCatchEnv
    
    print("=" * 70)
    print("  排球接球 RL 测试 (启发式策略)")
    print("=" * 70)
    print(f"  场地: 标准排球场 (18m x 9m) + 球网")
    print(f"  机器人: URDF 舵轮底盘 (N4)")
    print(f"  排球: 完整空气动力学模型")
    print(f"  Episodes: {args.episodes}")
    print("-" * 70)
    
    # 创建环境 (测试模式默认显示详细信息)
    env = VolleyballCatchEnv(
        simulation_app=simulation_app,
        render_mode="human" if not args.headless else None,
        verbose=True
    )
    
    catches = 0
    misses = 0
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action = heuristic_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            if not simulation_app.is_running():
                break
        
        if not simulation_app.is_running():
            break
        
        result = info.get("result", "unknown")
        if result == "catch":
            catches += 1
        elif result == "miss":
            misses += 1
        
        if (ep + 1) % 10 == 0:
            total = catches + misses
            rate = catches / total * 100 if total > 0 else 0
            print(f"\n  Progress: {ep+1}/{args.episodes} | Catch Rate: {rate:.1f}%")
    
    # 最终统计
    stats = env.get_stats()
    print("\n" + "=" * 70)
    print(f"  测试完成")
    print(f"     总 episodes: {stats['episodes']}")
    print(f"     接球: {stats['catches']} | 未接: {stats['misses']}")
    print(f"     接球率: {stats['catch_rate']*100:.1f}%")
    print("=" * 70)
    
    env.close()
    simulation_app.close()


def run_train(args):
    """运行 RSL-RL PPO 训练"""
    import yaml
    import torch
    
    # 检查 RSL-RL 可用性
    try:
        from rsl_rl.runners import OnPolicyRunner
    except ImportError:
        print("错误: 需要安装 rsl-rl-lib")
        print("运行: pip install rsl-rl-lib")
        return
    
    from isaacsim import SimulationApp
    
    # 启动仿真
    simulation_app = SimulationApp({
        "headless": args.headless, 
        "renderer": "RaytracedLighting"
    })
    
    from modules.catch_env import VolleyballCatchEnv
    from modules.rsl_rl_wrapper import GymToRslRlWrapper
    
    print("=" * 70)
    print("  排球接球 RL 训练 (RSL-RL PPO)")
    print("=" * 70)
    print(f"  场地: 标准排球场 (18m x 9m) + 球网")
    print(f"  机器人: URDF 舵轮底盘 (N4)")
    print(f"  框架: RSL-RL (ETH RSL)")
    print(f"  设备: {args.device}")
    print(f"  最大迭代: {args.max_iterations}")
    print("-" * 70)
    
    # 创建 Gymnasium 环境 (训练时默认不打印 Episode 详细信息)
    gym_env = VolleyballCatchEnv(
        simulation_app=simulation_app,
        render_mode="human" if not args.headless else None,
        verbose=False
    )
    
    # 包装为 RSL-RL VecEnv
    env = GymToRslRlWrapper(
        gym_env=gym_env,
        device=args.device,
        max_episode_length=120,  # 6s at 20Hz
    )
    
    # 日志目录
    log_root = os.path.join(os.path.dirname(__file__), "logs", "rsl_rl", "volleyball_catch")
    os.makedirs(log_root, exist_ok=True)
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "config", "ppo_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    
    # 覆盖配置中的迭代次数
    train_cfg["runner"]["max_iterations"] = args.max_iterations
    
    # 保存配置到日志目录
    with open(os.path.join(log_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(train_cfg, f, allow_unicode=True)
    
    print(f"  日志目录: {log_dir}")
    print(f"  配置文件: {config_path}")
    
    # 初始化环境
    env.reset()
    
    # 创建 OnPolicyRunner
    runner = OnPolicyRunner(
        env=env,
        train_cfg=train_cfg["runner"],
        log_dir=log_dir,
        device=args.device,
    )
    
    # 加载检查点（如果指定）
    if args.resume:
        print(f"  加载检查点: {args.resume}")
        runner.load(args.resume)
    
    print(f"  开始训练...")
    print("-" * 70)
    
    # 训练
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    # 最终统计
    stats = gym_env.get_stats()
    print("\n" + "=" * 70)
    print(f"  训练完成")
    print(f"     总 episodes: {stats['episodes']}")
    print(f"     接球率: {stats['catch_rate']*100:.1f}%")
    print(f"     模型保存: {log_dir}")
    print("=" * 70)
    
    env.close()
    simulation_app.close()


def run_clean(args):
    """清理日志和检查点"""
    import shutil
    
    base_dir = os.path.dirname(__file__)
    logs_dir = os.path.join(base_dir, "logs")
    
    print("=" * 70)
    print("  清理日志和检查点")
    print("=" * 70)
    
    if os.path.exists(logs_dir):
        # 列出所有日志目录
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(logs_dir):
            for f in files:
                fp = os.path.join(root, f)
                total_size += os.path.getsize(fp)
                file_count += 1
        
        print(f"  日志目录: {logs_dir}")
        print(f"  文件数量: {file_count}")
        print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")
        print("-" * 70)
        
        confirm = input("  确认删除? [y/N]: ")
        if confirm.lower() == 'y':
            shutil.rmtree(logs_dir)
            print("  已清理所有日志!")
        else:
            print("  取消清理")
    else:
        print("  没有日志需要清理")
    
    print("=" * 70)


def main():
    args = parse_args()
    
    if args.clean:
        run_clean(args)
    elif args.train:
        run_train(args)
    elif args.test:
        run_test(args)
    else:
        # 默认运行测试
        args.test = True
        run_test(args)


if __name__ == "__main__":
    main()
