# 排球接球 Isaac Lab 环境

基于 Isaac Lab DirectRLEnv 的 GPU 并行排球接球强化学习环境。

## 特性

- **真正的 GPU 并行**: 使用 Isaac Lab 的 `InteractiveSceneCfg` 实现 GPU 上的批量并行仿真
- **排球空气动力学**: 包含阻力和马格努斯效应的物理模型
- **URDF 机器人**: 支持自定义 URDF 舵轮底盘机器人
- **RSL-RL 集成**: 使用 ETH RSL 的 PPO 训练算法

## 文件结构

```
volleyball/isaaclab/
├── __init__.py                 # 包初始化
├── volleyball_catch_cfg.py     # 环境配置
├── volleyball_catch_env.py     # 环境实现
├── train.py                    # 训练脚本
├── play.py                     # 评估/播放脚本
├── test_env.py                 # 环境测试脚本
└── README.md                   # 本文件
```

## 环境说明

### 观察空间 (8 维)
| 索引 | 名称 | 描述 |
|------|------|------|
| 0 | ball_rel_x | 球相对机器人的 X 位置 |
| 1 | ball_rel_y | 球相对机器人的 Y 位置 |
| 2 | ball_z | 球的绝对高度 |
| 3 | ball_vx | 球的 X 速度 |
| 4 | ball_vy | 球的 Y 速度 |
| 5 | ball_vz | 球的 Z 速度 |
| 6 | pred_land_rel_x | 预测落点相对 X |
| 7 | pred_land_rel_y | 预测落点相对 Y |

### 动作空间 (2 维)
| 索引 | 名称 | 范围 | 描述 |
|------|------|------|------|
| 0 | vx | [-1, 1] | 机器人 X 方向速度 (×3 m/s) |
| 1 | vy | [-1, 1] | 机器人 Y 方向速度 (×3 m/s) |

### 奖励函数
- **接住奖励**: +100 (球在机器人接球半径内且高度合适)
- **错过惩罚**: -10 - 5×距离 (球落地时的距离惩罚)
- **位置奖励**: +2/+1/+0.5 (根据距预测落点的距离)
- **接近奖励**: 基于朝落点移动的速度
- **等待奖励**: +0.5 (在落点附近等待)
- **时间惩罚**: -0.02/步

## 使用方法

### 1. 测试环境

```powershell
cd C:\Users\Rick\Desktop\isaacsim\volleyball\isaaclab
C:\IsaacSim\python.bat test_env.py --num_envs 4
```

### 2. 训练

```powershell
# 基础训练 (64 个并行环境)
C:\IsaacSim\python.bat train.py --num_envs 64 --max_iterations 3000

# 无头模式训练 (更快)
C:\IsaacSim\python.bat train.py --num_envs 256 --max_iterations 5000 --headless

# 从检查点继续训练
C:\IsaacSim\python.bat train.py --num_envs 64 --checkpoint logs/volleyball_catch/2025-01-01_12-00-00/model.pt
```

### 3. tensorboard

C:\IsaacSim\python.bat -m tensorboard.main --logdir=volleyball\isaaclab\logs\volleyball_catch\2026-02-13_15-56-02


## 配置参数

### 环境配置 (`VolleyballCatchEnvCfg`)

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `num_envs` | 64 | 并行环境数量 |
| `decimation` | 6 | 控制频率 = 物理频率 / decimation |
| `episode_length_s` | 6.0 | 每集最大时长 (秒) |
| `action_scale` | 3.0 | 动作缩放 (最大速度 m/s) |
| `catch_radius` | 0.40 | 接球有效半径 (m) |

### 训练配置 (RSL-RL)

训练配置可通过 YAML 文件指定:

```yaml
# config/train_cfg.yaml
num_steps_per_env: 24
max_iterations: 5000
actor:
  hidden_dims: [256, 256, 128]
  activation: elu
algorithm:
  learning_rate: 0.001
  gamma: 0.99
  entropy_coef: 0.01
```

使用配置文件:

```powershell
C:\IsaacSim\python.bat train.py --config config/train_cfg.yaml
```

## 性能建议

1. **GPU 利用率**: 增加 `--num_envs` 以提高 GPU 利用率，推荐 64-512
2. **无头模式**: 使用 `--headless` 进行快速训练
3. **内存**: 如果 GPU 内存不足，减少 `--num_envs`

## 依赖

- Isaac Sim 4.5+
- Isaac Lab 0.54+
- RSL-RL 4.0.1 (`pip install rsl-rl-lib`)
- PyTorch 2.0+

## 故障排除

### URDF 加载失败
确保 URDF 路径正确且文件存在:
```
C:\Users\Rick\Desktop\isaacsim\URDF\N4\urdf\robot_fixed.urdf
```

### 模块导入错误
确保在 `volleyball/isaaclab/` 目录下运行脚本，或将该目录添加到 PYTHONPATH。

### GPU 内存不足
减少并行环境数量:
```powershell
C:\IsaacSim\python.bat train.py --num_envs 16
```
