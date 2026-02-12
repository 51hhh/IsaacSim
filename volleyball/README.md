# 🏐 排球物理模拟 — RL 训练环境

基于 NVIDIA Isaac Sim 构建的高保真排球物理仿真环境，专为**击球机器人强化学习 (RL) 训练**设计。

## 核心特性

- **真实空气动力学**: 包含阻力危机 (Drag Crisis)、马格努斯力 (Magnus Effect) 和飘球效应 (Knuckleball)。
- **GPU 并行加速**: 启用 PhysX GPU Dynamics，支持多 agent 大规模并行训练。
- **RL 训练优化**:
    - **几何重置**: 碰网、出界或超时立即重置 Episode。
    - **落地继续**: 球落地后保留物理交互（弹跳/滚动），支持连续击球任务。
    - **状态同步**: 直接通过 `RigidPrim` 读取 PhysX 状态，保证气动力计算的实时性。

## 物理模型

### 核心公式

| 模型 | 公式 | 说明 |
|------|------|------|
| **空气阻力** | F = ½ρv²ACd | 含阻力危机（Cd 随 Re 动态变化） |
| **马格努斯力** | F = ½ρv²ACl·(ω×v) | 旋转产生的升力/下压力 |
| **飘球效应** | 低频正弦 + 随机扰动 | 低旋转时 Knuckleball 效应 |
| **旋转衰减** | dω/dt = -C·|v|·ω | 空气对旋转的阻力矩 |

### 阻力危机 (Drag Crisis)

排球阻力系数随雷诺数动态变化（非常数！）：

```
速度 <7  m/s  → Cd ≈ 0.50 (亚临界：层流)
速度 7-14 m/s → Cd: 0.50→0.20 (过渡区)
速度 14-20 m/s → Cd: 0.20→0.12 (阻力危机)
速度 >20 m/s  → Cd ≈ 0.12 (超临界：湍流)
```

### 发球类型

| 类型 | 球速 | 旋转 | 特点 |
|------|------|------|------|
| 力量跳发 | 73-100 km/h | 40-90 rad/s | 上旋，快速下坠 |
| 飘球发球 | 43-65 km/h | <1.5 rad/s | 不规则飘移 |
| 上手发球 | 50-68 km/h | 5-25 rad/s | 平衡型 |

## 运行

```bash
cd volleyball
C:\IsaacSim\python.bat volleyball_simple.py
```

## 接球 RL 训练

### 任务描述

舵轮底盘移动到排球落点位置接球。

| 组件 | 设计 |
|------|------|
| **观察空间** (8D) | 预测落点(x,y) + 到达时间 + 底盘位置(x,y) + 速度(vx,vy) + 球高度 |
| **动作空间** (2D) | 底盘速度指令 (vx, vy)，范围 [-1, 1] |
| **接球奖励** | +100 |
| **未接住惩罚** | -10 - 距离×2 |
| **终止条件** | 排球触底 / 接住球 / 超时(6s) |

### 运行接球环境

```bash
# 测试（URDF舵轮底盘）
C:\IsaacSim\python.bat run_catch_rl.py --test

# 安装rsl_rl
cd rsl_rl
C:\IsaacSim\python.bat -m pip install -e .

# PPO训练
C:\IsaacSim\python.bat run_catch_rl.py --train

# 更多参数
C:\IsaacSim\python.bat run_catch_rl.py --help
```

### 落点预测

使用欧拉积分 + 空气阻力估算球在机器人高度 (0.1m) 处的落点和到达时间：

```python
def predict_landing_point(ball_pos, ball_vel, target_height):
    # 积分公式: a = g + drag, v += a*dt, pos += v*dt
    # 当 pos.z <= target_height 时返回 (x, y, time)
```

## 文件结构

```
volleyball/
├── volleyball_simple.py           # 排球仿真主脚本（独立运行）
├── run_catch_rl.py                # 接球 RL 运行入口
├── modules/                       # 模块化组件
│   ├── __init__.py
│   ├── volleyball_physics.py      # 排球空气动力学模型
│   ├── volleyball_court.py        # 场地创建模块
│   ├── swerve_robot.py            # URDF舵轮底盘
│   └── catch_env.py               # 接球 Gymnasium 环境
├── VOLLEYBALL_PHYSICS_MODEL.md    # 物理模型设计文档
└── README.md                      # 本文件
```

## 模块说明

### volleyball_physics.py

- `VolleyballAerodynamics`: 空气动力学计算（阻力、马格努斯力、飘球）
- `VolleyballRigidBody`: 排球刚体封装
- `generate_serve()`: 发球生成
- `predict_landing_point()`: 落点预测

### volleyball_court.py

- `VolleyballCourt`: 完整排球场（地面、球网、边线）
- `SimpleCourt`: 简化场地（仅地面，用于训练）

### swerve_robot.py

- `SwerveRobot`: URDF舵轮底盘加载与控制
- `SwerveKinematics`: 四轮舵轮运动学

## 参考文献

- Asai et al. (2010) — 排球空气动力学风洞数据
- Hong et al. (2014) — 新型排球 Cd 曲线
- The Sport Journal — 发球速度生物力学数据
- Frontiers in Sports (2021) — 飘球轨迹分析
