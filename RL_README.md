# 强化学习 (RL) 集成实现路径与完整流程

## 一、项目背景与目标

### 1.1 现有系统概述

当前项目是一套基于 ROS2 的电机控制系统，架构分为三层：

```
┌──────────────────────────────────────────────────────────┐
│                      应用层                               │
│  motor_control_node    motor_monitor_node                │
│  chassis_control_node  omni_chassis_control_node         │
│  joystick_control_node                                   │
└──────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────────────────────────────────────┐
│                      控制层                               │
│  CascadeController (串级 PID: 位置环 + 速度环)            │
│  OmniWheelKinematics (全向轮逆/正运动学)                  │
│  SteerWheelKinematics (舵轮运动学)                        │
└──────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────────────────────────────────────┐
│                      驱动层                               │
│  DJIMotor (GM6020/GM3508)  DamiaoMotor (DM4340/DM4310)  │
│  UnitreeMotor (A1/GO-8010)                               │
└──────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────────────────────────────────────┐
│                      硬件层                               │
│  CANInterface (USB-CAN, 921600bps)                       │
│  SerialInterface (RS485, 4000000bps)                     │
│  HardwareManager (设备管理、热插拔)                       │
└──────────────────────────────────────────────────────────┘
```

**已支持的电机**：

| 电机型号 | 接口 | 控制模式 | 输出范围 |
|---------|------|---------|---------|
| DJI GM6020 | CAN | 电压控制 | -30000 ~ 30000 |
| DJI GM3508 | CAN | 电流控制 | -16384 ~ 16384 |
| 达妙 DM4340/DM4310 | CAN | MIT 模式 (kp/kd/pos/vel/torque) |
| 宇树 A1/GO-8010 | RS485 | 力位混合 (kp/kd/pos/vel/torque) |

**已有控制能力**：
- 200Hz 控制循环频率
- 串级 PID 控制器（位置环 + 速度环）
- X 形全向轮底盘运动学（逆运动学/正运动学）
- 多接口并行通信（多 CAN/多串口）
- YAML 配置化电机管理

### 1.2 RL 集成目标

参考论文 *"Learning coordinated badminton skills for legged manipulators"* (ETH Zurich, Science Robotics 2025) 的方法论，将 RL 策略引入本系统，实现：

1. **替代传统 PID**：用 RL 策略网络直接输出关节力矩/电流命令
2. **全身协调控制**：底盘运动 + 机械臂/末端执行器协同
3. **Sim-to-Real 部署**：仿真训练 → 真实硬件零样本迁移
4. **感知驱动行为**：结合视觉/传感器的闭环策略

---

## 二、总体实现路径

```
阶段 1: 仿真环境搭建          阶段 2: RL 训练框架         阶段 3: Sim-to-Real 部署
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│ URDF/MJCF 建模   │    │ 观测空间设计      │    │ ROS2 推理节点         │
│ IsaacGym/MuJoCo  │ →  │ 奖励函数设计      │ →  │ 策略网络加载          │
│ 电机动力学建模    │    │ PPO/N-P3O 训练   │    │ 200Hz 实时推理        │
│ 域随机化         │    │ 约束 RL           │    │ 安全监督器            │
└─────────────────┘    └──────────────────┘    └──────────────────────┘
```

---

## 三、阶段 1：仿真环境搭建

### 3.1 机器人模型构建

#### 3.1.1 URDF/MJCF 模型

```
rl_sim/
├── assets/
│   ├── urdf/
│   │   └── robot.urdf          # 机器人模型描述
│   ├── mjcf/
│   │   └── robot.xml           # MuJoCo 格式（如需要）
│   └── meshes/                 # 碰撞/视觉网格
├── cfg/
│   ├── env_cfg.yaml            # 环境配置
│   ├── train_cfg.yaml          # 训练超参数
│   └── reward_cfg.yaml         # 奖励权重
└── envs/
    └── motor_env.py            # Gym 环境定义
```

**关键参数映射**（从真实硬件到仿真）：

| 真实参数 | 仿真对应 | 来源 |
|---------|---------|------|
| GM3508 减速比 19:1 | joint gear_ratio | `motor_base.hpp` |
| GM6020 编码器 8192 线/圈 | joint encoder resolution | `dji_motor.hpp` |
| CAN 通信延迟 ~2ms | action delay | `can_interface.hpp` |
| 控制频率 200Hz | control_dt = 0.005s | `control_params.yaml` |
| 全向轮底盘尺寸 | chassis geometry | `omni_chassis_params.yaml` |
| 轮半径 0.075m | wheel radius | `omni_chassis_params.yaml` |

#### 3.1.2 电机动力学建模

根据论文方法，需为每种电机建立精确的执行器模型：

```python
# 电机动力学参数（需通过系统辨识获取）
class MotorDynamicsModel:
    """电机仿真模型，参数通过 CMA-ES 辨识"""
    def __init__(self, motor_type):
        if motor_type == "GM3508":
            self.gear_ratio = 19.0
            self.max_current = 16384      # 原始单位
            self.friction = 0.0           # 待辨识
            self.damping = 0.0            # 待辨识
            self.armature = 0.0           # 待辨识
            self.torque_constant = 0.0    # 待辨识 (Nm/A)
        elif motor_type == "GM6020":
            self.gear_ratio = 1.0
            self.max_voltage = 30000      # 原始单位
            self.friction = 0.0
            self.damping = 0.0
            self.armature = 0.0
```

#### 3.1.3 系统辨识流程

参考论文 Section "Sim-to-real practicalities"，使用 CMA-ES 优化仿真参数：

```
步骤 1: 数据采集
  - 在真实电机上执行正弦波扫频命令 (0.5Hz ~ 50Hz)
  - 记录: 命令值、关节位置、关节速度、时间戳
  - 通过 ROS2 话题录制: ros2 bag record /dji_motor_states /dji_motor_command

步骤 2: 参数优化
  - 在仿真中重放相同命令序列
  - 用 CMA-ES 最小化仿真与真实轨迹的位置误差
  - 优化变量: friction, damping, armature, delay

步骤 3: 验证
  - 使用新的命令序列（非训练集）验证仿真精度
  - 目标: 位置跟踪误差 < 5% (均方根)
```

### 3.2 仿真平台选型

#### 方案 A: IsaacGym (推荐)

论文使用 IsaacGym + legged_gym 框架，优势：
- GPU 并行仿真 (4096 环境并行)
- 训练时间短 (~4.81h / RTX 2080Ti)
- 成熟的 legged_gym 生态

```bash
# 安装依赖
pip install isaacgym  # 需从 NVIDIA 官网下载
pip install rsl-rl    # legged_gym 的 RL 库

# 目录结构
rl_training/
├── legged_gym/
│   ├── envs/
│   │   └── usb2can_robot/
│   │       ├── usb2can_robot_config.py    # 环境 + 训练配置
│   │       └── usb2can_robot.py           # 环境实现
│   └── scripts/
│       ├── train.py
│       └── play.py
└── resources/
    └── robots/
        └── usb2can_robot/
            └── urdf/
```

#### 方案 B: MuJoCo + Gymnasium

适合轻量级验证和快速原型：
- 开源免费
- 物理精度高
- 社区活跃

### 3.3 域随机化 (Domain Randomization)

参考论文做法，训练时随机化以下参数以增强 Sim-to-Real 迁移：

| 随机化参数 | 范围 | 说明 |
|-----------|------|------|
| 关节摩擦 | ±50% | 每个 episode 开始时采样 |
| 关节阻尼 | ±30% | |
| 地面摩擦系数 | [0.5, 1.5] | |
| 底盘附加质量 | [-1kg, +3kg] | 模拟负载变化 |
| 控制延迟 | [0, 10ms] | 模拟通信抖动 |
| 编码器噪声 | σ = 0.01 rad | 高斯噪声 |
| 外部推力 | 随机脉冲 | 增强鲁棒性 |
| 地面不平度 | max 0.06m | 参考论文配置 |

---

## 四、阶段 2：RL 训练框架

### 4.1 MDP 设计

#### 4.1.1 观测空间 (Observation Space)

参考论文 Table S3，按功能分组设计观测向量：

```python
# Actor 观测（部署时可用的信息）
actor_obs = {
    # 本体感知 (Proprioception)
    "base_linear_velocity":    3,   # 基座线速度 (m/s)
    "base_angular_velocity":   3,   # 基座角速度 (rad/s)
    "gravity_vector":          3,   # 重力方向（基座坐标系）
    "joint_positions":         N,   # 关节位置偏移 (rad)
    "joint_velocities":        N,   # 关节速度 (rad/s)
    "previous_actions":        N,   # 上一步动作

    # 任务相关
    "target_command":          M,   # 目标命令（取决于任务）
    "heading_vector":          2,   # 朝向向量
}

# Critic 额外观测（仅训练时使用，参考论文非对称 Actor-Critic）
critic_extra_obs = {
    "noiseless_joint_states":  2*N, # 无噪声关节状态
    "domain_randomization":    K,   # 域随机化参数
    "privileged_info":         P,   # 特权信息
}
```

**针对本系统的具体映射**（以 4 轮全向底盘为例）：

| 观测量 | 维度 | 数据来源 | ROS2 话题 |
|-------|------|---------|----------|
| 4 个 GM3508 关节角度 | 4 | DJIMotorState.angle | `/dji_motor_states` |
| 4 个 GM3508 关节速度 | 4 | DJIMotorState.rpm | `/dji_motor_states` |
| 底盘速度 (vx, vy, wz) | 3 | 正运动学计算 | 内部计算 |
| 目标速度命令 | 3 | cmd_vel | `/cmd_vel` |
| 上一步动作 | 4 | 缓存 | 内部 |
| **总计** | **18** | | |

#### 4.1.2 动作空间 (Action Space)

```python
# 方案 1: 直接力矩/电流输出（推荐，最灵活）
# 优点: 端到端学习，无需 PID 中间层
# 缺点: 训练难度稍高
action_space = {
    "joint_torques": N,  # 归一化到 [-1, 1]
    # 实际输出 = action * max_output
    # GM3508: action * 16384 → 电流命令
    # GM6020: action * 30000 → 电压命令
}

# 方案 2: 目标关节位置（更稳定，底层用 PD 控制器跟踪）
# 优点: 训练更稳定
# 缺点: 依赖底层 PD 参数
action_space = {
    "target_positions": N,  # 归一化的目标位置偏移
    # 实际位置 = default_pos + action * action_scale
}
```

**动作频率配置**：

| 配置项 | 值 | 说明 |
|-------|---|------|
| 仿真步长 (sim_dt) | 0.0025s (400Hz) | 物理仿真精度 |
| 控制步长 (control_dt) | 0.005s (200Hz) | 与真实系统一致 |
| 策略推理频率 | 50-100Hz | 策略网络输出频率 |
| decimation | 2-4 | sim_dt / policy_dt |

#### 4.1.3 奖励函数设计

参考论文 Table S2 的奖励结构，按任务分层设计：

```python
class RewardFunction:
    """奖励函数（以底盘速度跟踪任务为例）"""

    def __init__(self):
        self.scales = {
            # === 任务奖励 (正向激励) ===
            "linear_velocity_tracking":   1.0,   # 线速度跟踪
            "angular_velocity_tracking":  0.5,   # 角速度跟踪

            # === 正则化惩罚 (负向约束) ===
            "joint_torques":             -1e-5,  # 力矩平滑
            "joint_acceleration":        -1e-6,  # 加速度平滑
            "action_rate":               -0.03,  # 动作变化率
            "collision":                 -2.0,   # 碰撞惩罚

            # === 约束惩罚 ===
            "joint_position_limit":      -1.0,   # 关节限位
            "joint_torque_limit":        -1e-3,  # 力矩限制
        }

    def compute_linear_velocity_tracking(self, cmd_vel, actual_vel):
        """线速度跟踪奖励 (论文公式 S3 风格)"""
        error = torch.norm(cmd_vel[:, :2] - actual_vel[:, :2], dim=1)
        return 1.0 / (1.0 + error / 0.25)

    def compute_angular_velocity_tracking(self, cmd_wz, actual_wz):
        """角速度跟踪奖励"""
        error = torch.abs(cmd_wz - actual_wz)
        return 1.0 / (1.0 + error / 0.25)

    def compute_action_rate(self, action, last_action):
        """动作平滑惩罚"""
        return torch.sum(torch.square(action - last_action), dim=1)
```

### 4.2 训练算法

#### 4.2.1 PPO (基础方案)

```python
# rsl-rl 框架下的 PPO 配置
class PPOConfig:
    seed = 1
    num_envs = 4096               # GPU 并行环境数
    num_steps_per_env = 24        # 每次收集的步数
    max_iterations = 10000        # 最大迭代次数

    # 网络结构（参考论文 Table S4）
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"

    # PPO 超参数
    learning_rate = 1e-3          # 自适应学习率
    discount_factor = 0.995
    gae_lambda = 0.975
    entropy_coef = 0.0016
    entropy_coef_decay = 0.99993
    clip_param = 0.2
    num_mini_batches = 4
    num_epochs = 5
    optimizer = "AdamW"
```

#### 4.2.2 N-P3O (约束 RL，推荐)

论文使用 N-P3O 处理硬件约束（如电流限制），适合本系统：

```python
# 约束定义
constraints = {
    # GM3508 电流约束
    "max_current_3508": {
        "type": "inequality",
        "limit": 16384,
        "expression": lambda action: torch.abs(action * 16384),
    },
    # GM6020 电压约束
    "max_voltage_6020": {
        "type": "inequality",
        "limit": 30000,
        "expression": lambda action: torch.abs(action * 30000),
    },
}
```

#### 4.2.3 非对称 Actor-Critic (Asymmetric AC)

参考论文核心方法，解决感知与特权信息的不对称问题：

```
训练时:
  Actor  → 接收: 有噪声的本体感知 + 任务目标
  Critic → 接收: Actor 观测 + 无噪声状态 + 域随机化参数 + MDP 信息

部署时:
  仅 Actor 网络 → 接收真实传感器数据 → 输出关节命令
```

### 4.3 训练流程

```
1. 环境初始化
   ├── 加载 URDF 到 IsaacGym
   ├── 创建 4096 个并行环境
   └── 初始化域随机化参数

2. 数据收集 (Rollout)
   ├── Actor 根据观测输出动作
   ├── 仿真执行动作，推进物理
   ├── 计算奖励和约束违反
   └── 存入经验缓冲区

3. 策略优化
   ├── 计算 GAE 优势估计
   ├── N-P3O / PPO 策略梯度更新
   ├── 更新 Actor + Critic 网络
   └── 记录训练指标 (TensorBoard)

4. 评估与保存
   ├── 每 N 步评估策略性能
   ├── 保存最佳模型 checkpoint
   └── 导出 ONNX / TorchScript 模型

5. 预计训练时间
   ├── RTX 2080Ti: ~5h (基础任务)
   ├── RTX 3090/4090: ~2h
   └── 复杂任务 (全身协调): 1-2 天
```

---

## 五、阶段 3：Sim-to-Real 部署

### 5.1 整体部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                    ROS2 部署架构                              │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 感知模块      │    │ RL 推理节点   │    │ 安全监督器     │  │
│  │ (传感器输入)  │ →  │ (策略网络)    │ →  │ (约束检查)    │  │
│  └──────────────┘    └──────────────┘    └───────────────┘  │
│                              │                    │         │
│                              ↓                    ↓         │
│                    ┌──────────────────────────────────┐     │
│                    │     motor_control_node (现有)     │     │
│                    │     200Hz 控制循环               │     │
│                    │     CAN/串口 通信               │     │
│                    └──────────────────────────────────┘     │
│                              │                              │
│                    ┌──────────────────────────────────┐     │
│                    │         硬件层                    │     │
│                    │   GM3508 / GM6020 / DM4340 / A1  │     │
│                    └──────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 RL 推理节点实现

#### 5.2.1 C++ 推理节点 (推荐，低延迟)

```cpp
// src/motor_control_ros2/src/rl_inference_node.cpp
#include <torch/script.h>  // LibTorch
#include <rclcpp/rclcpp.hpp>

class RLInferenceNode : public rclcpp::Node {
public:
    RLInferenceNode() : Node("rl_inference_node") {
        // 加载 TorchScript 模型
        model_ = torch::jit::load("policy.pt");
        model_.eval();

        // 200Hz 推理定时器（与现有控制频率一致）
        timer_ = this->create_wall_timer(
            std::chrono::microseconds(5000),  // 5ms = 200Hz
            std::bind(&RLInferenceNode::inferenceCallback, this)
        );

        // 订阅电机状态
        state_sub_ = this->create_subscription<DJIMotorState>(
            "/dji_motor_states", 10,
            std::bind(&RLInferenceNode::stateCallback, this, _1)
        );

        // 发布电机命令
        cmd_pub_ = this->create_publisher<DJIMotorCommand>(
            "/dji_motor_command", 10
        );
    }

private:
    void inferenceCallback() {
        // 1. 构建观测向量
        auto obs = buildObservation();

        // 2. 策略推理
        auto action = model_.forward({obs}).toTensor();

        // 3. 动作后处理 + 安全裁剪
        action = torch::clamp(action, -1.0, 1.0);

        // 4. 映射到电机命令
        publishCommand(action);
    }

    torch::jit::script::Module model_;
    // ... 订阅者、发布者、定时器
};
```

#### 5.2.2 Python 推理节点 (快速原型验证)

```python
# src/motor_control_ros2/scripts/rl_inference_node.py
import rclpy
import torch
import numpy as np

class RLInferenceNode(Node):
    def __init__(self):
        super().__init__('rl_inference_node')

        # 加载策略模型
        self.policy = torch.jit.load('policy.pt')
        self.policy.eval()

        # 200Hz 推理定时器
        self.timer = self.create_timer(0.005, self.inference_callback)

        # 观测缓存
        self.obs_buffer = np.zeros(OBS_DIM, dtype=np.float32)
        self.last_action = np.zeros(ACT_DIM, dtype=np.float32)

    def inference_callback(self):
        # 构建观测
        obs = self.build_observation()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        # 推理
        with torch.no_grad():
            action = self.policy(obs_tensor).squeeze(0).numpy()

        # 安全裁剪
        action = np.clip(action, -1.0, 1.0)

        # 发布命令
        self.publish_command(action)
        self.last_action = action.copy()
```

### 5.3 安全监督器

部署时必须有独立的安全层，防止策略输出异常命令：

```python
class SafetySupervisor:
    """安全监督器 - 独立于 RL 策略运行"""

    def __init__(self):
        # 电机安全限制
        self.limits = {
            "GM3508": {
                "max_current": 10000,       # 保守限制 (满量程 16384)
                "max_rpm": 8000,            # RPM 上限
                "max_temperature": 60,      # 温度上限 (°C)
                "max_current_rate": 5000,   # 电流变化率限制 (/step)
            },
            "GM6020": {
                "max_voltage": 20000,       # 保守限制 (满量程 30000)
                "max_rpm": 300,
                "max_temperature": 60,
                "max_voltage_rate": 8000,
            },
        }

    def check_and_clip(self, motor_type, command, state, last_command):
        """检查并裁剪命令"""
        limit = self.limits[motor_type]

        # 1. 幅值限制
        command = np.clip(command, -limit["max_current"], limit["max_current"])

        # 2. 变化率限制（防止突变）
        rate = command - last_command
        max_rate = limit["max_current_rate"]
        rate = np.clip(rate, -max_rate, max_rate)
        command = last_command + rate

        # 3. 温度保护（超温降功率）
        if state.temperature > limit["max_temperature"]:
            command *= 0.5  # 降功率 50%

        # 4. 超速保护
        if abs(state.rpm) > limit["max_rpm"]:
            command = 0  # 紧急停止

        return command
```

### 5.4 与现有系统的集成方式

#### 方式 A: 替换 PID 输出（最小改动）

```
现有流程: 目标位置 → PID → 电流/电压 → motor_control_node → 硬件
RL 流程:  观测 → RL Policy → 电流/电压 → motor_control_node → 硬件
                                   ↓
                             安全监督器裁剪
```

只需修改 `motor_control_node` 中的命令来源，从 PID 输出切换为 RL 输出。

#### 方式 B: 独立 RL 节点 + 话题通信（推荐，解耦）

```
rl_inference_node  →  /dji_motor_command  →  motor_control_node  →  硬件
       ↑                                           │
  /dji_motor_states  ←────────────────────────────-─┘
```

优点：
- 不修改现有 motor_control_node 代码
- 可随时切换 PID / RL 控制
- 支持热切换（ROS2 参数动态切换模式）

#### 对应 ROS2 话题接口

| 话题 | 类型 | 方向 | 说明 |
|------|------|------|------|
| `/dji_motor_states` | DJIMotorState | 订阅 | 电机状态反馈 |
| `/dji_motor_command` | DJIMotorCommand | 发布 | 直接输出模式 |
| `/dji_motor_command_advanced` | DJIMotorCommandAdvanced | 发布 | 支持模式选择 |
| `/cmd_vel` | geometry_msgs/Twist | 订阅 | 底盘速度目标 |
| `/rl_policy_status` | 自定义 | 发布 | 策略运行状态 |

### 5.5 模型导出与格式

```python
# 训练完成后导出模型
# 方式 1: TorchScript (推荐，C++ 可用)
traced_model = torch.jit.trace(policy.actor, example_obs)
traced_model.save("policy.pt")

# 方式 2: ONNX (跨平台)
torch.onnx.export(policy.actor, example_obs, "policy.onnx",
                  input_names=["observations"],
                  output_names=["actions"])

# 方式 3: TensorRT (最快推理速度，NVIDIA 平台)
# 适合 Jetson 等嵌入式部署
```

---

## 六、目录结构规划

```
USB2CAN_motor/
├── src/
│   └── motor_control_ros2/          # 现有 ROS2 包 (保持不变)
│       ├── config/
│       ├── include/
│       ├── src/
│       │   ├── motor_control_node.cpp
│       │   ├── rl_inference_node.cpp      # [新增] C++ RL 推理节点
│       │   └── ...
│       ├── scripts/
│       │   ├── rl_inference_node.py       # [新增] Python RL 推理节点
│       │   └── safety_supervisor.py       # [新增] 安全监督器
│       └── msg/
│           └── RLPolicyStatus.msg         # [新增] 策略状态消息
│
├── rl_training/                           # [新增] RL 训练工程
│   ├── envs/
│   │   ├── base_env.py                    # 基础环境
│   │   ├── omni_chassis_env.py            # 全向底盘环境
│   │   └── manipulator_env.py             # 机械臂环境
│   ├── cfg/
│   │   ├── env/
│   │   │   ├── omni_chassis.yaml          # 底盘环境配置
│   │   │   └── manipulator.yaml           # 机械臂环境配置
│   │   └── train/
│   │       ├── ppo.yaml                   # PPO 超参数
│   │       └── np3o.yaml                  # N-P3O 超参数
│   ├── rewards/
│   │   ├── velocity_tracking.py           # 速度跟踪奖励
│   │   └── position_tracking.py           # 位置跟踪奖励
│   ├── models/
│   │   └── actor_critic.py                # 网络结构
│   ├── scripts/
│   │   ├── train.py                       # 训练入口
│   │   ├── play.py                        # 仿真可视化
│   │   └── export.py                      # 模型导出
│   └── resources/
│       └── robots/
│           ├── urdf/                      # 机器人 URDF
│           └── meshes/                    # 网格文件
│
├── rl_sysid/                              # [新增] 系统辨识工具
│   ├── data_collection.py                 # 数据采集脚本
│   ├── cmaes_optimization.py              # CMA-ES 参数优化
│   └── validation.py                      # 辨识结果验证
│
├── rl_models/                             # [新增] 训练好的模型
│   ├── omni_chassis_v1.pt                 # TorchScript 格式
│   ├── omni_chassis_v1.onnx              # ONNX 格式
│   └── training_logs/                     # TensorBoard 日志
│
├── README.md                              # 原有文档
└── RL_README.md                           # 本文档
```

---

## 七、分阶段实施计划

### 阶段 1: 基础验证 (预计 2-3 周)

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1.1 | 构建全向底盘 URDF 模型 | `robot.urdf` |
| 1.2 | 搭建 IsaacGym 或 MuJoCo 仿真环境 | `omni_chassis_env.py` |
| 1.3 | 实现速度跟踪 PPO 训练 | 训练好的策略模型 |
| 1.4 | 仿真内验证策略效果 | 验证报告 |

### 阶段 2: 系统辨识与迁移 (预计 2-3 周)

| 步骤 | 内容 | 产出 |
|------|------|------|
| 2.1 | GM3508 电机系统辨识数据采集 | ROS2 bag 录制数据 |
| 2.2 | CMA-ES 参数优化 | 仿真电机参数 |
| 2.3 | 域随机化训练 | 鲁棒策略模型 |
| 2.4 | 非对称 Actor-Critic 训练 | 最终策略 |

### 阶段 3: 真实部署 (预计 2-3 周)

| 步骤 | 内容 | 产出 |
|------|------|------|
| 3.1 | 实现 RL 推理 ROS2 节点 | `rl_inference_node` |
| 3.2 | 实现安全监督器 | `safety_supervisor` |
| 3.3 | 低速保守测试 (限幅 30%) | 安全验证 |
| 3.4 | 逐步放宽限幅，完整性能测试 | 部署报告 |

### 阶段 4: 进阶任务 (长期)

| 步骤 | 内容 | 说明 |
|------|------|------|
| 4.1 | 视觉感知集成 | 摄像头 + RL 闭环 |
| 4.2 | 全身协调控制 | 底盘 + 机械臂联合策略 |
| 4.3 | 感知噪声建模 | 参考论文感知模型方法 |
| 4.4 | 约束 RL (N-P3O) | 硬件约束安全保障 |

---

## 八、关键技术要点总结

### 8.1 从论文中提炼的核心方法

| 方法 | 论文做法 | 本系统应用 |
|------|---------|-----------|
| 非对称 Actor-Critic | Critic 接收特权信息 (位置真值、域随机化参数等) | Critic 接收无噪声电机状态 + 域随机化参数 |
| 时间奖励 (Time-based) | 在指定时刻激活 swing 奖励 | 可用于需要时序协调的任务 |
| 感知噪声模型 | 回归真实相机噪声 → 训练中注入 | 注入编码器/IMU 噪声到训练 |
| 约束 RL (N-P3O) | 限制手臂电流 < 8A | 限制 GM3508 电流 / GM6020 电压 |
| CMA-ES 系统辨识 | 正弦波扫频 → 仿真参数匹配 | 同样方法辨识 GM3508/GM6020 动力学 |
| 多目标训练 | 每 episode 6 次 swing 目标 | 多目标速度/位置跟踪 |
| 状态依赖动作标准差 | 临近目标时降低探索 | 提升收敛稳定性 |

### 8.2 训练超参数参考 (来自论文 Table S4)

| 参数 | 值 |
|------|---|
| discount factor | 0.995 |
| GAE lambda | 0.975 |
| learning rate | adaptive (KLD target = 0.01) |
| entropy coefficient | 0.0016 (decay 0.99993) |
| num targets per episode | 6 |
| control dt | 0.01s (100Hz) → 本系统 0.005s (200Hz) |
| terrain max height diff | 0.06m |
| num envs | 4096 |
| actor MLP | (512, 256, 128) |
| critic MLP | (512, 256, 128) |
| activation | ELU |
| optimizer | AdamW |

### 8.3 常见问题与解决方案

| 问题 | 解决方案 |
|------|---------|
| Sim-to-Real 差距大 | 加强域随机化 + 系统辨识 |
| 训练不稳定/不收敛 | 减小学习率 / 增大 entropy / 简化奖励 |
| 策略在真实硬件震荡 | 增加 action_rate 惩罚 / 降低推理频率 |
| 电机过热 | 安全监督器温度保护 + 训练时加入温度约束 |
| 延迟导致不稳定 | 训练时随机化通信延迟 (0-10ms) |
| 编码器跳变 | 训练时注入编码器噪声 |

---

## 九、参考资源

1. **论文**: *Learning coordinated badminton skills for legged manipulators*, Ma et al., Science Robotics 2025
2. **框架**: [legged_gym](https://github.com/leggedrobotics/legged_gym) (IsaacGym RL 训练)
3. **框架**: [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (PPO / N-P3O 实现)
4. **仿真**: [IsaacGym](https://developer.nvidia.com/isaac-gym) (NVIDIA GPU 仿真)
5. **仿真**: [MuJoCo](https://mujoco.org/) (开源物理仿真)
6. **推理**: [LibTorch](https://pytorch.org/cppdocs/) (C++ PyTorch 推理)
7. **辨识**: [CMA-ES](https://github.com/CMA-ES/pycma) (演化策略优化)

---

**文档版本**: v1.0
**创建时间**: 2026-02-09
