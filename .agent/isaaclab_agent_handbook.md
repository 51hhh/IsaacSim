# Isaac Lab Agent 编程参考手册

> 基于 Isaac Lab 官方文档（2026-02-10）整理
> 文档来源：https://isaac-sim.github.io/IsaacLab/main/

---

## 目录

1. [架构总览](#1-架构总览)
2. [核心概念](#2-核心概念)
3. [仓库结构](#3-仓库结构)
4. [两种任务设计模式](#4-两种任务设计模式)
5. [核心 API 模块一览](#5-核心-api-模块一览)
6. [环境类体系](#6-环境类体系)
7. [Manager-Based 环境详解](#7-manager-based-环境详解)
8. [Direct 环境详解](#8-direct-环境详解)
9. [预置环境列表](#9-预置环境列表)
10. [RL 库集成 (Wrappers)](#10-rl-库集成-wrappers)
11. [Standalone 应用开发模板](#11-standalone-应用开发模板)
12. [Extension 开发规范](#12-extension-开发规范)
13. [How-to 实用指南索引](#13-how-to-实用指南索引)
14. [教程索引](#14-教程索引)
15. [常用命令速查](#15-常用命令速查)
16. [关键注意事项与最佳实践](#16-关键注意事项与最佳实践)

---

## 1. 架构总览

### 1.1 Isaac 生态定位

```
NVIDIA Omniverse (平台层)
  └── PhysX (GPU 加速物理引擎)
  └── USD (通用场景描述)
  └── 渲染引擎 (光线追踪等)
      └── Isaac Sim (机器人仿真工具包)
          └── Isaac Lab (机器人学习框架) ← 我们使用的层级
```

- **Isaac Lab 不是仿真器**，而是构建在 Isaac Sim 之上的**机器人学习框架**
- 替代了已废弃的 IsaacGymEnvs、OmniIsaacGymEnvs 和 Orbit 框架
- 提供统一的、模块化的接口，用于 RL（强化学习）、IL（模仿学习）和运动规划

### 1.2 核心特性

| 特性 | 说明 |
|------|------|
| GPU 加速仿真 | 基于 PhysX 的端到端 GPU 流水线 |
| 向量化环境 | 支持数千个并行环境实例 |
| 执行器动力学 | 在仿真中模拟真实执行器特性 |
| 程序化地形生成 | 自动生成多种地形（平坦、崎岖等） |
| 传感器仿真 | 相机、激光雷达、接触传感器等 |
| 多频率仿真 | 传感器可以不同频率运行 |
| Tiled Rendering | 向量化渲染支持 |
| Hydra 配置管理 | 灵活的配置系统 |
| 遥操作接口 | 支持多种遥操作设备 |
| 人类演示数据采集 | 内置数据采集支持 |

---

## 2. 核心概念

### 2.1 任务 (Task)

任务通过特定的观测和动作接口定义了环境与智能体之间的交互。环境向智能体提供当前观测，并执行智能体发出的动作来推进仿真。

### 2.2 关键抽象类

| 类名 | 用途 |
|------|------|
| `AppLauncher` | 启动 Omniverse 应用 |
| `SimulationContext` | 控制仿真步进 |
| `InteractiveScene` | 高级场景管理接口 |
| `AssetBase` | 资产基类 |
| `RigidObject` | 刚体对象 |
| `Articulation` | 关节体（机器人） |
| `DeformableObject` | 可变形对象 |
| `SensorBase` | 传感器基类 |
| `Camera` | 相机传感器 |
| `RayCaster` | 射线投射传感器 |

### 2.3 执行器 (Actuators)

Isaac Lab 提供多种执行器模型，可以模拟真实机器人的执行器动力学特性，而不仅是理想化的关节控制。

### 2.4 运动生成器 (Motion Generators)

- **Differential IK (差分逆运动学)**：任务空间控制器
- **OSC (操作空间控制器)**：操作空间控制

---

## 3. 仓库结构

```
IsaacLab/
├── .vscode/
├── isaaclab.bat / isaaclab.sh    # 平台入口脚本
├── pyproject.toml
├── docs/
├── docker/
├── source/                       # 扩展源码
│   ├── isaaclab/                 # 核心扩展：执行器、资产、传感器、环境等
│   ├── isaaclab_assets/          # 预配置资产
│   ├── isaaclab_tasks/           # 预配置环境/任务
│   ├── isaaclab_mimic/           # 模仿学习数据生成
│   └── isaaclab_rl/              # RL 库 wrapper
├── scripts/                      # 独立脚本
│   ├── benchmarks/               # 基准测试
│   ├── demos/                    # 演示
│   ├── environments/             # 环境运行脚本
│   ├── imitation_learning/       # 模仿学习工作流
│   ├── reinforcement_learning/   # RL 训练/评估脚本
│   ├── tools/                    # 工具（资产转换、数据集生成等）
│   └── tutorials/                # 教程脚本
├── tools/
└── VERSION
```

### 3.1 核心扩展说明

| 扩展名 | 用途 |
|--------|------|
| `isaaclab` | 核心接口：执行器、资产、控制器、传感器、场景、环境 |
| `isaaclab_assets` | 预配置的机器人和对象资产 |
| `isaaclab_tasks` | 预置 RL/IL 任务环境 |
| `isaaclab_rl` | RL 库封装（RSL-RL, SKRL, SB3, RL-Games） |
| `isaaclab_mimic` | 模仿学习数据生成 |
| `isaaclab_contrib` | 社区贡献的执行器、资产、传感器 |

---

## 4. 两种任务设计模式

### 4.1 Manager-Based（管理器模式）

**适用场景**：原型设计、模块化协作、需要灵活切换配置

- 环境被分解为独立的管理器组件
- 用户主要编写**配置类**，框架负责协调
- 更模块化、更易于团队协作
- 组件可独立开发和替换

**核心类**：
- `ManagerBasedEnv` / `ManagerBasedEnvCfg`
- `ManagerBasedRLEnv` / `ManagerBasedRLEnvCfg`（RL 专用，添加奖励、终止等）

**五大管理器**：

| 管理器 | 职责 |
|--------|------|
| **Scene Manager** | 创建和管理虚拟世界（机器人、对象、传感器） |
| **Observation Manager** | 从仿真状态和传感器生成观测（可含特权信息） |
| **Action Manager** | 处理原始动作→低级命令（关节力矩/位置/末端执行器位姿） |
| **Event Manager** | 基于事件的操作（重置、随机推力、域随机化） |
| **Recorder Manager** | 记录仿真数据，按 episode/环境区分 |

**配置示例（Cartpole RewardsCfg）**：

```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # (1) 存活奖励
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) 失败惩罚
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) 主任务：保持杆竖直
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "target": 0.0,
        },
    )
    # (4) 塑形：降低小车速度
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) 塑形：降低杆角速度
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
```

### 4.2 Direct（直接模式）

**适用场景**：性能优先、复杂逻辑、类似 IsaacGymEnvs 风格

- 用户在单一类中实现所有功能
- 不需要管理器，完全自控
- 可使用 PyTorch JIT 和 Warp 优化
- 更透明，逻辑集中

**核心类**：
- `DirectRLEnv` / `DirectRLEnvCfg`（单智能体）
- `DirectMARLEnv` / `DirectMARLEnvCfg`（多智能体）

**奖励函数示例（Cartpole Direct）**：

```python
def _get_rewards(self) -> torch.Tensor:
    total_reward = compute_rewards(
        self.cfg.rew_scale_alive,
        self.cfg.rew_scale_terminated,
        self.cfg.rew_scale_pole_pos,
        self.cfg.rew_scale_cart_vel,
        self.cfg.rew_scale_pole_vel,
        self.joint_pos[:, self._pole_dof_idx[0]],
        self.joint_vel[:, self._pole_dof_idx[0]],
        self.joint_pos[:, self._cart_dof_idx[0]],
        self.joint_vel[:, self._cart_dof_idx[0]],
        self.reset_terminated,
    )
    return total_reward

@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(
        torch.square(pole_pos).unsqueeze(dim=1), dim=-1
    )
    rew_cart_vel = rew_scale_cart_vel * torch.sum(
        torch.abs(cart_vel).unsqueeze(dim=1), dim=-1
    )
    rew_pole_vel = rew_scale_pole_vel * torch.sum(
        torch.abs(pole_vel).unsqueeze(dim=1), dim=-1
    )
    total_reward = (
        rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    )
    return total_reward
```

### 4.3 模式对比

| 维度 | Manager-Based | Direct |
|------|---------------|--------|
| 模块化 | ✅ 高度模块化 | ❌ 单类实现 |
| 性能 | 一般 | ✅ 可 JIT 优化 |
| 易用性 | ✅ 配置驱动 | 需要更多手写代码 |
| 灵活性 | 组件可插拔 | ✅ 完全控制 |
| 团队协作 | ✅ 适合多人 | 适合个人 |
| 代码复用 | ✅ 管理器复用 | 需手动抽象 |
| 迁移来源 | Isaac Lab 原生 | IsaacGymEnvs 风格 |

---

## 5. 核心 API 模块一览

### `isaaclab` 主扩展模块

| 模块 | 路径 | 说明 |
|------|------|------|
| `isaaclab.app` | `isaaclab.app` | 应用启动功能 |
| `isaaclab.actuators` | `isaaclab.actuators` | 执行器模型（理想、隐式、显式、神经网络等） |
| `isaaclab.assets` | `isaaclab.assets` | 资产类（`RigidObject`, `Articulation`, `DeformableObject`） |
| `isaaclab.controllers` | `isaaclab.controllers` | 控制器和运动生成器 |
| `isaaclab.devices` | `isaaclab.devices` | 遥操作设备接口 |
| `isaaclab.envs` | `isaaclab.envs` | 环境定义（Manager-Based / Direct） |
| `isaaclab.envs.mdp` | `isaaclab.envs.mdp` | MDP 组件（预置的观测/奖励/终止函数） |
| `isaaclab.managers` | `isaaclab.managers` | 环境管理器 |
| `isaaclab.markers` | `isaaclab.markers` | 可视化标记工具 |
| `isaaclab.scene` | `isaaclab.scene` | 交互式场景定义（`InteractiveScene`） |
| `isaaclab.sensors` | `isaaclab.sensors` | 传感器（相机、射线投射等） |
| `isaaclab.sim` | `isaaclab.sim` | 仿真功能（`SimulationContext`、spawners） |
| `isaaclab.terrains` | `isaaclab.terrains` | 程序化地形生成 |
| `isaaclab.utils` | `isaaclab.utils` | 通用工具函数 |

---

## 6. 环境类体系

```
gym.Env
├── ManagerBasedEnv          # 管理器模式基础环境
│   └── ManagerBasedRLEnv    # 管理器模式 RL 环境（奖励、终止、信息）
├── DirectRLEnv              # 直接模式 RL 环境
└── DirectMARLEnv            # 直接模式多智能体 RL 环境
```

### 通用属性

| 属性 | 说明 |
|------|------|
| `num_envs` | 并行子环境数量 |
| `device` | 运行设备（cuda/cpu） |
| `max_episode_length_s` | 最大 episode 时长(秒) |
| `max_episode_length` | 最大 episode 步数 |
| `physics_dt` | 物理时间步长(秒) |
| `step_dt` | 环境步进时间步长(秒) |
| `is_vector_env` | 是否为向量化环境（始终为 True） |

### 关键方法

| 方法 | 说明 |
|------|------|
| `reset()` | 重置环境（**仅在首次 step 前调用一次**） |
| `step(actions)` | 执行动作，返回 (obs, reward, terminated, truncated, info) |
| `close()` | 关闭环境 |

> ⚠️ **重要**：向量化环境中，`reset()` 只在创建后调用一次。之后 `step()` 自动处理已终止子环境的重置。仿真器不支持单独重置某个子环境。

---

## 7. Manager-Based 环境详解

### 7.1 环境配置类 (`ManagerBasedRLEnvCfg`)

一个完整的 Manager-Based RL 环境配置通常包含以下部分：

```python
@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    """环境配置"""

    # 场景配置
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP 组件配置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # 仿真参数
    sim: SimulationCfg = SimulationCfg(dt=0.005)
    episode_length_s: float = 5.0
    decimation: int = 4  # step_dt = dt * decimation
```

### 7.2 场景配置

```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """场景配置"""
    # 地面
    ground = AssetBaseCfg(prim_path="/World/Ground", ...)
    # 机器人
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path="path/to/robot.usd"),
        actuators={"joints": ImplicitActuatorCfg(...)},
    )
    # 传感器
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*")
```

### 7.3 预置 MDP 函数 (`isaaclab.envs.mdp`)

以下是可在 Manager-Based 环境中直接使用的预置函数：

**观测函数** (`ObsTerm`)：
- `mdp.joint_pos` - 关节位置
- `mdp.joint_vel` - 关节速度
- `mdp.root_pos_w` - 根节点世界坐标位置
- `mdp.root_quat_w` - 根节点世界坐标四元数
- `mdp.root_lin_vel_w` - 根节点线速度
- `mdp.root_ang_vel_w` - 根节点角速度

**奖励函数** (`RewTerm`)：
- `mdp.is_alive` - 存活奖励
- `mdp.is_terminated` - 终止惩罚
- `mdp.joint_pos_target_l2` - 关节位置目标 L2 距离
- `mdp.joint_vel_l1` - 关节速度 L1 范数

**终止函数** (`DoneTerm`)：
- `mdp.time_out` - 超时终止

**事件函数** (`EventTerm`)：
- `mdp.reset_scene_to_default` - 重置到默认状态
- `mdp.randomize_rigid_body_mass` - 随机化质量
- `mdp.push_by_setting_velocity` - 随机推力

---

## 8. Direct 环境详解

### 8.1 必须实现的方法

继承 `DirectRLEnv` 后，需要实现以下方法：

```python
class MyDirectEnv(DirectRLEnv):
    cfg: MyDirectEnvCfg

    def __init__(self, cfg: MyDirectEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 初始化自定义变量

    def _setup_scene(self):
        """设置场景：生成资产、添加到场景"""
        pass

    def _pre_physics_step(self, actions: torch.Tensor):
        """物理步之前：处理动作"""
        pass

    def _apply_action(self):
        """应用处理后的动作到仿真"""
        pass

    def _get_observations(self) -> dict:
        """计算并返回观测"""
        return {"policy": obs_tensor}

    def _get_rewards(self) -> torch.Tensor:
        """计算奖励"""
        return reward_tensor

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """计算终止标志"""
        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """重置指定环境"""
        pass
```

### 8.2 配置类

```python
@configclass
class MyDirectEnvCfg(DirectRLEnvCfg):
    # 仿真参数
    sim: SimulationCfg = SimulationCfg(dt=0.005)
    decimation: int = 4
    episode_length_s: float = 5.0

    # 场景
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )

    # 观测和动作空间
    num_observations: int = 4
    num_actions: int = 1

    # 自定义参数
    rew_scale_alive: float = 1.0
    rew_scale_terminated: float = -2.0
```

---

## 9. 预置环境列表

### 9.1 经典控制 (Classic)

| 环境 ID | 模式 | 描述 |
|---------|------|------|
| `Isaac-Cartpole-v0` | Manager | 经典 Cartpole 倒立摆 |
| `Isaac-Cartpole-Direct-v0` | Direct | 经典 Cartpole 倒立摆 |
| `Isaac-Cartpole-RGB-v0` | Manager | Cartpole + 视觉观测 |
| `Isaac-Ant-v0` | Manager | MuJoCo 蚂蚁运动 |
| `Isaac-Ant-Direct-v0` | Direct | MuJoCo 蚂蚁运动 |
| `Isaac-Humanoid-v0` | Manager | MuJoCo 人形机器人 |
| `Isaac-Humanoid-Direct-v0` | Direct | MuJoCo 人形机器人 |

### 9.2 操控 (Manipulation)

| 环境 ID | 描述 |
|---------|------|
| `Isaac-Reach-Franka-v0` | Franka 末端到达目标位姿 |
| `Isaac-Reach-UR10-v0` | UR10 末端到达目标位姿 |
| `Isaac-Lift-Cube-Franka-v0` | Franka 抓取方块至目标位置 |
| `Isaac-Lift-Cube-Franka-IK-Abs-v0` | Franka + 绝对 IK 控制 |
| `Isaac-Lift-Cube-Franka-IK-Rel-v0` | Franka + 相对 IK 控制 |
| `Isaac-Stack-Cube-Franka-v0` | Franka 堆叠三个方块 |
| `Isaac-Open-Drawer-Franka-v0` | Franka 打开抽屉 |
| `Isaac-Franka-Cabinet-Direct-v0` | Franka 开柜门 (Direct) |
| `Isaac-Repose-Cube-Allegro-v0` | Allegro 手内操控 |
| `Isaac-Deploy-Reach-UR10e-v0` | UR10e 到达（已部署到真机） |

### 9.3 运动 (Locomotion)

| 环境 ID | 描述 |
|---------|------|
| `Isaac-Velocity-Flat-Anymal-B/C/D-v0` | Anymal 系列平坦地形速度跟踪 |
| `Isaac-Velocity-Rough-Anymal-B/C/D-v0` | Anymal 系列崎岖地形速度跟踪 |
| `Isaac-Velocity-Flat-Unitree-A1/Go1/Go2-v0` | Unitree 系列平坦地形 |
| `Isaac-Velocity-Rough-Unitree-A1/Go1/Go2-v0` | Unitree 系列崎岖地形 |
| `Isaac-Velocity-Flat-Spot-v0` | Boston Dynamics Spot |
| `Isaac-Velocity-Flat/Rough-H1-v0` | Unitree H1 人形 |
| `Isaac-Velocity-Flat/Rough-G1-v0` | Unitree G1 人形 |
| `Isaac-Velocity-Flat/Rough-Digit-v0` | Agility Digit |

### 9.4 导航 (Navigation)

| 环境 ID | 描述 |
|---------|------|
| `Isaac-Navigation-Flat-Anymal-C-v0` | Anymal C 导航至目标位置 |

### 9.5 无人机 (Multirotor)

| 环境 ID | 描述 |
|---------|------|
| `Isaac-Quadcopter-Direct-v0` | Crazyflie 四旋翼悬停 |
| `Isaac-TrackPositionNoObstacles-ARL-Robot-1-v0` | ARL 无人机位置追踪 |

### 9.6 其他

| 环境 ID | 描述 |
|---------|------|
| `Isaac-Humanoid-AMP-Dance/Run/Walk-Direct-v0` | 对抗运动先验（AMP）- 人形模仿 |

> **列出所有环境**：  
> `isaaclab.bat -p scripts\environments\list_envs.py --keyword <搜索词>`  
> （Linux: `./isaaclab.sh -p scripts/environments/list_envs.py --keyword <搜索词>`）

---

## 10. RL 库集成 (Wrappers)

`isaaclab_rl` 扩展提供以下 RL 库的封装：

| 库 | Wrapper 模块 | 说明 |
|----|-------------|------|
| **RSL-RL** | `isaaclab_rl.rsl_rl` | ETH Zurich 的 RL 库，四足常用 |
| **SKRL** | `isaaclab_rl.skrl` | 支持 AMP 等高级算法 |
| **Stable-Baselines3** | `isaaclab_rl.sb3` | 流行的 RL 库 |
| **RL-Games** | `isaaclab_rl.rl_games` | NVIDIA 出品 |

### 使用示例

训练脚本通常位于 `scripts/reinforcement_learning/` 目录下，可通过命令行运行：

```bash
# Windows
isaaclab.bat -p scripts/reinforcement_learning/train.py --task Isaac-Cartpole-v0

# Linux
./isaaclab.sh -p scripts/reinforcement_learning/train.py --task Isaac-Cartpole-v0
```

> **注意**：AMP 算法仅通过 SKRL 库支持。使用 `--algorithm AMP` 参数启用。  
> 评估时使用 `--real-time` 可实时运行交互循环。

---

## 11. Standalone 应用开发模板

Standalone 应用允许完全控制仿真步进，适合 RL 训练等需要同步控制的场景。

### 11.1 基础模板

```python
"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

# 1. 启动仿真器（必须在所有其他 import 之前）
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""
# 2. 只有启动后才能导入其他模块
from isaaclab.sim import SimulationContext

if __name__ == "__main__":
    # 3. 获取仿真上下文
    simulation_context = SimulationContext()
    # 4. 重置并开始仿真
    simulation_context.reset()
    # 5. 步进仿真
    simulation_context.step()
    # 6. 停止并关闭
    simulation_context.stop()
    simulation_app.close()
```

> ⚠️ **关键**：`AppLauncher` 必须在所有 Isaac Lab/Omniverse 模块导入之前调用。Omniverse 模块通过热加载方式在仿真器启动时才可用。

### 11.2 带环境的模板

```python
"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""
import gymnasium as gym
import isaaclab_tasks  # 注册所有任务

if __name__ == "__main__":
    # 创建环境
    env = gym.make("Isaac-Cartpole-v0", num_envs=64)
    
    # 重置
    obs, info = env.reset()
    
    # 训练循环
    for step in range(1000):
        actions = env.action_space.sample()  # 随机策略
        obs, rewards, terminated, truncated, info = env.step(actions)
    
    env.close()
    simulation_app.close()
```

---

## 12. Extension 开发规范

### 12.1 扩展目录结构

```
<extension-name>/
├── config/
│   └── extension.toml          # 扩展元数据（名称、版本、依赖等）
├── docs/
│   ├── CHANGELOG.md
│   └── README.md
├── <extension-name>/           # Python 包
│   ├── __init__.py
│   ├── ...
│   └── scripts/                # Omniverse 加载的脚本
├── setup.py                    # setuptools 构建脚本
└── tests/                      # 单元测试（unittest）
```

### 12.2 扩展入口类

```python
import omni.ext

class MyExt(omni.ext.IExt):
    """自定义扩展"""

    def on_startup(self, ext_id):
        """扩展加载时调用"""
        pass

    def on_shutdown(self):
        """扩展卸载时调用，清理资源"""
        pass
```

### 12.3 依赖管理

在 `extension.toml` 中指定非 Python 依赖：

```toml
[isaac_lab_settings]
# apt 依赖
apt_deps = ["libboost-all-dev"]
# ROS 工作空间
ros_ws = "/home/user/catkin_ws"
```

安装依赖：
```bash
python tools/install_deps.py all ${ISAACLAB_PATH}/source
```

---

## 13. How-to 实用指南索引

| 主题 | 说明 | 文档链接 |
|------|------|---------|
| 导入新资产 | URDF/MJCF → USD → 配置 | [import_new_asset](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html) |
| 编写 Articulation 配置 | 配置关节体资产 | [write_articulation_cfg](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html) |
| 创建固定资产 | 将浮动基座→固定基座 | [make_fixed_prim](https://isaac-sim.github.io/IsaacLab/main/source/how-to/make_fixed_prim.html) |
| 生成多种资产 | 在不同环境中放置不同对象 | [multi_asset_spawning](https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html) |
| 保存相机输出 | 保存渲染图像和 3D 重投影 | [save_camera_output](https://isaac-sim.github.io/IsaacLab/main/source/how-to/save_camera_output.html) |
| 配置渲染 | 渲染模式预设和自定义 | [configure_rendering](https://isaac-sim.github.io/IsaacLab/main/source/how-to/configure_rendering.html) |
| 绘制标记 | 使用 `VisualizationMarkers` | [draw_markers](https://isaac-sim.github.io/IsaacLab/main/source/how-to/draw_markers.html) |
| 封装 RL 环境 | 为不同 RL 库封装环境 | [wrap_rl_env](https://isaac-sim.github.io/IsaacLab/main/source/how-to/wrap_rl_env.html) |
| 添加自定义 RL 库 | 集成新的学习库 | [add_own_library](https://isaac-sim.github.io/IsaacLab/main/source/how-to/add_own_library.html) |
| 录制动画/视频 | 录制仿真过程 | [record_animation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/record_animation.html) |
| 课程学习 | 动态修改环境参数 | [curriculums](https://isaac-sim.github.io/IsaacLab/main/source/how-to/curriculums.html) |
| 优化 Stage 创建 | Fabric Cloning + Stage in Memory | [optimize_stage_creation](https://isaac-sim.github.io/IsaacLab/main/source/how-to/optimize_stage_creation.html) |
| 相机数量估算 | 估算机器可支持的相机数 | [estimate_cameras](https://isaac-sim.github.io/IsaacLab/main/source/how-to/estimate_how_many_cameras_can_run.html) |

---

## 14. 教程索引

### 14.1 基础仿真设置

| 教程 | 涉及 API |
|------|---------|
| 创建空场景 | `AppLauncher`, `SimulationContext` |
| 在场景中生成 Prims | `spawners` |
| 深入 AppLauncher | `AppLauncher` 详解 |

### 14.2 资产交互

| 教程 | 涉及 API |
|------|---------|
| 添加新机器人 | `ArticulationCfg` |
| 与刚体交互 | `RigidObject` |
| 与关节体交互 | `Articulation` |
| 与可变形体交互 | `DeformableObject` |
| 使用表面抓取器 | `SurfaceGripper` |

### 14.3 场景创建

| 教程 | 涉及 API |
|------|---------|
| 使用交互式场景 | `InteractiveScene` |

### 14.4 环境设计

| 教程 | 涉及 API |
|------|---------|
| 创建 Manager-Based 基础环境 | `ManagerBasedEnv` |
| 创建 Manager-Based RL 环境 | `ManagerBasedRLEnv` |
| 创建 Direct RL 环境 | `DirectRLEnv` |
| 注册环境 | Gymnasium 注册 |
| RL 训练 | 训练工作流 |
| 配置 RL Agent | Agent 超参数 |
| 修改现有 Direct 环境 | 环境自定义 |
| USD 环境中的策略推理 | 部署推理 |

### 14.5 传感器集成

| 教程 | 涉及 API |
|------|---------|
| 在机器人上添加传感器 | `Camera`, `RayCaster` |

### 14.6 运动生成器

| 教程 | 涉及 API |
|------|---------|
| 使用任务空间控制器 | Differential IK |
| 使用操作空间控制器 | OSC |

---

## 15. 常用命令速查

### 15.1 环境操作

```bash
# 列出所有可用环境
isaaclab.bat -p scripts\environments\list_envs.py

# 按关键词过滤
isaaclab.bat -p scripts\environments\list_envs.py --keyword Cartpole

# 运行随机策略测试
isaaclab.bat -p scripts\environments\random_agent.py --task Isaac-Cartpole-v0

# 运行零动作策略
isaaclab.bat -p scripts\environments\zero_agent.py --task Isaac-Cartpole-v0
```

### 15.2 训练与评估

```bash
# 训练（使用默认 RL 库）
isaaclab.bat -p scripts\reinforcement_learning\train.py --task Isaac-Cartpole-v0

# 评估/播放已训练策略
isaaclab.bat -p scripts\reinforcement_learning\play.py --task Isaac-Cartpole-v0

# 实时评估
isaaclab.bat -p scripts\reinforcement_learning\play.py --task Isaac-Cartpole-v0 --real-time

# 使用 AMP 算法（仅 SKRL）
isaaclab.bat -p scripts\reinforcement_learning\train.py --task Isaac-Humanoid-AMP-Walk-Direct-v0 --algorithm AMP
```

### 15.3 视觉环境

```bash
# 启用相机（视觉观测任务必须）
isaaclab.bat -p scripts\reinforcement_learning\train.py --task Isaac-Cartpole-RGB-v0 --enable_cameras
```

### 15.4 工具

```bash
# 转换 URDF 到 USD
isaaclab.bat -p scripts\tools\convert_urdf.py input.urdf output.usd

# 安装扩展依赖
python tools/install_deps.py all ${ISAACLAB_PATH}/source
```

---

## 16. 关键注意事项与最佳实践

### 16.1 Import 顺序

```python
# ✅ 正确：先启动 AppLauncher，再导入其他模块
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# 此后才能导入
from isaaclab.sim import SimulationContext
import isaaclab_tasks
```

```python
# ❌ 错误：在启动前导入会导致模块缺失错误
from isaaclab.sim import SimulationContext  # ModuleNotFoundError!
from isaaclab.app import AppLauncher
```

### 16.2 向量化环境

- 所有环境都是向量化的，观测和动作都是批量 Tensor
- `reset()` 仅在第一次 `step()` 前调用一次
- `step()` 内部自动处理已终止子环境的重置
- 不支持单独重置某个子环境

### 16.3 动作空间抽象层

同一个任务可以配置不同的动作空间：
- **Joint Position** (`joint_pos`)：直接关节位置控制
- **IK Absolute** (`ik_abs`)：绝对逆运动学控制
- **IK Relative** (`ik_rel`)：相对逆运动学控制
- **Joint Torque**：关节力矩控制

### 16.4 性能优化

- Direct 模式下使用 `@torch.jit.script` 加速计算密集函数
- 使用 Fabric Cloning 和 Stage in Memory 优化场景创建
- 合理设置 `decimation`（控制频率 = 物理频率 / decimation）
- 视觉任务启用 Tiled Rendering

### 16.5 Gymnasium 注册

环境需要通过 Gymnasium 注册才能使用 `gym.make()`：

```python
import gymnasium as gym

gym.register(
    id="My-Custom-Env-v0",
    entry_point="my_package.envs:MyCustomEnv",
    kwargs={"cfg": MyCustomEnvCfg},
)
```

### 16.6 `.bat` vs `.sh`

| 平台 | 入口脚本 |
|------|---------|
| Windows | `isaaclab.bat -p <script>` |
| Linux | `./isaaclab.sh -p <script>` |

> `-p` 参数用于指定要运行的 Python 脚本，确保使用 Isaac Sim 内置的 Python 解释器。

---

## 附录 A：文档链接速查

| 资源 | URL |
|------|-----|
| Isaac Lab 主页 | https://isaac-sim.github.io/IsaacLab/main/ |
| API 参考 | https://isaac-sim.github.io/IsaacLab/main/source/api/index.html |
| 教程 | https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html |
| How-to 指南 | https://isaac-sim.github.io/IsaacLab/main/source/how-to/index.html |
| 环境列表 | https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html |
| 核心概念 | https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/index.html |
| 任务工作流 | https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/task_workflows.html |
| 开发者指南 | https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/index.html |
| 仓库结构 | https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/repo_structure.html |
| 生态说明 | https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html |
| GitHub 仓库 | https://github.com/isaac-sim/IsaacLab |

---

*本手册基于 Isaac Lab 官方文档（最后更新：2026-02-10）整理，旨在为 AI Agent 编写 IsaacLab 相关代码提供快速参考。*
