# Isaac Sim 项目环境配置文档

## 环境信息

| 项目 | 版本/路径 |
|------|----------|
| Isaac Sim | 5.1.0-rc.19 (`C:\IsaacSim`) |
| Isaac Lab | 0.54.3 (`C:\IsaacLab`) |
| 内置 Python | 3.11.13 |
| PyTorch | 2.7.0+cu128 |
| 操作系统 | Windows |
| WSL ROS2 | Ubuntu 22.04 + ROS2 Humble |

## 目录结构

```
isaacsim/
├── .vscode/
│   └── settings.json           # VS Code 配置
├── cartpole_simple.py          # 简单倒立摆（PD 控制，Isaac Sim 原生 API）
├── cartpole_rl_train.py        # RL 训练倒立摆（PPO，Isaac Lab）
└── README.md                   # 本文档
```

---

## Python 环境使用

### 核心原则

**必须使用 Isaac Sim 内置的 Python**，不要用系统 Python 或 conda。

```bat
:: 运行脚本
C:\IsaacSim\python.bat your_script.py

:: pip 安装包
C:\IsaacSim\python.bat -m pip install <package>

:: 进入交互模式
C:\IsaacSim\python.bat
```

### 验证命令

```bat
:: 检查 Python 版本
cmd /c "C:\IsaacSim\python.bat --version"

:: 验证 isaacsim 可导入
cmd /c "C:\IsaacSim\python.bat -c ""import isaacsim; print('OK')"""
```

---

## Isaac Lab 安装（已完成 ✅）

Isaac Lab 已安装于 `C:\IsaacLab`，包含以下组件：

| 包 | 版本 | 说明 |
|----|------|------|
| isaaclab | 0.54.3 | 核心框架 |
| isaaclab_assets | 0.2.4 | 机器人资产（Cartpole, Franka 等） |
| isaaclab_tasks | 0.11.13 | 预定义任务环境 |
| isaaclab_rl | 0.4.7 | 强化学习工具 |

### 验证安装

```bat
cmd /c "C:\IsaacSim\python.bat -c ""import isaaclab; print('Isaac Lab OK')"""
```

### 重新安装（如需要）

```bat
cd C:\IsaacLab
set ISAACSIM_PATH=C:\IsaacSim
C:\IsaacSim\python.bat -m pip install -e source/isaaclab
C:\IsaacSim\python.bat -m pip install -e source/isaaclab_assets
C:\IsaacSim\python.bat -m pip install -e source/isaaclab_tasks
C:\IsaacSim\python.bat -m pip install -e source/isaaclab_rl
```

---

## ROS2 集成配置

### 架构

```
┌─────────────────────────┐     DDS (FastRTPS)     ┌──────────────────────┐
│   Windows: Isaac Sim    │ ◄──────────────────► │  WSL: ROS2 Humble    │
│   ROS2 Bridge 扩展      │   localhost 通信      │  ROS2 节点           │
└─────────────────────────┘                       └──────────────────────┘
```

### Windows 端设置

```bat
:: 方式一：运行 ROS 环境脚本后再运行仿真
call C:\IsaacSim\setup_ros_env.bat
C:\IsaacSim\python.bat your_ros_script.py
```

Python 脚本中启用 ROS2 Bridge：

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni
omni.isaac.core.utils.extensions.enable_extension("isaacsim.ros2.bridge")
```

### WSL 端设置

```bash
source /opt/ros/humble/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# 验证话题
ros2 topic list
```

---

## 运行示例

### 1. 简单倒立摆 - PD 控制（Isaac Sim 原生 API）

```bat
C:\IsaacSim\python.bat C:\Users\Rick\Desktop\isaacsim\cartpole_simple.py
```

### 2. 强化学习倒立摆 - PPO 训练（Isaac Lab）

```bat
:: 训练 PPO 策略（1024 并行环境，GPU 加速）
C:\IsaacSim\python.bat C:\Users\Rick\Desktop\isaacsim\cartpole_rl_train.py --num_envs 1024

:: 调试模式（更少环境，有渲染）
C:\IsaacSim\python.bat C:\Users\Rick\Desktop\isaacsim\cartpole_rl_train.py --num_envs 64

:: 快速训练（无头模式，不渲染）
C:\IsaacSim\python.bat C:\Users\Rick\Desktop\isaacsim\cartpole_rl_train.py --num_envs 4096 --headless
```

### 3. 使用 Isaac Lab 官方训练脚本


```bat
:: 使用 SKRL PPO
C:\IsaacLab\isaaclab.bat -p C:\IsaacLab\scripts\reinforcement_learning\skrl\train.py --task Isaac-Cartpole-Direct-v0 --num_envs 1024

:: 使用 RSL-RL PPO
C:\IsaacLab\isaaclab.bat -p C:\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Cartpole-Direct-v0 --num_envs 1024
```

测试模型

C:\IsaacSim\python.bat C:\IsaacLab\scripts\reinforcement_learning\skrl\play.py --task Isaac-Cartpole-Direct-v0 --num_envs 32 --checkpoint C:\logs\skrl\cartpole_direct\2026-02-09_14-28-22_ppo_torch\checkpoints\best_agent.pt

### 4. 官方入门示例

```bat
C:\IsaacSim\python.bat C:\IsaacSim\standalone_examples\tutorials\getting_started.py
```

---

## VS Code 配置

已配置 `.vscode/settings.json`：
- Python 解释器指向 Isaac Sim 内置 Python
- extraPaths 包含 isaacsim 模块路径

如果智能提示不工作，按 `Ctrl+Shift+P` → `Python: Select Interpreter` → 选择 `C:\IsaacSim\kit\python\python.exe`

---

## 常见问题

### GPU 物理加速未启用

Isaac Lab 默认使用 GPU 物理加速。如果看到警告说没有使用 GPU，检查：

1. **NVIDIA 驱动** >= 525.60
2. **CUDA 版本** >= 11.8
3. 在环境配置中确认：
   ```python
   sim: SimulationCfg = SimulationCfg(device="cuda:0")
   ```

### ModuleNotFoundError: No module named 'isaaclab'

运行以下命令重新安装：

```bat
cd C:\IsaacLab
set ISAACSIM_PATH=C:\IsaacSim
C:\IsaacSim\python.bat -m pip install -e source/isaaclab
```

### ROS2 话题在 WSL 中看不到

1. 确保两端都使用 `rmw_fastrtps_cpp`
2. 检查 Windows 防火墙是否放行 DDS 端口
3. 在脚本中确认已启用 ROS2 Bridge 扩展

### GPU 内存不足

减少并行环境数量或使用 headless 模式：

```bat
C:\IsaacSim\python.bat script.py --num_envs 256 --headless
```

### URDF 导入警告 "No mass specified"

这是 URDF 文件中某些 link 没有定义质量，通常不影响仿真功能。

### pip 安装包时报错 "ModuleNotFoundError: No module named 'pkg_resources'"

这是由于新版 setuptools (>=82.0.0) 移除了 `pkg_resources` 模块导致的。解决方案：

**方法一：降级 setuptools（推荐）**

```bat
C:\IsaacSim\python.bat -m pip install setuptools==69.5.1
```

**方法二：使用 --no-build-isolation 参数**

```bat
C:\IsaacSim\python.bat -m pip install <package> --no-build-isolation
```

此参数让 pip 直接使用系统已安装的 setuptools，而不是创建隔离的构建环境。

### isaaclab 依赖缺失或版本冲突

如果看到 isaaclab 相关依赖警告，运行以下命令安装/修复：

```bat
:: 安装缺失的依赖
C:\IsaacSim\python.bat -m pip install einops flaky junitparser pytest pytest-mock --no-build-isolation
C:\IsaacSim\python.bat -m pip install hidapi==0.14.0.post2 prettytable==3.3.0 pyglet==1.5.29 --no-build-isolation
C:\IsaacSim\python.bat -m pip install onnx transformers==4.57.6 warp-lang --no-build-isolation
C:\IsaacSim\python.bat -m pip install docstring-parser==0.16 --no-build-isolation

:: 修复版本兼容问题
C:\IsaacSim\python.bat -m pip install gymnasium==1.2.1 pillow==11.3.0 starlette==0.49.1 --no-build-isolation
C:\IsaacSim\python.bat -m pip install numpy==1.26.4 packaging==23.2 --no-build-isolation
```

**已验证的依赖版本（isaaclab 0.54.3）**：

| 包 | 版本 |
|----|------|
| numpy | 1.26.4 |
| gymnasium | 1.2.1 |
| pillow | 11.3.0 |
| starlette | 0.49.1 |
| packaging | 23.2 |
| transformers | 4.57.6 |
| hidapi | 0.14.0.post2 |
| prettytable | 3.3.0 |
| pyglet | 1.5.29 |

---

## 相关资源

- [Isaac Sim 文档](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [内置示例路径](C:\IsaacSim\standalone_examples)

---

## Isaac Sim vs Isaac Lab 对比

| 特性 | Isaac Sim 原生 API | Isaac Lab |
|------|-------------------|-----------|
| **安装** | 内置 | 需单独安装 |
| **环境数量** | 单个或少量 | 数千并行环境 |
| **数据结构** | NumPy 数组 | PyTorch 张量（GPU） |
| **用途** | 可视化、调试、简单控制 | 强化学习训练 |
| **自动克隆** | ❌ 手动 | ✅ 自动 |
| **控制方式** | PD/PID 手动控制 | 神经网络策略 |
| **示例** | `cartpole_simple.py` | `cartpole_rl_train.py` |

---

## 示例脚本说明

| 脚本 | 说明 | 控制方式 | GPU 加速 |
|------|------|----------|---------|
| `cartpole_simple.py` | Isaac Sim 原生 API，单环境 | PD 控制器 | ✅ |
| `cartpole_rl_train.py` | Isaac Lab + SKRL PPO，多环境并行 | 神经网络策略 | ✅ |

**Isaac Lab 不随 Isaac Sim 一起安装**，它是独立的机器人学习框架，构建在 Isaac Sim 之上。
