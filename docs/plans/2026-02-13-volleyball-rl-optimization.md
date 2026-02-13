# 排球接球 RL 训练优化实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 优化排球接球 RL 训练的奖励设计、观察空间和训练机制，提升训练效率和接球成功率

**Architecture:** 基于 ETH 羽毛球论文的约束 RL 设计，扩展观察空间到 14 维，使用非线性距离奖励，增加网络容量到 256x256x128，添加完整惩罚机制

**Tech Stack:** Isaac Lab, PyTorch, RSL-RL, USD/PhysX

---

## 前置依赖

确保以下文件存在且可访问：
- `volleyball/isaaclab/volleyball_catch_env.py` - 主环境
- `volleyball/isaaclab/volleyball_catch_cfg.py` - 环境配置
- `volleyball/config/ppo_config.yaml` - PPO 训练配置

---

## Task 1: 扩展观察空间到 14 维

**Files:**
- Modify: `volleyball/isaaclab/volleyball_catch_cfg.py:136-137`
- Modify: `volleyball/isaaclab/volleyball_catch_env.py:247-272`

**Step 1: 更新配置中的观察空间维度**

```python
# volleyball/isaaclab/volleyball_catch_cfg.py 第 136-137 行
observation_space = 14  # 从 8 改为 14
```

**Step 2: 在 _get_observations 中添加新观察项**

```python
def _get_observations(self) -> Dict[str, torch.Tensor]:
    ball_pos = self.ball.data.root_pos_w
    ball_vel = self.ball.data.root_lin_vel_w
    ball_ang_vel = self.ball.data.root_ang_vel_w
    robot_pos = self.robot.data.root_pos_w
    robot_vel = self.robot.data.root_lin_vel_w
    robot_quat = self.robot.data.root_quat_w  # 四元数
    
    # 计算偏航角 (Wz) 从四元数
    robot_yaw = torch.atan2(
        2.0 * (robot_quat[:, 3] * robot_quat[:, 2] + robot_quat[:, 0] * robot_quat[:, 1]),
        1.0 - 2.0 * (robot_quat[:, 1] ** 2 + robot_quat[:, 2] ** 2)
    )
    
    pred_land = self._predict_landing_point(ball_pos, ball_vel)
    self.predicted_landing = pred_land
    
    t_land = self._calc_time_to_landing(ball_pos, ball_vel)
    self.time_to_landing = t_land
    
    # 14 维观察空间
    obs = torch.cat([
        # 原有观察 (8D)
        pred_land[:, 0:1] - robot_pos[:, 0:1],  # rel_x
        pred_land[:, 1:2] - robot_pos[:, 1:2],  # rel_y
        t_land.unsqueeze(-1),                    # time_to_land
        robot_pos[:, 0:1],                       # robot_x (场地坐标系)
        robot_pos[:, 1:2],                       # robot_y
        robot_vel[:, 0:1],                       # robot_vx
        robot_vel[:, 1:2],                       # robot_vy
        ball_pos[:, 2:3],                        # ball_z
        # 新增观察 (6D)
        robot_yaw.unsqueeze(-1),                 # robot_yaw (Wz)
        ball_vel[:, 0:1],                        # ball_vx
        ball_vel[:, 1:2],                        # ball_vy
        ball_vel[:, 2:3],                        # ball_vz
        ball_ang_vel[:, 0:1],                    # ball_ang_vel_x
        ball_ang_vel[:, 1:2],                    # ball_ang_vel_y
    ], dim=-1)
    
    self._prev_dist = torch.norm(
        pred_land[:, :2] - robot_pos[:, :2], dim=-1
    )
    
    return {"policy": obs}
```

**Step 3: 验证观察空间维度**

运行测试脚本验证观察空间正确性：
```bash
cd C:\Users\Rick\Desktop\isaacsim\volleyball\isaaclab
python test_env.py --num_envs 2
```

Expected: 无错误，观察空间维度为 14

**Step 4: Commit**

```bash
git add volleyball/isaaclab/volleyball_catch_env.py volleyball/isaaclab/volleyball_catch_cfg.py
git commit -m "feat: extend observation space to 14D with robot yaw and ball velocities"
```

---

## Task 2: 实现非线性距离奖励函数

**Files:**
- Modify: `volleyball/isaaclab/volleyball_catch_env.py:302-363`
- Modify: `volleyball/isaaclab/volleyball_catch_cfg.py:167-178`

**Step 1: 更新奖励配置参数**

```python
# volleyball/isaaclab/volleyball_catch_cfg.py 第 167-178 行
# 移除阶梯式奖励参数，添加非线性奖励参数
rew_catch = 100.0
rew_miss_base = -10.0
rew_miss_dist_scale = -5.0
rew_timeout = -20.0
rew_dist_exp_scale = 2.0      # 距离奖励指数系数
rew_dist_max = 3.0            # 最大距离奖励
rew_approach_scale = 2.0
rew_wait = 0.5
rew_time_penalty = -0.02
rew_boundary = -10.0          # 增加出界惩罚
rew_net_collision = -5.0
rew_robot_ball_collision = -3.0  # 新增车球碰撞惩罚
```

**Step 2: 重写 _get_rewards 方法**

```python
def _get_rewards(self) -> torch.Tensor:
    ball_pos = self.ball.data.root_pos_w
    ball_vel = self.ball.data.root_lin_vel_w
    robot_pos = self.robot.data.root_pos_w
    robot_vel = self.robot.data.root_lin_vel_w
    
    pred_land = self.predicted_landing
    dist_to_land = torch.norm(pred_land - robot_pos[:, :2], dim=-1)
    dist_to_ball = torch.norm(ball_pos[:, :2] - robot_pos[:, :2], dim=-1)
    ball_z = ball_pos[:, 2]
    robot_z = robot_pos[:, 2]
    
    reward = torch.zeros(self.num_envs, device=self.device)
    
    # ========== 终止条件检测 ==========
    catch_height_thresh = robot_z + self.cfg.catch_height_margin
    grounded = ball_z < self.cfg.ground_threshold
    
    # 接球成功：球在接球高度范围内且在接球半径内
    caught = (ball_z <= catch_height_thresh) & (ball_z > self.cfg.ground_threshold) & (dist_to_ball < self.cfg.catch_radius)
    
    # 接球失败：球触地但未接住
    missed = grounded & ~caught
    
    # ========== 非线性距离奖励 ==========
    # 使用指数衰减：越接近落点，奖励增长越快
    # 公式: r = r_max * exp(-scale * distance)
    dist_reward = self.cfg.rew_dist_max * torch.exp(
        -self.cfg.rew_dist_exp_scale * dist_to_land
    )
    
    # 距离小于 0.5m 时额外奖励
    close_bonus = torch.where(
        dist_to_land < 0.5,
        self.cfg.rew_wait * (1.0 - dist_to_land / 0.5),  # 线性衰减到 0.5m
        torch.zeros_like(reward)
    )
    
    # ========== 接近奖励 ==========
    # 计算向落点移动的进度
    approach_reward = (self._prev_dist - dist_to_land) * self.cfg.rew_approach_scale
    approach_reward = torch.clamp(approach_reward, -1.0, 1.0)
    
    # ========== 时间惩罚 ==========
    time_penalty = torch.full_like(reward, self.cfg.rew_time_penalty)
    
    # ========== 出界惩罚 ==========
    robot_out = (
        (robot_pos[:, 0] < self.cfg.robot_bounds_x[0]) |
        (robot_pos[:, 0] > self.cfg.robot_bounds_x[1]) |
        (robot_pos[:, 1] < self.cfg.robot_bounds_y[0]) |
        (robot_pos[:, 1] > self.cfg.robot_bounds_y[1])
    )
    boundary_penalty = torch.where(
        robot_out, 
        torch.full_like(reward, self.cfg.rew_boundary), 
        torch.zeros_like(reward)
    )
    
    # ========== 碰网惩罚 ==========
    net_collision = self._check_net_collision(robot_pos)
    net_penalty = torch.where(
        net_collision,
        torch.full_like(reward, self.cfg.rew_net_collision),
        torch.zeros_like(reward)
    )
    
    # ========== 车球碰撞惩罚（非接球情况）==========
    robot_ball_collision = self._check_robot_ball_collision(
        robot_pos, ball_pos, dist_to_ball, caught
    )
    collision_penalty = torch.where(
        robot_ball_collision,
        torch.full_like(reward, self.cfg.rew_robot_ball_collision),
        torch.zeros_like(reward)
    )
    
    # ========== 汇总持续奖励 ==========
    ongoing_reward = (
        dist_reward + 
        close_bonus + 
        approach_reward + 
        time_penalty + 
        boundary_penalty + 
        net_penalty + 
        collision_penalty
    )
    
    # ========== 终止奖励/惩罚 ==========
    # 成功接球
    reward = torch.where(caught, torch.full_like(reward, self.cfg.rew_catch), reward)
    
    # 未接住（球触地）
    miss_penalty = self.cfg.rew_miss_base + self.cfg.rew_miss_dist_scale * dist_to_ball
    reward = torch.where(missed, miss_penalty, reward)
    
    # 持续回合奖励
    ongoing_mask = ~caught & ~missed
    reward = torch.where(ongoing_mask, ongoing_reward, reward)
    
    return reward
```

**Step 3: 添加碰撞检测辅助方法**

```python
def _check_net_collision(self, robot_pos: torch.Tensor) -> torch.Tensor:
    """检测机器人是否与球网碰撞"""
    # 球网位置在 x=0，宽度为 court_width + 1.0
    net_x_range = 0.3  # 网厚度 + 机器人半径
    net_y_range = (self.cfg.court_width / 2) + 0.5
    
    net_collision = (
        (robot_pos[:, 0].abs() < net_x_range) &
        (robot_pos[:, 1].abs() < net_y_range)
    )
    return net_collision

def _check_robot_ball_collision(
    self, 
    robot_pos: torch.Tensor, 
    ball_pos: torch.Tensor,
    dist_to_ball: torch.Tensor,
    caught: torch.Tensor
) -> torch.Tensor:
    """检测机器人与球的非接球碰撞"""
    # 机器人半径约 0.3m，球半径 0.105m
    robot_radius = 0.3
    ball_radius = 0.105
    collision_dist = robot_radius + ball_radius
    
    # 碰撞检测：距离小于碰撞距离且不是成功接球
    collision = (dist_to_ball < collision_dist) & ~caught
    return collision
```

**Step 4: 运行测试验证奖励计算**

```bash
cd C:\Users\Rick\Desktop\isaacsim\volleyball\isaaclab
python test_env.py --num_envs 2 --verbose
```

Expected: 奖励值合理，无 NaN 或异常值

**Step 5: Commit**

```bash
git add volleyball/isaaclab/volleyball_catch_env.py volleyball/isaaclab/volleyball_catch_cfg.py
git commit -m "feat: implement exponential distance reward and penalty mechanisms"
```

---

## Task 3: 增加网络容量到 256x256x128

**Files:**
- Modify: `volleyball/config/ppo_config.yaml:28-43`

**Step 1: 更新 Actor 网络配置**

```yaml
# volleyball/config/ppo_config.yaml
actor:
  class_name: MLPModel
  hidden_dims: [256, 256, 128]  # 从 [128, 128] 增加
  activation: elu
  obs_normalization: true
  stochastic: true
  init_noise_std: 0.5
  noise_std_type: "scalar"
  state_dependent_std: false
```

**Step 2: 更新 Critic 网络配置**

```yaml
critic:
  class_name: MLPModel
  hidden_dims: [256, 256, 128]  # 从 [128, 128] 增加
  activation: elu
  obs_normalization: true
  stochastic: false
```

**Step 3: 记录网络参数数量**

训练时会自动打印网络结构，检查参数数量是否合理增长。

**Step 4: Commit**

```bash
git add volleyball/config/ppo_config.yaml
git commit -m "feat: increase network capacity to 256x256x128"
```

---

## Task 4: 实现固定发球（诱导学习）

**Files:**
- Modify: `volleyball/isaaclab/volleyball_catch_env.py:395-430`

**Step 1: 修改 _reset_idx 方法使用固定发球**

```python
def _reset_idx(self, env_ids: torch.Tensor):
    super()._reset_idx(env_ids)
    
    num_reset = len(env_ids)
    
    # ========== 固定发球参数（诱导学习）==========
    # 所有环境使用相同的固定发球
    serve_y = torch.zeros(num_reset, device=self.device)  # 固定 Y=0
    serve_z = self.cfg.ball_cfg.init_state.pos[2]         # 固定高度
    
    # 发球速度和角度固定
    serve_speed = self.cfg.ball_cfg.init_state.lin_vel[0]
    serve_angle_rad = math.radians(20.0)  # 固定 20 度角
    
    # 计算发球初速度
    ball_pos = torch.zeros(num_reset, 3, device=self.device)
    ball_pos[:, 0] = self.cfg.ball_cfg.init_state.pos[0]  # SERVE_X
    ball_pos[:, 1] = serve_y
    ball_pos[:, 2] = serve_z
    
    ball_vel = torch.zeros(num_reset, 3, device=self.device)
    ball_vel[:, 0] = serve_speed  # x 方向速度
    ball_vel[:, 2] = serve_speed * math.sin(serve_angle_rad)  # z 方向速度
    
    # 无旋转（固定）
    ball_ang_vel = torch.zeros(num_reset, 3, device=self.device)
    
    # 应用初始状态
    default_ball_state = self.ball.data.default_root_state[env_ids]
    default_ball_state[:, :3] = ball_pos
    default_ball_state[:, 7:10] = ball_vel
    default_ball_state[:, 10:13] = ball_ang_vel
    
    self.ball.write_root_state_to_sim(default_ball_state, env_ids)
    
    # ========== 固定机器人初始位置 ==========
    # 在接发球区域固定位置
    robot_x = torch.full((num_reset,), 5.0, device=self.device)  # 固定 X=5.0
    robot_y = torch.zeros(num_reset, device=self.device)         # 固定 Y=0
    
    default_robot_state = self.robot.data.default_root_state[env_ids]
    default_robot_state[:, 0] = robot_x
    default_robot_state[:, 1] = robot_y
    default_robot_state[:, 2] = self.cfg.robot_cfg.init_state.pos[2]
    default_robot_state[:, 7:13] = 0.0
    
    self.robot.write_root_state_to_sim(default_robot_state, env_ids)
    
    # 重置关节状态
    joint_pos = self.robot.data.default_joint_pos[env_ids]
    joint_vel = self.robot.data.default_joint_vel[env_ids]
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    # 重置距离跟踪
    self._prev_dist[env_ids] = 0.0
```

**Step 2: 验证固定发球**

```bash
cd C:\Users\Rick\Desktop\isaacsim\volleyball\isaaclab
python test_env.py --num_envs 4 --episodes 5 --verbose
```

Expected: 每次 episode 发球位置、速度、角度完全相同

**Step 3: Commit**

```bash
git add volleyball/isaaclab/volleyball_catch_env.py
git commit -m "feat: use fixed serve and robot position for inductive learning"
```

---

## Task 5: 添加场地坐标系转换（左后角为原点）

**Files:**
- Modify: `volleyball/isaaclab/volleyball_catch_env.py:247-272`
- Modify: `volleyball/isaaclab/volleyball_catch_cfg.py:32-42`

**Step 1: 定义场地坐标系参数**

```python
# volleyball/isaaclab/volleyball_catch_cfg.py
# 场地坐标系：左后角为原点
COURT_ORIGIN_X = -9.0  # 场地中心在 x=0，所以左边界在 -9.0
COURT_ORIGIN_Y = -4.5  # 场地中心在 y=0，所以后边界在 -4.5

ROBOT_BOUNDS_X = (0.0, COURT_LENGTH)  # 机器人活动范围（场地坐标系）
ROBOT_BOUNDS_Y = (0.0, COURT_WIDTH)
```

**Step 2: 添加世界坐标系到场地坐标系的转换**

```python
def _world_to_court_coords(self, world_pos: torch.Tensor) -> torch.Tensor:
    """将世界坐标转换为场地坐标（左后角为原点）"""
    court_pos = world_pos.clone()
    # 世界坐标: 中心在 (0,0)
    # 场地坐标: 左后角在 (0,0)，所以 x += 9.0, y += 4.5
    court_pos[:, 0] = world_pos[:, 0] + 9.0  # X: 0 to 18
    court_pos[:, 1] = world_pos[:, 1] + 4.5  # Y: 0 to 9
    return court_pos
```

**Step 3: 在观察中使用场地坐标**

```python
def _get_observations(self) -> Dict[str, torch.Tensor]:
    # ... 获取世界坐标 ...
    
    # 转换为场地坐标
    robot_pos_court = self._world_to_court_coords(robot_pos)
    ball_pos_court = self._world_to_court_coords(ball_pos)
    
    # 计算预测落点的场地坐标
    pred_land_court = pred_land.clone()
    pred_land_court[:, 0] = pred_land[:, 0] + 9.0
    pred_land_court[:, 1] = pred_land[:, 1] + 4.5
    
    # 使用场地坐标构建观察
    obs = torch.cat([
        # ...
        robot_pos_court[:, 0:1],  # 场地 X (0-18)
        robot_pos_court[:, 1:2],  # 场地 Y (0-9)
        # ...
    ], dim=-1)
```

**Step 4: Commit**

```bash
git add volleyball/isaaclab/volleyball_catch_env.py volleyball/isaaclab/volleyball_catch_cfg.py
git commit -m "feat: add court coordinate system with origin at left-back corner"
```

---

## Task 6: 运行完整测试验证

**Files:**
- Create: `volleyball/isaaclab/test_optimized_env.py`

**Step 1: 编写测试脚本**

```python
#!/usr/bin/env python3
"""优化后的环境测试脚本"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from volleyball_catch_env import VolleyballCatchEnv
from volleyball_catch_cfg import VolleyballCatchEnvCfg


def test_observation_space():
    """测试观察空间维度"""
    print("\n=== Testing Observation Space ===")
    cfg = VolleyballCatchEnvCfg()
    cfg.scene.num_envs = 4
    
    env = VolleyballCatchEnv(cfg=cfg, headless=True)
    obs_dict = env.reset()
    obs = obs_dict[0]["policy"] if isinstance(obs_dict, tuple) else obs_dict["policy"]
    
    expected_dim = 14
    actual_dim = obs.shape[-1]
    
    print(f"Expected observation dim: {expected_dim}")
    print(f"Actual observation dim: {actual_dim}")
    assert actual_dim == expected_dim, f"Observation dim mismatch!"
    print("✓ Observation space test passed!")


def test_reward_function():
    """测试奖励函数"""
    print("\n=== Testing Reward Function ===")
    cfg = VolleyballCatchEnvCfg()
    cfg.scene.num_envs = 2
    
    env = VolleyballCatchEnv(cfg=cfg, headless=True)
    obs = env.reset()
    
    # 随机动作
    actions = torch.randn(2, 2, device=env.device)
    next_obs, rewards, terminated, truncated, info = env.step(actions)
    
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"No NaN in rewards: {not torch.isnan(rewards).any()}")
    
    assert rewards.shape[0] == 2, "Reward batch size mismatch!"
    assert not torch.isnan(rewards).any(), "NaN in rewards!"
    print("✓ Reward function test passed!")


def test_fixed_serve():
    """测试固定发球"""
    print("\n=== Testing Fixed Serve ===")
    cfg = VolleyballCatchEnvCfg()
    cfg.scene.num_envs = 4
    
    env = VolleyballCatchEnv(cfg=cfg, headless=True)
    
    # 重置两次，检查发球是否相同
    obs1 = env.reset()
    ball_pos1 = env.ball.data.root_pos_w.clone()
    ball_vel1 = env.ball.data.root_lin_vel_w.clone()
    
    obs2 = env.reset()
    ball_pos2 = env.ball.data.root_pos_w.clone()
    ball_vel2 = env.ball.data.root_lin_vel_w.clone()
    
    pos_diff = torch.abs(ball_pos1 - ball_pos2).max()
    vel_diff = torch.abs(ball_vel1 - ball_vel2).max()
    
    print(f"Ball position diff: {pos_diff:.6f}")
    print(f"Ball velocity diff: {vel_diff:.6f}")
    
    assert pos_diff < 1e-5, "Fixed serve position changed!"
    assert vel_diff < 1e-5, "Fixed serve velocity changed!"
    print("✓ Fixed serve test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Volleyball Catch Environment Optimization Tests")
    print("=" * 60)
    
    try:
        test_observation_space()
        test_reward_function()
        test_fixed_serve()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

**Step 2: 运行测试**

```bash
cd C:\Users\Rick\Desktop\isaacsim\volleyball\isaaclab
python test_optimized_env.py
```

Expected: 所有测试通过

**Step 3: Commit**

```bash
git add volleyball/isaaclab/test_optimized_env.py
git commit -m "test: add comprehensive tests for optimized environment"
```

---

## Task 7: 运行训练验证

**Step 1: 启动短周期训练测试**

```bash
cd C:\Users\Rick\Desktop\isaacsim\volleyball\isaaclab
python train.py --num_envs 64 --max_iterations 100 --headless
```

**Step 2: 监控训练指标**

Expected:
- 无 NaN 损失
- 奖励值稳步上升
- 接球率随训练提升

**Step 3: 检查 TensorBoard**

```bash
tensorboard --logdir=logs/rsl_rl/volleyball_catch
```

检查指标：
- `rewards/mean` 应该上升
- `episode_length/mean` 应该稳定
- 无异常波动

---

## 总结

实施完成后，环境将具备：

✅ **14 维观察空间**（机器人偏航角 + 球速度 + 旋转）
✅ **非线性距离奖励**（指数衰减函数）
✅ **256x256x128 网络**（更大容量）
✅ **完整惩罚机制**（出界、碰网、车球碰撞）
✅ **固定发球**（诱导学习）
✅ **场地坐标系**（左后角为原点）

这些优化应该显著提升训练效率和策略性能。
