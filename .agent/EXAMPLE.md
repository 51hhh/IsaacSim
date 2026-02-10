1. 官方基准环境 (Official Tasks)
Isaac Lab 核心代码库中自带了超过 30 个可以直接训练的环境案例，涵盖了从基础到复杂的各种机器人形态： 
双足/四足机器人 (Locomotion):
ANYmal (C/D/Low-state): 使用 RSL-RL 训练经典的四足机器人行走任务。
Unitree (Go2/H1/G1): 支持宇树科技的人形和四足机器人，适用于全身控制（Whole-body control）训练。
机械臂操作 (Manipulation):
Franka Emika: 包括基础的 Lift（提升）和 Reach（触达）任务。
Allegro Hand: 复杂的灵巧手多指操作任务。
入门级项目:
Cartpole: 最基础的倒立摆平衡，适合熟悉 Stable-Baselines3 工作流。
Ant: 训练蚂蚁机器人向前快速奔跑。 


## 官方文档
https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html

## 示例项目位置
isaaclab: C:\IsaacLab
isaacsim: C:\IsaacSim

生成不同的手臂并应用随机的关节位置指令：
C:\IsaacSim\python.bat C:\IsaacLab\scripts\demos\arms.py
生成不同的可变形（软体）物体，并让它们从高处坠落：
C:\IsaacSim\python.bat C:\IsaacLab\scripts\demos\deformables.py
使用交互场景，在各个环境中生成不同的资源：
C:\IsaacSim\python.bat C:\IsaacLab\scripts\demos\multi_asset.py
利用互动场景生成一个简单的平行机器人进行拣选和放置：
C:\IsaacSim\python.bat C:\IsaacLab\scripts\demos\pick_and_place.py
在默认环境中生成一架四旋翼：
C:\IsaacSim\python.bat C:\IsaacLab\scripts\demos\quadcopter.py


isaac-lab-circular-cartpole sim4.5迁移的多级倒立摆实例
https://gitee.com/reinforced-kane/isaac-lab-circular-cartpole
./isaac-lab-circular-cartpole

WheeledLab 开源移动机器人的环境、资产和工作流程，集成于 IsaacLab。
https://github.com/UWRobotLearning/WheeledLab
./WheeledLab
