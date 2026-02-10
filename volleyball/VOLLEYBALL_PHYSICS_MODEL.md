# 🏐 排球运动学物理模型设计文档

> 基于学术论文和实验数据整理的真实排球物理模型参数

---

## 1. 排球基本参数（FIVB 官方规格）

| 参数 | 值 | 说明 |
|------|-----|------|
| 直径 (d) | 0.210 m (21.0 cm) | FIVB 标准: 20.5–21.5 cm |
| 半径 (r) | 0.105 m | |
| 质量 (m) | 0.270 kg | FIVB 标准: 260–280 g |
| 内部气压 | 0.300–0.325 kgf/cm² | 约 4.26–4.61 psi |
| 截面积 (A) | π × r² = 0.03464 m² | |
| 转动惯量 | I = ⅔mr² = 0.00199 kg·m² | 薄壳球近似 |

---

## 2. 空气动力学模型

### 2.1 阻力模型 (Drag Force)

排球飞行中的空气阻力力公式：

```
F_drag = ½ · ρ · v² · A · Cd
```

其中：
- **ρ** = 1.225 kg/m³（海平面标准空气密度）
- **v** = 球的飞行速度 (m/s)
- **A** = π × r² = 0.03464 m² (截面积)
- **Cd** = 阻力系数（动态变化）

#### 阻力系数 Cd 的动态模型（阻力危机 Drag Crisis）

排球阻力系数不是常数！它随雷诺数（Reynolds Number）动态变化：

```
Re = (d × v) / ν
```
其中 ν = 1.516 × 10⁻⁵ m²/s（20°C 空气运动黏度）

| 速度范围 (m/s) | Re 范围 | Cd 值 | 说明 |
|---------------|---------|-------|------|
| 0–7 (低速) | 0–1.0×10⁵ | 0.45–0.50 | 层流 |
| 7–14 (中速) | 1.0–2.0×10⁵ | 0.50→0.20 | 过渡区（阻力危机） |
| 14–20 (临界速度) | 2.0–2.8×10⁵ | 0.20→0.10 | 阻力急剧下降 |
| >20 (高速) | >2.8×10⁵ | 0.10–0.17 | 湍流，超临界 |

**采用的模型**（分段函数 + 平滑过渡）：

```python
def drag_coefficient(speed):
    """基于雷诺数的动态阻力系数"""
    Re = (0.210 * speed) / 1.516e-5
    
    if Re < 1.0e5:           # 低速层流
        Cd = 0.50
    elif Re < 2.0e5:         # 过渡区
        t = (Re - 1.0e5) / 1.0e5
        Cd = 0.50 - 0.30 * t  # 线性插值 0.50→0.20
    elif Re < 2.8e5:         # 阻力危机区
        t = (Re - 2.0e5) / 0.8e5
        Cd = 0.20 - 0.08 * t  # 0.20→0.12
    else:                    # 超临界
        Cd = 0.12
    
    return Cd
```

数据来源：
- Asai et al. (2010) "Fundamental aerodynamics of the soccer ball"
- 文献 Cd 范围: 0.10（超临界）~0.50（亚临界）
- 排球临界 Re ≈ 2.2–3.1 × 10⁵

---

### 2.2 马格努斯力模型 (Magnus Effect)

旋转的排球会在飞行中产生侧向力（马格努斯力）：

```
F_magnus = ½ · ρ · v² · A · Cl · (ω̂ × v̂)
```

其中：
- **Cl** = 升力系数（与自旋比有关）
- **ω̂** = 角速度方向单位向量
- **v̂** = 速度方向单位向量
- 力的方向 = ω × v 的方向

#### 自旋比 (Spin Ratio / Spin Parameter)

```
S = (ω × r) / v
```

- ω = 角速度 (rad/s)
- r = 球半径 (m)  
- v = 飞行速度 (m/s)

#### 升力系数 Cl 与自旋比的关系

| 自旋比 S | Cl 值 | 说明 |
|---------|-------|------|
| 0 | 0 | 无旋转（飘球） |
| 0.1 | ~0.05 | 低旋转 |
| 0.2 | ~0.12 | 中等旋转 |
| 0.3 | ~0.20 | 较强旋转 |
| 0.5 | ~0.30 | 强上旋/下旋 |
| 1.0 | ~0.40 | 极强旋转（罕见） |

**采用的模型**：

```python
def lift_coefficient(spin_ratio):
    """基于自旋比的马格努斯升力系数"""
    Cl = 0.60 * spin_ratio  # 线性近似，S < 0.5时精度好
    Cl = min(Cl, 0.40)       # 上限截断
    return Cl
```

数据来源：
- 实验值: Cl ≈ 0.25（排球典型值）
- 自旋系数 Cs 范围: 0.25–1.0（球体一般值）
- Watts & Ferrer (1987) for spinning spheres

---

### 2.3 飘球效应 (Float Effect)

飘球（无旋转发球）的不规则飞行轨迹：

当排球**几乎不旋转**时（ω < 2 rad/s），气流不稳定会导致球的飞行轨迹产生随机偏移。这就是飘球之所以难以接住的原因。

**模型**：对无旋转球施加随机侧向扰动力

```python
def float_perturbation(speed, spin_rate):
    """飘球效应：低旋转时的随机气动扰动"""
    if spin_rate > 3.0:  # rad/s
        return (0, 0, 0)  # 高旋转时无飘球效应
    
    # 扰动强度与速度有关（速度越高，扰动越大）
    amplitude = 0.5 * speed * (1.0 - spin_rate / 3.0)
    
    # 随机侧向和垂直扰动（模拟Knuckleball效应）
    fx = random.gauss(0, amplitude * 0.3)
    fy = random.gauss(0, amplitude)
    fz = random.gauss(0, amplitude * 0.5)
    
    return (fx, fy, fz)
```

数据来源：
- 飘球发球中观察到在垂直和水平方向偏移可达 1–2 米
- Frontiers in Sports and Active Living (2021)

---

## 3. 碰撞模型

### 3.1 恢复系数 (COR - Coefficient of Restitution)

```
COR = v_rebound / v_impact
```

| 碰撞表面 | COR | 说明 |
|---------|-----|------|
| 硬木地板 | 0.72–0.78 | 典型室内排球场 |
| 混凝土 | 0.75–0.80 | 更硬的表面 |
| 沙地 | 0.15–0.30 | 沙滩排球（大量能量吸收） |
| 人体（接球） | 0.40–0.55 | 手臂接球 |
| 球网 | 0.15–0.25 | 球网几乎不反弹 |

**COR 随碰撞速度的变化**：

```python
def dynamic_restitution(base_cor, impact_speed):
    """COR 随碰撞速度增大而减小"""
    # 高速碰撞 → 更大形变 → 更多能量耗散
    speed_factor = max(0, 1.0 - 0.005 * impact_speed)
    return base_cor * speed_factor
```

数据来源：
- COR 随碰撞速度增大而降低（非线性）
- ResearchGate: "Stiffness, energy loss and COR of volleyball"
- 气压越高 → COR 越高

### 3.2 碰撞摩擦

| 参数 | 值 | 说明 |
|------|-----|------|
| 静摩擦系数 | 0.55–0.65 | 球与地板 |
| 动摩擦系数 | 0.45–0.55 | 球与地板 |
| 球网摩擦 | 0.70–0.80 | 球与球网 |

---

## 4. 发球类型与参数

### 4.1 力量跳发 (Jump Serve / Power Serve)

| 参数 | 男子 | 女子 |
|------|------|------|
| 球速 | 73–104 km/h (20.3–28.9 m/s) | 66–89 km/h (18.3–24.7 m/s) |
| 旋转速率 | 40–100 rad/s (6–16 rev/s) | 30–80 rad/s |
| 发球高度 | 3.0–3.5 m | 2.5–3.0 m |
| 接球反应时间 | 0.52–0.74 秒 | |
| 旋转类型 | 上旋 (topspin) | |

### 4.2 飘球发球 (Float Serve)

| 参数 | 男子 | 女子 |
|------|------|------|
| 球速 | 42–75 km/h (11.7–20.8 m/s) | 40–61 km/h (11.1–16.9 m/s) |
| 旋转速率 | <2 rad/s (几乎不旋转) | <2 rad/s |
| 发球高度 | 2.3–2.8 m | 2.1–2.5 m |
| 接球反应时间 | 0.72–1.35 秒 | |
| 特点 | 飘忽不定的轨迹 | |

### 4.3 普通上手发球 (Standing Overhand)

| 参数 | 值 |
|------|-----|
| 球速 | 50–70 km/h (13.9–19.4 m/s) |
| 旋转速率 | 5–30 rad/s |
| 发球高度 | 2.3–2.8 m |

---

## 5. 完整力学方程 (运动方程)

排球飞行中受力分析：

```
m · a = F_gravity + F_drag + F_magnus + F_float
```

展开为：

```
m · (dvx/dt) = -½ρv²ACd·(vx/|v|) + F_magnus_x + F_float_x
m · (dvy/dt) = -½ρv²ACd·(vy/|v|) + F_magnus_y + F_float_y
m · (dvz/dt) = -mg - ½ρv²ACd·(vz/|v|) + F_magnus_z + F_float_z
```

其中：
- F_gravity = (0, 0, -mg) = (0, 0, -2.6487 N)
- F_drag 方向与速度方向相反
- F_magnus 方向 = ω × v 的方向
- F_float = 随机扰动（仅低旋转时）

角速度衰减（空气中的旋转衰减）：

```
dω/dt = -C_spin_decay · |v| · |ω|
```

C_spin_decay ≈ 0.001–0.005（经验系数）

---

## 6. 在 Isaac Sim 中的实现策略

### 6.1 PhysX 原生支持 vs 自定义力

| 特性 | PhysX 原生 | 自定义力（每帧施加） |
|------|-----------|-------------------|
| 重力 | ✅ 内置 | ✅ |
| 碰撞 | ✅ 内置 | ✅ |
| 线性阻尼 | ✅ 但为常数 | 需要自定义变速阻力 |
| 马格努斯力 | ❌ 不支持 | **必须自定义** |
| 飘球效应 | ❌ 不支持 | **必须自定义** |
| 阻力危机 | ❌ 不支持 | **必须自定义** |

### 6.2 实现方案

1. **重力 + 碰撞**：使用 PhysX 内置
2. **将 PhysX 线性阻尼设为 0**：完全由自定义模型控制
3. **每物理帧计算**：空气阻力 + 马格努斯力 + 飘球扰动
4. **通过 `PhysxForceAPI` 或修改速度**来施加自定义力

---

## 7. 参考文献

1. Asai, T., Ito, S., Seo, K., & Hitotsubashi, A. (2010). "Aerodynamics of a new volleyball." *Procedia Engineering*.
2. Hong, S., et al. (2014). "Aerodynamic properties of new volleyballs." *MDPI Applied Sciences*.
3. Wei, Q., et al. (2020). "The drag crisis and aerodynamics of volleyball." *Research Gate*.
4. Kao, S., et al. (2013). "Determination of coefficient of restitution of bouncing volleyball."
5. Frontiers in Sports and Active Living (2021). "Float serve trajectory analysis."
6. The Sport Journal. "Volleyball serve speeds: float vs power jump serve."
