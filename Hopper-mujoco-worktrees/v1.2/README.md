# Hopper-MuJoCo

基于 MuJoCo 的 Hopper 机器人仿真环境，实现 Raibert + 虚拟弹簧控制器，支持 zero-shot sim-to-real。

## 项目结构

```
Hopper-mujoco/
├── mjcf/
│   └── hopper_serial.xml      # 串联等效腿 MJCF 模型
├── config/
│   └── hopper_config.py       # 机器人参数配置
├── controllers/
│   ├── raibert_controller.py  # Raibert + 虚拟弹簧控制器
│   ├── forward_kinematics.py  # 3-RSR 正运动学
│   └── com_filter.py          # 互补滤波器
├── utils/
│   └── mujoco_interface.py    # MuJoCo 状态接口
└── scripts/
    └── run_raibert_mj.py      # 主入口（键盘遥控）
```

## 理论基础

### 1. 串联仿真 + Torque Mapping（论文 §3）

本项目采用与论文相同的两层架构：
- **仿真层**：使用等效串联腿（Roll/Pitch/Shift）进行仿真
- **映射层**：通过 Jacobian 将串联扭矩映射为真实 3-RSR 电机扭矩

```
τ_3RSR = J^{-T} @ f_foot
```

### 2. Raibert 控制器（论文 §2.3）

- **Flight Phase**：Raibert 足端放置
  ```
  x_target = Kv * v_current + Kr * v_desired
  ```
- **Stance Phase**：虚拟弹簧 + 能量环
  ```
  F_spring = -k * (l - l0) + Kp * error
  ```

### 3. Zero-Shot Sim-to-Real（论文 §4）

关键：仿真和真机使用相同的控制律，只在最后一步做 torque mapping。

## 运行方法

```bash
cd /home/abc/Hopper/Hopper-mujoco
python scripts/run_raibert_mj.py
```

### 键盘控制
- `Y`: +X 速度（前进）
- `H`: -X 速度（后退）
- `G`: -Y 速度（左移）
- `J`: +Y 速度（右移）
- `Space`: 速度归零

## 参考文献

- Mini Cheetah MPC + Raibert Heuristics
- PogoX: Parallel Leg Hopping Robot
- 论文中的 Serial-to-Parallel Torque Mapping

## 代码来源

- `Hopper4.py`: 原始 Raibert 控制器
- `forward_kinematics.py`: 3-RSR 正运动学
- `hopper.urdf`: 串联等效腿模型



