# Hopper_sim

Hopper 机器人仿真环境集合，包含两种不同的仿真模型。

## 📁 目录结构

```
Hopper_sim/
├── model_aero/          # LCM 虚拟通信的 Hopper-aero 仿真
│   ├── mujoco_lcm_fake_robot.py    # MuJoCo 仿真 + LCM 通信
│   ├── forward_kinematics.py       # 正运动学
│   └── motor_utils.py              # 电机模型（PWM ↔ 推力）
│
└── model_spring/        # Mode1 虚拟弹簧控制器（成功的 Raibert 实现）
    ├── controllers/
    │   └── raibert_controller.py   # Raibert + 虚拟弹簧控制器
    ├── scripts/
    │   ├── run_raibert_mj.py       # 主运行脚本
    │   └── record_task1.sh          # Task1 录制脚本
    ├── config/
    │   └── hopper_config.py        # 机器人参数配置
    └── mjcf/
        └── hopper_serial.xml        # MuJoCo 串联腿模型
```

## 🚀 model_aero: LCM 虚拟通信仿真

### 功能
- 使用 MuJoCo 仿真机器人物理
- 通过 LCM 与 ModeE 控制器通信（完全兼容真机 LCM 协议）
- 可以运行真实的 `run_modee.py` 控制器进行测试

### 使用方法

**终端 1 (仿真机器人):**
```bash
cd Hopper_sim/model_aero
python3 mujoco_lcm_fake_robot.py --arm --viewer
```

**终端 2 (ModeE 控制器):**
```bash
cd Hopper-aero/hopper_controller
python3 run_modee.py
```

### 特点
- ✅ 完全兼容真机 LCM 消息格式
- ✅ 支持 `hopper_data_lcmt`, `hopper_imu_lcmt`, `gamepad_lcmt`
- ✅ 支持 `hopper_cmd_lcmt`, `motor_pwm_lcmt` 命令
- ✅ 可以录制视频 (`--record-mp4`)
- ✅ 支持 HUD 显示 (`--hud`)

## 🌸 model_spring: Mode1 虚拟弹簧控制器

### 功能
- Raibert 足端放置 + 虚拟弹簧控制
- 成功的跳跃实现（Task1 优化参数）
- 支持键盘控制

### 使用方法

**运行仿真:**
```bash
cd Hopper_sim/model_spring
python3 scripts/run_raibert_mj.py
```

**录制 Task1 视频:**
```bash
cd Hopper_sim/model_spring
bash scripts/record_task1.sh
```

### 键盘控制
- `Y`: +X 速度（前进）
- `H`: -X 速度（后退）
- `G`: -Y 速度（左移）
- `J`: +Y 速度（右移）
- `Space`: 速度归零
- `R`: 重置机器人
- `Q/ESC`: 退出

### 特点
- ✅ 虚拟弹簧控制（k=1500, b=45）
- ✅ Raibert 足端放置（Kv=0.08, Kr=0.012）
- ✅ 姿态控制（hip torque）
- ✅ 成功的 Task1 实现（0.3m/s 前进 + 原地跳）

## 📝 依赖

### 共同依赖
- Python 3.8+
- NumPy
- MuJoCo Python bindings
- LCM (Lightweight Communications and Marshalling)

### model_aero 额外依赖
- `hopper_lcm_types` (LCM 消息定义)
- `modee.controllers.motor_utils` (电机模型)

### model_spring 额外依赖
- `controllers.com_filter` (互补滤波器)
- `utils.mujoco_interface` (MuJoCo 接口)
- `utils.state_estimator` (状态估计器)

## 🔗 相关项目

- **Hopper-aero**: 真机控制代码 (`/home/abc/Hopper/Hopper-aero/`)
- **Hopper-mujoco**: 完整 MuJoCo 仿真环境 (`/home/abc/Hopper/Hopper-mujoco/`)

## 📚 参考

- Raibert 控制器论文
- Mini Cheetah MPC + Raibert Heuristics
- PogoX: Parallel Leg Hopping Robot

