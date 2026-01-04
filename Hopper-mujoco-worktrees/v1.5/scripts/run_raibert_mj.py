#!/usr/bin/env python3
"""
Hopper MuJoCo 仿真主脚本

运行 Raibert + 虚拟弹簧控制器，支持键盘遥控

使用方法：
    python scripts/run_raibert_mj.py

键盘控制：
    Y: +X 速度（前进）
    H: -X 速度（后退）
    G: -Y 速度（左移）
    J: +Y 速度（右移）
    Space: 速度归零
    R: 重置机器人
    Q/ESC: 退出

参考论文：
- 串联仿真 + Torque Mapping（§3）
- Raibert 控制器（§2.3）
- Zero-Shot Sim-to-Real（§4）
"""

import os
import sys
import time
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'controllers'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'utils'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'config'))

import mujoco
import imageio
import socket
import json

from controllers.raibert_controller import RaibertController
from controllers.raibert_controller_mode2 import RaibertControllerMode2
from controllers.com_filter import ComplementaryFilter
from utils.state_estimator import HopperStateEstimator, EstimatorConfig
from utils.ekf_state_estimator import HopperESKF, EKFConfig
from utils.mujoco_interface import MuJoCoInterface
from utils.domain_randomization import DomainRandomizer, DomainRandomConfig

# 旋翼相关模块（仅在 prop/leg_prop 模式需要）
try:
    from controllers.se3_controller import SE3Controller, State as QuadState
    from controllers.propeller_mixer import PropellerMixer
    from controllers.tri_rotor_mixer import TriRotorMixer
    from controllers.motor_utils import MotorModel
    PROP_AVAILABLE = True
except ImportError:
    PROP_AVAILABLE = False
    SE3Controller = None
    QuadState = None
    PropellerMixer = None
    TriRotorMixer = None
    MotorModel = None


class HopperSimulation:
    """
    Hopper MuJoCo 仿真类
    
    集成：
    - MuJoCo 物理仿真
    - Raibert + 虚拟弹簧控制器
    - 互补滤波器 (Sim-to-Real State Estimation)
    - 键盘遥控
    - 实时可视化
    """
    
    def __init__(self, model_path=None, mode='leg', model=None, data=None, verbose=True, controller_mode: int = 1):
        """
        初始化仿真
        
        Args:
            model_path: MJCF 模型路径
        """
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'mjcf', 'hopper_serial.xml')
        self.model_path = model_path
        
        if model is None:
            if verbose:
                print(f"Loading model from: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        else:
            if verbose:
                print("Using externally provided MuJoCo model/data for HopperSimulation.")
            self.model = model
            self.data = data if data is not None else mujoco.MjData(model)
        
        # 创建接口
        self.interface = MuJoCoInterface(self.model, self.data)
        
        # ========== Domain Randomization (Sim-to-Real) ==========
        self.domain_rand_enabled = True  # 是否启用 Domain Randomization
        self.domain_rand_config = DomainRandomConfig()
        self.domain_randomizer = DomainRandomizer(self.model, self.data, self.domain_rand_config)
        
        # 模式切换：leg=仅腿控，prop=旋翼飞行，leg_prop=腿控+旋翼姿态辅助
        self.mode = mode  # 'leg', 'prop', 'leg_prop'
        
        # 检查旋翼模式是否可用
        if mode in ('prop', 'leg_prop') and not PROP_AVAILABLE:
            raise ImportError("旋翼模式需要安装 se3_controller, propeller_mixer, tri_rotor_mixer, motor_utils")
        
        self.leg_enabled = self.mode in ('leg', 'leg_prop')
        self.gravity = 9.81
        self.viewer_fixed_z = 0.8

        # 创建控制器（mode1/mode2）
        self.controller_mode = int(controller_mode)
        if self.controller_mode == 2:
            self.controller = RaibertControllerMode2()
        else:
            self.controller = RaibertController()
        self.enable_propeller = self.mode in ('prop', 'leg_prop')
        self.propeller_mixer = None

        # ========== 三旋翼参数 ==========
        # 电机参数（调整以匹配 Hopper 重量）
        # 原参考值 Ct=3.25e-4 太小，三电机最大推力只有 0.47N
        # Hopper 重量 ~32N，需要更大的电机
        # 设计：单电机最大推力 ~5N，三电机最大 ~15N（约 0.5*mg）
        self.Ct = 0.01          # 电机推力系数 (N/krpm^2)，增大约 30 倍
        self.Cd = 2.5e-4        # 电机反扭系数 (Nm/krpm^2)，相应增大
        self.max_motor_speed = 22.0  # 电机最大转速 (krpm)
        self.max_thrust_per_motor = self.Ct * self.max_motor_speed ** 2  # ~4.84 N
        
        # 姿态控制增益（Flight Phase 主动调正姿态）
        # 注意：这些是力矩增益，需要根据机器人惯量调整
        # Hopper 惯量约 Ixx≈0.5, Iyy≈0.5 kg·m²
        # 期望带宽 ~10 rad/s → Kp ≈ I * ωn² ≈ 0.5 * 100 = 50
        # 阻尼比 ζ=0.7 → Kd ≈ 2*ζ*ωn*I ≈ 2*0.7*10*0.5 = 7
        self.Kp_roll = 30.0     # Roll 比例增益 (Nm/rad)
        self.Kd_roll = 5.0      # Roll 微分增益 (Nm/(rad/s))
        self.Kp_pitch = 30.0    # Pitch 比例增益 (Nm/rad)
        self.Kd_pitch = 5.0     # Pitch 微分增益 (Nm/(rad/s))
        
        # 基础推力（提供一定升力辅助，减轻腿部负担）
        # 0.15 * mg ≈ 4.7N，约占体重 15%
        self.prop_base_thrust_ratio = 0.15  # 基础推力 = 0.15 * mg
        self.prop_torque_scale = 1.0  # prop/SE3 输出力矩缩放（默认 1.0）
        
        # 三旋翼混控器
        self.tri_rotor_mixer = None

        if self.mode == 'prop':
            # 纯旋翼飞行模式：使用 SE3 控制器
            self.se3_controller = SE3Controller()
            self.se3_controller.kx = 0.6
            self.se3_controller.kv = 0.4
            self.se3_controller.kR = 6.0
            self.se3_controller.kw = 1.0
            self.motor_model = MotorModel(ct=self.Ct, cd=self.Cd, max_speed=self.max_motor_speed)
            self.tri_rotor_mixer = TriRotorMixer(
                prop_positions=self.interface.prop_positions_body,
                Ct=self.Ct, Cd=self.Cd, max_speed=self.max_motor_speed
            )
            self.prop_goal_state = QuadState(
                position=np.array([0.0, 0.0, 1.2]),
                velocity=np.zeros(3),
                quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                omega=np.zeros(3)
            )
            self.prop_forward = np.array([1.0, 0.0, 0.0])
            self.last_prop_thrust = np.zeros(3)
            self.last_motor_speeds = np.zeros(3)
            self.last_prop_force_cmd = np.zeros(3)
            self.last_prop_torque_cmd = np.zeros(3)
        elif self.mode == 'leg_prop':
            # 腿控 + 旋翼姿态辅助模式：SE3(SO3误差)姿态控制 + 三旋翼混控
            self.se3_controller = SE3Controller()
            self.motor_model = MotorModel(ct=self.Ct, cd=self.Cd, max_speed=self.max_motor_speed)
            self.tri_rotor_mixer = TriRotorMixer(
                prop_positions=self.interface.prop_positions_body,
                Ct=self.Ct, Cd=self.Cd, max_speed=self.max_motor_speed
            )
            self.last_prop_thrust = np.zeros(3)
            self.last_motor_speeds = np.zeros(3)
            self.last_prop_force_cmd = np.zeros(3)
            self.last_prop_torque_cmd = np.zeros(3)
        else:
            # 纯腿控模式
            self.se3_controller = None
            self.motor_model = None
            self.tri_rotor_mixer = None
            self.last_prop_thrust = np.zeros(3)
            self.last_motor_speeds = np.zeros(3)
            self.last_prop_force_cmd = np.zeros(3)
            self.last_prop_torque_cmd = np.zeros(3)
        
        # 创建状态估计器
        self.com_filter = ComplementaryFilter()
        self.use_estimator = False  # 关闭估计器，使用真实速度（调参阶段）
        self.state_estimator = HopperStateEstimator(EstimatorConfig())
        self.ekf_estimator = HopperESKF(EKFConfig())
        self.estimator_kind = 'complementary'  # 'complementary' | 'ekf'
        # True: 用估计速度闭环控制；False: 仅对比/记录，不影响控制
        self.estimator_feed_control = False
        
        # 期望速度（向前走 demo）
        self.desired_vel = np.array([0.1, 0.0])  # 向前 0.1 m/s
        
        # 仿真参数
        self.dt = self.model.opt.timestep
        self.control_freq = 1000  # Hz
        self.control_dt = 1.0 / self.control_freq
        self.steps_per_control = max(1, int(self.control_dt / self.dt))
        
        # 日志
        self.log_data = {
            'time': [],
            'state': [],
            'body_pos': [],
            'body_vel': [],
            'joint_pos': [],
            'torque': [],
            'foot_pos': [],
            'target_foot_pos': [],
            'desired_vel': [],
            'body_quat': [],
            'body_rpy': [],
            'current_energy': [],
            'target_energy': [],
            'energy_error': [],
            'leg_len': [],
            'spring_force': [],
            'prop_force_cmd': [],
            'prop_torque_cmd': [],
            'prop_thrust': [],
            'motor_speed': []
        }
        
        # 状态
        self.running = True
        self.paused = False
        self.sim_time = 0.0
        
        # UDP 仪表盘
        self.dashboard_enabled = True
        self.dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dashboard_addr = ('127.0.0.1', 9999)
        self.dashboard_interval = 0.02  # 50 Hz
        self.last_dashboard_time = 0.0
        
        # UDP 遥控接收
        self.teleop_enabled = False
        self.teleop_socket = None
        self.teleop_vel = np.array([0.0, 0.0])  # 遥控期望速度
        
        print(f"Simulation initialized:")
        print(f"  - Timestep: {self.dt*1000:.2f} ms")
        print(f"  - Control freq: {self.control_freq} Hz")
        print(f"  - Steps per control: {self.steps_per_control}")
        if self.dashboard_enabled:
            print(f"  - Dashboard UDP: {self.dashboard_addr}")
    
    def enable_teleop(self, port=9998):
        """启用 UDP 遥控接收"""
        self.teleop_enabled = True
        self.teleop_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.teleop_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.teleop_socket.bind(('127.0.0.1', port))
        self.teleop_socket.setblocking(False)
        print(f"  - Teleop UDP listening on port {port}")
    
    def recv_teleop(self):
        """接收遥控命令（非阻塞）"""
        if not self.teleop_enabled or self.teleop_socket is None:
            return
        try:
            while True:  # 读取所有待处理的数据包
                data, _ = self.teleop_socket.recvfrom(1024)
                msg = json.loads(data.decode('utf-8'))
                self.teleop_vel[0] = msg.get('vx', 0.0)
                self.teleop_vel[1] = msg.get('vy', 0.0)
        except BlockingIOError:
            pass  # 没有更多数据
        except Exception:
            pass
    
    def reset(self):
        """重置仿真"""
        mujoco.mj_resetData(self.model, self.data)
        
        # ========== Domain Randomization ==========
        if self.domain_rand_enabled:
            self.domain_randomizer.reset()  # 随机化物理参数
        
        # 初始高度：腿控需要较高下落，纯 prop 模式保持低一些便于拉起
        init_height = 1.2 if self.leg_enabled else 0.5
        self.interface.reset(init_height=init_height, init_shift=0.0)
        if self.leg_enabled and self.controller is not None:
            self.controller.reset()  # 重置 Raibert 控制器
        self.sim_time = 0.0
        self.desired_vel = np.array([0.0, 0.0])
        
        # 重置滤波器位置为真实位置（模拟已知初始状态）
        if self.use_estimator:
            # 获取当前真实状态来初始化滤波器
            # 必须先 forward 一次以更新 xpos
            mujoco.mj_forward(self.model, self.data)
            state = self.interface.get_state()
            
            # 重置滤波器内部状态
            self.com_filter.last_pos = state['body_pos'].copy()
            self.com_filter.last_state = 1
            self.com_filter.stance_flag = 0
            self.com_filter.state_count = 0
            self.com_filter.pAng = 0.0
            self.com_filter.last_orient = np.array([1.0, 0.0, 0.0, 0.0])

            # 高层估计器：用当前真值四元数初始化（仅用于仿真对齐；真实机上可用 [1,0,0,0]）
            self.state_estimator.reset(quat_wxyz=state['body_quat'], vel_world=np.zeros(3))
            self.ekf_estimator.reset(p_world=state['body_pos'], v_world=np.zeros(3), q_wxyz=state['body_quat'])
        
        # 前向运动学更新
        mujoco.mj_forward(self.model, self.data)
        self.interface.clear_external_forces()

    def step(self, desired_vel):
        if self.mode == 'prop':
            return self.step_prop()
        elif self.mode == 'leg_prop':
            return self.step_leg_with_prop_assist(desired_vel)
        return self.step_leg(desired_vel)

    def compute_leg_control(self, state, desired_vel):
        """
        基于当前状态和期望速度计算腿部扭矩，但不执行仿真步进。
        """
        if self.use_estimator:
            # 保留真值（用于对比/录制），避免被估计值覆盖
            state['true_body_vel'] = state['body_vel'].copy()
            state['true_body_quat'] = state['body_quat'].copy()

            in_stance = (self.controller.state == 2)
            if self.estimator_kind == 'ekf':
                est_q, est_v = self.ekf_estimator.step(
                    dt=self.control_dt,
                    gyro_body=state.get('imu_gyro', state['body_ang_vel']),
                    acc_body=state['imu_acc'],
                    foot_pos_body=state.get('foot_pos_mj', state['foot_pos']),
                    foot_vel_rel_body=state.get('foot_vel_mj_rel', state.get('foot_vel_mj', state['foot_vel'])),
                    in_stance=in_stance,
                )
            else:
                est_q, est_v = self.state_estimator.update(
                    dt=self.control_dt,
                    imu_gyro_body=state.get('imu_gyro', state['body_ang_vel']),
                    imu_acc_body=state['imu_acc'],
                    # 注意：估计器必须使用与 quat/imu 同坐标系的 foot_pos/vel（未做 Z 翻转）
                    foot_pos_body=state.get('foot_pos_mj', state['foot_pos']),
                    # 且应使用“相对足端速度”（足端相对机体），更符合接触约束
                    foot_vel_body=state.get('foot_vel_mj_rel', state.get('foot_vel_mj', state['foot_vel'])),
                    in_stance=in_stance,
                )
            # 记录估计结果
            state['est_body_vel'] = est_v
            state['est_body_quat'] = est_q
            # 可选：用估计速度替换真值（用于 Raibert 速度闭环）
            if self.estimator_feed_control:
                state['body_vel'] = est_v

        # 与 Hopper4.py 的期望速度坐标约定保持一致：
        # 在当前 MuJoCo 模型里，直接使用 (+vx) 会导致 Raibert 目标落点方向相反。
        # 为了让“用户输入 +vx 表示向前”这一语义不变，这里在进入 controller 前做一次符号映射。
        desiredPos = np.array([-desired_vel[0], -desired_vel[1], 1.0])
        
        # Phase 切换在 controller.compute_torque() 内部通过腿长判断
        # （与 Hopper4.py 一致）

        # ========== Domain Randomization: 观测噪声 ==========
        if self.domain_rand_enabled:
            state = self.domain_randomizer.add_observation_noise(state)
            # IMU 偏移
            state['body_rpy'] = self.domain_randomizer.apply_imu_offset(state['body_rpy'])
            # 电机零点偏移 (与 RL 版本一致)
            state['joint_pos'] = self.domain_randomizer.apply_motor_offset(state['joint_pos'])
        
        # 支持 mode2: 共享期望姿态 desired_rpy（由 leg_prop 中计算并存入 self）
        desired_rpy = getattr(self, 'shared_desired_rpy', None)
        if hasattr(self.controller, 'compute_torque') and self.controller.__class__.__name__ == 'RaibertControllerMode2':
            torque, info = self.controller.compute_torque(
                X=state['foot_pos'],
                xdot=state['foot_vel'],
                joint=state['joint_pos'],
                jointVel=state['joint_vel'],
                vel=state['body_vel'],
                quat=state['body_quat'],
                angVel=state['body_ang_vel'],
                robotPos=state['body_pos'],
                desiredPos=desiredPos,
                rpy=state['body_rpy'],
                desired_rpy=desired_rpy,
            )
        else:
            torque, info = self.controller.compute_torque(
            X=state['foot_pos'],
            xdot=state['foot_vel'],
            joint=state['joint_pos'],
            jointVel=state['joint_vel'],
            vel=state['body_vel'],
            quat=state['body_quat'],
            angVel=state['body_ang_vel'],
            robotPos=state['body_pos'],
            desiredPos=desiredPos,
            rpy=state['body_rpy']
            )
        
        # ========== Domain Randomization: 电机强度 ==========
        if self.domain_rand_enabled:
            torque = self.domain_randomizer.apply_motor_strength(torque)
        
        if info is None:
            info = {}
        info.setdefault('state', self.controller.state)
        info.setdefault('leg_length', np.linalg.norm(state['foot_pos']))
        return torque, info, state

    def step_leg(self, desired_vel):
        """
        执行腿控控制步
        """
        state = self.interface.get_state()
        torque, info, state = self.compute_leg_control(state, desired_vel)
        self.interface.set_torque(torque)
        self.interface.clear_external_forces()

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt

        return state, torque, info

    def compute_prop_control(self, state):
        """
        计算三旋翼推力与力矩，不执行仿真步进。
        """
        if self.se3_controller is None or self.propeller_mixer is None or self.motor_model is None:
            raise RuntimeError("Propeller controller not initialized")

        quad_state = QuadState(
            position=state['body_pos'],
            velocity=state['body_vel'],
            quaternion=state['body_quat'],
            omega=state['body_ang_vel']
        )

        control_cmd = self.se3_controller.control_update(
            quad_state,
            self.prop_goal_state,
            self.control_dt,
            self.prop_forward
        )

        total_thrust = max(0.0, control_cmd.thrust * self.controller.m * self.gravity)
        torque_cmd = control_cmd.angular * self.prop_torque_scale
        roll_moment = torque_cmd[0]
        pitch_moment = torque_cmd[1]

        thrusts = self.propeller_mixer.allocate(
            total_force=total_thrust,
            roll_moment=roll_moment,
            pitch_moment=pitch_moment
        )
        thrusts = np.clip(thrusts, 0.0, None)
        motor_speeds = np.sqrt(np.where(self.motor_model.ct > 0, thrusts / self.motor_model.ct, 0.0))
        motor_speeds = self.motor_model.clamp_speed(motor_speeds)
        reaction = self.motor_model.torques_from_speeds(motor_speeds) * self.interface.prop_spin_dirs

        self.last_prop_force_cmd = np.array([0.0, 0.0, total_thrust])
        self.last_prop_torque_cmd = np.array([roll_moment, pitch_moment, 0.0])
        self.last_prop_thrust = thrusts
        self.last_motor_speeds = motor_speeds

        info = {
            'state': 'prop',
            'mode': 'prop',
            'thrust': total_thrust,
            'roll_moment': roll_moment,
            'pitch_moment': pitch_moment,
            'leg_length': np.linalg.norm(state['foot_pos']),
            'spring_force': 0.0,
            'motor_speed': motor_speeds.copy()
        }
        return thrusts, reaction, info

    def step_prop(self):
        """
        使用 SE(3) + 三桨混控 + 电机模型的控制步（忽略 yaw）
        """
        state = self.interface.get_state()
        torque = np.zeros(3)
        self.interface.set_torque(torque)

        thrusts, reaction, info = self.compute_prop_control(state)
        self.interface.apply_propeller_forces(thrusts, reaction)

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt

        return state, torque, info

    def step_leg_with_prop_assist(self, desired_vel):
        """
        腿控 + 旋翼姿态辅助模式
        
        完全按照 Quadrotor_SE3_Control-main 的逻辑，但适配三旋翼：
        - 腿部：正常 Raibert 控制
        - 旋翼：
          - 两个 Phase 都做姿态 PD 控制（强制水平）
          - 基础推力 = 0.1 * mg（一点点升力辅助）
          - Stance Phase 用较小增益，避免与腿部控制冲突
        
        参考：Quadrotor_SE3_Control-main/main.py + motor_mixer.py
        """
        # 1. 获取状态
        state = self.interface.get_state()
        
        # 2. 计算腿部扭矩（正常 Raibert 控制）
        leg_torque, info, state = self.compute_leg_control(state, desired_vel)
        self.interface.set_torque(leg_torque)
        
        # 3. 获取姿态信息
        rpy = state['body_rpy']
        omega = state['body_ang_vel']  # 角速度 [p, q, r]
        current_phase = self.controller.state  # 1=Flight, 2=Stance
        
        robot_mass = self.controller.m
        base_thrust = self.prop_base_thrust_ratio * robot_mass * self.gravity
        
        # ========== High-level: 由速度误差生成统一期望姿态 desired_rpy ==========
        # 期望姿态同时用于：
        # - 腿部 stance 的 hipTorque 姿态跟踪（mode2 可用）
        # - 旋翼 SO(3) 姿态控制（提供辅助力矩）
        #
        # 规则：想向前加速 -> 机体前倾（pitch>0），想向左加速 -> 左倾（roll>0）
        vel_world = state['body_vel']
        v_err = np.array([desired_vel[0] - vel_world[0], desired_vel[1] - vel_world[1]])
        max_tilt = np.radians(15.0)
        k_tilt = 0.8  # (rad)/(m/s) 速度误差->倾角，偏大更快收敛
        desired_roll = float(np.clip(k_tilt * v_err[1], -max_tilt, max_tilt))
        desired_pitch = float(np.clip(k_tilt * v_err[0], -max_tilt, max_tilt))
        desired_rpy = np.array([desired_roll, desired_pitch, 0.0], dtype=float)
        # 共享给腿部控制器（mode2 可接收）
        self.shared_desired_rpy = desired_rpy.copy()
        
        # 角速度
        p, q, r = omega
        
        # ===== 旋翼控制：两个 phase 都可用（stance 降低输出避免干扰腿）=====
        phase_thrust_scale = 0.4 if current_phase == 2 else 1.0
        total_thrust = base_thrust * phase_thrust_scale

        # SE3/SO3 姿态控制输出期望力矩（body frame）
        cmd = self.se3_controller.attitude_control(
            current_quat_wxyz=state['body_quat'],
            current_omega_body=omega,
            desired_rpy=desired_rpy,
            thrust_newton=total_thrust,
        )
        Mx, My, Mz = cmd.moment[0], cmd.moment[1], 0.0

        motor_speeds = self.tri_rotor_mixer.calculate(cmd.thrust, Mx, My, Mz)
        thrusts = np.array([self.tri_rotor_mixer.calc_thrust_from_speed(s) for s in motor_speeds])
        reaction_torques = np.zeros(3)

        # 限制总推力（避免把系统变成“飞行器”）
        max_total_thrust = 0.25 * robot_mass * self.gravity
        actual_total = float(np.sum(thrusts))
        if actual_total > max_total_thrust and actual_total > 1e-6:
            scale = max_total_thrust / actual_total
            thrusts = thrusts * scale
            motor_speeds = motor_speeds * np.sqrt(scale)
        
        # 应用旋翼推力
        self.interface.apply_propeller_forces(thrusts, reaction_torques)
        
        # 8. 保存日志数据
        self.last_prop_force_cmd = np.array([0.0, 0.0, np.sum(thrusts)])
        self.last_prop_torque_cmd = np.array([Mx, My, Mz])
        self.last_prop_thrust = thrusts
        self.last_motor_speeds = motor_speeds
        
        # 9. 物理仿真步进
        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt
        
        # 10. 更新 info
        info['mode'] = 'leg_prop'
        info['prop_thrust'] = np.sum(thrusts)
        info['prop_L'] = Mx
        info['prop_M'] = My
        info['prop_N'] = Mz
        info['desired_rpy'] = desired_rpy
        
        return state, leg_torque, info
    
    def log(self, state, torque, desired_vel, info):
        """记录数据"""
        self.log_data['time'].append(self.sim_time)
        self.log_data['state'].append(info['state'])
        self.log_data['body_pos'].append(state['body_pos'].copy())
        self.log_data['body_vel'].append(state['body_vel'].copy())
        self.log_data['joint_pos'].append(state['joint_pos'].copy())
        self.log_data['torque'].append(torque.copy())
        self.log_data['foot_pos'].append(state['foot_pos'].copy())
        
        # Log target foot position
        if hasattr(self.controller, 'flight_target_pos'):
            self.log_data['target_foot_pos'].append(self.controller.flight_target_pos.copy())
        else:
            self.log_data['target_foot_pos'].append(np.zeros(3))
            
        self.log_data['desired_vel'].append(desired_vel.copy())
        self.log_data['body_quat'].append(state['body_quat'].copy())
        self.log_data['body_rpy'].append(state['body_rpy'].copy())
        
        # Log energy data
        if hasattr(self.controller, 'current_energy'):
            self.log_data['current_energy'].append(self.controller.current_energy)
            self.log_data['target_energy'].append(self.controller.target_energy)
            self.log_data['energy_error'].append(self.controller.energy_error)
        else:
            self.log_data['current_energy'].append(0.0)
            self.log_data['target_energy'].append(0.0)
            self.log_data['energy_error'].append(0.0)

        leg_length = info.get('leg_length', np.linalg.norm(state['foot_pos']))
        self.log_data['leg_len'].append(leg_length)
        spring_force = info.get('spring_force', 0.0)
        self.log_data['spring_force'].append(spring_force)

        prop_force = getattr(self, 'last_prop_force_cmd', np.zeros(3))
        prop_torque = getattr(self, 'last_prop_torque_cmd', np.zeros(3))
        prop_thrust = getattr(self, 'last_prop_thrust', np.zeros(3))
        self.log_data['prop_force_cmd'].append(prop_force.copy())
        self.log_data['prop_torque_cmd'].append(prop_torque.copy())
        self.log_data['prop_thrust'].append(prop_thrust.copy())
        motor_speed = getattr(self, 'last_motor_speeds', np.zeros(3))
        self.log_data['motor_speed'].append(motor_speed.copy())
    
    def send_dashboard_data(self, state, info):
        """发送数据到仪表盘"""
        if not self.dashboard_enabled:
            return
        
        # 限制发送频率
        if self.sim_time - self.last_dashboard_time < self.dashboard_interval:
            return
        self.last_dashboard_time = self.sim_time
        
        # 计算 PWM (0-100%)
        motor_speeds = getattr(self, 'last_motor_speeds', np.zeros(3))
        pwm = (motor_speeds / self.max_motor_speed * 100).tolist()
        
        # 获取力矩命令
        prop_torque = getattr(self, 'last_prop_torque_cmd', np.zeros(3))
        prop_thrust = getattr(self, 'last_prop_thrust', np.zeros(3))
        
        # 直接从 MuJoCo 获取真实速度（不使用估计值）
        real_vz = float(self.data.qvel[2])
        
        # 构建数据包
        data = {
            'time': self.sim_time,
            'z': float(state['body_pos'][2]),
            'vz': real_vz,  # 使用真实速度
            'pwm': pwm,
            'roll': float(np.degrees(state['body_rpy'][0])),
            'pitch': float(np.degrees(state['body_rpy'][1])),
            'phase': int(self.controller.state),
            'Mx': float(prop_torque[0]),
            'My': float(prop_torque[1]),
            'thrust': float(np.sum(prop_thrust)),
        }
        
        try:
            msg = json.dumps(data).encode('utf-8')
            self.dashboard_socket.sendto(msg, self.dashboard_addr)
        except Exception as e:
            pass  # 忽略发送错误
    
    def run_with_viewer(self):
        """
        运行仿真（带可视化）
        """
        import mujoco.viewer
        
        print("\n" + "="*60)
        print("Hopper MuJoCo Simulation")
        print("="*60)
        print("Keyboard controls:")
        print("  Y: +X velocity (forward)")
        print("  H: -X velocity (backward)")
        print("  G: -Y velocity (left)")
        print("  J: +Y velocity (right)")
        print("  Space: zero velocity")
        print("  R: reset robot")
        print("  P: pause/resume")
        print("  Q/ESC: quit")
        print("="*60 + "\n")
        
        self.reset()
        
        last_print_time = 0
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                # 设置相机
                viewer.cam.distance = 10.0
                viewer.cam.elevation = -20
                viewer.cam.azimuth = 90
                viewer.cam.lookat[0] = 0.0
                viewer.cam.lookat[1] = 0.0
                viewer.cam.lookat[2] = self.viewer_fixed_z
                
                while viewer.is_running() and self.running:
                    step_start = time.time()
                    
                    # 接收遥控命令
                    self.recv_teleop()
                    if self.teleop_enabled:
                        self.desired_vel = self.teleop_vel.copy()
                    
                    if not self.paused:
                        # 让控制器内部处理相位切换
                        
                        # 执行控制步
                        state, torque, info = self.step(self.desired_vel)
                        viewer.cam.lookat[0] = state['body_pos'][0]
                        viewer.cam.lookat[1] = state['body_pos'][1]
                        viewer.cam.lookat[2] = self.viewer_fixed_z
                        
                        # 记录数据
                        self.log(state, torque, self.desired_vel, info)
                        
                        # 发送仪表盘数据
                        self.send_dashboard_data(state, info)
                        
                        # 打印状态（每秒一次）
                        if self.sim_time - last_print_time >= 1.0:
                            leg_len = info.get('leg_length', np.linalg.norm(state['foot_pos']))
                            spring_force = info.get('spring_force', 0.0)
                            if self.leg_enabled and self.controller is not None:
                                print(f"t={self.sim_time:.2f}s | "
                                      f"st={info['state']} | "
                                      f"z={state['body_pos'][2]:.3f}m | "
                                      f"leg={leg_len:.3f}m | "
                                      f"F={spring_force:.1f}N | "
                                      f"τ=[{torque[0]:.1f},{torque[1]:.1f},{torque[2]:.1f}]")
                            else:
                                thrust = info.get('thrust', 0.0)
                                roll = np.degrees(state['body_rpy'][0])
                                pitch = np.degrees(state['body_rpy'][1])
                                print(f"t={self.sim_time:.2f}s | mode=prop | "
                                      f"z={state['body_pos'][2]:.3f}m | "
                                      f"roll={roll:.1f}° | pitch={pitch:.1f}° | "
                                      f"T={thrust:.1f}N")
                            last_print_time = self.sim_time
                    
                    # 同步可视化
                    viewer.sync()
                    
                    # 控制仿真速度
                    elapsed = time.time() - step_start
                    if elapsed < self.control_dt:
                        time.sleep(self.control_dt - elapsed)
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user.")
        finally:
            print("\nSimulation ended.")
            self.save_log()
    
    def run_headless(self, duration=10.0):
        """
        无头模式运行仿真
        
        Args:
            duration: 仿真时长 (秒)
        """
        print(f"Running headless simulation for {duration}s...")
        
        self.reset()
        
        # 设置期望速度
        desired_vel = np.array([0.0, 0.0])  # 原地跳
        
        while self.sim_time < duration:
            # 让控制器内部处理相位切换
            # 不在外部强制设置状态
            
            state, torque, info = self.step(desired_vel)
            self.log(state, torque, desired_vel, info)
            
            # 摔倒检测 (Pitch or Roll > 60度)
            rpy_deg = np.degrees(state['body_rpy'])
            if abs(rpy_deg[0]) > 60 or abs(rpy_deg[1]) > 60:
                print(f"\n!!! Robot fell at t={self.sim_time:.2f}s !!!")
                print(f"State at fall: Roll={rpy_deg[0]:.1f}°, Pitch={rpy_deg[1]:.1f}°, Height={state['body_pos'][2]:.3f}m")
                break
            
            if int(self.sim_time * 1000) % 10 == 0:  # 每 0.01 秒打印一次
                if self.mode == 'prop':
                    thrust = info.get('thrust', 0.0)
                    roll = np.degrees(state['body_rpy'][0])
                    pitch = np.degrees(state['body_rpy'][1])
                    print(f"t={self.sim_time:.2f}s | mode=prop | z={state['body_pos'][2]:.3f} | "
                          f"roll={roll:.1f}° | pitch={pitch:.1f}° | T={thrust:.1f}N")
                else:
                    leg_len = info.get('leg_length', np.linalg.norm(state['foot_pos']))
                    rpy = np.degrees(state['body_rpy'])
                    spring_force = info.get('spring_force', 0.0)
                    
                    if hasattr(self.controller, 'flight_target_pos') and self.controller.state == 1:
                        tgt_x = self.controller.flight_target_pos[0]
                        real_x = state['foot_pos'][0]
                        err_x = real_x - tgt_x
                        print(f"t={self.sim_time:.2f}s | st={self.controller.state} | z={state['body_pos'][2]:.3f} | "
                              f"leg={leg_len:.3f} | F={spring_force:.1f} | TgtX={tgt_x:.3f} | RealX={real_x:.3f} | Err={err_x:.3f} | "
                              f"τ=[{torque[0]:.0f},{torque[1]:.0f},{torque[2]:.0f}]")
                    else:
                        print(f"t={self.sim_time:.2f}s | st={self.controller.state} | z={state['body_pos'][2]:.3f} | "
                              f"leg={leg_len:.3f} | F={spring_force:.1f} | P={rpy[1]:.1f}° | τ=[{torque[0]:.0f},{torque[1]:.0f},{torque[2]:.0f}]")
        
        print("Simulation complete.")
        self.save_log()

    def record_video(self, output_path, fps=30, width=1280, height=720, duration=None):
        """
        离线录制视频（无 GUI）。可通过 Ctrl+C 中断。
        """
        print(f"Recording video to {output_path} (fps={fps}, {width}x{height})")
        renderer = mujoco.Renderer(self.model, width, height)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth = 90
        cam.elevation = -20
        cam.distance = 5.0
        cam.lookat[:] = [0.0, 0.0, self.viewer_fixed_z]
        opt = mujoco.MjvOption()

        start_time = self.sim_time
        try:
            with imageio.get_writer(output_path, fps=fps) as writer:
                while True:
                    state, torque, info = self.step(self.desired_vel)
                    cam.lookat[0] = state['body_pos'][0]
                    cam.lookat[1] = state['body_pos'][1]
                    cam.lookat[2] = self.viewer_fixed_z

                    # 先 update_scene，再 render
                    renderer.update_scene(self.data, camera=cam, scene_option=opt)
                    frame = renderer.render()
                    writer.append_data(frame)

                    if duration is not None and self.sim_time - start_time >= duration:
                        print("Reached recording duration.")
                        break
        except KeyboardInterrupt:
            print("Recording interrupted by user.")
        finally:
            renderer.close()
            self.save_log()
    
    def save_log(self, filename=None):
        """保存日志"""
        if filename is None:
            filename = os.path.join(PROJECT_ROOT, 'hopper_log.npz')
        
        # 转换为 numpy 数组
        log_arrays = {}
        for key, value in self.log_data.items():
            if len(value) > 0:
                log_arrays[key] = np.array(value)
        
        if len(log_arrays) > 0:
            np.savez(filename, **log_arrays)
            print(f"Log saved to: {filename}")
            if 'time' in log_arrays and 'leg_len' in log_arrays:
                times = log_arrays['time']
                legs = log_arrays['leg_len']
                springs = log_arrays.get('spring_force', np.zeros_like(legs))
                print("\nLeg length & spring force vs time (down-sampled):")
                sample_count = min(50, len(times))
                indices = np.linspace(0, len(times) - 1, sample_count, dtype=int)
                for idx in indices:
                    print(f"t={times[idx]:.3f}s -> leg={legs[idx]:.4f} m, F={springs[idx]:.1f} N")


class SimulateUIRunner:
    """
    使用 MuJoCo Simulate GUI 的运行器，支持 Ctrl+V 录像、Ctrl 拖拽等原生功能。
    """

    def __init__(self, mode='leg', model_path=None):
        self.mode = mode
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'mjcf', 'hopper_serial.xml')
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.sim = HopperSimulation(
            model_path=self.model_path,
            mode=self.mode,
            model=self.model,
            data=self.data,
            verbose=False,
        )
        self.prev_time = 0.0
        self.last_print_time = 0.0

    def loader(self):
        self.sim.reset()
        self.prev_time = self.data.time
        self.last_print_time = 0.0
        return self.model, self.data

    def control_callback(self, model, data):
        # 检测外部重置（Simulate GUI 的 Reset）
        if data.time < self.prev_time - 1e-6:
            self.sim.controller.reset()
            self.sim.com_filter = ComplementaryFilter()
            self.sim.sim_time = data.time
            self.prev_time = data.time

        state = self.sim.interface.get_state()
        if self.mode == 'prop':
            thrusts, reaction, info = self.sim.compute_prop_control(state)
            data.ctrl[:] = 0.0
            self.sim.interface.apply_propeller_forces(thrusts, reaction)
            applied = np.zeros(3)
        else:
            torque, info, state = self.sim.compute_leg_control(state, self.sim.desired_vel)
            data.ctrl[:3] = torque
            self.sim.interface.clear_external_forces()
            applied = torque

        self.sim.sim_time = data.time
        self.sim.log(state, applied, self.sim.desired_vel, info)

        if data.time - self.last_print_time >= 1.0:
            if self.mode == 'prop':
                roll = np.degrees(state['body_rpy'][0])
                pitch = np.degrees(state['body_rpy'][1])
                thrust = info.get('thrust', 0.0)
                print(f"t={data.time:.2f}s | mode=prop | z={state['body_pos'][2]:.3f}m | "
                      f"roll={roll:.1f}° | pitch={pitch:.1f}° | T={thrust:.1f}N")
            else:
                leg_len = info.get('leg_length', np.linalg.norm(state['foot_pos']))
                spring_force = info.get('spring_force', 0.0)
                rpy = np.degrees(state['body_rpy'])
                print(f"t={data.time:.2f}s | st={self.sim.controller.state} | z={state['body_pos'][2]:.3f}m | "
                      f"leg={leg_len:.3f}m | F={spring_force:.1f}N | P={rpy[1]:.1f}°")
            self.last_print_time = data.time

        self.prev_time = data.time

    def run(self):
        import mujoco.viewer

        mujoco.set_mjcb_control(self.control_callback)
        try:
            mujoco.viewer.launch(loader=self.loader)
        finally:
            mujoco.set_mjcb_control(None)
        self.sim.save_log()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hopper MuJoCo Simulation')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration (headless mode)')
    parser.add_argument('--model', type=str, default=None, help='Path to MJCF model')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], default=1,
                        help='1: leg only, 2: propeller-only flight, 3: leg + propeller assist')
    parser.add_argument('--ui', choices=['viewer', 'simulate'], default='viewer',
                        help='viewer: 轻量级 GUI；simulate: MuJoCo 官方 Simulate GUI (Ctrl+V, Ctrl+拖动)')
    parser.add_argument('--record', type=str, help='录制视频到该 mp4 文件（无 GUI）')
    parser.add_argument('--record-duration', type=float, default=None,
                        help='录制时长（秒），默认一直录到 Ctrl+C')
    parser.add_argument('--record-fps', type=int, default=30, help='录制帧率')
    parser.add_argument('--record-width', type=int, default=1280, help='录制宽度')
    parser.add_argument('--record-height', type=int, default=720, help='录制高度')
    parser.add_argument('--teleop', action='store_true', 
                        help='启用 UDP 遥控接收（配合 teleop_gui.py 使用）')
    
    args = parser.parse_args()
    
    if args.headless and args.ui == 'simulate':
        raise ValueError('simulate UI cannot run in headless mode.')

    # 创建仿真
    if args.mode == 2:
        sim_mode = 'prop'
    elif args.mode == 3:
        sim_mode = 'leg_prop'
    else:
        sim_mode = 'leg'
    if args.ui == 'simulate':
        runner = SimulateUIRunner(mode=sim_mode, model_path=args.model)
        runner.run()
        return

    if args.record:
        sim = HopperSimulation(model_path=args.model, mode=sim_mode)
        sim.record_video(
            args.record,
            fps=args.record_fps,
            width=args.record_width,
            height=args.record_height,
            duration=args.record_duration,
        )
        return

    sim = HopperSimulation(model_path=args.model, mode=sim_mode)
    
    if args.teleop:
        sim.enable_teleop()

    if args.headless:
        sim.run_headless(duration=args.duration)
    else:
        sim.run_with_viewer()


if __name__ == '__main__':
    main()
