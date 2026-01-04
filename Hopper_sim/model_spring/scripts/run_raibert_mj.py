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

def _quat_wxyz_to_rpy(q_wxyz):
    """Convert quaternion (w,x,y,z) to roll-pitch-yaw (xyz, rad)."""
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = float(np.sign(sinp) * (np.pi / 2.0))
    else:
        pitch = float(np.arcsin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return np.array([roll, pitch, yaw], dtype=float)

from controllers.raibert_controller import RaibertController
from controllers.raibert_controller_mode2 import RaibertControllerMode2
from controllers.raibert_controller_mode3 import RaibertControllerMode3
from controllers.com_filter import ComplementaryFilter
from controllers.wbc_srb import SRBWBC, SRBWBCConfig
from controllers.wbc_qp_osqp import WBCQP, WBCQPConfig
from controllers.wbc_full_qp_osqp import FullBodyWBCQP, FullBodyWBCConfig
from utils.state_estimator import HopperStateEstimator, EstimatorConfig
from utils.ekf_state_estimator import HopperESKF, EKFConfig
from utils.mujoco_interface import MuJoCoInterface
from utils.domain_randomization import DomainRandomizer, DomainRandomConfig

# Pinocchio dynamics backend (optional; used to emulate real-robot dynamics inside sim)
try:
    from utils.pinocchio_dynamics import PinocchioHopperDynamics, PinocchioHopperConfig
    PINOCCHIO_AVAILABLE = True
except Exception:
    PinocchioHopperDynamics = None  # type: ignore
    PinocchioHopperConfig = None  # type: ignore
    PINOCCHIO_AVAILABLE = False

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

        # ========== High-level SRB/WBC parameters (for leg_prop) ==========
        # Target body height (world z)
        self.wbc_z_des = 1.0
        # PD for height -> desired vertical acceleration
        # For jumping (mode3), we need higher vertical tracking bandwidth.
        self.wbc_kp_z = 60.0
        self.wbc_kd_z = 10.0
        # Velocity -> desired horizontal acceleration (via desired tilt)
        self.wbc_kv_xy = 2.5
        # Max tilt from WBC (rad)
        self.wbc_max_tilt = np.radians(20.0)
        # Thrust limits (as fraction of m*g)
        self.wbc_min_thrust_ratio = 0.00
        self.wbc_max_thrust_ratio = 0.25

        # ========== Jumping as primary task (Mode3 QP-WBC) ==========
        # Use a periodic hop reference for COM height in world frame.
        # z_ref(t) = z0 + A * (1 - cos(2π f t))/2  (always >= z0)
        self.hop_enabled = True
        # Desired maximum BASE height (world z) during hop
        # Peak(z_ref) = hop_z0 + hop_amp; we compute hop_amp from hop_peak_z.
        self.hop_peak_z = 1.0
        self.hop_freq_hz = 1.6        # hop frequency
        self.hop_z0 = 0.60            # baseline height
        self.hop_amp = max(0.0, float(self.hop_peak_z - self.hop_z0))  # amplitude (peak - baseline)
        # weights: prioritize vertical motion over horizontal tracking
        self.hop_kv_xy_scale = 0.3    # default: reduce horizontal tracking during hopping (stability)
        if int(getattr(controller_mode, '__int__', lambda: controller_mode)()) == 3:
            # Allow overriding hop horizontal tracking scale (Task1 forward speed needs this higher).
            try:
                self.hop_kv_xy_scale = float(os.environ.get('MODE3_HOP_KV_XY_SCALE', str(self.hop_kv_xy_scale)))
            except Exception:
                pass
        self.wbc = SRBWBC(SRBWBCConfig(
            z_des=self.wbc_z_des,
            kp_z=self.wbc_kp_z,
            kd_z=self.wbc_kd_z,
            kv_xy=self.wbc_kv_xy,
            max_tilt=self.wbc_max_tilt,
            min_thrust_ratio=self.wbc_min_thrust_ratio,
            max_thrust_ratio=self.wbc_max_thrust_ratio,
        ))
        # QP-WBC will be initialized after tri-rotor parameters are available
        self.wbc_qp = None
        self.wbc_full_qp = None

        # ========== Mode3 phase gate (kinematic self-consistency) ==========
        # Goal: prevent false Flight->Stance switching caused by leg-length oscillations in flight.
        # We DO NOT use MuJoCo contact flag for control; instead we use a kinematic gate:
        # - foot world z is close to ground
        # - foot world speed is close to 0
        # This is a sim-side proxy for what you'd implement on the real robot using base EKF + leg kinematics.
        self.mode3_gate_enabled = True
        # NOTE: in this model, foot geom world-z can be >0 even during contact depending on geom frame.
        # We therefore gate primarily on low foot speed; z is a weak check.
        self.mode3_gate_foot_z_max = 0.30   # m
        # Allow horizontal slip; gate mainly on vertical foot motion (true contact should have small vz).
        self.mode3_gate_vfoot_z_max = 0.30  # m/s
        self.mode3_gate_vfoot_max = 5.0     # m/s (sanity cap only)
        self.mode3_gate_min_count = 5       # ticks @1kHz (~5ms)
        self._mode3_gate_count = 0
        # prefer foot geom for gating (closer to real contact point)
        try:
            self._mode3_gate_geom_id = int(self.model.geom('foot_collision').id)
        except Exception:
            self._mode3_gate_geom_id = None

        # 创建控制器（mode1/mode2/mode3）
        self.controller_mode = int(controller_mode)
        if self.controller_mode == 3:
            self.controller = RaibertControllerMode3()
            # Allow overriding "spring/energy" parameters from environment (for rapid tuning).
            # These parameters define the leg's spring-like behavior and vertical energy shaping (Raibert-style).
            try:
                if hasattr(self.controller, 'h'):
                    self.controller.h = float(os.environ.get('MODE3_SPRING_H', str(getattr(self.controller, 'h'))))
                if hasattr(self.controller, 'k'):
                    self.controller.k = float(os.environ.get('MODE3_SPRING_K', str(getattr(self.controller, 'k'))))
                if hasattr(self.controller, 'b'):
                    self.controller.b = float(os.environ.get('MODE3_SPRING_B', str(getattr(self.controller, 'b'))))
                if hasattr(self.controller, 'Kp'):
                    self.controller.Kp = float(os.environ.get('MODE3_ENERGY_KP', str(getattr(self.controller, 'Kp'))))
            except Exception:
                pass
            # Sign conventions for mapping desired horizontal accel -> desired roll/pitch.
            # (Different simulators/URDFs can differ in what +pitch means visually.)
            try:
                self.mode3_pitch_sign = float(os.environ.get('MODE3_PITCH_SIGN', str(getattr(self, 'mode3_pitch_sign', 1.0))))
                self.mode3_roll_sign = float(os.environ.get('MODE3_ROLL_SIGN', str(getattr(self, 'mode3_roll_sign', 1.0))))
            except Exception:
                self.mode3_pitch_sign = float(getattr(self, 'mode3_pitch_sign', 1.0))
                self.mode3_roll_sign = float(getattr(self, 'mode3_roll_sign', 1.0))

            # ---- Mode3 fixed defaults (structure lock) ----
            # These are the defaults we want when the user says "先把结构固定住".
            # Environment variables can still override for tuning, but the baseline is stable.
            self.mode3_use_ekf_everywhere = True
            # Hopper policy: prop is attitude assist by default (not in QP).
            self.mode3_prop_in_qp_default = False
            self.mode3_prop_thrust_ratio_max = float(os.environ.get('MODE3_PROP_THRUST_RATIO_MAX', str(getattr(self, 'mode3_prop_thrust_ratio_max', 0.25))))
            self.mode3_prop_base_thrust_ratio = float(os.environ.get('MODE3_PROP_BASE_THRUST_RATIO', str(getattr(self, 'mode3_prop_base_thrust_ratio', 0.25))))
            self.mode3_prop_kp_rp = float(os.environ.get('MODE3_PROP_KP_RP', str(getattr(self, 'mode3_prop_kp_rp', 220.0))))
            self.mode3_prop_kd_rp = float(os.environ.get('MODE3_PROP_KD_RP', str(getattr(self, 'mode3_prop_kd_rp', 40.0))))

            # stance 2-stage parameters (compression -> push-off)
            self.mode3_stance_comp_ratio = float(os.environ.get('MODE3_STANCE_COMP_RATIO', str(getattr(self, 'mode3_stance_comp_ratio', 0.82))))
            self.mode3_comp_az_up_max = float(os.environ.get('MODE3_COMP_AZ_UP_MAX', str(getattr(self, 'mode3_comp_az_up_max', 35.0))))
            self.mode3_push_az_up_max = float(os.environ.get('MODE3_PUSH_AZ_UP_MAX', str(getattr(self, 'mode3_push_az_up_max', 180.0))))
            self.mode3_stance_dleg_eps = float(os.environ.get('MODE3_STANCE_DLEG_EPS', str(getattr(self, 'mode3_stance_dleg_eps', 0.02))))
            self.mode3_stance_comp_max_ticks = int(os.environ.get('MODE3_STANCE_COMP_MAX_TICKS', str(getattr(self, 'mode3_stance_comp_max_ticks', 120))))
        elif self.controller_mode == 2:
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
        # Now that prop limits are known, init the QP-WBC (paper-style)
        thrust_ratio_hw_max = float((3.0 * self.max_thrust_per_motor) / (self.controller.m * self.gravity))
        self.wbc_qp = WBCQP(WBCQPConfig(
            mu=0.9,
            fz_max=220.0,
            thrust_total_ratio_max=thrust_ratio_hw_max,
        ))

        # Full-body WBC-QP (manipulator equation) config
        self.wbc_full_qp = FullBodyWBCQP(FullBodyWBCConfig(
            mu=0.9,
            fz_min=0.0,
            fz_max=400.0,
            # Torque limits are critical for "spring-like" leg energy storage + attitude recovery.
            tau_limit=80.0,
            shift_tau_limit=800.0,
            shift_unilateral=True,
            thrust_max_each=float(self.max_thrust_per_motor),
            thrust_sum_max=float(3.0 * self.max_thrust_per_motor),
            z_max=float(self.hop_peak_z),
            z_max_dt=0.01,
            w_base_lin=800.0,
            w_base_ang=80.0,
            # flight swing-foot should be strong enough to prevent the leg collapsing in the air
            w_swing_foot=3000.0,
            swing_kp=4000.0,
            swing_kd=120.0,
            w_shift_hold=0.0,
            shift_hold_kp=50.0,
            shift_hold_kd=10.0,
            w_tau=1e-4,
            w_tau_track=0.0,
            w_fc=1e-6,
            # Penalize prop usage: Hopper should hop (leg), prop is for balance/assist only.
            w_thrust=5e-3,
            formulation=os.environ.get("WBC_FORM", "condensed"),
        ))

        # Mode3 (Hopper) policy: propellers are NOT the primary vertical actuator.
        # Limit total prop thrust to a fraction of mg unless the user overrides it.
        # This keeps the behavior "hopping" instead of "flying".
        if int(getattr(self, 'controller_mode', 1)) == 3:
            try:
                self.mode3_prop_thrust_ratio_max = float(os.environ.get('MODE3_PROP_THRUST_RATIO_MAX', str(getattr(self, 'mode3_prop_thrust_ratio_max', 0.15))))
            except Exception:
                self.mode3_prop_thrust_ratio_max = float(getattr(self, 'mode3_prop_thrust_ratio_max', 0.15))

        # Optional: Pinocchio dynamics backend (URDF-based) for sim-side validation
        # This emulates the real-robot pipeline where MuJoCo internals are not available.
        self.pin_dyn = None
        if PINOCCHIO_AVAILABLE:
            try:
                self.pin_dyn = PinocchioHopperDynamics(PinocchioHopperConfig(
                    urdf_path=os.path.join(PROJECT_ROOT, 'urdf', 'hopper_serial.urdf'),
                    freeflyer_v_convention='world_lin_world_ang',
                ))
            except Exception:
                self.pin_dyn = None
        # Default to Pinocchio when available (real-robot style). Fall back to MuJoCo otherwise.
        self.full_wbc_dynamics_backend = 'pinocchio' if (PINOCCHIO_AVAILABLE and (self.pin_dyn is not None)) else 'mujoco'
        
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
        # mode2 默认更保守（避免触地阶段旋翼干扰导致翻车）
        if int(getattr(self, 'controller_mode', 1)) == 2:
            self.prop_base_thrust_ratio = 0.08  # 基础推力 = 0.08 * mg（仍会再被 <=0.10mg 限幅）
        else:
            self.prop_base_thrust_ratio = 0.15  # 基础推力 = 0.15 * mg
        self.prop_torque_scale = 1.0  # prop/SE3 输出力矩缩放（默认 1.0）

        # ========== Mode3: defaults for stability envelope ==========
        # NOTE: These are *controller-side* knobs; scripts may override them.
        if int(getattr(self, 'controller_mode', 1)) == 3:
            # Keep roll/pitch inside +/-15deg, start braking early.
            self.mode3_tilt_limit_rad = float(np.radians(15.0))
            self.mode3_tilt_margin_rad = float(np.radians(6.0))

            # Stronger attitude regulation is needed especially in stance to counter single-contact pitching torque.
            # Flight is prop-dominant; stance needs even more angular authority to keep the base level.
            self.mode3_w_base_ang_flight = float(getattr(self, 'mode3_w_base_ang_flight', 220.0))
            self.mode3_w_base_ang_stance = float(getattr(self, 'mode3_w_base_ang_stance', 260.0))
            self.mode3_kp_att_flight = float(getattr(self, 'mode3_kp_att_flight', 55.0))
            self.mode3_kd_att_flight = float(getattr(self, 'mode3_kd_att_flight', 11.0))
            self.mode3_kp_att_stance = float(getattr(self, 'mode3_kp_att_stance', 75.0))
            self.mode3_kd_att_stance = float(getattr(self, 'mode3_kd_att_stance', 14.0))

        # mode2: 更保守的“速度->倾角”与相位调度（保证稳定优先）
        # mode2: desired velocity shaping (helps EKF-closed-loop speed convergence without making Raibert gains unstable)
        # Default OFF (0.0): mode2 默认用纯 Raibert 参数收敛；如需更强速度收敛，可在脚本里打开
        self.mode2_vdes_i_gain = 0.0      # (m/s) / (m/s) / s
        self.mode2_vdes_bias_cap = 0.0    # m/s
        self._mode2_vdes_bias = np.zeros(2, dtype=float)
        self.mode2_phase_scale_stance = 0.2
        self.mode2_phase_scale_flight = 1.0
        # stance 的姿态力矩：触地瞬间为 0，随后小幅打开（靠 phase_duration_count 做 ramp）
        self.mode2_moment_scale_stance = 0.35
        self.mode2_moment_scale_flight = 1.0
        
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
        # 默认：所有 mode 都用 IMU/EKF 闭环（更接近真机）
        self.use_estimator = True
        self.state_estimator = HopperStateEstimator(EstimatorConfig())
        self.ekf_estimator = HopperESKF(EKFConfig())
        # default estimator selection: EKF (ESKF)
        self.estimator_kind = 'ekf'  # 'complementary' | 'ekf'
        # True: 用估计速度闭环控制；False: 仅对比/记录，不影响控制
        # mode1/mode2(Raibert) 对估计噪声更敏感：默认不喂给控制；mode3(WBC) 默认喂给控制
        self.estimator_feed_control = (int(controller_mode) == 3)
        # 估计速度低通（真机上也需要；直接用原始估计会把步态抖动带进落脚点）
        self.est_vel_lpf_alpha = 0.2
        self._vel_control = np.zeros(3, dtype=float)
        
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
    
    def _mode3_contact_gate(self) -> tuple[bool, float, float]:
        """Return (gate_ok, foot_z_world, |v_foot_world|). Uses kinematics only (no contact sensor)."""
        jacp = np.zeros((3, int(self.model.nv)), dtype=float)
        if self._mode3_gate_geom_id is not None:
            gid = int(self._mode3_gate_geom_id)
            foot_z = float(self.data.geom_xpos[gid][2])
            mujoco.mj_jacGeom(self.model, self.data, jacp, None, gid)
        else:
            foot_id = int(self.interface.foot_body_id)
            foot_z = float(self.data.xpos[foot_id][2])
            mujoco.mj_jacBody(self.model, self.data, jacp, None, foot_id)
        v_foot_w = jacp @ self.data.qvel
        vnorm = float(np.linalg.norm(v_foot_w))
        vz = float(v_foot_w[2])
        gate_ok = (
            (foot_z <= float(self.mode3_gate_foot_z_max))
            and (abs(vz) <= float(getattr(self, 'mode3_gate_vfoot_z_max', 0.30)))
            and (vnorm <= float(getattr(self, 'mode3_gate_vfoot_max', 5.0)))
        )
        return bool(gate_ok), foot_z, vnorm

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
        
        # 初始高度：
        # - 腿控默认需要较高下落（历史参数）
        # - 但对 mode3 的“跳跃主任务”，初始高度必须 <= hop_peak_z，否则一开始就违背高度目标并导致不稳定
        if bool(getattr(self, 'hop_enabled', False)) and int(getattr(self, 'controller_mode', 1)) == 3:
            init_height = float(getattr(self, 'hop_z0', 0.6))
        else:
            init_height = 1.2 if self.leg_enabled else 0.5
        # For spring-leg (shift joint has stiffness), start near the spring rest length so it doesn't "explode" at reset.
        # Shift: 0.0 is longest leg in this MJCF; larger shift compresses (shortens) the leg.
        init_shift = float(getattr(self, 'mode3_init_shift', 0.20)) if int(getattr(self, 'controller_mode', 1)) == 3 else 0.0
        self.interface.reset(init_height=init_height, init_shift=init_shift)
        if self.leg_enabled and self.controller is not None:
            self.controller.reset()  # 重置 Raibert 控制器
        self.sim_time = 0.0
        self.desired_vel = np.array([0.0, 0.0])
        self._vel_control = np.zeros(3, dtype=float)
        self._mode2_vdes_bias = np.zeros(2, dtype=float)
        # mode3: stance sub-phase (compression -> push-off)
        self._mode3_leglen_prev = None
        self._mode3_td_leglen = None
        self._mode3_td_leglen_min = None
        self._mode3_stance_subphase = 0  # 0=unknown/flight, 1=compression, 2=push-off
        # mode3: S2S apex-to-apex (foot placement) state
        self._mode3_s2s_offset_xy = np.zeros(2, dtype=float)
        self._mode3_s2s_last_vdes = np.zeros(2, dtype=float)
        self._mode3_s2s_vz_prev = None
        self._mode3_s2s_prev_in_flight = False
        # mode3: S2S touchdown pitch bias (A)
        self._mode3_s2s_pitch_bias = 0.0  # rad
        
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

        # mode3: record a flight leg-length reference at reset (\"original length\")
        # We use the same definition as logging: ||foot_pos|| (body-frame).
        # Then flight swing-foot target keeps this length constant, preventing leg collapse in the air.
        if int(getattr(self, 'controller_mode', 1)) == 3:
            try:
                state0 = self.interface.get_state()
                self.mode3_leglen_ref = float(np.linalg.norm(state0['foot_pos']))
                # also record shift joint reference (for flight leg-length hold in QP)
                self.mode3_shift_ref = float(np.asarray(state0['joint_pos'], dtype=float).reshape(3)[2])
            except Exception:
                self.mode3_leglen_ref = float(getattr(self.controller, 'l0', 0.464))
                self.mode3_shift_ref = float(0.0)
        else:
            self.mode3_leglen_ref = None
            self.mode3_shift_ref = None

        # mode3: flight leg extension hold (torque servo on shift joint)
        # This is NOT a spring model; it's a simple servo to keep the prismatic leg extended in flight.
        self.mode3_flight_shift_hold_enabled = True
        self.mode3_flight_shift_kp = 250.0
        self.mode3_flight_shift_kd = 25.0
        self.mode3_flight_shift_tau_limit = 300.0
        
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

            # EKF stance gate (real-robot compatible):
            # Only rely on the phase heuristic (leg-length hysteresis). No MuJoCo contact flag.
            # stance indicator for estimator update:
            # mode3 uses an additional kinematic gate to avoid false stance updates in flight.
            in_stance = (self.controller.state == 2)
            if int(getattr(self, 'controller_mode', 1)) == 3 and bool(getattr(self, 'mode3_gate_enabled', True)):
                gate_ok, foot_z, vfoot = self._mode3_contact_gate()
                if gate_ok:
                    self._mode3_gate_count = min(int(getattr(self, 'mode3_gate_min_count', 8)), self._mode3_gate_count + 1)
                else:
                    self._mode3_gate_count = max(0, self._mode3_gate_count - 1)
                in_stance = bool(in_stance and (self._mode3_gate_count >= int(getattr(self, 'mode3_gate_min_count', 8))))
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
            # EKF gating debug (for mode3: decide when to trust vz in height constraints)
            if self.estimator_kind == 'ekf':
                state['ekf_contact_update_ok'] = bool(getattr(self.ekf_estimator, 'last_contact_update_ok', False))
                state['ekf_contact_residual_norm'] = float(getattr(self.ekf_estimator, 'last_contact_y_norm', 0.0))
                state['ekf_contact_z_norm'] = float(np.linalg.norm(getattr(self.ekf_estimator, 'last_contact_z', np.zeros(3))))
            # Control velocity: ALWAYS use EKF velocity (MIT-style).
            # Do NOT overwrite the true MuJoCo velocity in state['body_vel'].
            a = float(np.clip(getattr(self, 'est_vel_lpf_alpha', 0.2), 0.0, 1.0))
            self._vel_control = a * np.asarray(est_v, dtype=float).reshape(3) + (1.0 - a) * np.asarray(self._vel_control, dtype=float).reshape(3)
            state['body_vel_ctrl'] = self._vel_control.copy()
                # Note: we do NOT override body_quat/foot_pos here, because this sim interface computes
                # foot_pos/foot_vel using the MuJoCo quaternion. Overriding quat without recomputing foot
                # kinematics would make the controller inputs inconsistent.

        # desiredPos follows the same semantic as user commands:
        # +vx means forward, +vy means right (MuJoCo world frame).
        # Note: historical sign flips were only needed under specific controller conventions.
        # mode2: shape desired velocity for Raibert when running EKF-closed-loop
        des_vel_ctrl = np.asarray(desired_vel, dtype=float).reshape(2).copy()
        # Mode3: allow explicit velocity sign mapping (some models use opposite x-forward convention).
        if int(getattr(self, 'controller_mode', 1)) == 3:
            try:
                sx = float(os.environ.get('MODE3_VEL_SIGN_X', '1.0'))
                sy = float(os.environ.get('MODE3_VEL_SIGN_Y', '1.0'))
                des_vel_ctrl[0] *= sx
                des_vel_ctrl[1] *= sy
            except Exception:
                pass
        if int(getattr(self, 'controller_mode', 1)) == 2 and bool(getattr(self, 'estimator_feed_control', False)):
            # use the velocity actually used by control (may be EKF)
            v_meas = np.asarray(state['body_vel'][0:2], dtype=float).reshape(2)
            e = des_vel_ctrl - v_meas
            dt = float(self.dt)
            ki = float(getattr(self, 'mode2_vdes_i_gain', 0.0))
            cap = float(getattr(self, 'mode2_vdes_bias_cap', 0.0))
            # decay bias when command is near zero (avoid drifting internal command)
            if float(np.linalg.norm(des_vel_ctrl)) < 1e-3:
                self._mode2_vdes_bias *= 0.98
            else:
                self._mode2_vdes_bias = self._mode2_vdes_bias + ki * e * dt
                if cap > 0.0:
                    self._mode2_vdes_bias = np.clip(self._mode2_vdes_bias, -cap, cap)
            des_vel_ctrl = des_vel_ctrl + self._mode2_vdes_bias

        desiredPos = np.array([des_vel_ctrl[0], des_vel_ctrl[1], 1.0])
        
        # Phase 切换在 controller.compute_torque() 内部通过腿长判断
        # （与 Hopper4.py 一致）

        # ========== Domain Randomization: 观测噪声 ==========
        if self.domain_rand_enabled:
            state = self.domain_randomizer.add_observation_noise(state)
            # IMU 偏移
            state['body_rpy'] = self.domain_randomizer.apply_imu_offset(state['body_rpy'])
            # 电机零点偏移 (与 RL 版本一致)
            state['joint_pos'] = self.domain_randomizer.apply_motor_offset(state['joint_pos'])
        
        # 支持 mode2/mode3: 共享期望姿态 desired_rpy（由 leg_prop 中计算并存入 self）
        desired_rpy = getattr(self, 'shared_desired_rpy', None)
        if hasattr(self.controller, 'compute_torque') and self.controller.__class__.__name__ in ('RaibertControllerMode2', 'RaibertControllerMode3'):
            torque, info = self.controller.compute_torque(
                X=state['foot_pos'],
                xdot=state['foot_vel'],
                joint=state['joint_pos'],
                jointVel=state['joint_vel'],
                vel=state.get('body_vel_ctrl', state['body_vel']),
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

        # Return the *post-step* state (useful for logging/video metrics)
        state_after = self.interface.get_state()
        return state_after, torque, info

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

        state_after = self.interface.get_state()
        return state_after, torque, info

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
        contact_flag = bool(self.interface.get_foot_contact())
        
        # 2. 计算腿部扭矩（默认 Raibert 控制，用于相位更新）
        leg_torque, info, state = self.compute_leg_control(state, desired_vel)
        
        # 3. 获取姿态信息
        rpy = state['body_rpy']
        omega = state['body_ang_vel']  # 角速度 [p, q, r]
        current_phase = self.controller.state  # 1=Flight, 2=Stance
        
        robot_mass = self.controller.m
        base_thrust_ratio = float(self.prop_base_thrust_ratio)
        # mode2: propellers are only for attitude stabilization; limit total thrust to <= 10% * m*g
        if int(getattr(self, 'controller_mode', 1)) == 2:
            base_thrust_ratio = min(base_thrust_ratio, 0.10)
        base_thrust = base_thrust_ratio * robot_mass * self.gravity
        
        # ========== Unified SRB/WBC solve (single dynamics constraint) ==========
        # mode3: 用同一个 SRB 动力学约束做 WBC。
        # 这里采用“稳定优先”的实现：腿仍由 Raibert/spring 产生接触力，
        # WBC 用 SRB 约束计算三旋翼 thrusts 去补偿残差（速度/高度/姿态收敛）。
        if self.controller_mode == 3:
            # ---- Full-body WBC-QP (manipulator equation, no SRB simplification) ----
            # Phase/contact for mode3:
            # User requirement (real robot has no foot force sensor): follow the same phase logic as mode2,
            # i.e., rely on controller.state (leg-length hysteresis) instead of MuJoCo contact detection.
            phase_heur = int(current_phase)
            # Contact source options:
            # - "gate" (default): phase heuristic + kinematic gate (real-robot friendly)
            # - "mujoco": use MuJoCo contact flag for WBC constraints (sim-only, much more reliable)
            contact_source = str(os.environ.get('MODE3_CONTACT_SOURCE', 'gate')).lower().strip()
            gate_ok, foot_z, vfoot = self._mode3_contact_gate() if bool(getattr(self, 'mode3_gate_enabled', True)) else (True, float('nan'), float('nan'))
            if contact_source == 'mujoco':
                in_contact = bool(contact_flag)
                # keep gate counter for logging only
                if in_contact and gate_ok:
                    self._mode3_gate_count = min(int(getattr(self, 'mode3_gate_min_count', 8)), self._mode3_gate_count + 1)
                elif in_contact and (not gate_ok):
                    self._mode3_gate_count = max(0, self._mode3_gate_count - 1)
                else:
                    self._mode3_gate_count = 0
            else:
                in_contact = bool(phase_heur == 2)
                # kinematic self-consistency gate (mode3): avoid false stance in flight
                if in_contact:
                    # integrate gate (debounce)
                    if gate_ok:
                        self._mode3_gate_count = min(int(getattr(self, 'mode3_gate_min_count', 8)), self._mode3_gate_count + 1)
                    else:
                        self._mode3_gate_count = max(0, self._mode3_gate_count - 1)
                    in_contact = bool(self._mode3_gate_count >= int(getattr(self, 'mode3_gate_min_count', 8)))
                else:
                    self._mode3_gate_count = 0
            phase_gated = 2 if in_contact else 1

            # --- touchdown bookkeeping (for soft-landing scheduling) ---
            prev_in_contact = bool(getattr(self, '_mode3_prev_in_contact', False))
            if in_contact and (not prev_in_contact):
                self._mode3_td_ticks = 0
                self._mode3_td_z = float(state['body_pos'][2])
                # mode3 stance split: record touchdown leg length reference
                try:
                    leg_len_now = float(np.linalg.norm(np.asarray(state['foot_pos'], dtype=float).reshape(3)))
                except Exception:
                    leg_len_now = float(info.get('leg_length', float('nan')))
                self._mode3_td_leglen = leg_len_now
                # target minimum leg length during compression (ratio of touchdown length)
                comp_ratio = float(os.environ.get('MODE3_STANCE_COMP_RATIO', '0.82'))  # <1 => allow compression
                comp_ratio = float(np.clip(comp_ratio, 0.5, 0.98))
                self._mode3_td_leglen_min = float(leg_len_now * comp_ratio) if np.isfinite(leg_len_now) else None
                self._mode3_stance_subphase = 1  # start with compression
            elif in_contact:
                self._mode3_td_ticks = int(getattr(self, '_mode3_td_ticks', 0)) + 1
            else:
                self._mode3_td_ticks = 0
                self._mode3_td_leglen = None
                self._mode3_td_leglen_min = None
                self._mode3_stance_subphase = 0
            self._mode3_prev_in_contact = bool(in_contact)

            td_ramp_ticks = int(getattr(self, 'mode3_td_ramp_ticks', 120))  # ~0.12s @ 1kHz
            td_alpha = float(np.clip(float(self._mode3_td_ticks) / max(1.0, float(td_ramp_ticks)), 0.0, 1.0))
            info['td_ticks'] = int(self._mode3_td_ticks)
            info['td_alpha'] = float(td_alpha)
            # 验证用：contact detection 与腿长相位是否一致（不影响控制）
            info['contact_flag'] = contact_flag
            info['phase_heuristic'] = int(phase_heur)
            info['phase_gated'] = int(phase_gated)
            info['phase_gate_ok'] = int(bool(gate_ok))
            info['phase_gate_count'] = int(self._mode3_gate_count)
            info['foot_z_world'] = float(foot_z) if np.isfinite(foot_z) else float('nan')
            info['foot_v_world_norm'] = float(vfoot) if np.isfinite(vfoot) else float('nan')
            info['phase_contact'] = 2 if contact_flag else 1
            info['phase_mismatch'] = int((info['phase_contact'] != info['phase_gated']))

            # ---- Mode3 stance sub-phase: compression -> push-off ----
            # Use leg length rate + a minimum compression target to decide when to switch.
            try:
                leg_len = float(np.linalg.norm(np.asarray(state['foot_pos'], dtype=float).reshape(3)))
            except Exception:
                leg_len = float(info.get('leg_length', float('nan')))
            if self._mode3_leglen_prev is None or (not np.isfinite(self._mode3_leglen_prev)):
                dleg = 0.0
            else:
                dleg = float((leg_len - float(self._mode3_leglen_prev)) / max(1e-6, float(self.dt)))
            self._mode3_leglen_prev = leg_len

            if in_contact:
                # If we reached min compression OR leg starts extending, switch to push-off
                eps_d = float(os.environ.get('MODE3_STANCE_DLEG_EPS', '0.02'))  # m/s
                min_len = self._mode3_td_leglen_min
                if self._mode3_stance_subphase != 2:
                    if (min_len is not None and np.isfinite(min_len) and np.isfinite(leg_len) and (leg_len <= float(min_len))) or (dleg > eps_d):
                        self._mode3_stance_subphase = 2
                # Safety: if stance lasts too long, force push-off
                max_comp_ticks = int(os.environ.get('MODE3_STANCE_COMP_MAX_TICKS', '120'))  # ~0.12s @1kHz
                if int(getattr(self, '_mode3_td_ticks', 0)) >= max_comp_ticks:
                    self._mode3_stance_subphase = 2
            else:
                self._mode3_stance_subphase = 0

            info['leg_length'] = float(leg_len)
            info['leg_dleg'] = float(dleg)
            info['stance_subphase'] = int(self._mode3_stance_subphase)  # 0 flight, 1 comp, 2 push-off

            q = state['body_quat']
            wq, xq, yq, zq = float(q[0]), float(q[1]), float(q[2]), float(q[3])
            R_wb = np.array([
                [wq*wq + xq*xq - yq*yq - zq*zq, 2*(xq*yq - wq*zq),       2*(xq*zq + wq*yq)],
                [2*(xq*yq + wq*zq),         wq*wq - xq*xq + yq*yq - zq*zq, 2*(yq*zq - wq*xq)],
                [2*(xq*zq - wq*yq),         2*(yq*zq + wq*xq),       wq*wq - xq*xq - yq*yq + zq*zq],
            ], dtype=float)

            # MIT-style: do NOT use MuJoCo velocity for control.
            # Use EKF velocity everywhere in WBC references.
            vel_w = np.asarray(state.get('body_vel_ctrl', state.get('est_body_vel', state.get('body_vel', np.zeros(3)))), dtype=float).reshape(3)
            pos_w = state['body_pos']

            # ---- Mode3 S2S (B: foot placement) update at APEX ----
            # Apex event (flight only): vz crosses from + to -.
            # Update a WORLD-frame XY offset added to the Raibert flight targetFootPos.
            if int(getattr(self, 'controller_mode', 1)) == 3:
                try:
                    # desired velocity in world (apply same sign mapping as WBC)
                    sx = float(os.environ.get('MODE3_VEL_SIGN_X', '1.0'))
                    sy = float(os.environ.get('MODE3_VEL_SIGN_Y', '1.0'))
                    vdes_xy = np.array([sx * float(desired_vel[0]), sy * float(desired_vel[1])], dtype=float)

                    # reset offset when command changes a lot (e.g., Task1 forward->inplace switch)
                    if np.linalg.norm(vdes_xy - self._mode3_s2s_last_vdes) > float(os.environ.get('MODE3_S2S_RESET_THRESH', '0.10')):
                        self._mode3_s2s_offset_xy[:] = 0.0
                    self._mode3_s2s_last_vdes = vdes_xy.copy()

                    in_flight = bool(not in_contact)
                    vz = float(vel_w[2])
                    apex = False
                    if self._mode3_s2s_vz_prev is not None:
                        if bool(self._mode3_s2s_prev_in_flight) and in_flight and (float(self._mode3_s2s_vz_prev) > 0.0) and (vz <= 0.0):
                            apex = True
                    self._mode3_s2s_prev_in_flight = in_flight
                    self._mode3_s2s_vz_prev = vz

                    if apex:
                        v_apex_xy = np.asarray(vel_w[0:2], dtype=float).reshape(2)
                        e = (vdes_xy - v_apex_xy)
                        # Use hop period to scale from (m/s) error to (m) placement change.
                        T = 1.0 / max(1e-6, float(self.hop_freq_hz))
                        k = float(os.environ.get('MODE3_S2S_K', '0.35'))  # dimensionless gain
                        dxy = k * e * (0.5 * T)
                        self._mode3_s2s_offset_xy = self._mode3_s2s_offset_xy + dxy
                        lim = float(os.environ.get('MODE3_S2S_OFFSET_LIM', '0.06'))
                        self._mode3_s2s_offset_xy = np.clip(self._mode3_s2s_offset_xy, -lim, lim)

                        # push to controller
                        if hasattr(self.controller, 's2s_offset_world'):
                            self.controller.s2s_offset_world[0] = float(self._mode3_s2s_offset_xy[0])
                            self.controller.s2s_offset_world[1] = float(self._mode3_s2s_offset_xy[1])
                            self.controller.s2s_offset_world[2] = 0.0

                    info['s2s_apex'] = int(apex)
                    info['s2s_offset_x'] = float(self._mode3_s2s_offset_xy[0])
                    info['s2s_offset_y'] = float(self._mode3_s2s_offset_xy[1])
                except Exception:
                    pass

            # ---- Mode3 S2S (A: touchdown pitch) update at APEX ----
            # Update a pitch bias (rad) once per step at apex: pitch_bias += k * (vdes - v_apex)
            try:
                s2s_a_on = bool(int(os.environ.get('MODE3_S2S_A', '1')))
                if s2s_a_on:
                    # reuse apex detection result above if present
                    apex_a = bool(info.get('s2s_apex', 0) == 1)
                    if apex_a:
                        vdes_x = float(os.environ.get('MODE3_VEL_SIGN_X', '1.0')) * float(desired_vel[0])
                        vx_apex = float(vel_w[0])
                        e = float(vdes_x - vx_apex)
                        k = float(os.environ.get('MODE3_S2S_A_K', '0.20'))  # rad per (m/s)
                        self._mode3_s2s_pitch_bias = float(self._mode3_s2s_pitch_bias + k * e)
                        lim = float(os.environ.get('MODE3_S2S_PITCH_LIM_DEG', '6.0'))
                        lim = float(np.radians(lim))
                        self._mode3_s2s_pitch_bias = float(np.clip(self._mode3_s2s_pitch_bias, -lim, lim))
                    info['s2s_pitch_bias_deg'] = float(np.degrees(self._mode3_s2s_pitch_bias))
            except Exception:
                pass

            # base linear accel reference (jump primary)
            if bool(getattr(self, 'hop_enabled', False)) and in_contact:
                w_hop = 2.0 * np.pi * float(self.hop_freq_hz)
                z_ref = float(self.hop_z0 + self.hop_amp * (1.0 - np.cos(w_hop * self.sim_time)) * 0.5)
                z_ref = float(np.clip(z_ref, self.hop_z0, self.hop_peak_z))
                vz_ref = float(self.hop_amp * np.sin(w_hop * self.sim_time) * w_hop * 0.5)
                az_ff = float(self.hop_amp * np.cos(w_hop * self.sim_time) * (w_hop ** 2) * 0.5)

                z_err = float(z_ref - pos_w[2])
                vz_err = float(vz_ref - vel_w[2])
                az_des = az_ff + self.wbc_kp_z * z_err + self.wbc_kd_z * vz_err
                # Allow stronger takeoff accel to reach higher apex (tuned by stability governors elsewhere).
                az_des = float(np.clip(az_des, -40.0, 140.0))

                # --- soft landing (mode3): blend to a "damped compression" objective right after touchdown ---
                if bool(getattr(self, 'mode3_soft_landing_enabled', False)) and (td_alpha < 1.0):
                    comp = float(getattr(self, 'mode3_land_compress_m', 0.05))  # allow ~5cm compression
                    z_td = float(getattr(self, '_mode3_td_z', pos_w[2]))
                    z_soft_ref = float(z_td - comp)
                    kp_land = float(getattr(self, 'mode3_land_kp', 60.0))
                    kd_land = float(getattr(self, 'mode3_land_kd', 16.0))
                    az_soft = (-0.15 * self.gravity) + kp_land * float(z_soft_ref - pos_w[2]) + kd_land * float(0.0 - vel_w[2])
                    az_soft = float(np.clip(az_soft, -40.0, 25.0))
                    az_des = float(td_alpha * az_des + (1.0 - td_alpha) * az_soft)
                    # early stance: avoid huge upward accel spikes
                    # NOTE: keep some upward authority so we can still reach high apex targets (e.g. 1.0m).
                    az_des = float(np.clip(az_des, -40.0, 80.0))

                # Two-stage stance:
                # - compression: soften upward accel to allow leg compression (store energy)
                # - push-off: allow aggressive upward accel for takeoff
                if int(getattr(self, '_mode3_stance_subphase', 0)) == 1:
                    # compression
                    az_des = float(np.clip(az_des, -60.0, float(os.environ.get('MODE3_COMP_AZ_UP_MAX', '40.0'))))
                elif int(getattr(self, '_mode3_stance_subphase', 0)) == 2:
                    # push-off / shooting
                    az_des = float(np.clip(az_des, -40.0, float(os.environ.get('MODE3_PUSH_AZ_UP_MAX', '180.0'))))
                kv_xy = float(self.wbc_kv_xy) * float(self.hop_kv_xy_scale)
            elif bool(getattr(self, 'hop_enabled', False)) and (not in_contact):
                # Flight: still track the same hop reference using prop thrust (no forced "push down").
                # This makes height regulation feasible in flight and improves stability (less aggressive vertical bias).
                w_hop = 2.0 * np.pi * float(self.hop_freq_hz)
                z_ref = float(self.hop_z0 + self.hop_amp * (1.0 - np.cos(w_hop * self.sim_time)) * 0.5)
                z_ref = float(np.clip(z_ref, self.hop_z0, self.hop_peak_z))
                vz_ref = float(self.hop_amp * np.sin(w_hop * self.sim_time) * w_hop * 0.5)
                az_ff = float(self.hop_amp * np.cos(w_hop * self.sim_time) * (w_hop ** 2) * 0.5)

                z_err = float(z_ref - pos_w[2])
                vz_err = float(vz_ref - vel_w[2])
                az_des = az_ff + self.wbc_kp_z * z_err + self.wbc_kd_z * vz_err
                az_des = float(np.clip(az_des, -40.0, 120.0))
                kv_xy = float(self.wbc_kv_xy) * float(self.hop_kv_xy_scale)
            else:
                z_err = float(self.wbc_z_des - pos_w[2])
                vz_err = float(0.0 - vel_w[2])
                az_des = self.wbc_kp_z * z_err + self.wbc_kd_z * vz_err
                kv_xy = float(self.wbc_kv_xy)

            # desired velocity: allow explicit sign mapping for this model (see MODE3_VEL_SIGN_X/Y)
            sx = float(os.environ.get('MODE3_VEL_SIGN_X', '1.0'))
            sy = float(os.environ.get('MODE3_VEL_SIGN_Y', '1.0'))
            des_vx = float(sx * float(desired_vel[0]))
            des_vy = float(sy * float(desired_vel[1]))
            ax_des = kv_xy * float(des_vx - vel_w[0])
            ay_des = kv_xy * float(des_vy - vel_w[1])

            # ---- Tilt safety governor (Mode3): keep roll/pitch within +/- 15 deg ----
            # If current tilt is close to the limit, reduce horizontal accel demand (so desired tilt shrinks).
            # If already beyond the limit, command upright (roll=pitch=0) and zero horizontal accel.
            tilt_limit = float(getattr(self, 'mode3_tilt_limit_rad', np.radians(15.0)))
            tilt_margin = float(getattr(self, 'mode3_tilt_margin_rad', np.radians(3.0)))  # start braking early
            roll_now = float(rpy[0])
            pitch_now = float(rpy[1])
            tilt_abs = float(max(abs(roll_now), abs(pitch_now)))
            if tilt_abs >= tilt_limit:
                tilt_scale = 0.0
                tilt_force_upright = True
            elif tilt_abs >= (tilt_limit - tilt_margin):
                tilt_scale = float(np.clip((tilt_limit - tilt_abs) / max(1e-6, tilt_margin), 0.0, 1.0))
                tilt_force_upright = False
            else:
                tilt_scale = 1.0
                tilt_force_upright = False

            ax_des *= tilt_scale
            ay_des *= tilt_scale

            # Additional tilt damping: if the base is already tilted, bias horizontal accel to recover.
            # This helps keep the *actual* pitch/roll inside the envelope when authority is limited.
            k_tilt_damp = float(getattr(self, 'mode3_tilt_damp', 0.8))
            ax_des = float(ax_des - k_tilt_damp * pitch_now * self.gravity)
            ay_des = float(ay_des + k_tilt_damp * roll_now * self.gravity)

            desired_base_lin_acc_w = np.array([ax_des, ay_des, az_des], dtype=float)
            info['mode3_tilt_abs_deg'] = float(np.degrees(tilt_abs))
            info['mode3_tilt_scale'] = float(tilt_scale)
            info['mode3_tilt_limit_deg'] = float(np.degrees(tilt_limit))

            # ---- Prop thrust tracking (PogoX-inspired) ----
            # Use propellers to assist vertical dynamics by tracking a desired total thrust corresponding to az_des.
            # Optionally, add a flight-only energy correction term (CLF-like power shaping).
            desired_thrusts = None
            if bool(getattr(self, 'hop_enabled', False)) and bool(getattr(self, 'mode3_thrust_track_enabled', True)):
                try:
                    m = float(getattr(self.controller, 'm', robot_mass))
                except Exception:
                    m = float(robot_mass)
                g = float(self.gravity)

                # Base thrust command from desired vertical acceleration:
                # Fz ~= m(g + az_des) * share
                share = float(np.clip(float(getattr(self, 'mode3_prop_thrust_share', 0.7)), 0.0, 1.0))
                Fz_cmd = float(m * (g + float(az_des)) * share)

                # Optional: flight-only energy shaping (Xiong/PogoX) to hit a desired apex energy.
                if (not in_contact) and bool(getattr(self, 'mode3_energy_enabled', False)):
                    z_apex_des = float(getattr(self, 'mode3_apex_z_des', float(self.hop_peak_z)))
                    z_now = float(pos_w[2])
                    vz_now = float(vel_w[2])
                    E = m * g * z_now + 0.5 * m * (vz_now ** 2)
                    Ed = m * g * z_apex_des
                    eE = float(E - Ed)
                    kE = float(getattr(self, 'mode3_energy_k', 6.0))
                    P_des = -kE * eE
                    vz_eff = float(np.sign(vz_now) * max(abs(vz_now), float(getattr(self, 'mode3_energy_vz_eps', 0.4))))
                    Fz_cmd = float(Fz_cmd + (P_des / vz_eff))
                    info['mode3_E'] = float(E)
                    info['mode3_Ed'] = float(Ed)
                    info['mode3_eE'] = float(eE)

                # clamp by available thrust sum
                Fz_min = 0.0
                Fz_max = float(getattr(self.wbc_full_qp.cfg, 'thrust_sum_max', 0.0) or (3.0 * float(getattr(self.wbc_full_qp.cfg, 'thrust_max_each', 0.0))))
                Fz_cmd = float(np.clip(Fz_cmd, Fz_min, Fz_max))
                desired_thrusts = np.ones(3, dtype=float) * (Fz_cmd / 3.0)
                info['mode3_Fz_cmd'] = float(Fz_cmd)

            # desired tilt from accel (small-angle)
            # Use a strict mode3 tilt limit (can be smaller than self.wbc_max_tilt).
            max_tilt_mode3 = float(min(float(self.wbc_max_tilt), tilt_limit))
            pitch_sign = float(getattr(self, 'mode3_pitch_sign', 1.0))
            roll_sign = float(getattr(self, 'mode3_roll_sign', 1.0))
            desired_pitch = float(pitch_sign * np.clip(ax_des / self.gravity, -max_tilt_mode3, max_tilt_mode3))
            desired_roll = float(roll_sign * np.clip(-ay_des / self.gravity, -max_tilt_mode3, max_tilt_mode3))
            # S2S-A: add touchdown pitch bias mainly in flight
            try:
                if bool(int(os.environ.get('MODE3_S2S_A', '1'))) and (not in_contact):
                    desired_pitch = float(desired_pitch + float(self._mode3_s2s_pitch_bias))
            except Exception:
                pass
            if tilt_force_upright:
                desired_roll, desired_pitch = 0.0, 0.0
            desired_rpy = np.array([desired_roll, desired_pitch, 0.0], dtype=float)
            self.shared_desired_rpy = desired_rpy.copy()

            # base angular accel reference (world)
            # IMPORTANT: yaw is effectively unactuated for this tri-rotor model (Mz≈0),
            # so using raw RPY subtraction can create artificial oscillations when yaw drifts.
            # We therefore build a desired rotation that keeps current yaw, and regulate only roll/pitch.
            # IMPORTANT (MuJoCo convention):
            # free-joint angular velocity qvel[3:6] is expressed in the BODY frame.
            # Convert to world before using it in a world-frame attitude task to avoid sign/coupling issues.
            omega_world = (R_wb @ np.asarray(state['body_ang_vel'], dtype=float).reshape(3)).copy()
            # Gains can be phase-dependent (stance usually needs stronger regulation).
            kp_att = float(getattr(self, 'mode3_kp_att', 45.0))
            kd_att = float(getattr(self, 'mode3_kd_att', 9.0))
            if in_contact:
                kp_att = float(getattr(self, 'mode3_kp_att_stance', kp_att))
                kd_att = float(getattr(self, 'mode3_kd_att_stance', kd_att))
            else:
                kp_att = float(getattr(self, 'mode3_kp_att_flight', kp_att))
                kd_att = float(getattr(self, 'mode3_kd_att_flight', kd_att))
            # When tilt governor activates, increase attitude bandwidth so we recover inside the envelope quickly.
            gain_boost = float(getattr(self, 'mode3_tilt_gain_boost', 2.0))
            boost = float(1.0 + (1.0 - float(tilt_scale)) * gain_boost)
            kp_att *= boost
            kd_att *= boost

            # --- Attitude task (Mode3): robust roll/pitch regulation ---
            # Within our enforced envelope (<= ~15deg), a simple roll/pitch PD in WORLD coordinates is more robust
            # than a mixed-frame vee-map here, and avoids sign issues across dynamics backends.
            desired_base_ang_acc_w = np.array([
                kp_att * float(desired_rpy[0] - roll_now) - kd_att * float(omega_world[0]),
                kp_att * float(desired_rpy[1] - pitch_now) - kd_att * float(omega_world[1]),
                0.0
            ], dtype=float)

            # flight swing foot target (world), for WBC swing task
            desired_swing_foot_pos_w = None
            if not in_contact:
                # Keep flight leg length at the reset reference (\"original length\")
                l0 = float(getattr(self, 'mode3_leglen_ref', None) or getattr(self.controller, 'l0', 0.464))
                # For strict leg-length holding, use a purely vertical target in world frame.
                # This reduces coupling with roll/pitch and prevents the leg from tucking up in the air.
                desired_swing_foot_pos_w = pos_w + np.array([0.0, 0.0, -l0], dtype=float)

            dof_ids = [self.model.jnt_dofadr[jid] for jid in self.interface.joint_ids]
            dyn_override = None
            if (self.full_wbc_dynamics_backend == 'pinocchio') and (self.pin_dyn is not None):
                q_pin, v_pin = self.pin_dyn.mujoco_state_to_qv(
                    base_pos_w=state['body_pos'],
                    base_quat_wxyz=state['body_quat'],
                    base_vel_w=state['body_vel'],
                    base_omega_w=self.data.qvel[3:6].copy(),
                    joint_pos=state['joint_pos'],
                    joint_vel=state['joint_vel'],
                )
                dyn_override = self.pin_dyn.compute_all_terms(q_pin, v_pin)

            # mode3: weight scheduling (flight/stance)
            # Keep this robust: record any overridden cfg fields and restore after solve.
            cfg = self.wbc_full_qp.cfg
            _cfg_restore: dict[str, float] = {}

            def _set_cfg(name: str, value: float) -> None:
                if name not in _cfg_restore:
                    _cfg_restore[name] = float(getattr(cfg, name))
                setattr(cfg, name, float(value))

            # Hard cap: total prop thrust <= ratio * m*g (Hopper hops; prop is balance assist).
            try:
                ratio = float(getattr(self, 'mode3_prop_thrust_ratio_max', 0.0))
                if ratio > 0.0:
                    cap = float(ratio * robot_mass * self.gravity)
                    if getattr(cfg, 'thrust_sum_max', None) is None:
                        _set_cfg('thrust_sum_max', cap)
                    else:
                        _set_cfg('thrust_sum_max', min(float(cfg.thrust_sum_max), cap))
            except Exception:
                pass

            # Option: remove prop from QP (leg-dominant hopping) and use a separate attitude assist.
            # This keeps hopping height mainly from the leg spring/actuation.
            prop_in_qp = bool(int(os.environ.get('MODE3_PROP_IN_QP', '1')))
            if not prop_in_qp:
                _set_cfg('thrust_max_each', 0.0)
                _set_cfg('thrust_sum_max', 0.0)

            if not in_contact:
                _set_cfg('w_swing_foot', 50000.0)
                _set_cfg('swing_kp', 12000.0)
                _set_cfg('swing_kd', 300.0)
                _set_cfg('w_base_lin', 150.0)
                # stronger attitude regulation in flight (propeller-dominant) to reduce oscillation/energy loss
                _set_cfg('w_base_ang', float(getattr(self, 'mode3_w_base_ang_flight', 220.0)))
                # keep leg posture in flight; but let prop handle most regulation
                _set_cfg('w_tau_track', float(getattr(self, 'mode3_w_tau_track_flight', 5e-2)))
                # track desired thrust profile in flight (energy shaping), plus smoothing in touchdown ramp
                if desired_thrusts is not None:
                    _set_cfg('w_thrust_track', float(getattr(self, 'mode3_w_thrust_track_flight', 5e-2)))
            else:
                # stance: prioritize keeping base level (single-contact is pitch-sensitive)
                _set_cfg('w_base_ang', float(getattr(self, 'mode3_w_base_ang_stance', 260.0)))
                # stance: encourage "spring-like" leg behavior by tracking the Raibert/virtual-spring torque prior
                _set_cfg('w_tau_track', float(getattr(self, 'mode3_w_tau_track_stance', 2e-1)))
                # stance: soften contact constraint right after touchdown, then ramp hard (reduces impulse)
                if bool(getattr(self, 'mode3_soft_landing_enabled', False)):
                    # Smooth only during touchdown ramp; do NOT smooth contact forces (keeps horizontal authority for speed tracking).
                    _set_cfg('w_fc_track', 0.0)
                    _set_cfg('w_thrust_track', float(getattr(self, 'mode3_w_thrust_track', 2e-3)) if (td_alpha < 1.0) else 0.0)

            # priors for smoothing (previous solution)
            # For touchdown: do NOT use a zero prior from flight; keep the last stance force (or mg) as a better prior.
            fc_prior = np.asarray(getattr(self, '_mode3_prev_fc', np.array([0.0, 0.0, robot_mass * self.gravity])), dtype=float).reshape(3)
            # thrust prior:
            # - if energy shaping provides a target, track that (flight).
            # - otherwise, smooth from previous solution.
            if desired_thrusts is not None:
                th_prior = np.asarray(desired_thrusts, dtype=float).reshape(3)
            else:
                th_prior = np.asarray(getattr(self, '_mode3_prev_thrusts', np.zeros(3)), dtype=float).reshape(3)
            sol = self.wbc_full_qp.solve(
                self.model,
                self.data,
                base_body_id=self.interface.base_body_id,
                foot_body_id=self.interface.foot_body_id,
                joint_dof_ids=dof_ids,
                prop_positions_body=self.interface.prop_positions_body,
                # MIT-style: always use EKF velocity for constraints too.
                base_vel_w=np.asarray(state.get('body_vel_ctrl', state.get('est_body_vel', np.zeros(3))), dtype=float).reshape(3),
                shift_ref=getattr(self, 'mode3_shift_ref', None),
                tau_prior=np.asarray(leg_torque, dtype=float).reshape(3),
                fc_prior=fc_prior,
                thrust_prior=th_prior,
                desired_base_lin_acc_w=desired_base_lin_acc_w,
                desired_base_ang_acc_w=desired_base_ang_acc_w,
                desired_swing_foot_pos_w=desired_swing_foot_pos_w,
                in_stance=in_contact,
                dyn_override=dyn_override,
            )
            # restore scheduled cfg values
            for _k, _v in _cfg_restore.items():
                setattr(cfg, _k, float(_v))

            leg_torque = np.asarray(sol['tau'], dtype=float).copy()
            thrusts = np.asarray(sol['thrusts'], dtype=float).copy()
            # store priors for next step smoothing
            if in_contact:
                self._mode3_prev_fc = np.asarray(sol.get('f_contact_w', np.zeros(3)), dtype=float).reshape(3)
            self._mode3_prev_thrusts = thrusts.copy()

            # mode3: flight leg extension hold (shift joint servo)
            if (not in_contact) and bool(getattr(self, 'mode3_flight_shift_hold_enabled', True)):
                try:
                    shift_ref = getattr(self, 'mode3_shift_ref', None)
                    if shift_ref is not None:
                        q_shift = float(np.asarray(state['joint_pos'], dtype=float).reshape(3)[2])
                        qd_shift = float(np.asarray(state['joint_vel'], dtype=float).reshape(3)[2])
                        tau_pd = float(getattr(self, 'mode3_flight_shift_kp', 250.0)) * (float(shift_ref) - q_shift) + \
                                 float(getattr(self, 'mode3_flight_shift_kd', 25.0)) * (0.0 - qd_shift)
                        tau_lim = float(getattr(self, 'mode3_flight_shift_tau_limit', 300.0))
                        tau_pd = float(np.clip(tau_pd, -tau_lim, tau_lim))
                        leg_torque[2] = float(np.clip(leg_torque[2] + tau_pd, -tau_lim, tau_lim))
                except Exception:
                    pass

            self.interface.set_torque(leg_torque)
            # Prop application:
            # - If prop is in QP: use QP thrusts directly.
            # - If prop is NOT in QP: use a lightweight attitude PD + small base thrust (balance assist).
            if prop_in_qp:
                self.interface.apply_propeller_forces(thrusts, np.zeros(3))
            else:
                # Small base thrust so the mixer can generate roll/pitch moments (but still << mg).
                base_ratio = float(os.environ.get('MODE3_PROP_BASE_THRUST_RATIO', '0.08'))
                total_thrust = float(max(0.0, base_ratio * robot_mass * self.gravity))

                # Attitude PD about roll/pitch (use world-frame rpy + body-frame omega)
                # NOTE: moments are applied in body frame by mixer model.
                kp_p = float(os.environ.get('MODE3_PROP_KP_RP', '80.0'))
                kd_p = float(os.environ.get('MODE3_PROP_KD_RP', '12.0'))
                e_roll = float(desired_rpy[0] - rpy[0])
                e_pitch = float(desired_rpy[1] - rpy[1])
                omega_b = np.asarray(state['body_ang_vel'], dtype=float).reshape(3)
                Mx = float(kp_p * e_roll - kd_p * float(omega_b[0]))
                My = float(kp_p * e_pitch - kd_p * float(omega_b[1]))

                # Enforce the same thrust ratio cap (if set)
                ratio_cap = float(getattr(self, 'mode3_prop_thrust_ratio_max', 0.0))
                if ratio_cap > 0.0:
                    total_thrust = float(min(total_thrust, ratio_cap * robot_mass * self.gravity))

                motor_speeds = self.tri_rotor_mixer.calculate(total_thrust, Mx, My, 0.0)
                thrusts_assist = np.array([self.tri_rotor_mixer.calc_thrust_from_speed(s) for s in motor_speeds], dtype=float)
                self.interface.apply_propeller_forces(thrusts_assist, np.zeros(3))
                # override info
                thrusts = thrusts_assist.copy()

            for _ in range(self.steps_per_control):
                mujoco.mj_step(self.model, self.data)
                self.sim_time += self.dt

            info['mode'] = 'leg_prop'
            info['desired_rpy'] = desired_rpy
            info['prop_thrust'] = float(np.sum(thrusts))
            info['wbc_status'] = sol.get('status', 'unknown')
            info['wbc_slack_norm'] = float(sol.get('slack_dyn_norm', 0.0)) + float(sol.get('slack_contact_norm', 0.0))
            info['wbc_f_contact'] = np.asarray(sol.get('f_contact_w', np.zeros(3)), dtype=float)
            state_after = self.interface.get_state()
            return state_after, leg_torque, info
        else:
            # mode2 and others: keep previous SE3 assist logic
            self.interface.set_torque(leg_torque)
        
        # ========== Legacy SE3 assist (mode2 / others) ==========
        # Desired attitude for prop assist: small tilt from velocity error (helps speed tracking and matches earlier stable behavior)
        vel_world = state['body_vel']
        v_err = np.array([desired_vel[0] - vel_world[0], desired_vel[1] - vel_world[1]], dtype=float)
        is_mode2 = int(getattr(self, 'controller_mode', 1)) == 2
        if is_mode2:
            # mode2: keep prop target attitude level; speed is handled by the leg (Raibert)
            desired_rpy_vel = np.zeros(3, dtype=float)
        else:
            k_tilt = 0.8
            max_tilt = float(self.wbc_max_tilt)
            desired_roll = float(np.clip(k_tilt * v_err[1], -max_tilt, max_tilt))
            desired_pitch = float(np.clip(k_tilt * v_err[0], -max_tilt, max_tilt))
            desired_rpy_vel = np.array([desired_roll, desired_pitch, 0.0], dtype=float)

        # mode2: hip(腿) 在 stance 阶段不追倾角（更像 mode1），prop 在 stance 通过 ramp 小幅参与稳定（避免 touchdown 翻车）
        if is_mode2 and current_phase == 2:
            desired_rpy_leg = np.zeros(3, dtype=float)
            desired_rpy_prop = desired_rpy_vel
        else:
            desired_rpy_leg = desired_rpy_vel
            desired_rpy_prop = desired_rpy_vel
        desired_rpy = desired_rpy_prop
        self.shared_desired_rpy = desired_rpy_leg.copy()

        # thrust schedule:
        # - mode2(B): flight+stance 都允许旋翼参与（你要求 ESC 有 PWM 输出），但 stance 仍稍微降低以减少触地干扰
        # - 其他模式保留历史设置（stance 降低但不为 0）
        # mode2: keep thrust small; still allow it in stance/flight, but always within the 10% m*g cap below
        if is_mode2:
            phase_scale = float(self.mode2_phase_scale_stance) if current_phase == 2 else float(self.mode2_phase_scale_flight)
        else:
            phase_scale = 0.6 if current_phase == 2 else 1.0
        total_thrust = float(base_thrust * phase_scale)

        cmd = self.se3_controller.attitude_control(
            current_quat_wxyz=state['body_quat'],
            current_omega_body=omega,
            desired_rpy=desired_rpy,
            thrust_newton=total_thrust,
        )
        Mx, My, Mz = float(cmd.moment[0]), float(cmd.moment[1]), 0.0
        # mode2: stance 阶段关闭姿态力矩（只保留少量推力），flight 再开启
        if is_mode2:
            if current_phase == 2:
                # touchdown / early stance: ramp from 0 -> stance_scale over ~0.1s @1kHz
                phase_cnt = float(getattr(self.controller, 'phase_duration_count', 0))
                settle = 20.0
                ramp = 80.0
                ramp01 = float(np.clip((phase_cnt - settle) / ramp, 0.0, 1.0))
                stance_scale = float(self.mode2_moment_scale_stance) * ramp01
                Mx *= stance_scale
                My *= stance_scale
            else:
                Mx *= float(self.mode2_moment_scale_flight)
                My *= float(self.mode2_moment_scale_flight)

        # 全局力矩缩放（保留原有接口）
        Mx *= float(getattr(self, 'prop_torque_scale', 1.0))
        My *= float(getattr(self, 'prop_torque_scale', 1.0))

        motor_speeds = self.tri_rotor_mixer.calculate(cmd.thrust, Mx, My, Mz)
        thrusts = np.array([self.tri_rotor_mixer.calc_thrust_from_speed(s) for s in motor_speeds], dtype=float)
        reaction_torques = np.zeros(3, dtype=float)

        # safety thrust cap
        if int(getattr(self, 'controller_mode', 1)) == 2:
            max_total_thrust = 0.10 * robot_mass * self.gravity
        else:
            max_total_thrust = 0.25 * robot_mass * self.gravity
        actual_total = float(np.sum(thrusts))
        if actual_total > max_total_thrust and actual_total > 1e-9:
            scale = max_total_thrust / actual_total
            thrusts *= scale
            motor_speeds *= np.sqrt(scale)

        self.interface.apply_propeller_forces(thrusts, reaction_torques)

        # log
        self.last_prop_force_cmd = np.array([0.0, 0.0, float(np.sum(thrusts))], dtype=float)
        self.last_prop_torque_cmd = np.array([Mx, My, Mz], dtype=float)
        self.last_prop_thrust = thrusts
        self.last_motor_speeds = motor_speeds

        for _ in range(self.steps_per_control):
            mujoco.mj_step(self.model, self.data)
            self.sim_time += self.dt

        info['mode'] = 'leg_prop'
        info['prop_thrust'] = float(np.sum(thrusts))
        info['prop_L'] = Mx
        info['prop_M'] = My
        info['prop_N'] = Mz
        info['desired_rpy'] = desired_rpy
        state_after = self.interface.get_state()
        return state_after, leg_torque, info
    
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
    parser.add_argument('--dyn-backend', choices=['mujoco', 'pinocchio'], default='pinocchio',
                        help='mode3 full-body WBC 的动力学后端：mujoco(默认) 或 pinocchio(URDF)')
    
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
        sim.full_wbc_dynamics_backend = args.dyn_backend
        sim.record_video(
            args.record,
            fps=args.record_fps,
            width=args.record_width,
            height=args.record_height,
            duration=args.record_duration,
        )
        return

    sim = HopperSimulation(model_path=args.model, mode=sim_mode)
    sim.full_wbc_dynamics_backend = args.dyn_backend
    
    if args.teleop:
        sim.enable_teleop()

    if args.headless:
        sim.run_headless(duration=args.duration)
    else:
        sim.run_with_viewer()


if __name__ == '__main__':
    main()
