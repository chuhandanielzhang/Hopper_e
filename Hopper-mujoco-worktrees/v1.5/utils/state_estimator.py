"""
High-level state estimator for Hopper (IMU-only orientation + contact-aided velocity).

Goal:
- Inputs: IMU gyro (rad/s), IMU accel (m/s^2, specific force), leg/foot kinematics (foot_pos/foot_vel in body),
          and a stance indicator.
- Outputs: orientation quaternion (w,x,y,z) and horizontal velocity estimate in world frame.

Design (high-level):
1) Orientation: integrate gyro -> quaternion, correct roll/pitch drift using accelerometer gravity direction
   (Mahony-style complementary filter). Yaw is gyro-integrated (will drift without magnetometer).
2) Velocity: propagate by integrating world-frame linear acceleration.
   During stance, fuse a pseudo-measurement from the contact constraint:
     v_foot_world ≈ 0  =>  v_base_world ≈ -(R*v_foot_body + ω_world × (R*r_foot_body))
   This is essentially a simple contact-aided inertial odometry update.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # (w,x,y,z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def _quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_norm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    qn = q / n
    # keep scalar part positive for continuity
    if qn[0] < 0:
        qn = -qn
    return qn


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    # Returns R_world_from_body, q=(w,x,y,z)
    w, x, y, z = q
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ], dtype=float)


@dataclass
class EstimatorConfig:
    # Orientation correction (Mahony-style)
    ori_kp: float = 1.5
    ori_ki: float = 0.05
    accel_min_norm: float = 2.0     # m/s^2, reject near-zero
    accel_max_norm: float = 20.0    # m/s^2, reject heavy impacts

    # Velocity fusion
    vel_accel_alpha: float = 1.0    # 1.0 = full accel integration; <1 adds strong damping
    contact_vel_beta: float = 0.15  # stance correction strength (0..1)


class HopperStateEstimator:
    """
    High-level estimator with a minimal, clean API.

    update(...) returns:
      - quat_wxyz: np.ndarray shape (4,)
      - vel_world: np.ndarray shape (3,) (z is kept but you can ignore)
    """

    def __init__(self, cfg: EstimatorConfig | None = None):
        self.cfg = cfg or EstimatorConfig()
        self.reset()

    def reset(self, quat_wxyz: np.ndarray | None = None, vel_world: np.ndarray | None = None):
        self.q = _quat_norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if quat_wxyz is None else quat_wxyz)
        self.v_w = np.zeros(3, dtype=float) if vel_world is None else np.array(vel_world, dtype=float).copy()
        self._gyro_bias = np.zeros(3, dtype=float)
        self._err_int = np.zeros(3, dtype=float)

    def update(
        self,
        dt: float,
        imu_gyro_body: np.ndarray,
        imu_acc_body: np.ndarray,
        foot_pos_body: np.ndarray,
        foot_vel_body: np.ndarray,
        in_stance: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            dt: timestep (s)
            imu_gyro_body: body-frame angular rate (rad/s)
            imu_acc_body: body-frame specific force (m/s^2). At rest should be ~[0,0,9.81] if +Z up.
            foot_pos_body: foot position in body frame (m)
            foot_vel_body: foot velocity in body frame (m/s)
            in_stance: True if foot is in contact / stance phase
        """
        dt = float(dt)
        if dt <= 0:
            return self.q.copy(), self.v_w.copy()

        w_b = np.asarray(imu_gyro_body, dtype=float).reshape(3)
        a_b = np.asarray(imu_acc_body, dtype=float).reshape(3)

        # ---------- Orientation: gyro integration + accel correction (roll/pitch) ----------
        # Expected gravity direction in body frame given current orientation:
        # g_world_dir = [0,0,1] (up), so g_body_dir = R^T * g_world_dir
        # (using +Z up convention for gravity direction)
        R_wb = _rot_from_quat(self.q)          # world_from_body
        g_b_est = R_wb.T @ np.array([0.0, 0.0, 1.0])

        a_norm = float(np.linalg.norm(a_b))
        use_acc = self.cfg.accel_min_norm <= a_norm <= self.cfg.accel_max_norm
        if use_acc:
            a_b_dir = a_b / a_norm
            # error is cross between estimated gravity dir and measured gravity dir
            # If accel measures +g when upright, then a_b_dir ~ g_b_true
            err = np.cross(g_b_est, a_b_dir)
            self._err_int += err * dt
            w_b_corr = w_b + self.cfg.ori_kp * err + self.cfg.ori_ki * self._err_int - self._gyro_bias
        else:
            w_b_corr = w_b - self._gyro_bias

        # quaternion kinematics: q_dot = 0.5 * q ⊗ [0, ω]
        omega_quat = np.array([0.0, w_b_corr[0], w_b_corr[1], w_b_corr[2]], dtype=float)
        q_dot = 0.5 * _quat_mul(self.q, omega_quat)
        self.q = _quat_norm(self.q + q_dot * dt)

        # ---------- Velocity: accel integration + stance pseudo-measurement ----------
        R_wb = _rot_from_quat(self.q)
        a_w = R_wb @ a_b  # specific force in world
        # Convert specific force to linear acceleration: a_lin = a_w - g_world_specific
        # where g_world_specific = [0,0,9.81] if +Z up and specific force includes +g at rest.
        a_lin_w = a_w - np.array([0.0, 0.0, 9.81])

        self.v_w = self.v_w + (self.cfg.vel_accel_alpha * a_lin_w) * dt

        if in_stance:
            r_b = np.asarray(foot_pos_body, dtype=float).reshape(3)
            vfb_b = np.asarray(foot_vel_body, dtype=float).reshape(3)

            w_w = R_wb @ w_b_corr
            r_w = R_wb @ r_b
            vfb_w = R_wb @ vfb_b

            # contact constraint: v_foot_world ≈ v_base_world + ω×r + v_foot_rel_world ≈ 0
            v_base_meas = -(vfb_w + np.cross(w_w, r_w))

            beta = float(np.clip(self.cfg.contact_vel_beta, 0.0, 1.0))
            self.v_w = (1.0 - beta) * self.v_w + beta * v_base_meas

        return self.q.copy(), self.v_w.copy()


