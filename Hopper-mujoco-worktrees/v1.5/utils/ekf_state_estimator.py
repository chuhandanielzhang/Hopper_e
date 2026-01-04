"""
High-level full EKF (ESKF) for Hopper: fuse IMU + leg kinematics (contact constraint).

Inputs (minimal, matches your requirement):
- IMU gyro (rad/s) in body frame
- IMU accel (specific force, m/s^2) in body frame
- Foot position/velocity in body frame (use MuJoCo-consistent ones: foot_pos_mj, foot_vel_mj_rel)
- A stance indicator (contact/phase)

Outputs:
- Estimated quaternion (w,x,y,z)
- Estimated velocity in world frame (m/s)
- (Also keeps position & biases internally)

This is an error-state EKF (15-state error):
  x_nom = [p, v, q, bg, ba]
  δx    = [δp, δv, δθ, δbg, δba]  (15x1)

Propagation (nominal):
  q_{k+1} = q_k ⊗ Exp((ω - bg) dt)
  a_w     = R(q) (f - ba) + g
  v_{k+1} = v_k + a_w dt
  p_{k+1} = p_k + v_k dt + 0.5 a_w dt^2
  bg, ba are random walk

Contact update (stance):
  foot contact constraint in body frame:
    v_base_body ≈ -(v_foot_rel_body + ω_body × r_foot_body)
  measurement model:
    z = R^T v_world   (body-frame base velocity)
  so it couples attitude and velocity during update.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _skew(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0],
    ], dtype=float)


def _quat_norm(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def _so3_exp(phi: np.ndarray) -> np.ndarray:
    """Exponential map so(3)->quat (wxyz) using rotation vector phi (rad)."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-12:
        return np.array([1.0, 0.5*phi[0], 0.5*phi[1], 0.5*phi[2]], dtype=float)
    axis = phi / angle
    half = 0.5 * angle
    return _quat_norm(np.array([np.cos(half), *(np.sin(half) * axis)], dtype=float))


def _rot_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
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
class EKFConfig:
    g: float = 9.81

    # Continuous-time noise densities (rough, tune as needed)
    sigma_gyro: float = 0.10         # rad/s / sqrt(Hz)
    sigma_acc: float = 1.00          # m/s^2 / sqrt(Hz)
    sigma_bg: float = 0.01           # rad/s^2 / sqrt(Hz)
    sigma_ba: float = 0.10           # m/s^3 / sqrt(Hz)

    # Contact velocity measurement noise (body frame)
    sigma_v_contact: float = 0.25    # m/s

    # Gate contact update if residual too large
    max_contact_residual: float = 3.0  # m/s


class HopperESKF:
    def __init__(self, cfg: EKFConfig | None = None):
        self.cfg = cfg or EKFConfig()
        self.reset()

    def reset(
        self,
        p_world: np.ndarray | None = None,
        v_world: np.ndarray | None = None,
        q_wxyz: np.ndarray | None = None,
    ):
        self.p = np.zeros(3, dtype=float) if p_world is None else np.asarray(p_world, dtype=float).reshape(3).copy()
        self.v = np.zeros(3, dtype=float) if v_world is None else np.asarray(v_world, dtype=float).reshape(3).copy()
        self.q = _quat_norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q_wxyz is None else q_wxyz)
        self.bg = np.zeros(3, dtype=float)
        self.ba = np.zeros(3, dtype=float)

        # covariance of error-state δx=[δp,δv,δθ,δbg,δba]
        self.P = np.eye(15, dtype=float) * 0.1

    def predict(self, dt: float, gyro_body: np.ndarray, acc_body: np.ndarray):
        dt = float(dt)
        if dt <= 0:
            return

        w_m = np.asarray(gyro_body, dtype=float).reshape(3)
        f_m = np.asarray(acc_body, dtype=float).reshape(3)  # specific force

        w = w_m - self.bg
        f = f_m - self.ba

        # nominal propagation
        dq = _so3_exp(w * dt)
        self.q = _quat_norm(_quat_mul(self.q, dq))
        R = _rot_from_quat_wxyz(self.q)  # world_from_body
        g_w = np.array([0.0, 0.0, -self.cfg.g], dtype=float)
        a_w = R @ f + g_w

        self.p = self.p + self.v * dt + 0.5 * a_w * (dt * dt)
        self.v = self.v + a_w * dt

        # covariance propagation (discrete approx)
        F = np.zeros((15, 15), dtype=float)
        F[0:3, 3:6] = np.eye(3)  # δp_dot = δv
        F[3:6, 6:9] = -R @ _skew(f)  # δv_dot depends on δθ
        F[3:6, 12:15] = -R          # δv_dot depends on δba
        F[6:9, 6:9] = -_skew(w)     # δθ_dot
        F[6:9, 9:12] = -np.eye(3)   # δθ_dot depends on δbg

        # noise Jacobian
        G = np.zeros((15, 12), dtype=float)
        # accel noise -> velocity
        G[3:6, 0:3] = R
        # gyro noise -> attitude
        G[6:9, 3:6] = np.eye(3)
        # bg random walk
        G[9:12, 6:9] = np.eye(3)
        # ba random walk
        G[12:15, 9:12] = np.eye(3)

        sa2 = self.cfg.sigma_acc ** 2
        sg2 = self.cfg.sigma_gyro ** 2
        sbg2 = self.cfg.sigma_bg ** 2
        sba2 = self.cfg.sigma_ba ** 2
        Qc = np.diag([sa2, sa2, sa2, sg2, sg2, sg2, sbg2, sbg2, sbg2, sba2, sba2, sba2]).astype(float)

        Phi = np.eye(15) + F * dt
        Qd = (G @ Qc @ G.T) * dt
        self.P = Phi @ self.P @ Phi.T + Qd

    def update_contact_velocity(
        self,
        foot_pos_body: np.ndarray,
        foot_vel_rel_body: np.ndarray,
        gyro_body: np.ndarray,
    ) -> bool:
        """
        Contact measurement update using body-frame base velocity pseudo-measurement.

        z_b = v_base_body ≈ -(v_rel_body + ω×r)
        h(x) = R^T v_world
        """
        r_b = np.asarray(foot_pos_body, dtype=float).reshape(3)
        vrel_b = np.asarray(foot_vel_rel_body, dtype=float).reshape(3)
        w_m = np.asarray(gyro_body, dtype=float).reshape(3)
        w = w_m - self.bg

        z = -(vrel_b + np.cross(w, r_b))  # body-frame base velocity pseudo measurement

        R = _rot_from_quat_wxyz(self.q)  # world_from_body
        v_b_pred = R.T @ self.v

        y = z - v_b_pred
        if float(np.linalg.norm(y)) > float(self.cfg.max_contact_residual):
            return False

        # H: 3x15
        H = np.zeros((3, 15), dtype=float)
        H[:, 3:6] = R.T
        # δ(R^T v) ≈ -skew(R^T v) δθ
        H[:, 6:9] = -_skew(v_b_pred)

        Rm = np.eye(3, dtype=float) * (self.cfg.sigma_v_contact ** 2)

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y

        # inject
        self.p += dx[0:3]
        self.v += dx[3:6]
        dtheta = dx[6:9]
        self.q = _quat_norm(_quat_mul(self.q, _so3_exp(dtheta)))
        self.bg += dx[9:12]
        self.ba += dx[12:15]

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T
        return True

    def step(
        self,
        dt: float,
        gyro_body: np.ndarray,
        acc_body: np.ndarray,
        foot_pos_body: np.ndarray,
        foot_vel_rel_body: np.ndarray,
        in_stance: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.predict(dt, gyro_body, acc_body)
        if in_stance:
            self.update_contact_velocity(foot_pos_body, foot_vel_rel_body, gyro_body)
        return self.q.copy(), self.v.copy()


