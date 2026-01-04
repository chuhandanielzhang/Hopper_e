"""
High-level SE3/SO3-style attitude controller (adapted from Quadrotor_SE3_Control-main).

For Hopper Mujoco tri-rotor assist:
- We mainly use the SO(3) attitude error formulation:
    e_R = 0.5 * vee(R_d^T R - R^T R_d)
    e_w = w - R^T R_d w_d
    M = -kR * e_R - kw * e_w + feedforward

- Thrust is provided externally (e.g., base_thrust_ratio * m*g), but we optionally compensate for tilt.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .se3_geometry import veemap, rot_from_quat_wxyz, rot_from_rpy


@dataclass
class State:
    """Quaternion uses MuJoCo convention: (w,x,y,z). omega is body-frame rad/s."""
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray  # (w,x,y,z)
    omega: np.ndarray       # body frame


@dataclass
class ControlCommand:
    thrust: float                 # Newton
    moment: np.ndarray            # body moments (Mx, My, Mz) in Nm
    desired_rpy: np.ndarray       # (roll,pitch,yaw) used to build R_d (for sharing with leg)


class SE3Controller:
    def __init__(self):
        # gains (tuned for assist)
        self.kR = 30.0
        self.kw = 5.0

        # tilt compensation on thrust (keep vertical component ~ const)
        self.enable_tilt_comp = True

    def attitude_control(
        self,
        current_quat_wxyz: np.ndarray,
        current_omega_body: np.ndarray,
        desired_rpy: np.ndarray,
        thrust_newton: float,
        desired_omega_body: np.ndarray | None = None,
    ) -> ControlCommand:
        desired_rpy = np.asarray(desired_rpy, dtype=float).reshape(3)
        w_d = np.zeros(3, dtype=float) if desired_omega_body is None else np.asarray(desired_omega_body, dtype=float).reshape(3)

        R = rot_from_quat_wxyz(np.asarray(current_quat_wxyz, dtype=float).reshape(4))    # world_from_body
        R_d = rot_from_rpy(desired_rpy[0], desired_rpy[1], desired_rpy[2])

        # so3 errors (body frame)
        e_R = 0.5 * veemap(R_d.T @ R - R.T @ R_d)
        e_w = np.asarray(current_omega_body, dtype=float).reshape(3) - (R.T @ R_d @ w_d)

        M = -self.kR * e_R - self.kw * e_w

        T = float(max(0.0, thrust_newton))
        if self.enable_tilt_comp:
            # compensate so that world vertical component stays close to desired
            zb = R @ np.array([0.0, 0.0, 1.0])
            cos_tilt = float(np.clip(zb[2], 0.2, 1.0))
            T = T / cos_tilt

        return ControlCommand(thrust=T, moment=M, desired_rpy=desired_rpy)


