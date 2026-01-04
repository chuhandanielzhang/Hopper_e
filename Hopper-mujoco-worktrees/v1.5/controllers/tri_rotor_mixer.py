import numpy as np


class TriRotorMixer:
    """
    Allocate (Fz, Mx, My) to 3 rotor thrusts.
    Assumptions:
    - Each rotor thrust points along +Z body axis.
    - Rotor positions are given in body frame (meters).
    - We ignore yaw moment allocation by default (Mz).
    """

    def __init__(self, prop_positions: np.ndarray, Ct: float, Cd: float, max_speed: float):
        self.prop_positions = np.asarray(prop_positions, dtype=float).reshape(3, 3)
        self.Ct = float(Ct)
        self.Cd = float(Cd)
        self.max_speed = float(max_speed)
        self.max_thrust_per_motor = self.Ct * (self.max_speed ** 2)

        # Build allocation matrix for [Fz, Mx, My]^T = A * thrusts
        # For rotor at r=(x,y,0): torque from thrust f*z is r×F => [ y*f, -x*f, 0 ]
        A = np.zeros((3, 3), dtype=float)
        for i in range(3):
            x, y, _ = self.prop_positions[i]
            A[0, i] = 1.0
            A[1, i] = y
            A[2, i] = -x
        self.A = A
        self.A_inv = np.linalg.inv(A)

    def calculate(self, total_thrust: float, Mx: float, My: float, Mz: float = 0.0) -> np.ndarray:
        # ignore Mz (over-actuated constraint)
        u = np.array([total_thrust, Mx, My], dtype=float)
        thrusts = self.A_inv @ u

        # Saturation handling: keep non-negative and within max, scale moments if needed
        thrusts = np.maximum(thrusts, 0.0)
        max_val = float(np.max(thrusts))
        if max_val > self.max_thrust_per_motor:
            scale = self.max_thrust_per_motor / max_val
            thrusts = thrusts * scale

        # Convert to motor speeds (same convention as Ct)
        motor_speeds = np.sqrt(np.where(self.Ct > 0, thrusts / self.Ct, 0.0))
        motor_speeds = np.clip(motor_speeds, 0.0, self.max_speed)
        return motor_speeds

    def calc_thrust_from_speed(self, speed: float) -> float:
        s = float(speed)
        return self.Ct * s * s


