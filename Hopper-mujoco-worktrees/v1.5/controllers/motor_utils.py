import numpy as np


class MotorModel:
    """Simple motor model: thrust = Ct * omega^2, reaction torque = Cd * omega^2."""

    def __init__(self, ct: float, cd: float, max_speed: float):
        self.ct = float(ct)
        self.cd = float(cd)
        self.max_speed = float(max_speed)  # krpm or rad/s depending on upstream convention

    def clamp_speed(self, motor_speeds: np.ndarray) -> np.ndarray:
        s = np.asarray(motor_speeds, dtype=float)
        return np.clip(s, 0.0, self.max_speed)

    def thrusts_from_speeds(self, motor_speeds: np.ndarray) -> np.ndarray:
        s = np.asarray(motor_speeds, dtype=float)
        return self.ct * s * s

    def torques_from_speeds(self, motor_speeds: np.ndarray) -> np.ndarray:
        s = np.asarray(motor_speeds, dtype=float)
        return self.cd * s * s


