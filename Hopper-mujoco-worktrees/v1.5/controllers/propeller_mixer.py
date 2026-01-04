import numpy as np


class PropellerMixer:
    """
    Simple 4-rotor mixer placeholder (for prop mode compatibility).
    Not used for tri-rotor assist; kept to satisfy imports if prop mode is enabled.
    """

    def __init__(self, arm_length: float = 0.1):
        self.arm_length = float(arm_length)

    def allocate(self, total_force: float, roll_moment: float, pitch_moment: float, yaw_moment: float = 0.0) -> np.ndarray:
        # Minimal equal split (no real yaw). This is intentionally simple.
        f = max(0.0, float(total_force)) / 4.0
        return np.array([f, f, f, f], dtype=float)


