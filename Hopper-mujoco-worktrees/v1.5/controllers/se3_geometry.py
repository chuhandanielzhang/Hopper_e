import numpy as np


def veemap(A: np.ndarray) -> np.ndarray:
    """vee map for so(3): takes a skew-symmetric matrix -> R^3."""
    return np.array([A[2, 1], A[0, 2], A[1, 0]], dtype=float)


def rot_from_quat_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrix R_world_from_body from quaternion (w,x,y,z)."""
    w, x, y, z = q_wxyz
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    return np.array([
        [ww + xx - yy - zz, 2*(xy - wz),       2*(xz + wy)],
        [2*(xy + wz),       ww - xx + yy - zz, 2*(yz - wx)],
        [2*(xz - wy),       2*(yz + wx),       ww - xx - yy + zz],
    ], dtype=float)


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R_world_from_body from roll-pitch-yaw (xyz intrinsic, same as scipy 'xyz')."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ], dtype=float)


