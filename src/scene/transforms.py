import numpy as np
import math
from typing import List, Optional

def rotation_matrix_from_euler(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    # Rotation order Z (rz) then Y (ry) then X (rx): R = Rz * Ry * Rx
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return Rz @ Ry @ Rx


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = axis.astype(float)
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    theta = math.radians(angle_deg)
    c = math.cos(theta)
    s = math.sin(theta)
    v = 1.0 - c
    return np.array([
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ])


def resolve_rotation_matrix(rotation: Optional[List[float]]) -> np.ndarray:
    if not rotation:
        return np.eye(3)
    if len(rotation) == 3:
        return rotation_matrix_from_euler(rotation[0], rotation[1], rotation[2])
    if len(rotation) == 4:
        axis = np.array(rotation[:3], dtype=float)
        angle = float(rotation[3])
        return rotation_matrix_from_axis_angle(axis, angle)
    return np.eye(3)


def scale_matrix(x, y, z) -> np.ndarray:
    return np.diag([x, y, z])


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t