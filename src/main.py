

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


Color = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]


@dataclass
class SceneObject:
    type: str
    color: Color
    position: Vec3
    rotation: Optional[List[float]]
    # Specific params
    length: Optional[float] = None
    radius: Optional[float] = None


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


def create_cube_vertices(length: float) -> np.ndarray:
    half = length / 2.0
    # 8 vertices of a cube centered at origin
    return np.array([
        [-half, -half, -half],
        [half, -half, -half],
        [half, half, -half],
        [-half, half, -half],
        [-half, -half, half],
        [half, -half, half],
        [half, half, half],
        [-half, half, half],
    ])


def cube_faces_indices() -> List[List[int]]:
    # Each face is a quad of 4 vertex indices
    return [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [3, 0, 4, 7],  # left
    ]


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def add_cube(ax, obj: SceneObject, bounds_accum: List[np.ndarray]):
    if obj.length is None:
        raise ValueError("Cube requires 'length'.")
    vertices = create_cube_vertices(obj.length)
    R = resolve_rotation_matrix(obj.rotation)
    t = np.array(obj.position, dtype=float)
    vertices_world = apply_transform(vertices, R, t)

    faces = cube_faces_indices()
    face_polys = [[vertices_world[idx] for idx in face] for face in faces]

    poly = Poly3DCollection(face_polys, alpha=0.9)
    poly.set_facecolor(obj.color)
    poly.set_edgecolor((0.1, 0.1, 0.1))
    ax.add_collection3d(poly)

    bounds_accum.append(vertices_world)


def add_sphere(ax, obj: SceneObject, bounds_accum: List[np.ndarray]):
    if obj.radius is None:
        raise ValueError("Sphere requires 'radius'.")
    # Parametric sphere at origin
    u = np.linspace(0.0, 2.0 * math.pi, 48)
    v = np.linspace(0.0, math.pi, 24)
    uu, vv = np.meshgrid(u, v)
    x = obj.radius * np.cos(uu) * np.sin(vv)
    y = obj.radius * np.sin(uu) * np.sin(vv)
    z = obj.radius * np.cos(vv)
    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    R = resolve_rotation_matrix(obj.rotation)
    t = np.array(obj.position, dtype=float)
    pts_world = apply_transform(pts, R, t)
    xw = pts_world[:, 0].reshape(x.shape)
    yw = pts_world[:, 1].reshape(y.shape)
    zw = pts_world[:, 2].reshape(z.shape)

    ax.plot_surface(xw, yw, zw, color=obj.color, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True)

    bounds_accum.append(pts_world)


def set_axes_equal(ax):
    # Equal aspect ratio for 3D
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def compute_scene_bounds(bounds_accum: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    all_pts = np.concatenate(bounds_accum, axis=0) if bounds_accum else np.zeros((1, 3))
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    return mins, maxs


def load_scene_objects(json_path: str) -> List[SceneObject]:
    with open(json_path, "r") as f:
        data = json.load(f)
    scene_list = data.get("scene", [])
    objects: List[SceneObject] = []
    for item in scene_list:
        obj_type = str(item.get("type", "")).lower()
        color = tuple(item.get("color", [0.7, 0.7, 0.7]))  # default gray
        position = tuple(item.get("position", [0.0, 0.0, 0.0]))
        rotation = item.get("rotation")

        scene_obj = SceneObject(
            type=obj_type,
            color=(float(color[0]), float(color[1]), float(color[2])),
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=[float(v) for v in rotation] if rotation is not None else None,
            length=float(item.get("length")) if "length" in item else None,
            radius=float(item.get("radius")) if "radius" in item else None,
        )
        objects.append(scene_obj)
    return objects


def render_scene(objects: List[SceneObject], title: str = "3D Scene") -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    bounds_accum: List[np.ndarray] = []

    for obj in objects:
        if obj.type == "cube":
            add_cube(ax, obj, bounds_accum)
        elif obj.type == "sphere":
            add_sphere(ax, obj, bounds_accum)
        else:
            raise ValueError(f"Unsupported object type: {obj.type}")

    mins, maxs = compute_scene_bounds(bounds_accum)
    center = 0.5 * (mins + maxs)
    extent = max(1.0, float(np.max(maxs - mins)))
    pad = 0.2 * extent
    ax.set_xlim(center[0] - extent - pad, center[0] + extent + pad)
    ax.set_ylim(center[1] - extent - pad, center[1] + extent + pad)
    ax.set_zlim(center[2] - extent - pad, center[2] + extent + pad)
    set_axes_equal(ax)

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a 3D scene from a JSON file.")
    parser.add_argument("config", type=str, help="Path to scene JSON file")
    parser.add_argument("--title", type=str, default="3D Scene", help="Window title")
    return parser.parse_args()


def main():
    args = parse_args()
    objects = load_scene_objects(args.config)
    render_scene(objects, title=args.title)


if __name__ == '__main__':
    main()