

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d


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
        [0, 3, 2, 1],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [1, 2, 6, 5],  # right
        [3, 0, 4, 7],  # left
    ]


def apply_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def create_cube_mesh(obj: SceneObject) -> o3d.geometry.TriangleMesh:
    if obj.length is None:
        raise ValueError("Cube requires 'length'.")
    # Build cube as a mesh from our vertices and faces
    vertices_local = create_cube_vertices(obj.length)
    R = resolve_rotation_matrix(obj.rotation)
    t = np.array(obj.position, dtype=float)
    vertices_world = apply_transform(vertices_local, R, t)

    # Triangulate each quad face into two triangles
    quads = cube_faces_indices()
    triangles = []
    for a, b, c, d in quads:
        triangles.append([a, b, c])
        triangles.append([a, c, d])

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices_world.astype(float)),
        triangles=o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32)),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(obj.color, dtype=float))
    return mesh


def create_sphere_mesh(obj: SceneObject) -> o3d.geometry.TriangleMesh:
    if obj.radius is None:
        raise ValueError("Sphere requires 'radius'.")
    # Use Open3D's sphere primitive, then apply our transform
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(obj.radius), resolution=40)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(np.array(obj.color, dtype=float))
    R = resolve_rotation_matrix(obj.rotation)
    mesh.rotate(R, center=(0.0, 0.0, 0.0))
    mesh.translate(np.array(obj.position, dtype=float))
    return mesh


def set_axes_equal(*args, **kwargs):
    # No-op placeholder retained for API compatibility
    return None


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
    geometries: List[o3d.geometry.Geometry] = []
    bounds_accum: List[np.ndarray] = []

    for obj in objects:
        if obj.type == "cube":
            mesh = create_cube_mesh(obj)
            geometries.append(mesh)
            # accumulate bounds
            bounds_accum.append(np.asarray(mesh.vertices))
        elif obj.type == "sphere":
            mesh = create_sphere_mesh(obj)
            geometries.append(mesh)
            bounds_accum.append(np.asarray(mesh.vertices))
        else:
            raise ValueError(f"Unsupported object type: {obj.type}")

    # Optional: center camera roughly to the scene by using bounds
    if bounds_accum:
        mins, maxs = compute_scene_bounds(bounds_accum)
        _ = (mins, maxs)  # kept in case of future camera setup

    o3d.visualization.draw_geometries(geometries, window_name=title)


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