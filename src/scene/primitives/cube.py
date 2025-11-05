import numpy as np
from typing import List
from src.scene.transforms import resolve_rotation_matrix, apply_transform
from src.scene.base import SceneObject
import open3d as o3d
from src.scene.base import Color, Vec3
from typing import Optional
from dataclasses import dataclass

@dataclass
class CubeObject(SceneObject):
    length: float

    def get_vertices(self) -> np.ndarray:
        half = self.length / 2.0
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

    def get_indices(self) -> np.ndarray:
        # Each face is a quad of 4 vertex indices
        return [
            [0, 3, 2, 1],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [1, 2, 6, 5],  # right
            [3, 0, 4, 7],  # left
        ]

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        assert self.length is not None, "Cube requires 'length'."
        # Build cube as a mesh from our vertices and faces
        vertices = self.get_vertices()
        R = resolve_rotation_matrix(self.rotation)
        t = np.array(self.position, dtype=float)
        vertices_world = apply_transform(vertices, R, t)

        # Triangulate each quad face into two triangles
        quads = self.get_indices()
        triangles = []
        for a, b, c, d in quads:
            triangles.append([a, b, c])
            triangles.append([a, c, d])

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices_world.astype(float)),
            triangles=o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32)),
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array(self.color, dtype=float))
        return mesh


def parse_cube_object(item: dict) -> CubeObject:
    return CubeObject(
        type="cube",
        color=tuple(item.get("color", [0.7, 0.7, 0.7])),
        position=tuple(item.get("position", [0.0, 0.0, 0.0])),
        rotation=[float(v) for v in item.get("rotation", [0.0, 0.0, 0.0, 0.0])],
        length=float(item.get("length", 1.0)),
    )
