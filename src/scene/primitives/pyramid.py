import numpy as np
from typing import List
from src.scene.transforms import resolve_rotation_matrix, apply_transform
from src.scene.base import SceneObject
import open3d as o3d
from src.scene.base import Color, Vec3
from typing import Optional
from dataclasses import dataclass

@dataclass
class PyramidObject(SceneObject):
    base_length: float
    height: float

    def get_vertices(self) -> np.ndarray:
        return np.array([
            [self.base_length/2, 0, self.base_length/2],
            [-self.base_length/2, 0, self.base_length/2],
            [-self.base_length/2, 0, -self.base_length/2],
            [self.base_length/2, 0, -self.base_length/2],
            [0, self.height, 0],
        ])

    def get_indices(self) -> np.ndarray:
        return np.array([
            [0, 1, 2],
            [0, 2, 3],
            [4, 1, 0],
            [4, 2, 1],
            [4, 3, 2],
            [4, 0, 3],
        ])

    def get_mesh(self) -> o3d.geometry.TriangleMesh:
        assert self.base_length is not None, "Pyramid requires 'base_length'."
        assert self.height is not None, "Pyramid requires 'height'."
        # Build pyramid as a mesh from our vertices and faces
        vertices = self.get_vertices()
        R = resolve_rotation_matrix(self.rotation)
        t = np.array(self.position, dtype=float)
        vertices_world = apply_transform(vertices, R, t)

        # Triangulate each quad face into two triangles
        quads = self.get_indices()
        triangles = [] # each face is a triangle
        for a, b, c in quads:
            triangles.append([a, b, c])

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices_world.astype(float)),
            triangles=o3d.utility.Vector3iVector(np.array(triangles, dtype=np.int32)),
        )
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array(self.color, dtype=float))
        return mesh


def parse_pyramid_object(item: dict) -> PyramidObject:
    return PyramidObject(
        type="pyramid",
        color=tuple(item.get("color", [0.7, 0.7, 0.7])),
        position=tuple(item.get("position", [0.0, 0.0, 0.0])),
        rotation=[float(v) for v in item.get("rotation", [0.0, 0.0, 0.0, 0.0])],
        base_length=float(item.get("base_length", 1.0)),
        height=float(item.get("height", 1.0)),
    )
