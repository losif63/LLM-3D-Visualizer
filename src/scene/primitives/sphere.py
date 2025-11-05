from src.scene.base import SceneObject
from src.scene.transforms import resolve_rotation_matrix
import numpy as np
import open3d as o3d
from dataclasses import dataclass

@dataclass
class SphereObject(SceneObject):
    radius: float

    def get_mesh(self):
        assert self.radius is not None, "Sphere requires 'radius'."
        # Use Open3D's sphere primitive, then apply our transform
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=float(self.radius), resolution=40)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(np.array(self.color, dtype=float))
        R = resolve_rotation_matrix(self.rotation)
        mesh.rotate(R, center=(0.0, 0.0, 0.0))
        mesh.translate(np.array(self.position, dtype=float))
        return mesh


def parse_sphere_object(item: dict) -> SphereObject:
    return SphereObject(
        type="sphere",
        color=tuple(item.get("color", [0.7, 0.7, 0.7])),
        position=tuple(item.get("position", [0.0, 0.0, 0.0])),
        rotation=[float(v) for v in item.get("rotation", [0.0, 0.0, 0.0, 0.0])],
        radius=float(item.get("radius", 1.0)),
    )