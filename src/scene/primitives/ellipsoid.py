from src.scene.base import SceneObject
from src.scene.transforms import resolve_rotation_matrix
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from typing import Tuple

@dataclass
class EllipsoidObject(SceneObject):
    rx: float
    ry: float
    rz: float

    def get_mesh(self):
        assert self.radii is not None, "Ellipsoid requires 'radii'."
        # Create a unit sphere, then scale it non-uniformly to create an ellipsoid
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
        mesh.compute_vertex_normals()
        
        # Scale the sphere to create an ellipsoid
        scale_matrix = np.eye(4)
        scale_matrix[0, 0] =  self.rx
        scale_matrix[1, 1] =  self.ry
        scale_matrix[2, 2] =  self.rz
        mesh.transform(scale_matrix)
        
        mesh.paint_uniform_color(np.array(self.color, dtype=float))
        R = resolve_rotation_matrix(self.rotation)
        mesh.rotate(R, center=(0.0, 0.0, 0.0))
        mesh.translate(np.array(self.position, dtype=float))
        return mesh


def parse_ellipsoid_object(item: dict) -> EllipsoidObject:
    rx = item.get('rx', 1.0)
    ry = item.get('ry', 1.0)
    rz = item.get('rz', 1.0)
    
    return EllipsoidObject(
        type="ellipsoid",
        color=tuple(item.get("color", [0.7, 0.7, 0.7])),
        position=tuple(item.get("position", [0.0, 0.0, 0.0])),
        rotation=[float(v) for v in item.get("rotation", [0.0, 0.0, 0.0])] if item.get("rotation") is not None else None,
        rx=rx,
        ry=ry,
        rz=rz
    )