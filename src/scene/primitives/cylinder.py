from src.scene.base import SceneObject
from src.scene.transforms import resolve_rotation_matrix
import numpy as np
import open3d as o3d
from dataclasses import dataclass

@dataclass
class CylinderObject(SceneObject):
    radius: float
    height: float

    def get_mesh(self):
        assert self.radius is not None, "Cylinder requires 'radius'."
        assert self.height is not None, "Cylinder requires 'height'."
        # Use Open3D's cylinder primitive (creates along Z-axis by default)
        # Rotate 90 degrees around X-axis to make it stand up along Y-axis
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=float(self.radius), height=float(self.height), resolution=40, split=1)
        mesh.compute_vertex_normals()
        
        # Default rotation: 90 degrees around X-axis to make cylinder stand up (Z -> Y)
        default_rotation = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0]
        ])
        mesh.rotate(default_rotation, center=(0.0, 0.0, 0.0))
        
        mesh.paint_uniform_color(np.array(self.color, dtype=float))
        
        # Apply user-specified rotation on top of default orientation
        if self.rotation is not None:
            R = resolve_rotation_matrix(self.rotation)
            mesh.rotate(R, center=(0.0, 0.0, 0.0))
        
        mesh.translate(np.array(self.position, dtype=float))
        return mesh


def parse_cylinder_object(item: dict) -> CylinderObject:
    return CylinderObject(
        type="cylinder",
        color=tuple(item.get("color", [0.7, 0.7, 0.7])),
        position=tuple(item.get("position", [0.0, 0.0, 0.0])),
        rotation=[float(v) for v in item.get("rotation", [0.0, 0.0, 0.0])] if item.get("rotation") is not None else None,
        radius=float(item.get("radius", 1.0)),
        height=float(item.get("height", 1.0)),
    )

