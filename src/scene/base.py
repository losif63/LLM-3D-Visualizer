from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import json

Color = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]

@dataclass
class SceneObject:
    type: str
    color: Color
    position: Vec3
    rotation: Optional[List[float]]

    def get_indices(self):
        raise NotImplementedError("Function get_indices not implemented for base SceneObject class")
    
    
    def get_vertices(self):
        raise NotImplementedError("Function get_vertices not implemented for base SceneObject class")
    
    
    def get_mesh(self):
        raise NotImplementedError("Function get_mesh not implemented for base SceneObject class")


def compute_scene_bounds(bounds_accum: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    all_pts = np.concatenate(bounds_accum, axis=0) if bounds_accum else np.zeros((1, 3))
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    return mins, maxs


def load_scene_objects(json_path: str) -> List[SceneObject]:
    with open(json_path, "r") as f:
        scene_dict = json.load(f)
    return objects_from_scene_dict(scene_dict)


def objects_from_scene_dict(scene_dict: dict) -> List[SceneObject]:
    # Import here to avoid circular import (primitives import SceneObject from base)
    from src.scene.primitives import parse_cube_object, parse_sphere_object, parse_cuboid_object, parse_ellipsoid_object
    
    scene_list = scene_dict.get("scene", [])
    objects: List[SceneObject] = []
    for item in scene_list:
        match str(item.get("type", "")).lower():
            case "cube":
                objects.append(parse_cube_object(item))
            case "sphere":
                objects.append(parse_sphere_object(item))
            case "cuboid":
                objects.append(parse_cuboid_object(item))
            case "ellipsoid":
                objects.append(parse_ellipsoid_object(item))
            case _:
                raise ValueError(f"Primitive {str(item.get("type", "")).lower()} does not match any of the provided options.")
    return objects