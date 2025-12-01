import sys
from pathlib import Path

# Add the project root to Python path so we can import src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime
import numpy as np
import open3d as o3d
import os
import re
from dotenv import load_dotenv

from src.scene.base import SceneObject, compute_scene_bounds, load_scene_objects, objects_from_scene_dict

def extract_json_block(text: str) -> str:
    # Try to recover a JSON object from a model response
    code_fence = re.search(r"```(?:json)?\n([\s\S]*?)```", text)
    if code_fence:
        return code_fence.group(1).strip()
    # Fallback: find first '{' and last '}'
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text.strip()


def generate_scene_with_openai(prompt: str, model: str, temperature: float, title: str) -> List[SceneObject]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("The 'openai' package is required. Please install it with 'pip install openai'.") from e
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=api_key)

    system_prompt = (
        """
        You are a tool that MUST output ONLY valid JSON. 
        NO comments. NO explanations. NO trailing text. NO deviations.

        Create a 3D scene according to the prompt, using cube, cuboid, sphere, ellipsoid, or pyramid primitives.
        Return the information of the primitives in the scene as a JSON object.
        The output MUST be a JSON object of the exact form:

        {
        "scene": [
            { "type": "cube", "length": 1.0, "color": [0.0, 0.0, 0.0], "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0] },
            { "type": "cuboid", "x_length": 1.0, "y_length": 0.5, "z_length": 2.0, "color": [0.0, 0.0, 0.0], "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0] },
            { "type": "sphere", "radius": 1.0, "color": [0.0, 0.0, 0.0], "position": [0.0, 0.0, 0.0] },
            { "type": "ellipsoid", "rx": 1.0, "ry": 1.0, "rz": 1.0, "color": [0.0, 0.0, 0.0], "position": [0.0, 0.0, 0.0] },
            { "type": "pyramid", "base_length": 1.0, "height": 1.0, "color": [0.0, 0.0, 0.0], "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0] }
        ]
        }

        This is an EXAMPLE. Replace values to create the scene.
        The scene should be created in a way that is visually appealing and realistic.
        The scene should be created in a way that is easy to understand and visualize.
        You may use as many primitives as you like. Focus on making the scene complete and realistic.

        IMPORTANT RULES:
        - Output ONLY valid JSON. 
        - DO NOT include comments.
        - DO NOT include any text before or after the JSON.
        - DO NOT include trailing commas.
        - If you include anything other than valid JSON, that is considered an error.
        """
    )

    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    content = completion.choices[0].message.content or "{}"
    os.makedirs("response", exist_ok=True)
    safe_title = re.sub(r"[^A-Za-z0-9_-]+", "_", title)
    with open(f"response/{safe_title}_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt", "w") as f:
        f.write(content)
    json_str = extract_json_block(content)
    scene_dict = json.loads(json_str)
    with open(f"configs/scene_{safe_title}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as jf:
        json.dump(scene_dict, jf, indent=4)
    return objects_from_scene_dict(scene_dict)

def render_scene(objects: List[SceneObject], title: str = "3D Scene") -> None:
    geometries: List[o3d.geometry.Geometry] = []
    bounds_accum: List[np.ndarray] = []

    for obj in objects:
        mesh = obj.get_mesh()
        geometries.append(mesh)
        # accumulate bounds
        bounds_accum.append(np.asarray(mesh.vertices))

    # Optional: center camera roughly to the scene by using bounds
    if bounds_accum:
        mins, maxs = compute_scene_bounds(bounds_accum)
        _ = (mins, maxs)  # kept in case of future camera setup

    o3d.visualization.draw_geometries(geometries, window_name=title)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a 3D scene from a JSON file.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--config", type=str, help="Path to scene JSON file")
    group.add_argument("--prompt", type=str, help="LLM prompt to generate a scene")
    parser.add_argument("--title", type=str, default="3D Scene", help="Window title")
    parser.add_argument("--model", type=str, default="gpt-4.1", help="OpenAI model for generation")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for LLM")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt:
        objects = generate_scene_with_openai(args.prompt, model=args.model, temperature=args.temperature, title=args.title)
    elif args.config:
        objects = load_scene_objects(args.config)
    else:
        raise SystemExit("Provide either --config <file> or --prompt \"text\"")
    render_scene(objects, title=args.title)


if __name__ == '__main__':
    main()