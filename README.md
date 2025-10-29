# LLM-3D-Visualizer

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python src/main.py configs/example1.json --title "Example Scene"
```

## JSON Format

```json
{
  "scene": [
    {
      "type": "cube",
      "length": 2.0,
      "color": [0.2, 0.4, 0.8],
      "position": [0, 0, 0],
      "rotation": [0, 0, 0]           // Euler degrees [rx, ry, rz]
      // or axis-angle: [ax, ay, az, angle_deg]
    },
    {
      "type": "sphere",
      "radius": 1.0,
      "color": [0.9, 0.2, 0.2],
      "position": [3, 0, 0],
      "rotation": [0, 0, 0]
    }
  ]
}
```

Notes:
- `rotation` is optional; defaults to no rotation.
- `rotation` supports 3-length Euler degrees [rx, ry, rz] or 4-length axis-angle [ax, ay, az, angle_deg].
