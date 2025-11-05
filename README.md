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

The viewer uses Open3D and supports interactive orbit, pan, and zoom with the mouse.

### LLM-powered scene generation (OpenAI)

1) Set your API key:

```bash
export OPENAI_API_KEY=sk-...   # or use a .env/your shell profile
```

2) Generate from a prompt:

```bash
python src/main.py --prompt "three blue cubes in a row and a red sphere above" --model gpt-4o-mini --title "LLM Scene"
```

Notes:
- The model will output JSON matching the same schema as `configs/example1.json`.
- You can adjust `--temperature` (default 0.2) to control randomness.

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
