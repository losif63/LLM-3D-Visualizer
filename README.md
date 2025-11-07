# LLM-3D-Visualizer

## Setup

```bash
conda create -n llmviz python=3.12
pip install -r requirements.txt
```

## Loading pre-generated scenes

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
python src/app/main.py --prompt "three blue cubes in a row and a red sphere above" --title "LLM Scene"
```

Notes:
- The model will output JSON matching the same schema as `configs/example1.json`.
- You can adjust `--temperature` (default 0.2) to control randomness.
