# location: /README.md

# AnyGAN

AnyGAN is a modular Streamlit playground for real AI image generation. It now uses Stable Diffusion through Hugging Face Diffusers instead of fake/random GAN placeholders.

## Features

- Modern Streamlit UI with playful styling
- Stable Diffusion-only model selection
- Optional Hugging Face model ID or URL input
- Prompt and negative prompt controls
- Context-aware prompt enhancement
- Style presets: Photorealistic, Cyberpunk, Fantasy, Anime
- Deterministic generation with `torch.Generator.manual_seed(seed)`
- Guidance scale control from 7.5 to 9.0
- Inference step control from 30 to 40
- Single-image generation mode
- Side-by-side comparison mode for two prompts or two seeds
- Cached model loading with `st.cache_resource`
- CPU/GPU auto-selection through `.env`
- Hugging Face token support
- Basic Python logging
- Local output saving to `experiments/`
- Fun fact and balloons after successful generation

## Project Structure

```text
AnyGAN/
  app.py
  assets/
    styles.css
  models/
    base_model.py
    diffusion_model.py
  utils/
    config.py
    fun_facts.py
    helpers.py
    model_loader.py
    prompt_engine.py
    ui_components.py
  tests/
  experiments/
  requirements.txt
  .env.example
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

On macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

## Configuration

Useful `.env` values:

```text
HF_TOKEN=
DEVICE=auto
DEFAULT_DIFFUSION_MODEL=runwayml/stable-diffusion-v1-5
ALLOW_EXTERNAL_MODELS=True
DIFFUSION_STEPS=35
SAVE_OUTPUTS=True
OUTPUT_DIR=experiments/
LOG_LEVEL=INFO
```

`DEVICE=auto` uses CUDA when available and CPU otherwise.

## Hugging Face Models

The sidebar accepts either a repo ID:

```text
runwayml/stable-diffusion-v1-5
```

or a full Hugging Face URL:

```text
https://huggingface.co/runwayml/stable-diffusion-v1-5
```

The model must be compatible with `StableDiffusionPipeline`.

## Testing

```bash
pytest -q -p no:cacheprovider
coverage run --source=models,utils -m pytest -q -p no:cacheprovider
coverage report -m
```

Tests mock the heavy Stable Diffusion pipeline so validation does not download model weights.
