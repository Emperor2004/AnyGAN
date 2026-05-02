# location: /README.md

# AnyGAN

AnyGAN is a modular Streamlit playground for experimenting with GAN-style image generation and Hugging Face Diffusers models.

The built-in DCGAN, StyleGAN, CycleGAN, and CGAN classes are seeded procedural placeholders with a shared `generate(params)` interface. They are intentionally structured so real PyTorch GAN checkpoints can be plugged in later. Stable Diffusion and custom Hugging Face model IDs use `diffusers`.

## Features

- Streamlit web UI with a playful custom stylesheet
- Sidebar model selector for DCGAN, StyleGAN, CycleGAN, CGAN, and Stable Diffusion
- Optional Hugging Face model ID or URL input
- Prompt input for diffusion models
- Seed and noise/variation sliders
- Cached model loading via `st.cache_resource`
- Common model class interface
- CPU/GPU device selection through `.env`
- Hugging Face token support
- Basic Python logging
- Local output saving to `experiments/`
- Random GAN fun facts after generation

## Project Structure

```text
AnyGAN/
  app.py
  assets/
    styles.css
  models/
    base_model.py
    dcgan.py
    stylegan.py
    cyclegan.py
    cgan.py
    diffusion_model.py
  utils/
    config.py
    fun_facts.py
    helpers.py
    model_loader.py
    ui_components.py
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
DIFFUSION_STEPS=25
SAVE_OUTPUTS=True
OUTPUT_DIR=experiments/
LOG_LEVEL=INFO
```

`DEVICE=auto` uses CUDA when available and CPU otherwise.

## Extending With Real GANs

Each built-in GAN wrapper extends `ProceduralGAN` from `models/base_model.py`. To add a real model:

1. Implement checkpoint loading in `load_checkpoint`.
2. Replace procedural image generation with a PyTorch inference pass.
3. Keep the public `generate(params) -> PIL.Image.Image` contract.
4. Register the model in `utils/model_loader.py`.

This keeps the UI independent from backend model details.
