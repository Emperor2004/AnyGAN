# location: /utils/ui_components.py

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.helpers import normalize_hf_model_id


def inject_custom_css() -> None:
    """Load the project stylesheet into Streamlit."""
    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_app_header() -> None:
    """Render the page heading."""
    st.title("🎨 AnyGAN")
    st.caption("A real AI image generation playground powered by Stable Diffusion.")


def model_selector(model_names: list[str]) -> tuple[str, str | None]:
    """Render sidebar controls for model selection."""
    st.sidebar.header("Model")
    model_name = st.sidebar.selectbox("Choose a model", model_names, index=0)

    hf_value = st.sidebar.text_input(
        "Optional Hugging Face model ID",
        placeholder="runwayml/stable-diffusion-v1-5",
        help="Paste a Diffusers-compatible repo ID or Hugging Face model URL.",
    )

    return model_name, normalize_hf_model_id(hf_value)


def generation_controls() -> dict:
    """Render image generation controls and return normalized params."""
    st.subheader("Controls")

    mode = st.radio(
        "Mode",
        ["Single image", "Side-by-side compare"],
        horizontal=True,
        label_visibility="collapsed",
    )
    compare_mode = mode == "Side-by-side compare"

    prompt = st.text_area(
        "Prompt",
        value="A cinematic neon city at sunset, reflective glass towers, ultra detailed",
        height=104,
        help="This controls the generated image content.",
    )

    prompt_b = ""
    if compare_mode:
        prompt_b = st.text_area(
            "Comparison prompt",
            value="A cinematic floating garden city at sunrise, soft clouds, ultra detailed",
            height=104,
            help="Used for the right-side comparison image.",
        )

    negative_prompt = st.text_input(
        "Negative prompt",
        value="blurry, low quality, distorted, extra limbs, watermark",
        help="Optional text describing what the model should avoid.",
    )

    col_seed, col_seed_b, col_noise = st.columns(3)
    with col_seed:
        seed = st.slider("Seed", min_value=0, max_value=999_999, value=42, step=1)
    with col_seed_b:
        seed_b = st.slider(
            "Comparison seed",
            min_value=0,
            max_value=999_999,
            value=43,
            step=1,
            disabled=not compare_mode,
        )
    with col_noise:
        noise = st.slider(
            "Creativity",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="Maps to Stable Diffusion guidance strength.",
        )

    return {
        "compare_mode": compare_mode,
        "prompt": prompt,
        "prompt_b": prompt_b,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "seed_b": seed_b,
        "noise": noise,
    }


def params_for_side(params: dict, side: str) -> dict:
    """Build the parameter payload for a single or comparison image."""
    payload = {
        "prompt": params["prompt"],
        "negative_prompt": params.get("negative_prompt"),
        "seed": params["seed"],
        "noise": params["noise"],
    }
    if side == "right":
        payload["prompt"] = params.get("prompt_b") or params["prompt"]
        payload["seed"] = params.get("seed_b", params["seed"])
    return payload


def render_model_summary(model_name: str, hf_model_id: str | None) -> None:
    """Show the selected backend before generation."""
    with st.expander("Selected model details", expanded=False):
        st.write(f"**Model:** {model_name}")
        st.write(f"**Hugging Face ID:** {hf_model_id or 'default from .env'}")
        st.write("**Backend:** Hugging Face Diffusers StableDiffusionPipeline")


def render_output(image, caption: str, saved_path=None) -> None:
    """Render a generated image and optional saved path."""
    st.image(image, caption=caption, use_column_width=True)
    if saved_path:
        st.success(f"Saved to {saved_path}")


def render_sidebar_help() -> None:
    """Render concise sidebar guidance."""
    st.sidebar.divider()
    st.sidebar.caption(
        "Only real Stable Diffusion generation is enabled. "
        "Use comparison mode to test two prompts or seeds side by side."
    )
