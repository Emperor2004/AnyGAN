# location: /utils/ui_components.py

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.helpers import normalize_hf_model_id
from utils.prompt_engine import STYLE_PRESETS, enhance_prompt, get_negative_prompt


def inject_custom_css() -> None:
    """Load the project stylesheet into Streamlit."""
    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_app_header() -> None:
    """Render the page heading."""
    st.title("AnyGAN")
    st.caption("Ultra-realistic, context-aware image generation with Stable Diffusion XL.")


def model_selector(model_names: list[str]) -> tuple[str, str | None]:
    """Render sidebar controls for model selection."""
    st.sidebar.header("Model")
    model_name = st.sidebar.selectbox("Choose a model", model_names, index=0)

    hf_value = st.sidebar.text_input(
        "Optional Hugging Face model ID",
        placeholder="stabilityai/stable-diffusion-xl-base-1.0",
        help="Paste an SDXL-compatible Diffusers repo ID or Hugging Face model URL.",
    )

    return model_name, normalize_hf_model_id(hf_value)


def generation_controls() -> dict:
    """Render generation controls and return model-ready parameters."""
    st.subheader("Controls")

    mode = st.radio(
        "Mode",
        ["Single image", "Image comparison"],
        horizontal=True,
        label_visibility="collapsed",
    )
    compare_mode = mode == "Image comparison"

    style = st.selectbox("Style preset", list(STYLE_PRESETS.keys()), index=0)

    prompt = st.text_area(
        "Prompt",
        value="a futuristic city",
        height=104,
        help="Write the core subject. AnyGAN expands it into a richer model prompt.",
    )

    prompt_b = ""
    if compare_mode:
        prompt_b = st.text_area(
            "Comparison prompt",
            value="a futuristic city at sunrise",
            height=104,
            help="Used for the right-side comparison image.",
        )

    col_seed, col_seed_b = st.columns(2)
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

    col_guidance, col_steps = st.columns(2)
    with col_guidance:
        guidance_scale = st.slider(
            "Guidance scale",
            min_value=7.5,
            max_value=9.0,
            value=8.0,
            step=0.1,
            help="Higher values follow the enhanced prompt more strongly.",
        )
    with col_steps:
        num_inference_steps = st.slider(
            "Inference steps",
            min_value=35,
            max_value=50,
            value=40,
            step=1,
            help="More steps can improve detail but take longer.",
        )

    negative_prompt = get_negative_prompt()
    enhanced_prompt = enhance_prompt(prompt, style)
    enhanced_prompt_b = enhance_prompt(prompt_b or prompt, style)

    with st.expander("Enhanced prompt transparency", expanded=True):
        st.write("**Prompt sent to model:**")
        st.code(enhanced_prompt, language="text")
        if compare_mode:
            st.write("**Comparison prompt sent to model:**")
            st.code(enhanced_prompt_b, language="text")
        st.write("**Negative prompt:**")
        st.code(negative_prompt, language="text")

    return {
        "compare_mode": compare_mode,
        "style": style,
        "prompt": prompt,
        "prompt_b": prompt_b,
        "enhanced_prompt": enhanced_prompt,
        "enhanced_prompt_b": enhanced_prompt_b,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "seed_b": seed_b,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "width": 1024,
        "height": 1024,
    }


def params_for_side(params: dict, side: str) -> dict:
    """Build the parameter payload for a single or comparison image."""
    payload = {
        "prompt": params["prompt"],
        "enhanced_prompt": params["enhanced_prompt"],
        "negative_prompt": params["negative_prompt"],
        "seed": params["seed"],
        "guidance_scale": params["guidance_scale"],
        "num_inference_steps": params["num_inference_steps"],
        "width": params.get("width", 1024),
        "height": params.get("height", 1024),
    }
    if side == "right":
        payload["prompt"] = params.get("prompt_b") or params["prompt"]
        payload["enhanced_prompt"] = params.get("enhanced_prompt_b") or params["enhanced_prompt"]
        payload["seed"] = params.get("seed_b", params["seed"])
    return payload


def render_model_summary(model_name: str, hf_model_id: str | None) -> None:
    """Show the selected backend before generation."""
    with st.expander("Selected model details", expanded=False):
        st.write(f"**Model:** {model_name}")
        st.write(f"**Hugging Face ID:** {hf_model_id or 'default from .env'}")
        st.write("**Backend:** Hugging Face Diffusers StableDiffusionXLPipeline")


def render_output(image, caption: str, saved_path=None) -> None:
    """Render a generated image and optional saved path."""
    st.image(image, caption=caption, use_column_width=True)
    if saved_path:
        st.success(f"Saved to {saved_path}")


def render_sidebar_help(has_hf_token: bool = False) -> None:
    """Render concise sidebar guidance."""
    st.sidebar.divider()
    token_status = "configured" if has_hf_token else "not set"
    st.sidebar.caption(
        "SDXL is cached after the first load. Image comparison generates two images "
        f"with separate prompts or seeds. Hugging Face token: {token_status}."
    )
