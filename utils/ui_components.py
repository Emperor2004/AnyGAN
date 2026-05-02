# location: /utils/ui_components.py

from __future__ import annotations

from pathlib import Path

import streamlit as st

from utils.helpers import normalize_hf_model_id


def inject_custom_css() -> None:
    css_path = Path("assets/styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_app_header() -> None:
    st.title("AnyGAN")
    st.caption("A playful, modular playground for GAN-style generation and diffusion models.")


def model_selector(model_names: list[str]) -> tuple[str, str | None]:
    st.sidebar.header("Model")
    model_name = st.sidebar.selectbox("Choose a model", model_names, index=0)

    hf_value = st.sidebar.text_input(
        "Optional Hugging Face model ID",
        placeholder="runwayml/stable-diffusion-v1-5",
        help="Paste a repo ID or Hugging Face model URL. External IDs are loaded as Diffusers pipelines.",
    )

    return model_name, normalize_hf_model_id(hf_value)


def generation_controls(is_diffusion: bool) -> dict:
    st.subheader("Controls")

    prompt = st.text_area(
        "Prompt",
        value="A bright futuristic city made of glass, neon, and soft evening light",
        height=92,
        disabled=not is_diffusion,
        help="Used by Stable Diffusion and Hugging Face Diffusers models.",
    )

    col_seed, col_noise = st.columns(2)
    with col_seed:
        seed = st.slider("Seed", min_value=0, max_value=999_999, value=42, step=1)
    with col_noise:
        noise = st.slider(
            "Noise / variation",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            help="For placeholder GANs this controls visual variation. For diffusion it nudges guidance.",
        )

    return {
        "prompt": prompt,
        "seed": seed,
        "noise": noise,
    }


def render_model_summary(model) -> None:
    with st.expander("Loaded model details", expanded=False):
        st.write(f"**Name:** {model.display_name}")
        st.write(f"**Type:** {model.model_type}")
        st.write(f"**Device:** {model.device}")
        st.write(model.description)


def render_sidebar_help() -> None:
    st.sidebar.divider()
    st.sidebar.caption(
        "Tip: DCGAN, StyleGAN, CycleGAN, and CGAN are extensible demo wrappers. "
        "Stable Diffusion and Hugging Face IDs use Diffusers."
    )
