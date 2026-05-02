# location: /utils/ui_components.py

import streamlit as st

def model_selector():
    st.sidebar.subheader("🤖 Model Selection")

    model_name = st.sidebar.selectbox(
        "Choose Model",
        ["DCGAN", "StyleGAN", "CycleGAN", "CGAN", "Stable Diffusion"]
    )

    hf_link = st.sidebar.text_input("🔗 Hugging Face Link")

    return model_name, hf_link


def generation_controls():
    import streamlit as st

    st.subheader("🎮 Generation Controls")

    prompt = st.text_input(
        "📝 Prompt",
        "A futuristic city with neon lights"
    )

    seed = st.slider("Seed 🎲", 0, 100, 42)
    noise = st.slider("Noise Level 🌪️", 0.0, 1.0, 0.5)

    return {
        "prompt": prompt,
        "seed": seed,
        "noise": noise
    }