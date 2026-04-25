# location: /utils/ui_components.py

import streamlit as st

def model_selector():
    st.sidebar.subheader("🤖 Model Selection")

    model_name = st.sidebar.selectbox(
        "Choose Model",
        ["DCGAN", "StyleGAN", "CycleGAN", "CGAN"]
    )

    hf_link = st.sidebar.text_input("🔗 Hugging Face Link")

    return model_name, hf_link


def generation_controls():
    st.subheader("🎮 Generation Controls")

    col1, col2 = st.columns(2)

    with col1:
        seed = st.slider("Seed 🎲", 0, 100, 42)

    with col2:
        noise = st.slider("Noise Level 🌪️", 0.0, 1.0, 0.5)

    return {
        "seed": seed,
        "noise": noise
    }