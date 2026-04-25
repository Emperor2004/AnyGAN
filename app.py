# location: /app.py

import streamlit as st
from utils.model_loader import load_model
from utils.ui_components import model_selector, generation_controls
from utils.fun_facts import get_fun_fact
from utils.helpers import save_output
from utils.config import config

st.set_page_config(page_title="AnyGAN", layout="wide")

st.title("🎨 AnyGAN — GAN Playground")
st.caption("Play. Compare. Understand GANs.")

# Sidebar
model_name, hf_link = model_selector()

# Load model
model = load_model(model_name, hf_link)

# Controls
params = generation_controls()

# Generate
if st.button("🚀 Generate"):
    if model:
        output = model.generate(params)

        st.image(output, caption=f"{model_name} Output", use_column_width=True)

        # Save output
        if config.SAVE_OUTPUTS:
            path = save_output(output, config.OUTPUT_DIR)
            st.success(f"Saved to {path}")

        # Fun fact
        st.info(f"💡 {get_fun_fact()}")

    else:
        st.error("❌ Model failed to load")