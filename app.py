# location: /app.py

import logging

import streamlit as st

from utils.config import config, configure_logging
from utils.fun_facts import get_fun_fact
from utils.helpers import save_output
from utils.model_loader import AVAILABLE_MODELS, ModelLoadError, load_model
from utils.ui_components import (
    generation_controls,
    inject_custom_css,
    model_selector,
    render_app_header,
    render_model_summary,
    render_sidebar_help,
)

configure_logging()
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def cached_load_model(model_name: str, hf_model_id: str | None):
    """Cache expensive model instances across Streamlit reruns."""
    return load_model(model_name=model_name, hf_model_id=hf_model_id)


def main() -> None:
    st.set_page_config(
        page_title=config.APP_NAME,
        page_icon="AnyGAN",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()
    render_app_header()

    model_name, hf_model_id = model_selector(AVAILABLE_MODELS)
    is_diffusion = model_name == "Stable Diffusion" or bool(hf_model_id)
    params = generation_controls(is_diffusion=is_diffusion)

    render_sidebar_help()

    st.subheader("Generation")
    st.caption("Choose a model, tune the controls, and generate a fresh image.")

    model = None
    load_error = None
    with st.spinner(f"Loading {hf_model_id or model_name}..."):
        try:
            model = cached_load_model(model_name, hf_model_id)
            render_model_summary(model)
        except ModelLoadError as exc:
            load_error = str(exc)
            logger.exception("Model loading failed")
        except Exception as exc:
            load_error = "Unexpected model loading error. Check logs for details."
            logger.exception("Unexpected model loading error: %s", exc)

    if load_error:
        st.error(load_error)
        st.stop()

    generate = st.button(
        "Generate image",
        type="primary",
        use_container_width=True,
        disabled=model is None,
    )

    if generate and model is not None:
        with st.spinner("Generating image..."):
            try:
                output = model.generate(params)
            except Exception as exc:
                logger.exception("Image generation failed: %s", exc)
                st.error("Image generation failed. Try another seed, prompt, or model.")
                st.stop()

        st.image(output, caption=f"{model.display_name} output", use_column_width=True)

        if config.SAVE_OUTPUTS:
            try:
                path = save_output(output, config.OUTPUT_DIR, prefix=model.slug)
                st.success(f"Saved image to {path}")
            except Exception as exc:
                logger.exception("Saving generated image failed: %s", exc)
                st.warning("Image generated, but saving it locally failed.")

        st.info(get_fun_fact())


if __name__ == "__main__":
    main()
