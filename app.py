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
    params_for_side,
    render_app_header,
    render_model_summary,
    render_output,
    render_sidebar_help,
)

configure_logging()
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def cached_load_model(model_name: str, hf_model_id: str | None):
    """Cache expensive Stable Diffusion pipeline loading across Streamlit reruns."""
    return load_model(model_name=model_name, hf_model_id=hf_model_id)


def _load_selected_model(model_name: str, hf_model_id: str | None):
    """Load the selected model and show a friendly Streamlit error on failure."""
    try:
        return cached_load_model(model_name, hf_model_id)
    except ModelLoadError as exc:
        logger.exception("Model loading failed")
        st.error(str(exc))
    except Exception as exc:
        logger.exception("Unexpected model loading error: %s", exc)
        st.error("Unexpected model loading error. Check logs for details.")
    return None


def _generate_and_save(model, params: dict, label: str):
    """Generate an image and persist it when saving is enabled."""
    image = model.generate(params)
    saved_path = None
    if config.SAVE_OUTPUTS:
        saved_path = save_output(image, config.OUTPUT_DIR, prefix=f"{model.slug}-{label}")
    return image, saved_path


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
    render_sidebar_help(has_hf_token=bool(config.HF_TOKEN))

    params = generation_controls()
    render_model_summary(model_name, hf_model_id)

    st.subheader("Output")
    st.caption("Generated images are saved locally in the experiments folder.")

    left_action, right_action = st.columns(2)
    generate_clicked = left_action.button(
        "Generate image",
        type="primary",
        use_container_width=True,
    )
    compare_clicked = right_action.button(
        "Compare",
        use_container_width=True,
        disabled=not params["compare_mode"],
    )

    if generate_clicked or compare_clicked:
        with st.spinner(f"Loading {hf_model_id or config.DEFAULT_DIFFUSION_MODEL}..."):
            model = _load_selected_model(model_name, hf_model_id)

        if model is None:
            st.stop()

        if compare_clicked:
            with st.spinner("Generating SDXL image comparison..."):
                try:
                    left_params = params_for_side(params, "left")
                    right_params = params_for_side(params, "right")
                    left_image, left_path = _generate_and_save(model, left_params, "compare-left")
                    right_image, right_path = _generate_and_save(model, right_params, "compare-right")
                except Exception as exc:
                    logger.exception("Comparison generation failed: %s", exc)
                    st.error("Comparison generation failed. Try another prompt, seed, or model ID.")
                    st.stop()

            col_left, col_right = st.columns(2)
            with col_left:
                render_output(left_image, f"{model.display_name} | image A", left_path)
            with col_right:
                render_output(right_image, f"{model.display_name} | image B", right_path)
            st.success("Image comparison generated successfully.")
            st.balloons()
            st.info(get_fun_fact())
            return

        with st.spinner("Generating image..."):
            try:
                image, saved_path = _generate_and_save(model, params_for_side(params, "left"), "single")
            except Exception as exc:
                logger.exception("Image generation failed: %s", exc)
                st.error("Image generation failed. Try another prompt, seed, or model ID.")
                st.stop()

        render_output(image, f"{model.display_name} output", saved_path)
        st.success("Image generated successfully.")
        st.balloons()
        st.info(get_fun_fact())


if __name__ == "__main__":
    main()
