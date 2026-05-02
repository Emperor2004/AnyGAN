# location: /utils/model_loader.py

from __future__ import annotations

import logging

from huggingface_hub.utils import HFValidationError, validate_repo_id

from models.diffusion_model import DiffusionModel
from utils.config import config

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a requested model cannot be loaded."""


AVAILABLE_MODELS = ["SDXL"]


def _validate_hf_model_id(model_id: str) -> None:
    try:
        validate_repo_id(model_id)
    except HFValidationError as exc:
        raise ModelLoadError(f"Invalid Hugging Face model ID: {model_id}") from exc


def load_model(model_name: str, hf_model_id: str | None = None) -> DiffusionModel:
    """Load SDXL from the default or user-provided Hugging Face repo."""
    logger.info("Loading model_name=%s hf_model_id=%s", model_name, hf_model_id)

    if model_name not in AVAILABLE_MODELS:
        raise ModelLoadError(f"Unsupported model: {model_name}. Only SDXL is available.")

    model_id = None
    if hf_model_id:
        if not config.ALLOW_EXTERNAL_MODELS:
            raise ModelLoadError("External Hugging Face models are disabled by configuration.")
        _validate_hf_model_id(hf_model_id)
        model_id = hf_model_id

    try:
        return DiffusionModel(model_id=model_id)
    except ModelLoadError:
        raise
    except Exception as exc:
        logger.exception("Failed to load Stable Diffusion model")
        raise ModelLoadError(
            "Failed to load SDXL. Check the Hugging Face model ID, token, "
            "network access, and available memory."
        ) from exc
