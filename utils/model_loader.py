# location: /utils/model_loader.py

from __future__ import annotations

import logging
from collections.abc import Callable

from huggingface_hub.utils import HFValidationError, validate_repo_id

from models.cgan import CGANModel
from models.cyclegan import CycleGANModel
from models.dcgan import DCGANModel
from models.diffusion_model import DiffusionModel
from models.stylegan import StyleGANModel
from utils.config import config

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when a requested model cannot be loaded."""


MODEL_REGISTRY: dict[str, Callable[[], object]] = {
    "DCGAN": DCGANModel,
    "StyleGAN": StyleGANModel,
    "CycleGAN": CycleGANModel,
    "CGAN": CGANModel,
    "Stable Diffusion": DiffusionModel,
}

AVAILABLE_MODELS = list(MODEL_REGISTRY.keys())


def _validate_hf_model_id(model_id: str) -> None:
    try:
        validate_repo_id(model_id)
    except HFValidationError as exc:
        raise ModelLoadError(f"Invalid Hugging Face model ID: {model_id}") from exc


def load_model(model_name: str, hf_model_id: str | None = None):
    """Load a built-in model or a Hugging Face Diffusers model."""
    logger.info("Loading model_name=%s hf_model_id=%s", model_name, hf_model_id)

    if hf_model_id:
        if not config.ALLOW_EXTERNAL_MODELS:
            raise ModelLoadError("External Hugging Face models are disabled by configuration.")
        _validate_hf_model_id(hf_model_id)
        return DiffusionModel(model_id=hf_model_id)

    model_factory = MODEL_REGISTRY.get(model_name)
    if model_factory is None:
        raise ModelLoadError(f"Unsupported model: {model_name}")

    try:
        return model_factory()
    except ModelLoadError:
        raise
    except Exception as exc:
        logger.exception("Failed to load %s", model_name)
        raise ModelLoadError(f"Failed to load {model_name}. Check logs for details.") from exc
