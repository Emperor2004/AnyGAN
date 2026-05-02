# location: /models/diffusion_model.py

from __future__ import annotations

import logging

from models.base_model import BaseGenerativeModel
from utils.config import config

logger = logging.getLogger(__name__)


class DiffusionModel(BaseGenerativeModel):
    """SDXL text-to-image model using Hugging Face Diffusers."""

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or config.DEFAULT_DIFFUSION_MODEL
        super().__init__(
            display_name=f"SDXL ({self.model_id})",
            slug="sdxl",
            model_type="Diffusers SDXL text-to-image pipeline",
            description="A real Stable Diffusion XL pipeline loaded from Hugging Face.",
            device="auto",
        )
        self.pipe = self._load_pipeline()

    def _resolve_device(self, torch_module) -> str:
        requested = config.DEVICE
        if requested == "cpu":
            return "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            logger.warning("DEVICE=cuda was requested but CUDA is unavailable. Falling back to CPU.")
            return "cpu"
        if requested in {"auto", "cuda"} and torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    def _authenticate(self) -> None:
        if not config.HF_TOKEN:
            logger.info("HF_TOKEN is not set. Public models will still load if they do not require auth.")
            return
        try:
            from huggingface_hub import login

            login(token=config.HF_TOKEN, add_to_git_credential=False)
        except Exception as exc:
            logger.warning("Hugging Face login failed; continuing with token-based loading: %s", exc)

    def _load_pipeline(self):
        """Load and move the Diffusers pipeline to the selected device."""
        try:
            import torch
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:
            raise RuntimeError(
                "SDXL requires torch and diffusers. Install requirements.txt first."
            ) from exc

        self._authenticate()
        self.device = self._resolve_device(torch)
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info("Loading SDXL pipeline %s on %s", self.model_id, self.device)
        pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            token=config.HF_TOKEN,
            torch_dtype=dtype,
        )
        pipe = pipe.to(self.device)

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if self.device == "cuda" and hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()

        return pipe

    def generate(self, params: dict):
        """Generate a deterministic image for a prompt and seed."""
        import torch

        prompt = params.get("enhanced_prompt") or params.get("prompt") or "A colorful generative artwork"
        seed = int(params.get("seed", 42))
        guidance_scale = float(params.get("guidance_scale", 8.0))
        guidance_scale = max(7.5, min(9.0, guidance_scale))
        steps = int(params.get("num_inference_steps", config.DIFFUSION_STEPS))
        steps = max(35, min(50, steps))
        width = int(params.get("width", 1024))
        height = int(params.get("height", 1024))
        negative_prompt = params.get("negative_prompt") or None

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        )
        return result.images[0]
