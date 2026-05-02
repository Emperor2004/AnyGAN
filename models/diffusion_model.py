# location: /models/diffusion_model.py

from __future__ import annotations

import logging

from models.base_model import BaseGenerativeModel
from utils.config import config

logger = logging.getLogger(__name__)


class DiffusionModel(BaseGenerativeModel):
    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or config.DEFAULT_DIFFUSION_MODEL
        super().__init__(
            display_name=f"Stable Diffusion ({self.model_id})",
            slug="stable-diffusion",
            model_type="Diffusers text-to-image pipeline",
            description="A real Diffusers Stable Diffusion pipeline loaded from Hugging Face.",
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
            return
        try:
            from huggingface_hub import login

            login(token=config.HF_TOKEN, add_to_git_credential=False)
        except Exception as exc:
            logger.warning("Hugging Face login failed; continuing with token-based loading: %s", exc)

    def _load_pipeline(self):
        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Stable Diffusion requires torch and diffusers. Install requirements.txt first."
            ) from exc

        self._authenticate()
        self.device = self._resolve_device(torch)
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info("Loading Diffusers pipeline %s on %s", self.model_id, self.device)
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            token=config.HF_TOKEN,
            torch_dtype=dtype,
        )
        pipe = pipe.to(self.device)

        if self.device == "cuda":
            pipe.enable_attention_slicing()

        return pipe

    def generate(self, params: dict):
        import torch

        prompt = params.get("prompt") or "A colorful generative artwork"
        seed = int(params.get("seed", 42))
        noise = float(params.get("noise", 0.45))
        guidance_scale = 5.0 + (noise * 4.0)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=config.DIFFUSION_STEPS,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        return result.images[0]
