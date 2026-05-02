# location: /models/diffusion_model.py

import torch
from diffusers import StableDiffusionPipeline
from utils.config import config

class DiffusionModel:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() and config.DEVICE != "cpu" else "cpu"

        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            use_auth_token=config.HF_TOKEN
        )

        self.pipe = self.pipe.to(self.device)

    def generate(self, params):
        prompt = params.get("prompt", "A futuristic GAN generated artwork")

        image = self.pipe(prompt).images[0]
        return image