# location: /models/cyclegan.py

from __future__ import annotations

import numpy as np

from models.base_model import ProceduralGAN


class CycleGANModel(ProceduralGAN):
    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(
            display_name="CycleGAN",
            slug="cyclegan",
            model_type="Procedural GAN placeholder",
            description=(
                "A stylized image-to-image placeholder inspired by CycleGAN. "
                "Swap in paired generator modules when real checkpoints are available."
            ),
            checkpoint_path=checkpoint_path,
        )

    def generate(self, params: dict):
        rng = self._rng(params)
        noise = self._noise_level(params)
        h = w = self.size
        y = np.linspace(0, 1, h)
        x = np.linspace(0, 1, w)
        xx, yy = np.meshgrid(x, y)

        sky = np.stack([80 + 90 * yy, 145 + 40 * yy, 210 + 25 * xx], axis=-1)
        waves = 35 * np.sin((xx * 18) + (yy * 14) + rng.random() * 5)
        texture = rng.normal(0, 25 * noise, size=(h, w, 3))
        image = sky + waves[..., None] + texture
        return self._to_image(image)


DummyCycleGAN = CycleGANModel
