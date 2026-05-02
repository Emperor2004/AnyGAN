# location: /models/dcgan.py

from __future__ import annotations

import numpy as np

from models.base_model import ProceduralGAN


class DCGANModel(ProceduralGAN):
    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(
            display_name="DCGAN",
            slug="dcgan",
            model_type="Procedural GAN placeholder",
            description=(
                "A seeded placeholder for a Deep Convolutional GAN. "
                "Replace load_checkpoint and generate with a trained PyTorch generator later."
            ),
            checkpoint_path=checkpoint_path,
        )

    def generate(self, params: dict):
        rng = self._rng(params)
        noise = self._noise_level(params)
        h = w = self.size
        base = rng.normal(loc=128, scale=55 + 60 * noise, size=(h, w, 3))

        yy, xx = np.mgrid[0:h, 0:w]
        rings = np.sin((xx**2 + yy**2) / (850 - 500 * noise) + rng.random() * 6.28)
        palette = np.stack(
            [
                120 + 80 * rings,
                95 + 90 * np.sin(xx / 15 + noise * 4),
                155 + 70 * np.cos(yy / 18),
            ],
            axis=-1,
        )
        image = (0.55 * base) + (0.45 * palette)
        return self._to_image(image)


DummyDCGAN = DCGANModel
