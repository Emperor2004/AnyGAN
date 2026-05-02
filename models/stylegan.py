# location: /models/stylegan.py

from __future__ import annotations

import numpy as np

from models.base_model import ProceduralGAN


class StyleGANModel(ProceduralGAN):
    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(
            display_name="StyleGAN",
            slug="stylegan",
            model_type="Procedural GAN placeholder",
            description=(
                "A smooth latent-style placeholder for StyleGAN. "
                "The class is ready to host a real style-based generator checkpoint."
            ),
            checkpoint_path=checkpoint_path,
        )

    def generate(self, params: dict):
        rng = self._rng(params)
        noise = self._noise_level(params)
        h = w = self.size
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        xx, yy = np.meshgrid(x, y)

        layers = np.zeros((h, w, 3))
        for channel in range(3):
            freq = rng.uniform(2.0, 6.0) + noise * 3
            phase = rng.uniform(0, 2 * np.pi)
            layers[..., channel] = np.sin(freq * xx + phase) + np.cos((freq + 1.5) * yy - phase)

        radial = np.exp(-2.8 * (xx**2 + yy**2))
        image = 140 + 48 * layers + 85 * radial[..., None]
        return self._to_image(image)


DummyStyleGAN = StyleGANModel
