# location: /models/cgan.py

from __future__ import annotations

import hashlib

import numpy as np

from models.base_model import ProceduralGAN


class CGANModel(ProceduralGAN):
    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(
            display_name="CGAN",
            slug="cgan",
            model_type="Procedural GAN placeholder",
            description=(
                "A conditional GAN placeholder. The prompt text is hashed as a lightweight condition "
                "until a real label-conditioned generator is added."
            ),
            checkpoint_path=checkpoint_path,
        )

    def generate(self, params: dict):
        prompt = params.get("prompt", "condition")
        condition_hash = hashlib.sha256(prompt.encode("utf-8")).digest()
        condition = np.frombuffer(condition_hash[:3], dtype=np.uint8).astype(float)

        rng = self._rng(params)
        noise = self._noise_level(params)
        h = w = self.size
        yy, xx = np.mgrid[0:h, 0:w]
        checker = ((xx // 24 + yy // 24) % 2)[..., None]
        colors = condition.reshape(1, 1, 3)
        jitter = rng.normal(0, 45 * noise, size=(h, w, 3))
        image = checker * colors + (1 - checker) * (255 - colors) + jitter
        return self._to_image(image)


DummyCGAN = CGANModel
