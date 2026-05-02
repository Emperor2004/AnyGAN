# location: /models/base_model.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class BaseGenerativeModel:
    display_name: str
    slug: str
    model_type: str
    description: str
    checkpoint_path: str | None = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.checkpoint_path:
            self.load_checkpoint(Path(self.checkpoint_path))

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Extension hook for real GAN checkpoints."""
        raise NotImplementedError("Checkpoint loading is not implemented for this model yet.")

    def generate(self, params: dict) -> Image.Image:
        raise NotImplementedError("Models must implement generate(params).")


class ProceduralGAN(BaseGenerativeModel):
    """Seeded image generator used until real GAN checkpoints are plugged in."""

    size: int = 256

    def _rng(self, params: dict) -> np.random.Generator:
        seed = int(params.get("seed", 42))
        return np.random.default_rng(seed)

    def _noise_level(self, params: dict) -> float:
        return float(np.clip(params.get("noise", 0.45), 0.0, 1.0))

    def _to_image(self, array: np.ndarray) -> Image.Image:
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")


BaseGAN = BaseGenerativeModel
