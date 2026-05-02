# location: /models/base_model.py

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class BaseImageModel:
    """Common interface for real image generation backends."""

    display_name: str
    slug: str
    model_type: str
    description: str
    device: str = "cpu"

    def generate(self, params: dict) -> Image.Image:
        """Generate an image from UI parameters."""
        raise NotImplementedError("Models must implement generate(params).")


BaseGenerativeModel = BaseImageModel
