# location: /utils/helpers.py

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower())
    return slug.strip("-") or "anygan"


def save_output(image: Image.Image, output_dir: str | Path = "experiments", prefix: str = "output") -> Path:
    directory = ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{slugify(prefix)}_{timestamp}.png"
    path = directory / filename
    image.save(path)
    return path


def normalize_hf_model_id(value: str | None) -> str | None:
    if not value:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    marker = "huggingface.co/"
    if marker in cleaned:
        cleaned = cleaned.split(marker, 1)[1]

    cleaned = cleaned.split("?", 1)[0].strip("/")
    return cleaned or None
