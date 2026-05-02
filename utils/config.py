# location: /utils/config.py

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    APP_NAME: str = os.getenv("APP_NAME", "AnyGAN")
    HF_TOKEN: str | None = os.getenv("HF_TOKEN") or None
    DEVICE: str = os.getenv("DEVICE", "auto").strip().lower()
    DEBUG: bool = _env_bool("DEBUG", False)
    SAVE_OUTPUTS: bool = _env_bool("SAVE_OUTPUTS", True)
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "experiments"))
    ALLOW_EXTERNAL_MODELS: bool = _env_bool("ALLOW_EXTERNAL_MODELS", True)
    DEFAULT_DIFFUSION_MODEL: str = os.getenv(
        "DEFAULT_DIFFUSION_MODEL",
        "runwayml/stable-diffusion-v1-5",
    )
    DIFFUSION_STEPS: int = _env_int("DIFFUSION_STEPS", 25)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


config = Config()


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=False,
    )
