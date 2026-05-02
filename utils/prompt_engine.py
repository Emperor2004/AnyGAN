# location: /utils/prompt_engine.py

from __future__ import annotations

STYLE_PRESETS = {
    "Photorealistic": (
        "photorealistic, natural light, realistic textures, lifelike detail, "
        "professional photography, sharp focus"
    ),
    "Cyberpunk": (
        "cyberpunk, neon lights, glass skyscrapers, rainy streets, reflective surfaces, "
        "cinematic lighting, ultra realistic"
    ),
    "Fantasy": (
        "epic fantasy, magical atmosphere, intricate worldbuilding, dramatic lighting, "
        "highly detailed environment, realistic materials"
    ),
    "Anime": (
        "high quality anime illustration, expressive composition, detailed background, "
        "clean linework, vibrant color grading, cinematic scene"
    ),
}

QUALITY_TERMS = (
    "highly detailed, ultra realistic, 8k, sharp focus, cinematic composition, "
    "rich color, depth of field, masterpiece"
)

NEGATIVE_PROMPT = "blurry, low quality, distorted, bad anatomy, watermark, artifacts"


def _clean_prompt(value: str | None) -> str:
    prompt = (value or "").strip()
    return prompt or "a beautiful cinematic scene"


def enhance_prompt(user_prompt: str, style: str = "Photorealistic") -> str:
    """Transform a short user prompt into a detailed image-generation prompt."""
    base_prompt = _clean_prompt(user_prompt)
    style_terms = STYLE_PRESETS.get(style, STYLE_PRESETS["Photorealistic"])
    return f"{base_prompt}, {style_terms}, {QUALITY_TERMS}"


def get_negative_prompt() -> str:
    """Return the shared negative prompt used to reduce common generation artifacts."""
    return NEGATIVE_PROMPT
