# location: /tests/test_prompt_engine.py

from utils.prompt_engine import STYLE_PRESETS, enhance_prompt, get_negative_prompt


def test_enhance_prompt_adds_style_and_quality_terms():
    prompt = enhance_prompt("a futuristic city", "Cyberpunk")

    assert "a futuristic city" in prompt
    assert "cyberpunk" in prompt
    assert "neon lights" in prompt
    assert "ultra realistic" in prompt
    assert "8k" in prompt


def test_enhance_prompt_uses_photorealistic_fallback():
    prompt = enhance_prompt("", "Unknown")

    assert "a beautiful cinematic scene" in prompt
    assert STYLE_PRESETS["Photorealistic"] in prompt


def test_get_negative_prompt_contains_required_terms():
    negative = get_negative_prompt()

    assert negative == "blurry, low quality, distorted, bad anatomy, watermark, artifacts"
