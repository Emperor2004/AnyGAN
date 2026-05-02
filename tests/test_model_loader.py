# location: /tests/test_model_loader.py

import pytest

import utils.model_loader as model_loader
from utils.model_loader import AVAILABLE_MODELS, ModelLoadError, load_model


class FakeDiffusionModel:
    def __init__(self, model_id=None):
        self.model_id = model_id
        self.display_name = "Stable Diffusion"


def test_available_models_only_include_stable_diffusion():
    assert AVAILABLE_MODELS == ["Stable Diffusion"]


def test_load_default_stable_diffusion(monkeypatch):
    monkeypatch.setattr(model_loader, "DiffusionModel", FakeDiffusionModel)

    model = load_model("Stable Diffusion")

    assert isinstance(model, FakeDiffusionModel)
    assert model.model_id is None


def test_load_hugging_face_model_id(monkeypatch):
    monkeypatch.setattr(model_loader, "DiffusionModel", FakeDiffusionModel)

    model = load_model("Stable Diffusion", hf_model_id="acme/demo-model")

    assert model.model_id == "acme/demo-model"


def test_load_unknown_model_raises_friendly_error():
    with pytest.raises(ModelLoadError, match="Only Stable Diffusion"):
        load_model("DCGAN")


def test_invalid_hugging_face_model_id_is_rejected_before_loading():
    with pytest.raises(ModelLoadError, match="Invalid Hugging Face model ID"):
        load_model("Stable Diffusion", hf_model_id="not a valid repo id")


def test_external_hugging_face_models_can_be_disabled(monkeypatch):
    fake_config = type("Config", (), {"ALLOW_EXTERNAL_MODELS": False})()
    monkeypatch.setattr(model_loader, "config", fake_config)

    with pytest.raises(ModelLoadError, match="disabled"):
        load_model("Stable Diffusion", hf_model_id="acme/demo-model")
