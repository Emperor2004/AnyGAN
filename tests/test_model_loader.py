# location: /tests/test_model_loader.py

import pytest

import utils.model_loader as model_loader
from models.dcgan import DCGANModel
from utils.model_loader import AVAILABLE_MODELS, ModelLoadError, load_model


def test_available_models_include_required_choices():
    assert AVAILABLE_MODELS == ["DCGAN", "StyleGAN", "CycleGAN", "CGAN", "Stable Diffusion"]


def test_load_builtin_model_by_name():
    model = load_model("DCGAN")

    assert isinstance(model, DCGANModel)
    assert model.display_name == "DCGAN"


def test_load_unknown_model_raises_friendly_error():
    with pytest.raises(ModelLoadError, match="Unsupported model"):
        load_model("UnknownGAN")


def test_invalid_hugging_face_model_id_is_rejected_before_loading():
    with pytest.raises(ModelLoadError, match="Invalid Hugging Face model ID"):
        load_model("Stable Diffusion", hf_model_id="not a valid repo id")


def test_external_hugging_face_models_can_be_disabled(monkeypatch):
    fake_config = type("Config", (), {"ALLOW_EXTERNAL_MODELS": False})()
    monkeypatch.setattr(model_loader, "config", fake_config)

    with pytest.raises(ModelLoadError, match="disabled"):
        load_model("Stable Diffusion", hf_model_id="acme/demo-model")
