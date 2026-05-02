# location: /tests/test_models.py

from PIL import Image

import models.diffusion_model as diffusion_module
from models.cgan import CGANModel
from models.cyclegan import CycleGANModel
from models.dcgan import DCGANModel
from models.diffusion_model import DiffusionModel
from models.stylegan import StyleGANModel


PROCEDURAL_MODELS = [DCGANModel, StyleGANModel, CycleGANModel, CGANModel]


def test_procedural_models_generate_rgb_images():
    params = {"seed": 42, "noise": 0.4, "prompt": "validation"}

    for model_cls in PROCEDURAL_MODELS:
        image = model_cls().generate(params)
        assert image.size == (256, 256)
        assert image.mode == "RGB"


def test_procedural_models_are_deterministic_for_same_seed():
    params = {"seed": 99, "noise": 0.25, "prompt": "same"}

    for model_cls in PROCEDURAL_MODELS:
        first = model_cls().generate(params)
        second = model_cls().generate(params)
        assert first.tobytes() == second.tobytes()


def test_procedural_models_change_with_seed():
    for model_cls in PROCEDURAL_MODELS:
        first = model_cls().generate({"seed": 1, "noise": 0.4, "prompt": "a"})
        second = model_cls().generate({"seed": 2, "noise": 0.4, "prompt": "a"})
        assert first.tobytes() != second.tobytes()


def test_diffusion_model_generate_uses_pipeline(monkeypatch):
    class FakeResult:
        images = [Image.new("RGB", (4, 4), color=(12, 34, 56))]

    class FakePipe:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return FakeResult()

    fake_pipe = FakePipe()
    monkeypatch.setattr(DiffusionModel, "_load_pipeline", lambda self: fake_pipe)

    model = DiffusionModel(model_id="fake/model")
    model.device = "cpu"
    image = model.generate({"prompt": "hello", "seed": 123, "noise": 0.5})

    assert image.size == (4, 4)
    assert fake_pipe.kwargs["prompt"] == "hello"
    assert fake_pipe.kwargs["guidance_scale"] == 7.0


def test_diffusion_device_resolution(monkeypatch):
    fake_config = type("Config", (), {"DEVICE": "cuda", "HF_TOKEN": None})()
    monkeypatch.setattr(diffusion_module, "config", fake_config)

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = type("Torch", (), {"cuda": FakeCuda})()
    model = object.__new__(DiffusionModel)

    assert model._resolve_device(fake_torch) == "cpu"


def test_diffusion_authenticate_noops_without_token(monkeypatch):
    fake_config = type("Config", (), {"HF_TOKEN": None})()
    monkeypatch.setattr(diffusion_module, "config", fake_config)
    model = object.__new__(DiffusionModel)

    assert model._authenticate() is None
