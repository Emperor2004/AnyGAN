# location: /tests/test_models.py

from PIL import Image

import models.diffusion_model as diffusion_module
from models.diffusion_model import DiffusionModel


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
    image = model.generate(
        {
            "prompt": "raw hello",
            "enhanced_prompt": "enhanced hello",
            "negative_prompt": "blur",
            "seed": 123,
            "guidance_scale": 8.2,
            "num_inference_steps": 36,
        }
    )

    assert image.size == (4, 4)
    assert fake_pipe.kwargs["prompt"] == "enhanced hello"
    assert fake_pipe.kwargs["negative_prompt"] == "blur"
    assert fake_pipe.kwargs["guidance_scale"] == 8.2
    assert fake_pipe.kwargs["num_inference_steps"] == 36


def test_diffusion_generation_params_are_clamped(monkeypatch):
    class FakeResult:
        images = [Image.new("RGB", (4, 4))]

    class FakePipe:
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return FakeResult()

    fake_pipe = FakePipe()
    monkeypatch.setattr(DiffusionModel, "_load_pipeline", lambda self: fake_pipe)

    model = DiffusionModel(model_id="fake/model")
    model.device = "cpu"
    model.generate({"prompt": "hello", "seed": 1, "guidance_scale": 99, "num_inference_steps": 99})

    assert fake_pipe.kwargs["guidance_scale"] == 9.0
    assert fake_pipe.kwargs["num_inference_steps"] == 40


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
