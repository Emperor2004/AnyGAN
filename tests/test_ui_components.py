# location: /tests/test_ui_components.py

from types import SimpleNamespace

import utils.ui_components as ui


class ContextValue:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeSidebar:
    def __init__(self):
        self.captions = []

    def header(self, value):
        self.header_value = value

    def selectbox(self, label, options, index=0):
        self.selectbox_args = (label, options, index)
        return options[index]

    def text_input(self, *args, **kwargs):
        self.text_input_args = (args, kwargs)
        return "https://huggingface.co/acme/demo-model?tab=files"

    def divider(self):
        self.divider_called = True

    def caption(self, value):
        self.captions.append(value)


class FakeStreamlit:
    def __init__(self):
        self.sidebar = FakeSidebar()
        self.markdown_calls = []
        self.title_calls = []
        self.caption_calls = []
        self.subheader_calls = []
        self.write_calls = []
        self.success_calls = []
        self.code_calls = []

    def markdown(self, *args, **kwargs):
        self.markdown_calls.append((args, kwargs))

    def title(self, value):
        self.title_calls.append(value)

    def caption(self, value):
        self.caption_calls.append(value)

    def subheader(self, value):
        self.subheader_calls.append(value)

    def radio(self, *args, **kwargs):
        return "Side-by-side compare"

    def selectbox(self, label, options, index=0):
        return options[index]

    def text_area(self, label, *args, **kwargs):
        return "prompt b" if "Comparison" in label else "prompt a"

    def columns(self, count):
        return [ContextValue() for _ in range(count)]

    def slider(self, label, *args, **kwargs):
        if label == "Seed":
            return 42
        if label == "Comparison seed":
            return 43
        if label == "Guidance scale":
            return 8.0
        if label == "Inference steps":
            return 35
        return kwargs.get("value")

    def expander(self, *args, **kwargs):
        return ContextValue()

    def write(self, value):
        self.write_calls.append(value)

    def code(self, value, language=None):
        self.code_calls.append((value, language))

    def image(self, *args, **kwargs):
        self.image_args = (args, kwargs)

    def success(self, value):
        self.success_calls.append(value)


def test_inject_custom_css_and_header(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)

    ui.inject_custom_css()
    ui.render_app_header()

    assert fake_st.markdown_calls
    assert fake_st.title_calls == ["AnyGAN"]
    assert fake_st.caption_calls


def test_model_selector_normalizes_hugging_face_url(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)

    model_name, hf_model_id = ui.model_selector(["Stable Diffusion"])

    assert model_name == "Stable Diffusion"
    assert hf_model_id == "acme/demo-model"


def test_generation_controls_and_sidebar_help(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)

    params = ui.generation_controls()
    ui.render_sidebar_help(has_hf_token=True)

    assert params["compare_mode"] is True
    assert params["style"] == "Photorealistic"
    assert params["prompt"] == "prompt a"
    assert params["prompt_b"] == "prompt b"
    assert params["negative_prompt"] == "blurry, low quality, distorted, bad anatomy, watermark, artifacts"
    assert params["seed"] == 42
    assert params["seed_b"] == 43
    assert params["guidance_scale"] == 8.0
    assert params["num_inference_steps"] == 35
    assert "prompt a" in params["enhanced_prompt"]
    assert fake_st.code_calls
    assert fake_st.sidebar.captions


def test_params_for_side_uses_right_prompt_and_seed():
    params = {
        "prompt": "left",
        "prompt_b": "right",
        "negative_prompt": "avoid",
        "enhanced_prompt": "enhanced left",
        "enhanced_prompt_b": "enhanced right",
        "seed": 1,
        "seed_b": 2,
        "guidance_scale": 8.0,
        "num_inference_steps": 35,
    }

    assert ui.params_for_side(params, "left")["prompt"] == "left"
    assert ui.params_for_side(params, "left")["enhanced_prompt"] == "enhanced left"
    assert ui.params_for_side(params, "right")["prompt"] == "right"
    assert ui.params_for_side(params, "right")["enhanced_prompt"] == "enhanced right"
    assert ui.params_for_side(params, "right")["seed"] == 2


def test_render_model_summary_and_output(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)
    image = SimpleNamespace()

    ui.render_model_summary("Stable Diffusion", "acme/demo")
    ui.render_output(image, "caption", "experiments/image.png")

    assert fake_st.write_calls
    assert fake_st.success_calls == ["Saved to experiments/image.png"]
