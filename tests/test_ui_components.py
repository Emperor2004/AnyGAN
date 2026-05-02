# location: /tests/test_ui_components.py

from types import SimpleNamespace

import utils.ui_components as ui


class ContextValue:
    def __init__(self, value=None):
        self.value = value
        self.writes = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def slider(self, *args, **kwargs):
        return self.value

    def write(self, value):
        self.writes.append(value)


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

    def markdown(self, *args, **kwargs):
        self.markdown_calls.append((args, kwargs))

    def title(self, value):
        self.title_calls.append(value)

    def caption(self, value):
        self.caption_calls.append(value)

    def subheader(self, value):
        self.subheader_calls.append(value)

    def text_area(self, *args, **kwargs):
        self.text_area_args = (args, kwargs)
        return "prompt"

    def columns(self, count):
        return [ContextValue(123), ContextValue(0.5)][:count]

    def slider(self, *args, **kwargs):
        return kwargs.get("value")

    def expander(self, *args, **kwargs):
        return ContextValue()

    def write(self, value):
        self.write_calls.append(value)


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

    model_name, hf_model_id = ui.model_selector(["DCGAN", "Stable Diffusion"])

    assert model_name == "DCGAN"
    assert hf_model_id == "acme/demo-model"


def test_generation_controls_and_sidebar_help(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)

    params = ui.generation_controls(is_diffusion=True)
    ui.render_sidebar_help()

    assert params == {"prompt": "prompt", "seed": 42, "noise": 0.45}
    assert fake_st.sidebar.captions


def test_render_model_summary(monkeypatch):
    fake_st = FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake_st)
    model = SimpleNamespace(
        display_name="Demo",
        model_type="test",
        device="cpu",
        description="A demo model.",
    )

    ui.render_model_summary(model)

    assert fake_st.write_calls
