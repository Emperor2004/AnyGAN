# location: /tests/test_helpers.py

from PIL import Image

from utils.helpers import normalize_hf_model_id, save_output, slugify


def test_slugify_normalizes_names():
    assert slugify("Stable Diffusion!") == "stable-diffusion"
    assert slugify("   ") == "anygan"


def test_normalize_hf_model_id_accepts_repo_ids_and_urls():
    assert normalize_hf_model_id("runwayml/stable-diffusion-v1-5") == "runwayml/stable-diffusion-v1-5"
    assert (
        normalize_hf_model_id("https://huggingface.co/runwayml/stable-diffusion-v1-5?x=1")
        == "runwayml/stable-diffusion-v1-5"
    )
    assert normalize_hf_model_id("") is None


def test_save_output_creates_experiment_file(tmp_path):
    image = Image.new("RGB", (8, 8), color=(255, 0, 128))
    path = save_output(image, tmp_path, prefix="Test Model")

    assert path.exists()
    assert path.parent == tmp_path
    assert path.name.startswith("test-model_")
    assert path.suffix == ".png"
