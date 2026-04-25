# location: /utils/model_loader.py

from models.dcgan import DummyDCGAN
from models.stylegan import DummyStyleGAN
from models.cyclegan import DummyCycleGAN
from models.cgan import DummyCGAN
from utils.config import config

def load_model(model_name, hf_link=None):
    try:
        if hf_link and config.ALLOW_EXTERNAL_MODELS:
            print(f"Loading external model: {hf_link}")
            return DummyDCGAN()  # placeholder

        if model_name == "DCGAN":
            return DummyDCGAN()

        elif model_name == "StyleGAN":
            return DummyStyleGAN()

        elif model_name == "CycleGAN":
            return DummyCycleGAN()

        elif model_name == "CGAN":
            return DummyCGAN()

        return None

    except Exception as e:
        print("Error loading model:", e)
        return None