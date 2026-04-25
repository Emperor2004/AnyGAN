# location: /utils/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    HF_TOKEN = os.getenv("HF_TOKEN")
    DEVICE = os.getenv("DEVICE", "cpu")
    DEBUG = os.getenv("DEBUG", "False") == "True"
    SAVE_OUTPUTS = os.getenv("SAVE_OUTPUTS", "True") == "True"
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "experiments/")
    ALLOW_EXTERNAL_MODELS = os.getenv("ALLOW_EXTERNAL_MODELS", "True") == "True"

config = Config()