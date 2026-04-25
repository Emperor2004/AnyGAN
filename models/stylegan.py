# location: /models/stylegan.py

from PIL import Image
import numpy as np
from models.base_model import BaseGAN

class DummyStyleGAN(BaseGAN):
    def generate(self, params):
        # Placeholder (replace with real StyleGAN later)
        img = np.random.randint(100, 255, (128, 128, 3), dtype=np.uint8)
        return Image.fromarray(img)