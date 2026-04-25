# location: /models/cyclegan.py

from PIL import Image
import numpy as np
from models.base_model import BaseGAN

class DummyCycleGAN(BaseGAN):
    def generate(self, params):
        img = np.random.randint(0, 200, (128, 128, 3), dtype=np.uint8)
        return Image.fromarray(img)