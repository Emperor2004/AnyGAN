# location: /models/cgan.py

from PIL import Image
import numpy as np
from models.base_model import BaseGAN

class DummyCGAN(BaseGAN):
    def generate(self, params):
        img = np.random.randint(50, 255, (64, 64, 3), dtype=np.uint8)
        return Image.fromarray(img)