# location: /models/dcgan.py

import numpy as np
from PIL import Image

class DummyDCGAN:
    def __init__(self):
        pass

    def generate(self, params):
        # Dummy image generation (replace with real model later)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return Image.fromarray(img)