# location: /utils/helpers.py

import os
from datetime import datetime
from PIL import Image

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_output(image: Image.Image, output_dir="experiments"):
    ensure_dir(output_dir)
    filename = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(output_dir, filename)
    image.save(path)
    return path