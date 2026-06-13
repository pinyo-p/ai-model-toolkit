import io
import zipfile
import os
import random
import numpy as np
import torch
from PIL import Image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image(image: Image.Image, path: str):
    image.save(path)


def load_image(path: str) -> Image.Image:
    return Image.open(path)


def create_zip_from_images(images: list[Image.Image], filenames: list[str]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for img, name in zip(images, filenames):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            zf.writestr(name, img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()