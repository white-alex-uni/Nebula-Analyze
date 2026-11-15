import numpy as np
from PIL import Image
from pathlib import Path

def image_2_rgb(image_path: Path) -> np.ndarray:
    """
    Завантажує зображення у форматі PNG і повертає єдиний список значень RGB [R, G, B].
    """
    with Image.open(image_path)as img:
        img_rgba = img.convert('RGB')

        # Single long list of N*3 [R,G,B] pixels
        pixel_data = np.array(img_rgba).reshape(-1,3)

    img.close()

    return pixel_data[:, :3]