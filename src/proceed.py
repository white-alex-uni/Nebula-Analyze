import os
import numpy as np
from PIL import Image
from pathlib import Path

def get_image() -> tuple[Image.Image, Path]:
    """Запитує шлях до вхідного зображення та повертає об'єкт Image й його шлях"""
    while True:
        path_str = input("Введіть шлях до зображення ('q' для скасування): ")
        path_str = path_str.replace("\"", "")
        if path_str.lower() == 'q':
            return None, None
        path = Path(path_str)
        if os.path.exists(path):
            try:
                img = Image.open(path)
                print(f"Зображення успішно завантажено! Розмір: {img.size}, Формат: {img.format}")
                return img, path
            except Exception as e:
                print(f"Помилка при відкритті зображення: {e}")
        else:
            print("Файл не знайдено. Спробуйте ще раз.")

def image_2_rgb(img: Image.Image) -> np.ndarray:
    """
    Завантажує зображення у форматі PNG і повертає список значень RGB [R, G, B].
    """
    img_rgba = img.convert('RGB')

    pixel_data = np.array(img_rgba, np.uint8)

    return pixel_data

def rgb_2_gray(rgb_img: np.ndarray) -> np.ndarray:
    """Допоміжна функція для обчислення яскравості Y = 0.299*R + 0.587*G + 0.114*B."""
    data = np.array(rgb_img, np.float32)
    return 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]

def array_2_image(array: np.ndarray):
    return Image.fromarray(array.astype(np.uint8))

def save_image(img: Image.Image, original_path: Path, output_path: str, suffix: str, default_format:str=None):
    """Генерує шлях для збереження зображення та зберігає зображення."""
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    dir_name = output_path
    base_name, img_format = os.path.splitext(os.path.basename(original_path))

    # Якщо передано нове розширення (при конвертації форматів)
    if default_format:
        img_format = default_format if default_format.startswith('.') else f".{default_format}"

    save_path = os.path.join(dir_name, f"{base_name}_{suffix}{img_format}")

    try:
        # Зберігаємо файл
        img.save(Path(save_path))
        print(f"Зображення успішно збережено за адресою: {save_path}")
    except Exception as e:
        print(f"Помилка при збереженні: {e}")