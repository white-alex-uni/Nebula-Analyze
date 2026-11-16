import os
import numpy as np
from PIL import Image
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_image(file_path: str | Path) -> tuple[Image.Image, Path]:
    """Завантажує одне зображення за шляхом та повертає (Image, Path)."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл не знайдено: {path}")
    img = Image.open(path)
    img.load()
    return img, path


def load_images_from_folder(folder_path: str | Path) -> list[tuple[Image.Image, Path]]:
    """Завантажує всі підтримувані зображення з вказаної папки."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Папку не знайдено: {folder}")

    items: list[tuple[Image.Image, Path]] = []
    for entry in sorted(folder.iterdir()):
        if entry.is_file() and entry.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                img = Image.open(entry)
                img.load()
                items.append((img, entry))
            except Exception as e:
                print(f"Пропущено файл {entry.name}: {e}")
    return items


def get_image() -> tuple[Image.Image | None, Path | None]:
    """Запитує шлях до вхідного зображення у консолі та повертає об'єкт Image й його шлях."""
    while True:
        path_str = input("Введіть шлях до зображення ('q' для скасування): ")
        path_str = path_str.replace("\"", "").strip()
        if path_str.lower() == 'q':
            return None, None
        path = Path(path_str)
        if path.exists():
            try:
                img, resolved_path = load_image(path)
                print(f"Зображення успішно завантажено! Розмір: {img.size}, Формат: {img.format}")
                return img, resolved_path
            except Exception as e:
                print(f"Помилка при відкритті зображення: {e}")
        else:
            print("Файл не знайдено. Спробуйте ще раз.")


def image_2_rgb(img: Image.Image) -> np.ndarray:
    """
    Конвертує об'єкт PIL Image (зокрема .jpg, .png, .bmp тощо)
    у триканальний масив NumPy RGB [H, W, 3] типу uint8.
    """
    img_rgb = img.convert('RGB')
    pixel_data = np.asarray(img_rgb, dtype=np.uint8)
    return pixel_data


def rgb_2_gray(rgb_img: np.ndarray) -> np.ndarray:
    """Допоміжна функція для обчислення яскравості Y = 0.299*R + 0.587*G + 0.114*B."""
    data = np.asarray(rgb_img, dtype=np.float32)
    return 0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]


def rgb_2_image(array: np.ndarray) -> Image.Image:
    """
    Перетворює масив NumPy (2D grayscale або 3D RGB) у PIL Image в режимі 'RGB'.
    Забезпечує єдиний вихідний формат Image ('RGB') та коректне збереження у .jpg/.png.
    """
    clipped = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    if clipped.ndim == 2:
        clipped = np.stack([clipped, clipped, clipped], axis=-1)
    elif clipped.ndim == 3 and clipped.shape[2] == 1:
        clipped = np.repeat(clipped, 3, axis=2)
    return Image.fromarray(clipped, mode='RGB')


def save_image(
    img: Image.Image,
    original_path: str | Path,
    output_path: str | Path = r'.\output',
    suffix: str = '',
    default_format: str | None = None
) -> Path | None:
    """Генерує шлях для збереження зображення та зберігає його (коректно обробляє .jpg та інші формати)."""
    os.makedirs(output_path, exist_ok=True)

    base_name, img_format = os.path.splitext(os.path.basename(original_path))

    # Якщо передано нове розширення (при конвертації форматів) або у файлу немає розширення
    if default_format:
        img_format = default_format if default_format.startswith('.') else f".{default_format}"
    elif not img_format:
        img_format = ".jpg"

    save_path = Path(output_path) / f"{base_name}{suffix}{img_format}"

    try:
        # Для JPEG/JPG обов'язково конвертуємо в RGB, оскільки формат не підтримує альфа-канал (RGBA/P)
        img_to_save = img
        if img_format.lower() in {'.jpg', '.jpeg'} and img.mode != 'RGB':
            img_to_save = img.convert('RGB')

        img_to_save.save(save_path)
        print(f"Зображення успішно збережено за адресою: {save_path}")
        return save_path
    except Exception as e:
        print(f"Помилка при збереженні: {e}")
        return None
