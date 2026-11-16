import time
import numpy as np
from PIL import Image

from src.proceed import image_2_rgb, rgb_2_image, rgb_2_gray
from src.methods import median_filter, huang_median_filter_oklab, clahe_channel_1d
from src.color_spaces import oklab_2_rgb, rgb_2_oklab


class ExecutionTimer:
    """Вбудований таймер на базі time.perf_counter() без зовнішніх залежностей."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.last_durations: dict[str, float] = {}

    def start(self, thread: str = "default") -> None:
        self._starts[thread] = time.perf_counter()

    def stop(self, thread: str = "default") -> float:
        start_time = self._starts.pop(thread, None)
        if start_time is None:
            return 0.0
        elapsed = time.perf_counter() - start_time
        self.last_durations[thread] = elapsed
        print(f"[{thread}] Час виконання: {elapsed:.4f} с")
        return elapsed


timer = ExecutionTimer()


def _ensure_rgb_array(input_img: Image.Image | np.ndarray) -> np.ndarray:
    """Приводить вхідні дані (PIL.Image або np.ndarray) до масиву RGB uint8 [H, W, 3]."""
    if isinstance(input_img, Image.Image):
        return image_2_rgb(input_img)
    arr = np.asarray(input_img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def run_basic_median(input_img: Image.Image | np.ndarray) -> Image.Image:
    """
    Запускає базовий медіанний фільтр (вікно 3x3).
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    rgb_arr = _ensure_rgb_array(input_img)
    gray_arr = rgb_2_gray(rgb_arr)

    timer.start(thread="basic_median")
    filtered_arr = median_filter(gray_arr)
    timer.stop(thread="basic_median")

    return rgb_2_image(filtered_arr)


def run_huang_median_filter(input_img: Image.Image | np.ndarray) -> Image.Image:
    """
    Запускає медіанний фільтр з алгоритмом Хуанга у просторі Oklab (вікно 3x3).
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    rgb_arr = _ensure_rgb_array(input_img)
    oklab_img = rgb_2_oklab(rgb_arr)

    timer.start(thread="huang_median")
    filtered_oklab = huang_median_filter_oklab(oklab_img)
    timer.stop(thread="huang_median")

    filtered_rgb = oklab_2_rgb(filtered_oklab)
    return rgb_2_image(filtered_rgb)


def run_clahe(input_img: Image.Image | np.ndarray) -> Image.Image:
    """
    Запускає адаптивне гістограмне вирівнювання (CLAHE) для каналу L у просторі Oklab.
    Повертає Image.Image ('RGB').
    """
    rgb_arr = _ensure_rgb_array(input_img)
    oklab_img = rgb_2_oklab(rgb_arr)

    timer.start(thread="clahe")
    filtered_oklab = oklab_img.copy()
    L_arr = oklab_img[:, :, 0]
    L_arr_enhanced = clahe_channel_1d(L=L_arr)
    filtered_oklab[:, :, 0] = L_arr_enhanced
    timer.stop(thread="clahe")

    filtered_rgb = oklab_2_rgb(filtered_oklab)
    return rgb_2_image(filtered_rgb)


ALGORITHMS: dict[str, tuple[callable, str]] = {
    "Адаптивне гістограмне вирівнювання (CLAHE)": (run_clahe, "_clahe"),
    "Медіанний фільтр (алгоритм Хуанга)": (run_huang_median_filter, "_huang_median"),
    "Базовий медіанний фільтр": (run_basic_median, "_basic_median"),
}