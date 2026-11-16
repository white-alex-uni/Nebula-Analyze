import time
import numpy as np
from PIL import Image

from src.proceed import image_2_rgb, rgb_2_image, rgb_2_gray
from src.methods import (
    median_filter,
    huang_median_filter_oklab,
    clahe_channel_1d,
    canny_edge_detection,
    otsu_thresholding,
    erosion_filter,
    wavelet_transform,
)
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


def run_basic_median(
    input_img: Image.Image | np.ndarray,
    kernel_size: int = 3,
) -> Image.Image:
    """
    Запускає базовий медіанний фільтр.
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1

    rgb_arr = _ensure_rgb_array(input_img)
    gray_arr = rgb_2_gray(rgb_arr)

    timer.start(thread="basic_median")
    filtered_arr = median_filter(gray_arr, kernel_size=k)
    timer.stop(thread="basic_median")

    return rgb_2_image(filtered_arr)


def run_huang_median_filter(
    input_img: Image.Image | np.ndarray,
    kernel_size: int = 3,
    bins: int = 256,
) -> Image.Image:
    """
    Запускає медіанний фільтр з алгоритмом Хуанга у просторі Oklab.
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1

    rgb_arr = _ensure_rgb_array(input_img)
    oklab_img = rgb_2_oklab(rgb_arr)

    timer.start(thread="huang_median")
    filtered_oklab = huang_median_filter_oklab(oklab_img, kernel_size=k, bins=int(bins))
    timer.stop(thread="huang_median")

    filtered_rgb = oklab_2_rgb(filtered_oklab)
    return rgb_2_image(filtered_rgb)


def run_clahe(
    input_img: Image.Image | np.ndarray,
    clip_limit: float = 2.5,
    grid_size: int | str | tuple[int, int] = (8, 8),
    n_bins: int = 1024,
) -> Image.Image:
    """
    Запускає адаптивне гістограмне вирівнювання (CLAHE) для каналу L у просторі Oklab.
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    if isinstance(grid_size, str) and "x" in grid_size.lower():
        parts = grid_size.lower().split("x")
        grid_tuple = (int(parts[0]), int(parts[1]))
    elif isinstance(grid_size, (int, float)):
        g = max(2, int(grid_size))
        grid_tuple = (g, g)
    else:
        grid_tuple = (int(grid_size[0]), int(grid_size[1]))

    rgb_arr = _ensure_rgb_array(input_img)
    oklab_img = rgb_2_oklab(rgb_arr)

    timer.start(thread="clahe")
    filtered_oklab = oklab_img.copy()
    L_arr = oklab_img[:, :, 0]
    L_arr_enhanced = clahe_channel_1d(
        L=L_arr,
        grid_size=grid_tuple,
        clip_limit=float(clip_limit),
        n_bins=int(n_bins),
    )
    filtered_oklab[:, :, 0] = L_arr_enhanced
    timer.stop(thread="clahe")

    filtered_rgb = oklab_2_rgb(filtered_oklab)
    return rgb_2_image(filtered_rgb)


def run_canny(
    input_img: Image.Image | np.ndarray,
    sigma: float = 1.4,
    kernel_size: int = 5,
    low_threshold: float = 0.08,
    high_threshold: float = 0.20,
) -> Image.Image:
    """
    Запускає виділення контурів методом Кенні.
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1

    rgb_arr = _ensure_rgb_array(input_img)
    gray_arr = rgb_2_gray(rgb_arr)

    timer.start(thread="canny")
    edges_arr = canny_edge_detection(
        gray_arr,
        sigma=float(sigma),
        kernel_size=k,
        low_threshold=float(low_threshold),
        high_threshold=float(high_threshold),
    )
    timer.stop(thread="canny")

    return rgb_2_image(edges_arr)


def run_otsu(
    input_img: Image.Image | np.ndarray,
    n_bins: int = 256,
) -> Image.Image:
    """
    Запускає порогову сегментацію методом Оцу.
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    rgb_arr = _ensure_rgb_array(input_img)
    gray_arr = rgb_2_gray(rgb_arr)

    timer.start(thread="otsu")
    binary_arr = otsu_thresholding(gray_arr, n_bins=int(n_bins))
    timer.stop(thread="otsu")

    return rgb_2_image(binary_arr)


def run_erosion(
    input_img: Image.Image | np.ndarray,
    kernel_size: int = 3,
    shape: str = "rect",
    iterations: int = 1,
) -> Image.Image:
    """
    Запускає морфологічну обробку — ерозію.
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1

    rgb_arr = _ensure_rgb_array(input_img)

    timer.start(thread="erosion")
    eroded_arr = erosion_filter(
        rgb_arr,
        kernel_size=k,
        shape=str(shape),
        iterations=max(1, int(iterations)),
    )
    timer.stop(thread="erosion")

    return rgb_2_image(eroded_arr)


def run_wavelet(
    input_img: Image.Image | np.ndarray,
    levels: int = 2,
    mode: str = "decompose",
    threshold_scale: float = 1.0,
) -> Image.Image:
    """
    Запускає 2D перетворення Вейвлета (багаторівневий DWT розклад Хаара).
    Контракт: приймає Image.Image (або ndarray) -> повертає Image.Image ('RGB').
    """
    rgb_arr = _ensure_rgb_array(input_img)
    mode_str = str(mode).lower()

    timer.start(thread="wavelet")
    if mode_str == "denoise":
        wavelet_arr = wavelet_transform(
            rgb_arr,
            levels=int(levels),
            mode="denoise",
            threshold_scale=float(threshold_scale),
        )
    else:
        gray_arr = rgb_2_gray(rgb_arr)
        wavelet_arr = wavelet_transform(
            gray_arr,
            levels=int(levels),
            mode="decompose",
            threshold_scale=float(threshold_scale),
        )
    timer.stop(thread="wavelet")

    return rgb_2_image(wavelet_arr)


ALGORITHMS: dict[str, tuple[callable, str]] = {
    "Адаптивне гістограмне вирівнювання (CLAHE)": (run_clahe, "_clahe"),
    "Медіанний фільтр (алгоритм Хуанга)": (run_huang_median_filter, "_huang_median"),
    "Базовий медіанний фільтр": (run_basic_median, "_basic_median"),
    "Метод Кенні (виділення контурів)": (run_canny, "_canny"),
    "Метод Оцу (порогова сегментація)": (run_otsu, "_otsu"),
    "Ерозія (морфологія)": (run_erosion, "_erosion"),
    "Перетворення Вейвлета": (run_wavelet, "_wavelet"),
}

ALGORITHM_PARAMS: dict[str, list[dict]] = {
    "Адаптивне гістограмне вирівнювання (CLAHE)": [
        {
            "name": "clip_limit",
            "label": "Clip Limit",
            "type": "slider",
            "min": 1.0,
            "max": 10.0,
            "step": 0.1,
            "default": 2.5,
        },
        {
            "name": "grid_size",
            "label": "Grid Size (N×N)",
            "type": "combo",
            "options": ["4x4", "8x8", "12x12", "16x16"],
            "default": "8x8",
        },
        {
            "name": "n_bins",
            "label": "Histogram Bins",
            "type": "combo",
            "options": ["256", "512", "1024", "2048"],
            "default": "1024",
        },
    ],
    "Медіанний фільтр (алгоритм Хуанга)": [
        {
            "name": "kernel_size",
            "label": "Kernel Size",
            "type": "slider",
            "min": 3,
            "max": 15,
            "step": 2,
            "default": 3,
        },
        {
            "name": "bins",
            "label": "Quantization Bins",
            "type": "combo",
            "options": ["64", "128", "256", "512"],
            "default": "256",
        },
    ],
    "Базовий медіанний фільтр": [
        {
            "name": "kernel_size",
            "label": "Kernel Size",
            "type": "slider",
            "min": 3,
            "max": 11,
            "step": 2,
            "default": 3,
        },
    ],
    "Метод Кенні (виділення контурів)": [
        {
            "name": "sigma",
            "label": "Gaussian Sigma",
            "type": "slider",
            "min": 0.5,
            "max": 5.0,
            "step": 0.1,
            "default": 1.4,
        },
        {
            "name": "kernel_size",
            "label": "Gaussian Kernel Size",
            "type": "slider",
            "min": 3,
            "max": 11,
            "step": 2,
            "default": 5,
        },
        {
            "name": "low_threshold",
            "label": "Low Threshold",
            "type": "slider",
            "min": 0.01,
            "max": 0.40,
            "step": 0.01,
            "default": 0.08,
        },
        {
            "name": "high_threshold",
            "label": "High Threshold",
            "type": "slider",
            "min": 0.05,
            "max": 0.80,
            "step": 0.01,
            "default": 0.20,
        },
    ],
    "Метод Оцу (порогова сегментація)": [
        {
            "name": "n_bins",
            "label": "Histogram Bins",
            "type": "combo",
            "options": ["64", "128", "256", "512"],
            "default": "256",
        },
    ],
    "Ерозія (морфологія)": [
        {
            "name": "kernel_size",
            "label": "Kernel Size",
            "type": "slider",
            "min": 3,
            "max": 15,
            "step": 2,
            "default": 3,
        },
        {
            "name": "shape",
            "label": "Structuring Element",
            "type": "combo",
            "options": ["rect", "cross", "disk"],
            "default": "rect",
        },
        {
            "name": "iterations",
            "label": "Iterations",
            "type": "slider",
            "min": 1,
            "max": 10,
            "step": 1,
            "default": 1,
        },
    ],
    "Перетворення Вейвлета": [
        {
            "name": "levels",
            "label": "Decomposition Levels",
            "type": "slider",
            "min": 1,
            "max": 5,
            "step": 1,
            "default": 2,
        },
        {
            "name": "mode",
            "label": "Mode",
            "type": "combo",
            "options": ["decompose", "denoise"],
            "default": "decompose",
        },
        {
            "name": "threshold_scale",
            "label": "Threshold Scale",
            "type": "slider",
            "min": 0.2,
            "max": 3.0,
            "step": 0.1,
            "default": 1.0,
        },
    ],
}
