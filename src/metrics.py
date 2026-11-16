from datetime import datetime
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image

from src.proceed import image_2_rgb, rgb_2_gray


def compute_image_metrics(
    orig_img: Image.Image | np.ndarray,
    proc_img: Image.Image | np.ndarray,
) -> dict[str, float | None]:
    """
    Обчислює 6 основних метрик для обробленого зображення відносно оригінального:
    - std: стандартне відхилення σ (міра глобального RMS-контрасту)
    - variance: дисперсія σ^2
    - mse: середньоквадратична похибка (стор. 105)
    - psnr_db: пікове відношення сигнал/шум у дБ (стор. 105)
    - energy: нормалізована енергія гістограми
    - entropy: інформаційна ентропія Шеннона (у бітах)
    """
    orig_rgb = image_2_rgb(orig_img) if isinstance(orig_img, Image.Image) else np.asarray(orig_img)
    proc_rgb = image_2_rgb(proc_img) if isinstance(proc_img, Image.Image) else np.asarray(proc_img)

    proc_gray = rgb_2_gray(proc_rgb).astype(np.float64)

    # Дисперсія σ^2 та стандартне відхилення σ
    variance = float(np.var(proc_gray))
    std = float(np.sqrt(variance))

    # Середньоквадратична похибка (MSE) та Пікове відношення сигнал/шум (PSNR)
    orig_f = orig_rgb.astype(np.float64)
    proc_f = proc_rgb.astype(np.float64)
    mse = float(np.mean((orig_f - proc_f) ** 2))

    if mse <= 1e-12:
        psnr_db: float | None = None
    else:
        psnr_db = float(10.0 * np.log10((255.0 ** 2) / mse))

    # Нормалізована енергія гістограми та інформаційна ентропія Шеннона (у бітах)
    quantized = np.clip(np.rint(proc_gray).astype(np.int32), 0, 255)
    hist = np.bincount(quantized.ravel(), minlength=256).astype(np.float64)
    total_pixels = float(proc_gray.size)
    probs = hist / total_pixels if total_pixels > 0 else np.zeros(256, dtype=np.float64)

    energy = float(np.sum(probs ** 2))
    non_zero_probs = probs[probs > 0.0]
    entropy = float(-np.sum(non_zero_probs * np.log2(non_zero_probs)))

    return {
        "std": round(std, 4),
        "variance": round(variance, 4),
        "mse": round(mse, 4),
        "psnr_db": round(psnr_db, 4) if psnr_db is not None else None,
        "energy": round(energy, 6),
        "entropy": round(entropy, 4),
    }


def save_image_metrics_json(
    original_path: str | Path,
    saved_image_path: str | Path | None,
    algorithm_name: str,
    parameters: dict,
    processing_time_sec: float,
    metrics: dict[str, float | None],
    output_dir: str | Path = r".\output",
) -> Path:
    """
    Зберігає метрики у файл 'output/metrics/<назва_зображення>.json'.
    Якщо файл вже існує — додає новий запис у список 'runs' (нову комірку).
    Якщо не існує — створює файл і записує першу комірку метрик.
    """
    orig_path = Path(original_path)
    base_name = orig_path.stem

    metrics_dir = Path(output_dir) / "metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    json_path = metrics_dir / f"{base_name}.json"

    try:
        rel_saved = os.path.relpath(saved_image_path, Path.cwd()) if saved_image_path else "-"
    except ValueError:
        rel_saved = Path(saved_image_path).name if saved_image_path else "-"

    new_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "algorithm": algorithm_name,
        "parameters": parameters,
        "saved_path": rel_saved,
        "processing_time_ms": round(processing_time_sec * 1000.0, 2),
        "metrics": metrics,
    }

    data: dict = {
        "image_name": orig_path.name,
        "runs": [],
    }

    if json_path.is_file():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and isinstance(loaded.get("runs"), list):
                data = loaded
            elif isinstance(loaded, list):
                data = {"image_name": orig_path.name, "runs": loaded}
        except Exception:
            pass

    data["runs"].append(new_entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path
