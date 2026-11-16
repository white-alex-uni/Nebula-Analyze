import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def median_filter(img: np.ndarray, kernel_size: int=3) -> np.ndarray:
    """
    Базовий медіанний фільтр зображення для відтінков сірого.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError('Kernel size must be odd number')

    height, width = img.shape[:2]
    output_img = np.zeros_like(img)
    radius = kernel_size // 2
    window_size = kernel_size * kernel_size
    median_index = window_size // 2

    window_shape = (window_size,) + img.shape[2:]

    for h in range(height):
        for w in range(width):
            window = np.empty(window_shape, dtype=img.dtype)
            count = 0

            for closest_h in range(-radius, radius + 1):
                for closest_w in range(-radius, radius + 1):
                    sample_h = min(max(h + closest_h, 0), height - 1)
                    sample_w = min(max(w + closest_w, 0), width - 1)

                    window[count] = img[sample_h][sample_w]
                    count += 1

            window.sort(axis=0)
            output_img[h][w] = window[median_index]

    return output_img

def huang_median_filter_oklab(oklab_img: np.ndarray, kernel_size: int = 3, bins: int = 256) -> np.ndarray:
    """
    Медіанний фільтр за алгоритмом Хуанга для зображення в просторі Oklab.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError('Kernel size must be odd number')

    # Межі каналів Oklab: L у [0, 1], a та b у [-0.4, 0.4]
    channel_mins = np.array([0.0, -0.4, -0.4], dtype=oklab_img.dtype)
    channel_maxs = np.array([1.0, 0.4, 0.4], dtype=oklab_img.dtype)

    scales = (bins - 1) / (channel_maxs - channel_mins)

    # Лінійне квантування float -> int32 у діапазон [0, bins - 1]
    clipped = np.clip(oklab_img, channel_mins, channel_maxs)
    quantized = np.rint((clipped - channel_mins) * scales).astype(np.int32)

    # Поканальна фільтрація за алгоритмом Хуанга
    filtered_quantized = np.empty_like(quantized)
    for channel in range(3):
        filtered_quantized[:, :, channel] = _huang_channel_vector(quantized[:, :, channel], kernel_size=kernel_size, bins=bins)

    # Деквантування індексів гістограми назад у простір Oklab
    filtered_oklab = filtered_quantized.astype(oklab_img.dtype) / scales + channel_mins
    return filtered_oklab

def _huang_channel_vector(quantized_channel: np.ndarray, kernel_size: int, bins: int) -> np.ndarray:
    """
    Алгоритм Хуанга для одного квантованого каналу векторизований по рядках.
    """
    h, w = quantized_channel.shape
    radius = kernel_size // 2
    threshold = (kernel_size * kernel_size) // 2

    padded = np.pad(quantized_channel, radius, mode="edge")
    output = np.empty((h, w), dtype=np.int32)
    row_idx = np.arange(h)

    # Векторизована ініціалізація гістограм для першого стовпця (x = 0)
    # Формуємо всі H вікон розміром (kernel_size, kernel_size) без копіювання пам'яті
    first_col_windows = sliding_window_view(
        padded[:, :kernel_size],
        window_shape=(kernel_size, kernel_size),
    )[:, 0].reshape(h, -1)

    # Зміщуємо значення кожного рядка на (row * bins), щоб порахувати всі H гістограм одним bincount
    row_offsets = (row_idx * bins)[:, None]
    hists = np.bincount(
        (first_col_windows + row_offsets).ravel(),
        minlength=h * bins,
    ).reshape(h, bins).astype(np.int32)

    # Початкова медіана та кількість елементів строго менших за неї
    cdf = np.cumsum(hists, axis=1)
    medians = np.argmax(cdf > threshold, axis=1).astype(np.int32)
    lt_medians = np.where(
        medians > 0,
        cdf[row_idx, np.maximum(0, medians - 1)],
        0,
    ).astype(np.int32)

    output[:, 0] = medians

    # Ковзання вікна вздовж осі X
    for x in range(1, w):
        old_col_idx = x - 1
        new_col_idx = x + kernel_size - 1

        for dy in range(kernel_size):
            old_vals = padded[dy: dy + h, old_col_idx]
            new_vals = padded[dy: dy + h, new_col_idx]

            np.add.at(hists, (row_idx, old_vals), -1)
            np.add.at(hists, (row_idx, new_vals), 1)

            lt_medians -= old_vals < medians
            lt_medians += new_vals < medians

        # Коригування вказівника медіани
        mask_left = lt_medians > threshold
        while np.any(mask_left):
            rows = row_idx[mask_left]
            medians[rows] -= 1
            lt_medians[rows] -= hists[rows, medians[rows]]
            mask_left = lt_medians > threshold

        mask_right = (lt_medians + hists[row_idx, medians]) <= threshold
        while np.any(mask_right):
            rows = row_idx[mask_right]
            lt_medians[rows] += hists[rows, medians[rows]]
            medians[rows] += 1
            mask_right = (lt_medians + hists[row_idx, medians]) <= threshold

        output[:, x] = medians

    return output

def clahe_channel_1d(L: np.ndarray, grid_size: tuple[int, int] = (8, 8), clip_limit: float = 2.5, n_bins: int = 1024):
    """
    Векторизований CLAHE для одноканального масиву L [0.0, 1.0].
    Використовує 1024 біни для збереження плавних градієнтів туманності.
    """
    orig_h, orig_w = L.shape
    grid_h, grid_w = grid_size

    # Доповнення (Symmetric Padding), щоб розміри ділилися на розмір сітки
    pad_h = (grid_h - (orig_h % grid_h)) % grid_h
    pad_w = (grid_w - (orig_w % grid_w)) % grid_w
    L_pad = np.pad(L, ((0, pad_h), (0, pad_w)), mode='reflect')

    H, W = L_pad.shape
    tile_h, tile_w = H // grid_h, W // grid_w

    # Квантування L [0, 1] в індекси бінів [0, n_bins - 1] із коректним округленням
    L_pad_clipped = np.clip(L_pad, 0.0, 1.0)
    bins = np.clip(np.rint(L_pad_clipped * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)

    # Розбиття на блоки через reshape + swapaxes
    tiles = bins.reshape(grid_h, tile_h, grid_w, tile_w).swapaxes(1, 2)
    luts = _compute_tile_luts(tiles, n_bins, clip_limit)

    # Векторизована білінійна інтерполяція між центрами тайлів
    y_coords = np.arange(H, dtype=np.float32)
    x_coords = np.arange(W, dtype=np.float32)

    # Координати відносно центрів тайлів
    ty = (y_coords + 0.5) / tile_h - 0.5
    tx = (x_coords + 0.5) / tile_w - 0.5

    r0 = np.clip(np.floor(ty).astype(np.int32), 0, grid_h - 1)
    r1 = np.clip(r0 + 1, 0, grid_h - 1)
    c0 = np.clip(np.floor(tx).astype(np.int32), 0, grid_w - 1)
    c1 = np.clip(c0 + 1, 0, grid_w - 1)

    # Вагові коефіцієнти інтерполяції
    wy = np.clip(ty - r0, 0.0, 1.0)[:, None]  # shape (H, 1)
    wx = np.clip(tx - c0, 0.0, 1.0)[None, :]  # shape (1, W)

    # Розширення індексів для 2D-вибірки з масиву luts
    r0_2d, r1_2d = r0[:, None], r1[:, None]
    c0_2d, c1_2d = c0[None, :], c1[None, :]

    # Вибірка значень яскравості з 4 сусідніх тайлів для кожного пікселя
    val_tl = luts[r0_2d, c0_2d, bins]  # Top-Left
    val_tr = luts[r0_2d, c1_2d, bins]  # Top-Right
    val_bl = luts[r1_2d, c0_2d, bins]  # Bottom-Left
    val_br = luts[r1_2d, c1_2d, bins]  # Bottom-Right

    # Білінійне змішування
    top = (1.0 - wx) * val_tl + wx * val_tr
    bottom = (1.0 - wx) * val_bl + wx * val_br
    L_out = (1.0 - wy) * top + wy * bottom

    # Повертаємо вихідний розмір без padding
    return np.clip(L_out[:orig_h, :orig_w], 0.0, 1.0)

def _compute_tile_luts(tiles: np.ndarray, n_bins: int, clip_limit: float) -> np.ndarray:
    """
    Обчислює таблиці перекодування (CDF LUT) для кожного тайлу з ітеративним обрізанням контрасту,
    гарантуючи, що жоден бін після перерозподілу не перевищує max_count.
    """
    grid_h, grid_w, tile_h, tile_w = tiles.shape
    n_pixels = tile_h * tile_w

    # Абсолютний поріг обрізання у пікселях (мінімум середнє заповнення біна)
    effective_clip = max(1.0, float(clip_limit))
    max_count = max(1.0, effective_clip * (n_pixels / n_bins))
    luts = np.zeros((grid_h, grid_w, n_bins), dtype=np.float32)

    for r in range(grid_h):
        for c in range(grid_w):
            hist = np.bincount(tiles[r, c].ravel(), minlength=n_bins).astype(np.float32)

            # Ітеративне обрізання піків (Clip Limit) та перерозподіл надлишку
            # лише між бінами, які не досягли max_count
            while True:
                excess = np.maximum(hist - max_count, 0.0)
                total_excess = float(np.sum(excess))
                if total_excess <= 1e-5:
                    break
                hist = np.minimum(hist, max_count)
                below_mask = hist < (max_count - 1e-6)
                n_below = int(np.count_nonzero(below_mask))
                if n_below == 0:
                    break
                hist[below_mask] += total_excess / n_below

            hist = np.minimum(hist, max_count)

            # Кумулятивна функція розподілу (CDF), нормалізована у [0.0, 1.0]
            cdf = np.cumsum(hist)
            cdf_min = cdf[0]
            total_sum = cdf[-1]
            denom = total_sum - cdf_min
            if denom > 1e-6:
                luts[r, c] = (cdf - cdf_min) / denom
            else:
                luts[r, c] = np.linspace(0.0, 1.0, n_bins, dtype=np.float32)

    return np.clip(luts, 0.0, 1.0)


def _to_grayscale_float(img: np.ndarray) -> np.ndarray:
    """Приводить вхідний 2D або 3D масив до 2D float32 у діапазоні [0.0, 255.0]."""
    arr = np.asarray(img, dtype=np.float32)
    if not np.issubdtype(np.asarray(img).dtype, np.integer) and np.max(arr) <= 1.0:
        arr = arr * 255.0
    if arr.ndim == 3:
        arr = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return np.clip(arr, 0.0, 255.0).astype(np.float32)


def _gaussian_blur_2d(gray: np.ndarray, kernel_size: int = 5, sigma: float = 1.4) -> np.ndarray:
    """Сепарабельне 2D гаусове згладжування за допомогою sliding_window_view."""
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be odd number >= 3")

    radius = kernel_size // 2
    sigma_val = max(float(sigma), 1e-3)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-0.5 * (x / sigma_val) ** 2)
    kernel_1d /= np.sum(kernel_1d)

    # Згортка по горизонталі, потім по вертикалі
    pad_h = np.pad(gray, ((0, 0), (radius, radius)), mode="reflect")
    blurred_h = sliding_window_view(pad_h, kernel_size, axis=1) @ kernel_1d

    pad_v = np.pad(blurred_h, ((radius, radius), (0, 0)), mode="reflect")
    blurred = sliding_window_view(pad_v, kernel_size, axis=0) @ kernel_1d
    return blurred.astype(np.float32)


def _sobel_gradients(smoothed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Обчислює магнітуду та кут градієнта [0, 180) за допомогою сепарабельного оператора Собеля."""
    s_smooth = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    s_diff = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    padded = np.pad(smoothed, ((1, 1), (1, 1)), mode="reflect")

    # Gx: згладжування по вертикалі, похідна по горизонталі
    gx_v = sliding_window_view(padded, 3, axis=0) @ s_smooth
    gx = sliding_window_view(gx_v, 3, axis=1) @ s_diff

    # Gy: похідна по вертикалі, згладжування по горизонталі
    gy_v = sliding_window_view(padded, 3, axis=0) @ s_diff
    gy = sliding_window_view(gy_v, 3, axis=1) @ s_smooth

    magnitude = np.hypot(gx, gy).astype(np.float32)
    angle = (np.rad2deg(np.arctan2(gy, gx)) % 180.0).astype(np.float32)
    return magnitude, angle


def canny_edge_detection(
    img: np.ndarray,
    sigma: float = 1.4,
    kernel_size: int = 5,
    low_threshold: float = 0.08,
    high_threshold: float = 0.20,
) -> np.ndarray:
    """
    Виділення контурів методом Кенні (Canny Edge Detection)
    1. Сепарабельний фільтр Гаусса.
    2. Оператор Собеля (магнітуда та напрямок градієнта).
    3. Векторизоване придушення немаксимумів (Non-Maximum Suppression).
    4. Подвійне порогування та трасування контурів гістерезисом.
    Повертає 2D масив uint8 зі значеннями 0 або 255.
    """
    gray = _to_grayscale_float(img)
    smoothed = _gaussian_blur_2d(gray, kernel_size=kernel_size, sigma=sigma)
    mag, angle = _sobel_gradients(smoothed)

    # Придушення немаксимумів (Non-Maximum Suppression) без циклів по пікселях
    mag_pad = np.pad(mag, 1, mode="constant", constant_values=0.0)

    east = mag_pad[1:-1, 2:]
    west = mag_pad[1:-1, :-2]
    south = mag_pad[2:, 1:-1]
    north = mag_pad[:-2, 1:-1]
    se = mag_pad[2:, 2:]
    nw = mag_pad[:-2, :-2]
    ne = mag_pad[:-2, 2:]
    sw = mag_pad[2:, :-2]

    dir_0 = (angle < 22.5) | (angle >= 157.5)
    dir_45 = (angle >= 22.5) & (angle < 67.5)
    dir_90 = (angle >= 67.5) & (angle < 112.5)
    dir_135 = (angle >= 112.5) & (angle < 157.5)

    n1 = np.select([dir_0, dir_45, dir_90, dir_135], [east, se, south, ne], default=0.0)
    n2 = np.select([dir_0, dir_45, dir_90, dir_135], [west, nw, north, sw], default=0.0)

    nms = np.where((mag >= n1) & (mag >= n2), mag, 0.0)

    max_mag = float(np.max(nms))
    if max_mag <= 1e-6:
        return np.zeros_like(gray, dtype=np.uint8)

    high_val = max_mag * high_threshold if high_threshold <= 1.0 else float(high_threshold)
    low_val = max_mag * low_threshold if low_threshold <= 1.0 else float(low_threshold)
    if low_val > high_val:
        low_val, high_val = high_val, low_val

    strong = nms >= high_val
    weak = (nms >= low_val) & (~strong)

    # Трасування контурів гістерезисом через векторизоване 8-сусіднє розширення
    edges = strong.copy()
    while True:
        p = np.pad(edges, 1, mode="constant", constant_values=False)
        dilated = (
            p[:-2, :-2] | p[:-2, 1:-1] | p[:-2, 2:]
            | p[1:-1, :-2] | p[1:-1, 1:-1] | p[1:-1, 2:]
            | p[2:, :-2] | p[2:, 1:-1] | p[2:, 2:]
        )
        next_edges = edges | (dilated & weak)
        if np.array_equal(next_edges, edges):
            break
        edges = next_edges

    return (edges.astype(np.uint8) * 255)

def compute_otsu_threshold(img: np.ndarray, n_bins: int = 256) -> float:
    """
    Обчислює оптимальний поріг бінаризації за методом Оцу (максимізація міжкласової дисперсії).
    Повертає значення порогу в діапазоні [0.0, 255.0].
    """
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")

    gray = _to_grayscale_float(img)
    scale = (n_bins - 1) / 255.0
    quantized = np.clip(np.rint(gray * scale).astype(np.int32), 0, n_bins - 1)

    hist = np.bincount(quantized.ravel(), minlength=n_bins).astype(np.float64)
    total = float(np.sum(hist))
    if total <= 0.0:
        return 127.5

    prob = hist / total
    bins = np.arange(n_bins, dtype=np.float64)

    omega0 = np.cumsum(prob)
    omega1 = 1.0 - omega0

    mu = np.cumsum(prob * bins)
    mu_t = mu[-1]

    denom = omega0 * omega1
    sigma_b_sq = np.zeros(n_bins, dtype=np.float64)
    valid = denom > 1e-12
    sigma_b_sq[valid] = ((mu_t * omega0[valid] - mu[valid]) ** 2) / denom[valid]

    max_var = float(np.max(sigma_b_sq))
    if max_var <= 0.0:
        best_bin = (n_bins - 1) / 2.0
    else:
        max_indices = np.flatnonzero(np.isclose(sigma_b_sq, max_var, rtol=1e-9))
        best_bin = float(np.mean(max_indices))

    return float(best_bin / scale)


def otsu_thresholding(img: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """
    Порогова сегментація зображення за методом Оцу.
    Повертає бінарну маску uint8 зі значеннями 0 та 255.
    """
    gray = _to_grayscale_float(img)
    threshold = compute_otsu_threshold(gray, n_bins=n_bins)
    return np.where(gray > threshold, 255, 0).astype(np.uint8)

def _build_structuring_element(kernel_size: int, shape: str = "rect") -> np.ndarray:
    """Формує бінарний структуруючий елемент ('rect', 'cross', 'disk') розміром kernel_size x kernel_size."""
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be odd number >= 3")

    radius = kernel_size // 2
    shape_norm = shape.lower()
    if shape_norm == "rect":
        return np.ones((kernel_size, kernel_size), dtype=bool)
    if shape_norm == "cross":
        se = np.zeros((kernel_size, kernel_size), dtype=bool)
        se[radius, :] = True
        se[:, radius] = True
        return se
    if shape_norm == "disk":
        y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
        return (x * x + y * y) <= (radius * radius)
    raise ValueError(f"Unknown structuring element shape: {shape}")


def erosion_filter(
    img: np.ndarray,
    kernel_size: int = 3,
    shape: str = "rect",
    iterations: int = 1,
) -> np.ndarray:
    """
    Морфологічна ерозія на базі NumPy (мінімум у межах структуруючого елемента).
    Підтримує як 2D (відтінки сірого / бінарні), так і 3D (поканальні) масиви.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be odd number >= 3")
    if iterations < 1:
        raise ValueError("Iterations must be >= 1")

    radius = kernel_size // 2
    se = _build_structuring_element(kernel_size, shape=shape)
    is_rect = shape.lower() == "rect"

    result = np.asarray(img).copy()
    is_2d = result.ndim == 2
    if is_2d:
        result = result[:, :, None]

    for _ in range(iterations):
        if is_rect:
            # Сепарабельний мінімум по рядках і стовпцях для прямокутного ядра
            pad_h = np.pad(result, ((0, 0), (radius, radius), (0, 0)), mode="edge")
            eroded_h = np.min(sliding_window_view(pad_h, kernel_size, axis=1), axis=-1)

            pad_v = np.pad(eroded_h, ((radius, radius), (0, 0), (0, 0)), mode="edge")
            result = np.min(sliding_window_view(pad_v, kernel_size, axis=0), axis=-1)
        else:
            padded = np.pad(result, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
            windows = sliding_window_view(padded, (kernel_size, kernel_size), axis=(0, 1))
            # windows має форму (H, W, C, kernel_size, kernel_size)
            result = np.min(windows[:, :, :, se], axis=-1)

    return result[:, :, 0] if is_2d else result
