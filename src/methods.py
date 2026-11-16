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

    # Квантування L [0, 1] в індекси бінів [0, n_bins - 1]
    bins = np.clip((L_pad * (n_bins - 1)).astype(np.int32), 0, n_bins - 1)

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
    return L_out[:orig_h, :orig_w]

def _compute_tile_luts(tiles: np.ndarray, n_bins: int, clip_limit: float) -> np.ndarray:
    """
    Обчислює таблиці перекодування (CDF LUT) для кожного тайлу з обрізанням контрасту.
    """
    grid_h, grid_w, tile_h, tile_w = tiles.shape
    n_pixels = tile_h * tile_w

    # Абсолютний поріг обрізання у пікселях
    max_count = max(1, int(clip_limit * (n_pixels / n_bins)))
    luts = np.zeros((grid_h, grid_w, n_bins), dtype=np.float32)

    for r in range(grid_h):
        for c in range(grid_w):
            hist = np.bincount(tiles[r, c].ravel(), minlength=n_bins).astype(np.float32)

            # Обрізання піків (Clip Limit) та рівномірний перерозподіл надлишку
            excess = np.maximum(hist - max_count, 0.0)
            total_excess = np.sum(excess)
            hist = np.minimum(hist, max_count)
            hist += total_excess / n_bins

            # Кумулятивна функція розподілу (CDF), нормалізована у [0.0, 1.0]
            cdf = np.cumsum(hist)
            cdf_min = cdf[0]
            denom = n_pixels - cdf_min
            if denom > 1e-6:
                luts[r, c] = (cdf - cdf_min) / denom
            else:
                luts[r, c] = np.linspace(0.0, 1.0, n_bins)

    return np.clip(luts, 0.0, 1.0)