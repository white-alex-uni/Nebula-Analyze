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