import numpy as np

def median_filter(img: np.ndarray, kernel_size: int) -> np.ndarray:
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

    ...