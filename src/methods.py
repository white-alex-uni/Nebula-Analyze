import numpy as np

def median_filter(img: np.ndarray, width: int, height: int, kernel_size: int) -> np.ndarray:
    """
    Базовий медіанний фільтр зображення для відтінков сірого.
    """
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError('Kernel size must be odd number')

    output_img = np.zeros((height, width, 3), np.uint8)
    radius = kernel_size // 2
    window_size = kernel_size * kernel_size
    median_index = window_size // 2

    for h in range(height):
        for w in range(width):
            window = np.ndarray(window_size)
            count = 0

            for closest_h in range(-radius, radius + 1):
                for closest_w in range(-radius, radius + 1):
                    sample_h = np.clip(h + closest_h, 0, height-1)
                    sample_w = np.clip(w + closest_w, 0, width-1)

                    window[count] = img[sample_h][sample_w]
                    count += 1

            window.sort(descending=False)
            output_img[h][w] = window[median_index]

    return output_img
