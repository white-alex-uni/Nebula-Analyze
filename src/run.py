from timer import function_timer

from src.proceed import image_2_rgb, rgb_2_image, rgb_2_gray
from src.methods import median_filter, huang_median_filter_oklab
from src.color_spaces import oklab_2_rgb, rgb_2_oklab

@function_timer(thread="basic median filter", decimals=3)
def run_basic_median(input_img):
    """
    Запускає базовий медіанний фільтр для вікна розміром 3.
    """
    rgb_arr = image_2_rgb(input_img)
    gray_arr = rgb_2_gray(rgb_arr)
    filtered_arr = median_filter(gray_arr)

    return rgb_2_image(filtered_arr)

@function_timer(thread="huang median filter", decimals=3)
def run_huang_median_filter(input_img):
    """
    Запускає медіанний фільтп з алгоритмом Хуанга для вікна розміром 3.
    """
    rgb_arr = image_2_rgb(input_img)
    oklab_img = rgb_2_oklab(rgb_arr)

    filtered_oklab = huang_median_filter_oklab(oklab_img)

    filtered_rgb = oklab_2_rgb(filtered_oklab)
    return rgb_2_image(filtered_rgb)