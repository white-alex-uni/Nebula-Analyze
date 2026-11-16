from timer import Timer

from src.proceed import image_2_rgb, rgb_2_image, rgb_2_gray
from src.methods import median_filter, huang_median_filter_oklab, clahe_channel_1d
from src.color_spaces import oklab_2_rgb, rgb_2_oklab

timer = Timer()

def run_basic_median(input_img):
    """
    Запускає базовий медіанний фільтр для вікна розміром 3.
    """
    rgb_arr = image_2_rgb(input_img)
    gray_arr = rgb_2_gray(rgb_arr)

    timer.start(thread="basic_median")
    filtered_arr = median_filter(gray_arr)
    timer.stop(thread="basic_median")

    return rgb_2_image(filtered_arr)

def run_huang_median_filter(input_img):
    """
    Запускає медіанний фільтп з алгоритмом Хуанга для вікна розміром 3.
    """
    rgb_arr = image_2_rgb(input_img)
    oklab_img = rgb_2_oklab(rgb_arr)

    timer.start(thread="huang_median")
    filtered_oklab = huang_median_filter_oklab(oklab_img)
    timer.stop(thread="huang_median")

    filtered_rgb = oklab_2_rgb(filtered_oklab)
    return rgb_2_image(filtered_rgb)

def run_clahe(input_img):
    rgb_arr = image_2_rgb(input_img)
    oklab_img = rgb_2_oklab(rgb_arr)

    filtered_oklab = oklab_img.copy()
    L_arr = oklab_img[:, : , 0]
    L_arr_enhanced = clahe_channel_1d(L=L_arr)
    filtered_oklab[:, : , 0] = L_arr_enhanced

    filtered_rgb = oklab_2_rgb(filtered_oklab)
    return rgb_2_image(filtered_rgb)