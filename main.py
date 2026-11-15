from src.proceed import *
from src.methods import median_filter

output_path = r'.\output'

img, image_path = get_image()

rgb_img = image_2_rgb(img)

gray_img = rgb_2_gray(rgb_img)

filtered_img = median_filter(gray_img, kernel_size=3)

save_img = array_2_image(filtered_img)

save_image(save_img, original_path=image_path, suffix="median",output_path=output_path)