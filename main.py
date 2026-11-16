from src.proceed import get_image, save_image
from src.run import run_basic_median, run_huang_median_filter
output_path = r'.\output'

def main():
    img, img_path = get_image()

    med_basic_filtered_img = run_basic_median(img)
    med_huang_filtered_img = run_huang_median_filter(img)

    save_image(img=med_basic_filtered_img, original_path=img_path, output_path=output_path, suffix='_basic_median')
    save_image(img=med_huang_filtered_img, original_path=img_path, output_path=output_path, suffix='_huang_median')


if __name__ == "__main__":
    main()