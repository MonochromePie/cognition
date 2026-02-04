from utils.image import read_image, write_image, resize_image, convolution, create_gaussian_kernel
from utils.color_space import rgb_to_oklab, oklab_to_rgb
import time

def main():
    image = read_image("frame1.jpg")
    blurred_image = resize_image(image, 100, 75)
    oklab_image = rgb_to_oklab(blurred_image)
    lightness = oklab_image[..., 0]
    write_image("oklab_image.jpg", lightness*255)
    rgb_image = oklab_to_rgb(oklab_image)
    write_image("rgb_image.jpg", rgb_image)
    # gaussian_kernel = create_gaussian_kernel(3, 1)
    # for _ in range(10):
    #     blurred_image = convolution(blurred_image, gaussian_kernel)
    # write_image("blurred_image.jpg", blurred_image)

if __name__ == "__main__":
    main()
