from utils.image import (
    gaussian_blur,
    read_image,
    write_image,
    resize_image,
    sobel_filter,
    image_to_binary
)
from utils.color_space import rgb_to_oklab, oklab_to_rgb, rgb_to_oklch
from utils.blob import blobize
import time

def main():
    image = read_image("frame1.jpg")
    image = resize_image(image, 100, 75)
    blurred_image = gaussian_blur(image, 5)
    oklch_image = rgb_to_oklch(blurred_image)
    oklc_image = oklch_image[..., :2]
    oklc_image[..., 1] = oklc_image[..., 1] / 0.37 # Giving a gain for Chroma
    edge_image = sobel_filter(oklc_image, 'magnitude')
    edge_image = image_to_binary(edge_image, 0.1)
    write_image("edge_image.png", edge_image)
    blob_images = blobize(image, edge_image)
    for i, blob_image in enumerate(blob_images):
        write_image(f"blob_{i}.png", blob_image)
if __name__ == "__main__":
    main()
