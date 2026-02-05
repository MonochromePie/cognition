from utils.image import (
    gaussian_blur,
    read_image,
    write_image,
    resize_image,
    sobel_filter,
)
from utils.color_space import rgb_to_oklch
from utils.blob import blobize, group_blobs
import time

def find_gradient(image):
    blurred_image = gaussian_blur(image, 5)
    oklch_image = rgb_to_oklch(blurred_image)
    oklc_image = oklch_image[..., :2]
    oklc_image[..., 1] = oklc_image[..., 1] / 0.37 # Giving a gain for Chroma
    gradient_image = sobel_filter(oklc_image, 'gradient')
    return gradient_image

def main():
    start_time = time.time()
    # Image 1
    image1 = read_image("frame1.jpg")
    image1 = resize_image(image1, 100, 75)
    gradient_image1 = find_gradient(image1)
    write_image("outputs/gradient_image1.png", gradient_image1)
    blobs1 = blobize(image1, gradient_image1)
    
    # Image 2
    image2 = read_image("frame2.jpg")
    image2 = resize_image(image2, 100, 75)
    gradient_image2 = find_gradient(image2)
    blobs2 = blobize(image2, gradient_image2)
    groups = group_blobs(blobs1, blobs2)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    for i, group in enumerate(groups):
        write_image(f"outputs/group_{i}.png", group[0])
        write_image(f"outputs/group_{i}_pair.png", group[1])
if __name__ == "__main__":
    main()
