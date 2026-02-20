from email import utils
from utils import optical_flow
from utils.image import (
    gaussian_blur,
    read_image,
    write_image,
    resize_image,
    sobel_filter,
)
from utils.color_space import rgb_to_oklch
from utils.blob import blobize, group_blobs
from utils.optical_flow import optical_flow_vector, optical_flow_magnitude, plot_optical_flow, optical_flow_pyramid
import time
import numpy as np

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
    for i, group in enumerate(groups):
        write_image(f"outputs/group_{i}.png", group[0])
        write_image(f"outputs/group_{i}_pair.png", group[1])


    test1 = read_image("test\small_step_1.jpg")
    test1 = resize_image(test1, 100, 75)

    test2 = read_image("test\small_step_2.jpg")
    test2 = resize_image(test2, 100, 75)

    print(False in (test1 == test2))

    #optical flow
    vector_field = optical_flow_pyramid(test1, test2, levels=3, initial_kernel_size=3, blur_iterations=0, eig_thresh=1e-4)

    vector_field_direction = optical_flow_vector( vector_field)
    magnitude = optical_flow_magnitude(vector_field)

    #stop time
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

    #plot optical flow magnitude
    print(f"Max magnitude location: {np.where(magnitude == np.max(magnitude))}")
    print(f"Optical Flow Vector Field Direction: {vector_field_direction}")
    write_image("outputs/test_1_re.png", gaussian_blur(test1))
    write_image("outputs/test_2_re.png", gaussian_blur(test2))
    plot_optical_flow(vector_field, step=1, background=test1)
    
if __name__ == "__main__":
    main()
