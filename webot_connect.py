import os
import sys
import numpy as np
import cv2
from dotenv import load_dotenv

from utils.image import (
    gaussian_blur,
    image_to_binary,
    read_image,
    write_image,
    resize_image,
    sobel_filter,
)
from utils.color_space import rgb_to_oklch
from utils.blob import blobize, filter_blobs_by_pixel_count, group_blobs
import time

def find_gradient(image):
    blurred_image = gaussian_blur(image, 5)
    oklch_image = rgb_to_oklch(blurred_image)
    oklc_image = oklch_image[..., :2]
    oklc_image[..., 1] = oklc_image[..., 1] / 0.37 # Giving a gain for Chroma
    gradient_image = sobel_filter(oklc_image, 'gradient')
    return gradient_image

load_dotenv()

webots_home = os.getenv("WEBOTS_HOME")
if not webots_home:
    raise ValueError("WEBOTS_HOME not found in .env file!")

sys.path.append(os.path.join(webots_home, 'lib', 'controller', 'python'))
from controller import Robot

def get_image(robot, width, height):
    camera = robot.getDevice("cam")
    raw_image = camera.getImage()
    if raw_image:
        image_array = np.frombuffer(raw_image, np.uint8).reshape((height, width, 4))
        # Drop the Alpha channel
        image_array = image_array[:, :, :3]
        return image_array
    return None

def run_robot():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    previous_image = None

    camera = robot.getDevice("cam")
    camera.enable(timestep)
    
    width = camera.getWidth()
    height = camera.getHeight()

    print("Camera initialized. Press 'q' in the OpenCV window to exit.")

    while robot.step(timestep) != -1:
        image_array = get_image(robot, width, height)
        if image_array is None:
            continue
        # Flip the image vertically
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        if previous_image is None:
            # No previous image, so we can't compare
            previous_image = image_rgb
            continue
        gradient_image1 = find_gradient(previous_image)
        blobs1 = blobize(previous_image, gradient_image1, 0.3)
        blobs1 = filter_blobs_by_pixel_count(blobs1, 25)
        
        # Image 2
        gradient_image2 = find_gradient(image_rgb)
        blobs2 = blobize(image_rgb, gradient_image2, 0.3)
        blobs2 = filter_blobs_by_pixel_count(blobs2, 25)
        groups = group_blobs(blobs1, blobs2)
        
        # Remove images from outputs folder
        for file in os.listdir("outputs"):
            os.remove(os.path.join("outputs", file))
        for i, group in enumerate(groups):
            write_image(f"outputs/group_{i}.png", group[0])
            write_image(f"outputs/group_{i}_pair.png", group[1])    
            
        previous_image = image_rgb
        cv2.imshow("Robot Vision", image_array)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_robot()