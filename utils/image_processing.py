import cv2
import numpy as np
from utils.draw_functions import *

def preprocess_img(image):
    green_channel = image[:, :, 1]  # Green channel is at index 1 (0 for blue, 2 for red)
    preprocessed_image = green_channel

    # Step 2: Reduce image resolution by downsampling
    downsampling_factor = 1
    downsampled_image = cv2.resize(preprocessed_image, None, fx=1/downsampling_factor, fy=1/downsampling_factor, interpolation=cv2.INTER_AREA)

    # Step 3: Apply high-pass filtering using Gaussian blur
    sigma = 1.25
    blurred_image = cv2.GaussianBlur(downsampled_image, (0, 0), sigma)
    high_pass_image = downsampled_image - blurred_image
    return high_pass_image


def define_l_points(image, ellipse):
    ellipse_mask = np.zeros_like(image)
    # Define red ellipse based on maximum expected rotation - todo: missing calculation with max exp rotation
    max_exp_rotation_ellipse_axes = [ellipse[0][0] / 1.2, ellipse[0][1] / 1.2]  # Change ellipse axes to remove outer parts of fruit
    max_exp_rotation_ellipse = [max_exp_rotation_ellipse_axes, *ellipse[1:]]
    draw_ellipse(ellipse_mask, max_exp_rotation_ellipse, True)   #True to show a filled ellipse
    selected_pixels = cv2.bitwise_and(image, image, mask=ellipse_mask)  # Points inside smaller ellipse
    # Choose pixels above 97th percentile
    percentile_97 = np.percentile(np.abs(selected_pixels), 97)
    mask_97 = cv2.inRange(selected_pixels, percentile_97, 255)
    return cv2.bitwise_and(selected_pixels, selected_pixels, mask=mask_97)  # Return mat obj with L points in white


