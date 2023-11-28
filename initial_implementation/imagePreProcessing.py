# This code implements section 3.1.1 of the paper: 3D rotation of fruits
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

image = cv2.imread("./data/oranges/obj0001/im00.png")

# Step 1: Reduce the number of color channels by keeping the green channel
green_channel = image[:, :, 1]  # Green channel is at index 1 (0 for blue, 2 for red)
preprocessed_image = green_channel

# Step 2: Reduce image resolution by downsampling
downsampling_factor = 4
downsampled_image = cv2.resize(preprocessed_image, None, fx=1/downsampling_factor, fy=1/downsampling_factor, interpolation=cv2.INTER_AREA)

# Step 3: Apply high-pass filtering using Gaussian blur
sigma = 1.25
blurred_image = cv2.GaussianBlur(downsampled_image, (0, 0), sigma)
high_pass_image = downsampled_image - blurred_image

plt.imshow(high_pass_image)