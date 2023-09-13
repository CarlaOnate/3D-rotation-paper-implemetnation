
# This code implements section 3.1.1 of the paper: 3D rotation of fruits
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

image = cv2.imread("./data/oranges/obj0001/im00.png")
plt.imshow(image)

# Pre Processing image -------------------------------
green_channel = image[:, :, 1]  # Green channel is at index 1 (0 for blue, 2 for red)
preprocessed_image = green_channel

downsampling_factor = 4
downsampled_image = cv2.resize(preprocessed_image, None, fx=1/downsampling_factor, fy=1/downsampling_factor, interpolation=cv2.INTER_AREA)

sigma = 1.25
blurred_image = cv2.GaussianBlur(downsampled_image, (0, 0), sigma)
high_pass_image = downsampled_image - blurred_image
plt.imshow(high_pass_image)


# Creating binary mask with threshold -------------------------------
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
unique_values = np.unique(mask)
print("UNIQUE VALUES MASK", unique_values)
plt.imshow(mask)


# Calculate covariance matrix -------------------------------
mean = np.mean(mask, axis=1)
cov_matrix = np.zeros((mask.shape[1], mask.shape[1]), dtype=np.float64)
flags = cv2.COVAR_NORMAL | cv2.COVAR_ROWS

cv2.calcCovarMatrix(samples=mask, mean=mean, covar=cov_matrix, flags=flags)
print(cov_matrix.shape)


# Getting the eigenvectos -------------------------------
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_eigenvalues = np.sort(eigenvalues)[::-1]
print(sorted_eigenvalues[0], sorted_eigenvalues[-1])






