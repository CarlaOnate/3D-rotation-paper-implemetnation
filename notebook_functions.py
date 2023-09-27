import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def create_mask_from_img (image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    return mask

def calculate_ellipse_from_mask (mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Finds multiple contours that match the mask
    ellipse = None

    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)  # Choose the contour with the maximum area
        if len(contour) >= 5:                         # The fitEllipse functions needs at least 5 points to create an ellipse
            ellipse = cv2.fitEllipse(contour)

    return ellipse

def draw_ellipse(image, ellipse, color = (0, 0, 255)):
    thickness = 1
    center, axes_length, angle = ellipse

    ellipse_box = ((center[0], center[1]), (axes_length[0], axes_length[1]), angle)   # ellipse axes, center and angle to be drawn on the img given
    cv2.ellipse(image, ellipse_box, color, thickness)
    return image

def draw_semi_axes_of_ellipse (image, ellipse):
    center, axes_length, angle = ellipse
    color = (0, 255, 0)  # Green color
    thickness = 2
    degrees = np.deg2rad(angle)                         # Turn radians to degrees
    minor_axis_length, major_axis_length = axes_length  # Assigns vars major and minor axis -> refers to the full length of the longer and shorter axis of an ellipse
    center_coord_int = (int(center[0]), int(center[1]))

    # Calculate minor axis start and end coordinates
    minor_axis_endpoint = (
        int(center[0] - minor_axis_length / 2 * np.cos(degrees)),   # Coordinate X - calculated by subtracting from the center the length of the semi-axis (axis/2) - multiplying degrees to fit rotated ellipse
        int(center[1] - minor_axis_length / 2 * np.sin(degrees))    # Coordinate Y
    )
    # Calculate major axis start and end coordinates
    major_axis_endpoint = (
        int(center[0] - major_axis_length / 2 * np.cos(degrees + np.pi / 2)),
        int(center[1] - major_axis_length / 2 * np.sin(degrees + np.pi / 2))
    )

    cv2.line(image, center_coord_int, major_axis_endpoint, color, thickness)    # Paints line on img given two points (center, endpoint)
    cv2.line(image, center_coord_int, minor_axis_endpoint, color, thickness)
    return image


def draw_axes_names (image, ellipse):
    center, axes_length, angle = ellipse

    # Font values
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_color = (255, 0, 0)  # Red color in BGR format
    thickness = 2

    degrees = np.deg2rad(angle)
    minor_axis_length, major_axis_length = axes_length
    minor_letter = (
        int(center[0] - minor_axis_length / 4 * np.cos(degrees)), # Coordinate X - calculated by subtracting from the center half the length of the semi-axis (axis/4) - multiplying degrees to fit rotated ellipse
        int(center[1] - minor_axis_length / 4 * np.sin(degrees))  # Coordinate Y
    )
    major_letter = (
        int(center[0] - major_axis_length / 4 * np.cos(degrees + np.pi / 2)),
        int(center[1] - major_axis_length / 4 * np.sin(degrees + np.pi / 2))
    )

    cv2.putText(image, "a", major_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(image, "b", minor_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)
    return image


def draw_spheroid_on_view(img_path, spheroid_semi_axes, color=None, img_mat=None):
    if img_mat is not None: img = img_mat
    else: img = cv2.imread(img_path)
    mask = create_mask_from_img(img)
    view_ellipse = calculate_ellipse_from_mask(mask)
    center, _, rotation_angle = view_ellipse
    center = (int(center[0]), int(center[1]))
    spheroid_axes = (spheroid_semi_axes[0] * 2, spheroid_semi_axes[1] * 2)

    if color:
        draw_ellipse(img, (center, spheroid_axes, rotation_angle + 90), color)
    else:
        draw_ellipse(img, (center, spheroid_axes, rotation_angle + 90))

    return img

def table_for_views_w_spheroid(images, descriptions, size=(25, 25)):
    num_images = len(images)
    num_cols = 2  # Number of columns (1 for image, 1 for description)
    num_rows = (num_images + 1) // num_cols  # Calculate the number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=size)

    # Populate the subplots with images and descriptions
    for i, (img, desc) in enumerate(zip(images, descriptions)):
        row = i // num_cols
        col = i % num_cols

        # ax = axes[row, col]
        ax = axes[row, col] if num_rows > 1 else axes[col]

        ax.imshow(img)
        ax.axis('off')  # Hide axis
        ax.set_title(desc, fontsize=15)

    # Adjust spacing
    plt.tight_layout()

    # Show the table
    plt.show()


def generate_random_colors(num_colors):
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]
