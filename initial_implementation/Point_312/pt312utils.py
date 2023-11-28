from Point_311 import pt311utils
import cv2
import matplotlib.pyplot as plt
import random

def draw_spheroid_on_view(img_path, spheroid_semi_axes, color=None, img_mat=None):
    if img_mat is not None: img = img_mat
    else: img = cv2.imread(img_path)
    mask = pt313utils.create_mask_from_img(img)
    view_ellipse = pt313utils.calculate_ellipse_from_mask(mask)
    center, _, rotation_angle = view_ellipse
    center = (int(center[0]), int(center[1]))
    spheroid_axes = (spheroid_semi_axes[0] * 2, spheroid_semi_axes[1] * 2)

    if color:
        pt313utils.draw_ellipse(img, (center, spheroid_axes, rotation_angle + 90), color)
    else:
        pt313utils.draw_ellipse(img, (center, spheroid_axes, rotation_angle + 90))

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

        ax = axes[row, col]

        ax.imshow(img)
        ax.axis('off')  # Hide axis
        ax.set_title(desc, fontsize=15)

    # Adjust spacing
    plt.tight_layout()

    # Show the table
    plt.show()


def generate_random_colors(num_colors):
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_colors)]
