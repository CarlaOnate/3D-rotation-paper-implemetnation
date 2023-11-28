import cv2
from utils.covariance_matrix import *
import matplotlib.pyplot as plt

def create_mask_from_img (image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    return mask


def draw_ellipse(image, image_ellipse, filled, color = (255, 0, 0)):
    axes, center, direction, _ = image_ellipse
    rotation_angle = int(np.degrees(np.arctan2(direction[0][1], direction[0][0])))
    ellipse_box = ((center[0], center[1]), (axes[0] * 2, axes[1] * 2), rotation_angle)   # ellipse axes, center and angle to be drawn on the img given
    if filled: cv2.ellipse(image, ellipse_box, color, -1)
    else: cv2.ellipse(image, ellipse_box, color, 2)


def draw_spheroid_on_view(img_path, spheroid_semi_axes, color=None, img_mat=None):
    if img_mat is not None: img = img_mat
    else: img = cv2.imread(img_path)
    mask = create_mask_from_img(img)
    view_ellipse = calculate_ellipse_from_mask(mask)
    _, center, direction, _ = view_ellipse
    rotation_angle = int(np.degrees(np.arctan2(direction[0][1], direction[0][0])))
    center = (int(center[0]), int(center[1]))
    spheroid_axes = (spheroid_semi_axes[0] * 2, spheroid_semi_axes[1] * 2)

    if color:
        draw_ellipse(img, (center, spheroid_axes, rotation_angle + 90), color)
    else:
        draw_ellipse(img, (center, spheroid_axes, rotation_angle + 90))

    return img


def draw_axes(img, img_ellipse):
    axes, center, direction, _ = img_ellipse
    a, b = axes
    direction_a, direction_b = direction

    end_point_a = (int(center[0] + a * direction_a[0]), int(center[1] + a * direction_a[1]))
    end_point_b = (int(center[0] + b * direction_b[0]), int(center[1] + b * direction_b[1]))

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_color = (255, 0, 0)  # Red color in BGR format
    thickness = 2

    major_letter = (
        int((center[0] + a * direction_a[0]) / 1.2),
        int((center[1] + a * direction_a[1]) / 1.2)
    )
    minor_letter = (
        int((center[0] + b * direction_b[0]) / 1.2),
        int((center[1] + b * direction_b[1]) / 1.2)
    )

    cv2.line(img, center, end_point_a, (0, 255, 0), 2)
    cv2.line(img, center, end_point_b, (0, 255, 0), 2)
    cv2.putText(img, "a", major_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(img, "b", minor_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)


def write_angle_on_img(img_to_draw, angle_text, org, color = (255, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 2
    cv2.putText(img_to_draw, str(angle_text), org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img_to_draw


def draw_circle(img, pixel_coord, color = (0, 0, 255)):
    radius = 5
    thickness = 7
    cv2.circle(img, (int(pixel_coord[0]), int(pixel_coord[1])), radius, color, thickness)

def get_spheroid_model (image, spheroid, image_name, path):  # Only works for sphere and oblate model, not prolate
    a = spheroid[0]  # Semi-major axis along x-axis
    b = spheroid[0]  # Semi-major axis along y-axis
    c = spheroid[1]  # Semi-major axis along z-axis

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    axs[0] = fig.add_subplot(121, projection='3d')
    axs[0].plot_surface(x, y, z)
    axs[0].view_init(elev=0, azim=0)
    axs[0].set_aspect('equal')

    axs[1].imshow(image)
    axs[1].set_title(image_name)
    plt.savefig(path)
    plt.close()


def create_z_coords_graph(angle_img, image_name, z_coordinates, path):
    # Create X and Y coordinates
    y = np.arange(0, angle_img.shape[0])
    x = np.arange(0, angle_img.shape[1])
    X, Y = np.meshgrid(x, y)

    # Create a figure with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Create a 3D surface plot on the first subplot
    axs[0] = fig.add_subplot(121, projection='3d')
    axs[0].set_xlim([0, angle_img.shape[0]])  # width
    axs[0].set_ylim([0, angle_img.shape[1]])

    axs[0].plot_surface(X, Y, z_coordinates)

    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')
    axs[0].set_zlabel('Z Value')

    # Display the image on the second subplot
    axs[1].imshow(angle_img)
    axs[1].set_title(image_name)
    plt.savefig(path)
    plt.close()