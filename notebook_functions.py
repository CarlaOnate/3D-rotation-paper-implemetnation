import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from statistics import mean

# GENERAL ELLIPSE FUNCTIONS

def create_mask_from_img (image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    return mask

def calculate_ellipse_from_mask(binary_mask):
    mask = binary_mask.astype(np.int32)
    y_indices, x_indices = np.indices(mask.shape)
    positive_pixels = mask == 255
    result_x = x_indices[positive_pixels]
    result_y = y_indices[positive_pixels]
    n = len(result_x)

    sx = np.sum(result_x)
    cx = sx / len(result_x)
    sxx = np.sum(np.square(result_x))

    sy = np.sum(result_y)
    cy = sy / len(result_y)
    syy = np.sum(np.square(result_y))

    mult_list = [x * y for x, y in zip(result_x, result_y)]
    sxy = np.sum(mult_list)

    sigma_x2 = (sxx / n) - cx ** 2
    sigma_y2 = (syy / n) - cy ** 2
    sigma_xy = (sxy / n) - (cx * cy)

    cov_matrix = [[sigma_x2, sigma_xy], [sigma_xy, sigma_y2]]
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    direction_a = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
    direction_b = eigenvectors[1] / np.linalg.norm(eigenvectors[1])

    a = int(2 * (math.sqrt(abs(eigenvalues[0]))))  # semi major axis of projected ellipse
    b = int(2 * (math.sqrt(abs(eigenvalues[1]))))  # semi minor axis of projected ellipse

    return [a, b], [int(cx), int(cy)], [direction_a, direction_b] ,[*eigenvectors],

def draw_ellipse(image, ellipse, color = (0, 0, 255)):
    thickness = 1
    center, axes, rotation_angle = ellipse
    ellipse_box = ((center[0], center[1]), (axes[0], axes[1]), rotation_angle)   # ellipse axes, center and angle to be drawn on the img given
    cv2.ellipse(image, ellipse_box, color, thickness)
    return image

def draw_axes(img, img_ellipse):
    axes, center, direction, _ = img_ellipse
    a, b = axes
    direction_a, direction_b = direction

    print(a, b, center, direction_a, direction_b)

    end_point_a = (int(center[0] + a * direction_a[0]), int(center[1] + a * direction_a[1]))
    end_point_b = (int(center[0] + b * direction_b[0]), int(center[1] + b * direction_b[1]))

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    font_scale = 1
    font_color = (255, 0, 0)  # Red color in BGR format
    thickness = 2

    major_letter = (
        int((center[0] + a * direction_a[0])),
        int((center[1] + a * direction_a[1]))
    )
    minor_letter = (
        int((center[0] + b * direction_b[0])),
        int((center[1] + b * direction_b[1]))
    )

    cv2.line(img, center, end_point_a, (0, 255, 0), 2)
    cv2.line(img, center, end_point_b, (0, 255, 0), 2)
    cv2.putText(img, "a", major_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(img, "b", minor_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)

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

# def calculate_ellipse_from_mask (mask):
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Finds multiple contours that match the mask
#     ellipse = None
#
#     if len(contours) > 0:
#         contour = max(contours, key=cv2.contourArea)  # Choose the contour with the maximum area
#         if len(contour) >= 5:                         # The fitEllipse functions needs at least 5 points to create an ellipse
#             ellipse = cv2.fitEllipse(contour)
#
#     return ellipse




# def draw_semi_axes_of_ellipse (image, ellipse):
#     center, axes_length, angle = ellipse
#     color = (0, 255, 0)  # Green color
#     thickness = 2
#     degrees = np.deg2rad(angle)                         # Turn radians to degrees
#     minor_axis_length, major_axis_length = axes_length  # Assigns vars major and minor axis -> refers to the full length of the longer and shorter axis of an ellipse
#     center_coord_int = (int(center[0]), int(center[1]))
#
#     # Calculate minor axis start and end coordinates
#     minor_axis_endpoint = (
#         int(center[0] - minor_axis_length / 2 * np.cos(degrees)),   # Coordinate X - calculated by subtracting from the center the length of the semi-axis (axis/2) - multiplying degrees to fit rotated ellipse
#         int(center[1] - minor_axis_length / 2 * np.sin(degrees))    # Coordinate Y
#     )
#     # Calculate major axis start and end coordinates
#     major_axis_endpoint = (
#         int(center[0] - major_axis_length / 2 * np.cos(degrees + np.pi / 2)),
#         int(center[1] - major_axis_length / 2 * np.sin(degrees + np.pi / 2))
#     )
#
#     cv2.line(image, center_coord_int, major_axis_endpoint, color, thickness)    # Paints line on img given two points (center, endpoint)
#     cv2.line(image, center_coord_int, minor_axis_endpoint, color, thickness)
#     return image


# def draw_axes_names (image, ellipse):
#     center, axes_length, angle = ellipse
#
#     # Font values
#     font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#     font_scale = 1
#     font_color = (255, 0, 0)  # Red color in BGR format
#     thickness = 2
#
#     degrees = np.deg2rad(angle)
#     minor_axis_length, major_axis_length = axes_length
#     minor_letter = (
#         int(center[0] - minor_axis_length / 4 * np.cos(degrees)), # Coordinate X - calculated by subtracting from the center half the length of the semi-axis (axis/4) - multiplying degrees to fit rotated ellipse
#         int(center[1] - minor_axis_length / 4 * np.sin(degrees))  # Coordinate Y
#     )
#     major_letter = (
#         int(center[0] - major_axis_length / 4 * np.cos(degrees + np.pi / 2)),
#         int(center[1] - major_axis_length / 4 * np.sin(degrees + np.pi / 2))
#     )
#
#     cv2.putText(image, "a", major_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)
#     cv2.putText(image, "b", minor_letter, font, font_scale, font_color, thickness, cv2.LINE_AA)
#     return image


# POINT 3.1.3 ANGLE ESTIMATION

# CALCULATE THE NEXT ANGLE IF NEEDED
def angle_estimation(index_view, oblate_angles, oblate_spheroid, axe_b_all_views):
    if oblate_angles[index_view] != -1: return oblate_angles[index_view]   # Return if already calculated in another cycle

    #Spheroid values
    A, B = oblate_spheroid
    b_curr = axe_b_all_views[index_view]
    fruit_rot = 'downwards'
    curr_cos_theta = math.sqrt((pow(b_curr, 2) - pow(B, 2))/(pow(A, 2) - pow(B, 2)))

    curr_theta_pos = np.degrees(np.arccos(curr_cos_theta))
    curr_theta_neg = np.degrees(-np.arccos(curr_cos_theta))
    current_view_angles = (curr_theta_pos, curr_theta_neg)

    rotation_per_view = 360 / len(axe_b_all_views) - 5
    correct_theta = lambda ascending_sequence: current_view_angles[0] if ascending_sequence and (fruit_rot == 'downwards') or (not ascending_sequence) and (fruit_rot == 'upwards') else current_view_angles[1]

    # Second ambiguity
    if 0 <= abs(current_view_angles[0]) <= rotation_per_view or abs(90 - rotation_per_view) <= abs(current_view_angles[0]) >= 0:   # Value calculated is in range or a local extrema
        if index_view < len(axe_b_all_views) - 1:
            temp_correct_theta_value = correct_theta(get_trend_for_view(index_view, axe_b_all_views, 0))
            oblate_angles[index_view] = temp_correct_theta_value
            next_view_angle = angle_estimation(index_view + 1, oblate_angles, oblate_spheroid, axe_b_all_views)
            if next_view_angle in [0.0, 90.0, -90.0] or oblate_angles[index_view - 1] in [0.0, 90.0, -90.0]:
                theta = correct_theta(get_trend_for_view(index_view, axe_b_all_views, 0))
            else:
                theta = choose_smooth_angle(oblate_angles[index_view - 1])
        else:
            theta = correct_theta(get_trend_for_view(index_view, axe_b_all_views, 0))
    else:
        theta = correct_theta(get_trend_for_view(index_view, axe_b_all_views, 0))

    oblate_angles[index_view] = theta
    return theta

def get_trend_for_view(index_view, axe_b_all_views, direction = 0, step = 0):
    b_axes = prev_post_values_b(index_view, axe_b_all_views, step, direction)  # Try first with direction 0, in other recursions does direction 1 - forward
    trend = estimate_b_axe_trend(b_axes)
    if trend is None:
        return get_trend_for_view(index_view, axe_b_all_views, 1, step + 1)
    else:
        return trend

def estimate_b_axe_trend(b_axes):
    mean_seq = mean(b_axes)
    if b_axes[0] < mean_seq < b_axes[-1]:
        return True   # "Ascending"
    elif b_axes[0] > mean_seq > b_axes[-1]:
        return False  # "Descending"
    else:
        return None   # No clear trend

def prev_post_values_b(current_b_index, b_list, steps, direction = 0):
    enlarged_b_list = [*b_list, *b_list]
    enlarged_index = current_b_index + len(b_list) - 1 if current_b_index < (len(b_list) / 2) else current_b_index
    total_steps = 3 + steps

    if direction < 0:
        b_axes = enlarged_b_list[(enlarged_index - total_steps) : enlarged_index]
    elif direction > 0:
        b_axes = enlarged_b_list[enlarged_index : (enlarged_index + total_steps)]
    else:
        b_axes = enlarged_b_list[enlarged_index - 2 : enlarged_index + 2]
    return b_axes

def choose_smooth_angle(calc_angle):
    diff_0 = abs(calc_angle - 0)
    diff_90 = abs(calc_angle - 90)
    diff_minus_90 = abs(calc_angle + 90)

    if diff_0 < diff_90 and diff_0 < diff_minus_90 and diff_0 < 40:  # Lower than 40 degrees of difference
        return 0
    elif diff_90 < diff_0 and diff_90 < diff_minus_90:
        return 90
    else:
        return -90

def join_images(images):
    min_height = min([image.shape[0] for image in images])
    resized_imgs = [cv2.resize(image, (image.shape[1], min_height)) for image in images]
    joined_imgs = cv2.hconcat(resized_imgs)
    return joined_imgs

def write_angle_on_img(img_to_draw, angle_text, color = (255, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 0.5
    thickness = 2

    cv2.putText(img_to_draw, str(angle_text), org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img_to_draw