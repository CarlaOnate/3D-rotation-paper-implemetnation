import cv2
import math
import numpy as np
import os
from statistics import mean
from Point_311 import pt311utils


def join_images(images):
    min_height = min([image.shape[0] for image in images])
    resized_imgs = [cv2.resize(image, (image.shape[1], min_height)) for image in images]
    joined_imgs = cv2.hconcat(resized_imgs)
    return joined_imgs

def prev_post_values_b(current_b_index, b_list, steps, direction = 0):
    enlarged_b_list = [*b_list, *b_list]
    enlarged_index = current_b_index + len(b_list) - 1 if current_b_index < (len(b_list) / 2) else current_b_index
    total_steps = 3 + steps

    if direction < 0:
        b_axes = enlarged_b_list[(enlarged_index - total_steps) : index]
    elif direction > 0:
        b_axes = enlarged_b_list[enlarged_index : (enlarged_index + total_steps)]
    else:
        b_axes = enlarged_b_list[enlarged_index - 1 : enlarged_index + 3]
    return b_axes


def estimate_b_axe_trend(b_axes):
    mean_seq = mean(b_axes)
    if b_axes[0] < mean_seq < b_axes[-1]:
        return True   # "Ascending"
    elif b_axes[0] > mean_seq > b_axes[-1]:
        return False  # "Descending"
    else:
        return None   # No clear trend


def angle_estimation(index_view):
    if oblate_angles[index_view] != -1: return oblate_angles[index_view]   # Return if already calculated in another cycle

    #Spheroid values
    A, B = oblate_spheroid
    b_curr = axe_b_all_views[index_view]
    fruit_rot = 'downwards'
    curr_cos_theta = math.sqrt((pow(b_curr, 2) - pow(B, 2))/(pow(A, 2) - pow(B, 2)))

    curr_theta_pos = np.degrees(np.arccos(curr_cos_theta))
    curr_theta_neg = np.degrees(-np.arccos(curr_cos_theta))
    current_view_angles = (curr_theta_pos, curr_theta_neg)

    rotation_per_view = 360 / len(axe_b_all_views) + 3
    correct_theta = lambda ascending_sequence: current_view_angles[0] if ascending_sequence and (fruit_rot == 'downwards') or (not ascending_sequence) and (fruit_rot == 'upwards') else current_view_angles[1]

    # Second ambiguity
    if 0 <= abs(current_view_angles[0]) <= rotation_per_view or abs(90 - rotation_per_view) <= abs(current_view_angles[0]) >= 0:   # Value calculated is in range or a local extrema
        if index_view < len(image_files) - 1:
            temp_correct_theta_value = correct_theta(get_trend_for_view(index_view, 0))
            oblate_angles[index_view] = temp_correct_theta_value
            next_view_angle = angle_estimation(index_view + 1)
            if next_view_angle in [0.0, 90.0, -90.0] or oblate_angles[index_view - 1] in [0.0, 90.0, -90.0]:
                theta = correct_theta(get_trend_for_view(index_view, 0))
            else:
                theta = choose_smooth_angle(oblate_angles[index_view - 1])
        else:
            theta = correct_theta(get_trend_for_view(index_view, 0))
    else:
        theta = correct_theta(get_trend_for_view(index_view, 0))

    oblate_angles[index_view] = theta
    return theta


def write_angle_on_img(img_to_draw, angle_text, color = (255, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1.1
    thickness = 2

    cv2.putText(img_to_draw, str(angle_text), org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img_to_draw


def calculate_oblate():
    a = mean(axe_a_all_views)
    b = min(axe_b_all_views)
    return [a, b]

def choose_smooth_angle(calc_angle):
    angles = [0.0, 90.0, -90.0]
    closest = min(angles, key=lambda x: abs(calc_angle - x))
    return closest


def get_trend_for_view(index_view, direction = 0, step = 0):
    b_axes = prev_post_values_b(index_view, axe_b_all_views, step, direction)  # Try first with direction 0, in other recursions does direction 1 - forward
    trend = estimate_b_axe_trend(b_axes)
    if trend is None:
        return get_trend_for_view(index_view, 1, step + 1)
    else:
        return trend


fruit_folders = ["../data/tomatoes/", "../data/mandarins/"]
fruit_names = ["Tomatoes", "Mandarins"]


for curr_fruit, fruit_path in enumerate(fruit_folders): # Cycle types of fruit folders
    fruit_name = fruit_names[curr_fruit]

    for view_folder in os.listdir(fruit_path):  # Cycle folders of mandarins, tomatoes
        image_files = []
        folder_path = os.path.join(fruit_path, view_folder)
        if os.path.isdir(folder_path): image_files = sorted([file for file in os.listdir(folder_path)])

        axe_a_all_views = []
        axe_b_all_views = []
        oblate_angles = [-1 for element in range(len(image_files))]
        oblate_spheroid = []
        oblate_edited_imgs = []

        for index, image_name in enumerate(image_files):
            file_path = os.path.join(fruit_path, view_folder, image_name)
            img = cv2.imread(file_path)
            mask = pt311utils.create_mask_from_img(img)
            ellipse = pt311utils.calculate_ellipse_from_mask(mask)
            center, axes_length, angle = ellipse
            minor_axis_length, major_axis_length = axes_length
            axe_b_all_views.append(minor_axis_length / 2)
            axe_a_all_views.append(major_axis_length / 2)

        # Calculate oblate from views
        A = mean(axe_a_all_views)
        B = min(axe_b_all_views)
        oblate_spheroid = [A, B]

        for index, image_name in enumerate(image_files):
            # Read current image
            file_path = os.path.join(fruit_path, view_folder, image_name)
            img = cv2.imread(file_path)
            img_axes = cv2.imread(file_path)

            # Estimate angle
            oblate_angle = angle_estimation(index)

            write_angle_on_img(img, oblate_angle, (0, 255, 0))  # (image, angle to write, color of the text)
            oblate_edited_imgs.append(img)

        # Join img and store with name
        joined_oblate_images = join_images(oblate_edited_imgs)
        joined_img_name = "./" + fruit_name + "/" + view_folder + ".png"
        cv2.imwrite(joined_img_name, joined_oblate_images)


