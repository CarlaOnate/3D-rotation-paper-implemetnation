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


def prev_post_values_b(current_b_index, b_list):
    b_axes = []
    # Calculate prev_b and post_b based on index
    if current_b_index == 0:  # Grab last 3 items
        b_axes = b_list[-3:]
    elif  current_b_index == 1:
        b_axes = [b_list[-1], *b_list[:current_b_index + 1]] # Grab last item and the first two
    elif current_b_index == len(b_list) - 1:
        b_axes = [*b_list[current_b_index - 2:], b_list[0]]  # Grab last two item and first one
    else:
        b_axes = b_list[current_b_index - 2 : current_b_index + 1]  # Grab prev item, current, last item
    return b_axes


def estimate_b_axe_trend(b_axes):
    mean_seq = mean(b_axes)
    if b_axes[0] < mean_seq < b_axes[-1]:
        return 1   # "Ascending"
    elif b_axes[0] > mean_seq > b_axes[-1]:
        return -1   # "Descending"
    else:
        return 0   # No clear trend


def angle_estimation(A, B, b_i, b_axes):
    fruit_rot = 'downwards'
    cos_theta = math.sqrt((pow(b_i, 2) - pow(B, 2))/(pow(A, 2) - pow(B, 2)))
    theta1 = np.arccos(cos_theta)
    theta1 = np.degrees(theta1)
    # theta2 = 360 - theta1
    theta2 = -np.arccos(cos_theta)
    theta2 = np.degrees(theta2)
    ascending_sequence = None
    b_axes_trend = estimate_b_axe_trend(b_axes)

    if b_axes_trend > 0:
        ascending_sequence = True
    elif b_axes_trend < 0:
        ascending_sequence = False

    print("\n\t b_axes:  ", b_axes)

    # if (b_axes[1] > b_axes[0]) and (fruit_rot == 'downwards') or (b_axes[0] > b_axes[1]) and (fruit_rot == 'upwards'):
    if ascending_sequence is None:
        print("NO CLEAR TREND, 2nd ambiguity??")
        theta = 999
    elif ascending_sequence and (fruit_rot == 'downwards') or (not ascending_sequence) and (fruit_rot == 'upwards'):
        theta = theta1
    else:
        theta = theta2
    return theta


def write_angle_on_img(img_to_draw, angle_text, color = (255, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    thickness = 2

    cv2.putText(img_to_draw, str(angle_text), org, font, fontScale, color, thickness, cv2.LINE_AA)
    return img_to_draw


def calculate_oblate(axes_a, axes_b):
    a = mean(axes_a)
    b = min(axes_b)
    return [a, b]


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
        oblate_angles = []
        oblate_spheroid = []
        oblate_edited_imgs = []

        for index, image_name in enumerate(image_files):
            # Get ellipse from view
            file_path = os.path.join(fruit_path, view_folder, image_name)
            img = cv2.imread(file_path)
            mask = pt311utils.create_mask_from_img(img)
            ellipse = pt311utils.calculate_ellipse_from_mask(mask)
            center, axes_length, angle = ellipse
            minor_axis_length, major_axis_length = axes_length
            axe_b_all_views.append(minor_axis_length / 2)
            axe_a_all_views.append(major_axis_length / 2)

        for index, image_name in enumerate(image_files):
            file_path = os.path.join(fruit_path, view_folder, image_name)
            img = cv2.imread(file_path)

            # Calculate oblate
            oblate_spheroid = calculate_oblate(axe_a_all_views, axe_b_all_views)

            # Calculate angle for view
            oblate_A = oblate_spheroid[0]
            oblate_B = oblate_spheroid[1]

            # Obtain prev b and post b
            b_values = prev_post_values_b(index, axe_b_all_views)

            # Estimate angle
            oblate_angle = angle_estimation(oblate_A, oblate_B, axe_b_all_views[index], b_values)

            # #Check 2nd ambiguity
            # if index > 0:
            #     oblate_angle = choose_smooth_angle(oblate_angles[index - 1], oblate_angle)
            #     print("OBLATE ANGLE AFTER CHECK ", oblate_angle, "\n")

            write_angle_on_img(img, oblate_angle, (0, 255, 0))  # (image, angle to write, color of the text)
            oblate_edited_imgs.append(img)

            # Store angle on list
            oblate_angles.append(oblate_angle)

        # Join img and store with name
        joined_oblate_images = join_images(oblate_edited_imgs)
        joined_img_name = "./" + fruit_name + "/" + view_folder + ".png"
        cv2.imwrite(joined_img_name, joined_oblate_images)


