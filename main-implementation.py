from utils.image_processing import *
from utils.fruit_angle import *
from utils.z_coords import *
from utils.rotation_matrix import *
from utils.draw_functions import *
import angles_data.mandarins
import angles_data.tomatoes
import cv2
import os
import timeit

def projected_ellipse_axes(args):
    # POINT 3.1.1
    for image_name in args["image_files"]:
        file_path = os.path.join(args["path"], image_name)
        color_img = cv2.imread(file_path)
        img = preprocess_img(color_img)
        _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        ellipse = calculate_ellipse_from_mask(mask)
        axes_length, _, _, _ = ellipse
        major_axis_length, minor_axis_length = axes_length
        args["axe_b_all_views"].append(minor_axis_length)
        args["axe_a_all_views"].append(major_axis_length)
        args["processed_imgs"].append(img)
        args["color_imgs"].append(color_img)
        args["projected_ellipses"].append(ellipse)
        axes_img = color_img.copy() # Comment this line to measure execution time properly
        draw_axes(axes_img, ellipse) # Comment this line to measure execution time properly
        cv2.imwrite("./results/" + args["fruit_type_path"] + "/projected-ellipses-views/" + image_name, axes_img) # Comment this line to measure execution time properly
        # SAVE PREPROCESSED IMG OF VIEW
        cv2.imwrite("./results/" + args["fruit_type_path"] + "/pre-processed-views/" + image_name, img) # Comment this line to measure execution time properly

    joined_points = join_images(args["color_imgs"])
    cv2.imwrite("./results/" + args["fruit_type_path"] + "/tracking-points.png", joined_points)

def spheroid_model(args):
    # POINT 3.1.2
    args["spheroid"] = calculate_spheroid(args["axe_a_all_views"], args["axe_b_all_views"], args["type_spheroid"])
    get_spheroid_model(args["color_imgs"][0], args["spheroid"], "im00", "./results/" + args["fruit_type_path"] + "/spheroid/spheroid-model-im000") # Comment this line to measure execution time properly

def fruits_angle_estimation(args):
    # POINT 3.1.3
    estimated_angles = [-1 for _ in range(len(args["image_files"]))]
    estimated_angles_images = []
    for index, angle in enumerate(args["angles"]):
        # INCORRECT ANGLE ESTIMATION FUNCTION
        # estimated_angle_img = args["color_imgs"][index].copy()
        # estimated_angle = angle_estimation(index, estimated_angles, args["axe_b_all_views"], args["spheroid"])
        # write_angle_on_img(estimated_angle_img, estimated_angle, (50, 50), (0, 255, 0))
        # estimated_angles_images.append(estimated_angle_img)

        # Manually added angles
        angle_img = args["color_imgs"][index].copy()
        write_angle_on_img(angle_img, angle, (50, 50), (0, 255, 0))
        args["angle_imgs"].append(angle_img)

        # DRAW SPHEROID ON VIEWS
        ellipse_img = args["color_imgs"][index].copy() # Comment this line to measure execution time properly
        spheroid_ellipse = [args["spheroid"], *args["projected_ellipses"][index][1:]] # Comment this line to measure execution time properly
        draw_ellipse(ellipse_img, spheroid_ellipse, False) # Comment this line to measure execution time properly
        cv2.imwrite("./results/" + args["fruit_type_path"] + "/spheroid/drawn-ellipse-" + str(index) + ".png", ellipse_img) # Comment this line to measure execution time properly

    # SAVE ESTIMATED ANGLES IMAGE
    # estimated_angles_joined_image = join_images(estimated_angles_images) # Comment this line to measure execution time properly
    # cv2.imwrite("./results/" + args["fruit_type_path"] + "/estimated-angles.png", estimated_angles_joined_image) # Comment this line to measure execution time properly


def calculate_estimated_rotation(args):
    processed_imgs = args["processed_imgs"]
    projected_ellipses = args["projected_ellipses"]
    spheroid = args["spheroid"]
    color_imgs = args["color_imgs"]
    point_tracking_imgs = args["point_tracking_imgs"]
    fruit_type_path = args["fruit_type_path"]
    angle_images = args["angle_imgs"]
    angles = args["angles"]

    # POINT 3.1.2
    ps = []
    for index, processed_img in enumerate(processed_imgs):
        if index < len(processed_imgs) - 1:
            print("Running image ", index)
            curr_angle = angles[index] if angles else 0
            curr_angle_img = angle_images[index] if angles else color_imgs[index]

            # Obtain selected points and calculate z coordiantes
            l_points = define_l_points(processed_img, projected_ellipses[index])
            l_points_indices = np.argwhere(l_points == 255)
            cv2.imwrite("./results/"  + fruit_type_path + "/l-points-img/" + "im" + str(index) + ".png", cv2.cvtColor(l_points, cv2.COLOR_BGR2RGB)) # Comment this line to measure execution time properly

            # Z coords for l-points - used to obtain results, not in final solution
            z_coordinates = np.zeros_like(processed_img)  # like source img
            calculate_z_coordinates(l_points_indices, curr_angle, projected_ellipses[index], spheroid, z_coordinates, args["type_spheroid"]) # Comment this line to measure execution time properly
            create_z_coords_graph(curr_angle_img, "im" + str(index), z_coordinates, "./results/" + args["fruit_type_path"] + "/z-graph-l-points/im" + str(index)) # Comment this line to measure execution time properly

            # Z coords for whole image - used in final solution
            calculate_z_coordinates_image(processed_img, curr_angle, projected_ellipses[index], spheroid, z_coordinates, args["type_spheroid"])
            create_z_coords_graph(curr_angle_img, "im" + str(index), z_coordinates,  "./results/" + args["fruit_type_path"] + "/z-graph/im" + str(index)) # Comment this line to measure execution time properly

            # Estimate initial rotation
            max_exp_rotation = -30
            views_ellipses = [projected_ellipses[index], projected_ellipses[index + 1]]

            estimated_rotation_matrix, estimated_rotation_error = find_best_rotation(max_exp_rotation, processed_imgs[index], processed_imgs[index + 1], l_points_indices, z_coordinates, views_ellipses)

            # Track points from source img to target using estimate rotation
            source_img = color_imgs[index].copy()
            target_img = color_imgs[index + 1].copy()

            if index == 0:
                _, center, _, _ = projected_ellipses[index]
                ps = np.array([*center, z_coordinates[center[1], center[0]]])
                draw_circle(source_img, ps, (255, 255, 0))
                point_tracking_imgs.append(source_img)

            pt = estimated_rotation_matrix @ ps
            if pt[2] < 0: # not visible in the next view
                draw_circle(target_img, pt, (0, 255, 0))
            else:
                draw_circle(target_img, pt, (255, 255, 0))
            point_tracking_imgs.append(target_img)
            ps = pt

    joined_points = join_images(point_tracking_imgs)
    # plt.imshow(joined_points)
    cv2.imwrite("./results/" + fruit_type_path + "/tracking-points.png", joined_points)

# IMPLEMENTATION
def main_implementation(path, results_path, angles = None):
    print("main for:  ", path)

    args = {
        "angles": angles,
        "path": path,
        "image_files": sorted([file for file in os.listdir(path)]),
        "axe_b_all_views" : [],
        "axe_a_all_views" : [],
        "projected_ellipses" : [],
        "oblate_spheroid" : [],
        "color_imgs" : [],
        "processed_imgs" : [],
        "angle_imgs" : [],
        "ellipses_imgs" : [],
        "correct_angle_imgs" : [],
        "estimated_angle_imgs" : [],
        "point_tracking_imgs" : [],
        "spheroid" : None,
        "fruit_type_path": results_path, # "mandarins/obj0014",
        "type_spheroid": 2 if angles else 1  # Oblate - 2, Sphere - 1
    }

    projected_ellipse_axes(args)
    spheroid_model(args)
    if angles: fruits_angle_estimation(args)
    calculate_estimated_rotation(args)

    # UNCOMMENT TO MEASURE EXECUTION TIME
    # axes_time = timeit.timeit(lambda: projected_ellipse_axes(args), number=1)
    # spheroid_time = timeit.timeit(lambda: spheroid_model(args), number=1)
    # fruits_angle = timeit.timeit(lambda: fruits_angle_estimation(args), number=1)
    # estimated_rotation_time = timeit.timeit(lambda: calculate_estimated_rotation(args), number=1)

    # print("AXES TIME", axes_time)
    # print("SPHEROID TIME", spheroid_time)
    # print("FRUITS ANGLE ELEVATION", fruits_angle)
    # print("ESTIMATED ROTATION TIME", estimated_rotation_time)

def run_complete_dataset():
    data_path = "./data/"
    # fruit_folders = os.listdir(data_path)
    fruit_folders = ["oranges", "mandarins"]
    for fruit_path in fruit_folders:  # Cycle types of fruit folders
        if os.path.isdir(data_path + fruit_path + "/"):
            for fruit_obj in os.listdir(data_path + fruit_path + "/"):  # Cycle folders of mandarins, tomatoes
                folder_path = os.path.join(data_path + fruit_path, fruit_obj + "/")

                if not os.path.isdir(folder_path): continue

                angles = None
                if fruit_path == "mandarins":
                    obj_num = int(fruit_obj[3:])  # Extract the numeric part from the folder name
                    attr_name = f"obj{obj_num:04d}"  # Reconstruct the variable name
                    angles = getattr(angles_data.mandarins, attr_name, None)
                elif fruit_path == "tomatoes":
                    obj_num = int(fruit_obj[3:])  # Extract the numeric part from the folder name
                    attr_name = f"obj{obj_num:04d}"  # Reconstruct the variable name
                    angles = getattr(angles_data.tomatoes, attr_name, None)

                results_path = fruit_path + "/" + fruit_obj
                path = folder_path
                main_implementation(path, results_path, angles)


# --------- UNCOMMENT TO RUN SINGLE FRUIT ------------
path = "./data/tomatoes/obj0006"
angles = angles_data.tomatoes.obj0006
results_path = "tomatoes/obj0006"
main_implementation(path, results_path, angles)
# UNCOMMENT TO MEASURE EXECUTION TIME
# total_time = timeit.timeit(lambda: main_implementation(path, results_path, angles), number=1)
# print("TOTAL_TIME", total_time)

# ---------- RUN ENTIRE DATASET ------------
# run_complete_dataset()