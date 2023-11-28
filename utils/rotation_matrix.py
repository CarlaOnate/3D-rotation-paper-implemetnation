import cv2
import numpy as np

def find_best_rotation_image(max_exp_rotation, source_image, target_image, z_coordinates, views_ellipses):
    # rx component 0 - max expected rotation
    # ry component - between -x degrees and +degrees
    # rz component - 0
    best_rotation_error = []
    best_rotation_matrix = None

    for rotation in range(max_exp_rotation):
        print("Rotation  ", rotation, "\n", best_rotation_error)
        for degrees in range(-10, 10):
            if rotation != 0 or degrees != 0:
                rotation_rad = np.deg2rad(rotation)
                degrees_rad = np.deg2rad(degrees)
                rotation_vector = np.array([rotation_rad, degrees_rad, 0], dtype=np.float32)
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Based on the rotation matrix obtain the target points for relevant_point in l_points:
                _, center_source, _, _ = views_ellipses[0]  # source view ellipse

                total_difference = 0
                s_points_count = 0

                for y_pos in range(source_image.shape[0]):
                    for x_pos in range(source_image.shape[1]):
                        x_source_pos_centered = x_pos - center_source[0]
                        y_source_pos_centered = y_pos - center_source[1]
                        z_pos = z_coordinates[y_pos][x_pos]   # z_coordinates are stored in a 2D matrix in the position (x,y) of each pixel
                        ps = np.array([x_source_pos_centered, y_source_pos_centered, z_pos])  # x', y' , z positions
                        pt = np.dot(rotation_matrix, ps).astype(int)
                        if pt[2] > 0 and is_coordinate_in_bounds(pt, target_image):
                            s_points_count += 1
                            x_target = pt[0]
                            y_target = pt[1]
                            source_pixel = source_image[y_pos][x_pos]
                            target_pixel = target_image[y_target][x_target]
                            diff = source_pixel.astype(int) - target_pixel.astype(int)
                            total_difference += diff

                rotation_error = total_difference / s_points_count

                if best_rotation_matrix is None or best_rotation_error > rotation_error:
                    best_rotation_error = rotation_error
                    best_rotation_matrix = rotation_matrix

    return best_rotation_matrix, best_rotation_error


def find_best_rotation(max_exp_rotation, source_image, target_image, l_points_indices, z_coordinates, views_ellipses):
    # rx component 0 - max expected rotation
    # ry component - between -x degrees and +degrees
    # rz component - 0
    best_rotation_error = []
    best_rotation_matrix = None

    for rotation in range(max_exp_rotation, -1):  # FIXME: ADD IF UPWARDS ROTATIONS, NOW ITS DOWNWARDS ROTATION
        for degrees in range(-10, 10):
            if rotation != 0 or degrees != 0:
                rotation_rad = np.deg2rad(rotation)
                degrees_rad = np.deg2rad(degrees)
                rotation_vector = np.array([rotation_rad, degrees_rad, 0], dtype=np.float32)
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Based on the rotation matrix obtain the target points for relevant_point in l_points:
                _, center_source, _, _ = views_ellipses[0]  # source view ellipse

                total_difference = 0
                s_points_count = 0

                for relevant_point in l_points_indices:
                    x_pos = relevant_point[1]
                    y_pos = relevant_point[0]
                    x_source_pos_centered = x_pos - center_source[0]
                    y_source_pos_centered = y_pos - center_source[1]
                    z_pos = z_coordinates[y_pos][x_pos]   # z_coordinates are stored in a 2D matrix in the position (x,y) of each pixel
                    ps = np.array([x_source_pos_centered, y_source_pos_centered, z_pos])  # x', y' , z positions
                    pt = np.dot(rotation_matrix, ps).astype(int)
                    if pt[2] > 0:
                        s_points_count += 1
                        x_target = pt[0]
                        y_target = pt[1]
                        source_pixel = source_image[y_pos][x_pos]
                        target_pixel = target_image[y_target][x_target]
                        diff = source_pixel.astype(int) - target_pixel.astype(int)
                        total_difference += diff

                rotation_error = total_difference / s_points_count

                if best_rotation_matrix is None or best_rotation_error > rotation_error:
                    best_rotation_error = rotation_error
                    best_rotation_matrix = rotation_matrix

    return best_rotation_matrix, best_rotation_error


def rotationMatrixToEulerAngles(R) :
    sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    # Convert to degrees
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)
    return np.array([x, y, z])


def show_target_point():
    # pt = Rps
    # How can I define ps??
    # ps, rotation_matrix, target_image
    return None



def pixel_picker(img):
    # Create a callback function for mouse click
    def on_mouse_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Print the pixel coordinates
            print(f"Pixel coordinates: x={x}, y={y}")

    # Create a window and set the mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", on_mouse_click)

    # Display the image
    while True:
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()


def is_coordinate_in_bounds(coordinate, image):
    height, width = image.shape[:2]  # get the height and width of the image
    x, y, _ = coordinate
    return 0 <= x < width and 0 <= y < height


def track_point(bin_imgs, list_imgs, z_list, rot_mat_list, total_height, max_width):
    image_with_marker_list = []
    # image_with_marker_list.append(image_dot)

    for i, image in enumerate(bin_imgs):

        contours, _ = cv2.findContours(image, 1, 2)
        # Draw the contours
        # Select the largest contour
        selected_contour = max(contours, key=cv2.contourArea)

        ####Calculate the centroid of the selected contour
        M = cv2.moments(selected_contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])

        # Initial Point to track
        # if i == 0: x,y,z = centroid_x,centroid_y, 0
        # else: x,y,z = x_new,y_new, 0
        z_mat = z_list[i]
        if i == 0:
            x, y, z = centroid_x, 0, z_mat[0, centroid_x]
        else:
            x, y, z = x_new, y_new, z_mat[y_new, x_new]

        tracking_point = np.array([[x], [y], [z]])
        print("x,y,z", [x, y, z])
        # Apply the rotation matrix to transform the point
        new_point = rot_mat_list[i] @ tracking_point
        # Calculate the corresponding 2D pixel coordinates
        x_new = int(new_point[0, 0])
        y_new = int(new_point[1, 0])

        ##Draw the ellipse on the original image
        center_x, center_y = centroid_x, centroid_y
        center = (centroid_x, centroid_y)

        # Draw a marker at the new pixel coordinates to visualize the point's position in the next image
        image_with_marker = list_imgs[i].copy()  # Create a copy of the original image to draw on
        color = (0, 0, 255)  # Red color
        radius = 5
        thickness = 7
        cv2.circle(image_with_marker, (x, y), radius, color, thickness)

        # Draw properties
        color = (0, 255, 0)
        thickness = 2

        # Display image
        image_with_marker_list.append(image_with_marker)

    ###Stack images horizontal
    x_offset = 0
    canvas = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    for image in image_with_marker_list:
        canvas[0:image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1]

    cv2.imshow('Canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()