import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from statistics import mean

def calculate_z_coordinates_image(image, angle, ellipse, spheroid, z_coordinates, fruit_type):
    a_axe, b_axe = spheroid
    _, center, _, unit_eigenvectors = ellipse

    if fruit_type == 2:
        rad_angle = np.radians(angle)

        p11 = unit_eigenvectors[1][0] * np.sin(rad_angle)
        p12 = unit_eigenvectors[1][1] * np.sin(rad_angle)
        p13 = np.cos(rad_angle)

        p1 = [p11, p12, p13]
        p2 = [unit_eigenvectors[0][0], unit_eigenvectors[0][1], 0.0]
        p3 = np.cross(p1, p2)

        pose_matrix = np.array([p1, p2, p3])
        b_2 = b_axe ** 2
        a_2 = a_axe ** 2
        equation_matrix = [[1 / b_2, 0.0, 0.0], [0.0, 1 / a_2, 0.0], [0.0, 0.0, 1 / a_2]]

        matrix_a = pose_matrix.T @ equation_matrix @ pose_matrix

    for y_pos in range(image.shape[0]):
        for x_pos in range(image.shape[1]):
            xp = x_pos - center[0]
            yp = y_pos - center[1]

            if fruit_type == 2:
                xpyp = np.array([xp, yp])

                # Calculate C
                # sub_matrixa_c = matrix_a[:2, :2]
                a11 = matrix_a[0][0]
                a12 = matrix_a[0][1]
                a21 = matrix_a[1][0]
                a22 = matrix_a[1][1]
                sub_matrixa_c = [[a11, a12], [a21, a22]]
                c = xpyp @ sub_matrixa_c @ xpyp.T - 1

                # Calculate B
                # sub_matrixa_b_1 = matrix_a[:2, 2:3]
                a13 = matrix_a[0][2]
                a23 = matrix_a[1][2]
                sub_matrixa_b_1 = [[a13], [a23]]

                # sub_matrixa_b_2 = matrix_a[2, :2]
                a31 = matrix_a[2][0]
                a32 = matrix_a[2][1]
                sub_matrixa_b_2 = [a31, a32]

                b_const = xpyp @ sub_matrixa_b_1 + sub_matrixa_b_2 @ xpyp.T
                b_const = b_const[0]

                # Calculate A
                a33 = matrix_a[2][2]

                # Calculate Z
                sqrt = (b_const ** 2) - (4 * a33 * c)

                if sqrt >= 0:
                    z = (-b_const + math.sqrt(sqrt)) / (2 * a33)
                else:
                    z = 0
            else:
                radius = spheroid[0]

                # Calculate Z
                z_sq = radius ** 2 - xp ** 2 - yp ** 2

                if z_sq >= 0:
                    z = math.sqrt(z_sq)
                else:
                    z = 0

            z_coordinates[y_pos][x_pos] = z

def calculate_z_coordinates(relevant_points_indices, angle, ellipse, spheroid, z_coordinates, fruit_type):
    a_axe, b_axe = spheroid
    _, center, _, unit_eigenvectors = ellipse

    if fruit_type == 2:
        rad_angle = np.radians(angle)

        p11 = unit_eigenvectors[1][0] * np.sin(rad_angle)
        p12 = unit_eigenvectors[1][1] * np.sin(rad_angle)
        p13 = np.cos(rad_angle)

        p1 = [p11, p12, p13]
        p2 = [unit_eigenvectors[0][0], unit_eigenvectors[0][1], 0.0]
        p3 = np.cross(p1, p2)

        pose_matrix = np.array([p1, p2, p3])
        b_2 = b_axe ** 2
        a_2 = a_axe ** 2
        equation_matrix = [[1 / b_2, 0.0, 0.0], [0.0, 1 / a_2, 0.0], [0.0, 0.0, 1 / a_2]]

        matrix_a = pose_matrix.T @ equation_matrix @ pose_matrix

    for relevant_point in relevant_points_indices:
        x_pos = relevant_point[1]
        y_pos = relevant_point[0]
        xp = x_pos - center[0]
        yp = y_pos - center[1]

        if fruit_type == 2:
            xpyp = np.array([xp, yp])

            # Calculate C
            # sub_matrixa_c = matrix_a[:2, :2]
            a11 = matrix_a[0][0]
            a12 = matrix_a[0][1]
            a21 = matrix_a[1][0]
            a22 = matrix_a[1][1]
            sub_matrixa_c = [[a11, a12], [a21, a22]]
            c = xpyp @ sub_matrixa_c @ xpyp.T - 1

            # Calculate B
            # sub_matrixa_b_1 = matrix_a[:2, 2:3]
            a13 = matrix_a[0][2]
            a23 = matrix_a[1][2]
            sub_matrixa_b_1 = [[a13], [a23]]

            # sub_matrixa_b_2 = matrix_a[2, :2]
            a31 = matrix_a[2][0]
            a32 = matrix_a[2][1]
            sub_matrixa_b_2 = [a31, a32]

            b_const = xpyp @ sub_matrixa_b_1 + sub_matrixa_b_2 @ xpyp.T
            b_const = b_const[0]

            # Calculate A
            a33 = matrix_a[2][2]

            # Calculate Z
            sqrt = (b_const ** 2) - (4 * a33 * c)

            if sqrt >= 0:
                z = (-b_const + math.sqrt(sqrt)) / (2 * a33)
            else:
                z = 0
        else:
            radius = spheroid[0]

            # Calculate Z
            z_sq = radius ** 2 - xp ** 2 - yp ** 2

            if z_sq >= 0:
                z = math.sqrt(z_sq)
            else:
                z = 0

        z_coordinates[y_pos][x_pos] = z