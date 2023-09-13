# FUNCTION TO OBTAIN AXES OF THE PROJECTED ELLIPSE
import math
import numpy as np


def calculate_projected_axes(binary_mask):
    squared_mask = binary_mask ** 2
    nx = binary_mask.shape[0]
    ny = binary_mask.shape[1]
    nxy = binary_mask.shape[0] * binary_mask.shape[1]

    # ELEMENT (0,0) OF MATRIX
    sx = np.sum(np.sum(binary_mask, axis=1))
    sxx = np.sum(np.sum(squared_mask, axis=1))
    cx = sx / nx
    cov_matrix_00 = sxx / (nx - cx ** 2)

    # ELEMENT (1,1) OF MATRIX
    sy = np.sum(np.sum(binary_mask, axis=0))
    syy = np.sum(np.sum(squared_mask, axis=0))
    cy = sy / ny
    cov_matrix_11 = syy / (ny - cy ** 2)

    # ELEMENT (0, 1) and (1, 0) OF MATRIX
    sxy = np.sum(binary_mask)
    cov_matrix_10_01 = sxy / nxy - (cx * cy)

    # COVARIANCE MATRIX
    cov_matrix = [[cov_matrix_00, cov_matrix_10_01], [cov_matrix_10_01, cov_matrix_11]]
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    major_eigenvalue = sorted_eigenvalues[0]
    minor_eigenvalue = sorted_eigenvalues[-1]
    a = 2 * (math.sqrt(abs(major_eigenvalue)))  # semi major axis of projected ellipse
    b = 2 * (math.sqrt(abs(minor_eigenvalue)))  # semi minor axis of projected ellipse

    return [a, b]
