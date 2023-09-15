# FUNCTION TO OBTAIN AXES OF THE PROJECTED ELLIPSE
import math
import numpy as np

def calculate_projected_axes(binary_mask):
    squared_mask = binary_mask ** 2
    x_indices, y_indices = np.indices(binary_mask.shape)
    n = np.count_nonzero(binary_mask)

    # ELEMENT (0,0) OF MATRIX
    sx = np.sum(x_indices * binary_mask)
    sxx = np.sum(x_indices * squared_mask)
    cx = sx / n
    cov_matrix_00 = (sxx / n) - cx ** 2

    # ELEMENT (1,1) OF MATRIX
    sy = np.sum(y_indices * binary_mask)
    syy = np.sum(y_indices * squared_mask)
    cy = sy / n
    cov_matrix_11 = (syy / n) - cy ** 2

    # ELEMENT (0, 1) and (1, 0) OF MATRIX
    sxy = np.sum(x_indices * y_indices * binary_mask)
    cov_matrix_10_01 = (sxy / n) - cx * cy

    # COVARIANCE MATRIX
    cov_matrix = [[cov_matrix_00, cov_matrix_10_01], [cov_matrix_10_01, cov_matrix_11]]
    eigenvalues = np.linalg.eigvals(cov_matrix)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    major_eigenvalue = sorted_eigenvalues[0]
    minor_eigenvalue = sorted_eigenvalues[-1]

    a = 2 * (math.sqrt(abs(major_eigenvalue)))  # semi major axis of projected ellipse
    b = 2 * (math.sqrt(abs(minor_eigenvalue)))  # semi minor axis of projected ellipse

    return [a, b]


