import numpy as np


def create_assortative_matrix(off_diagonal_values, matrix_dimensions):
    """
    Create a matrix with all values the same except for the diagonal elements, which are greater, according to the
    requested value to go in the off-diagonal elements. To be used for creating a standard assortative mixing matrix
    according to any number of interacting groups.
    """

    assert 0. <= off_diagonal_values <= 1. / len(matrix_dimensions)
    off_diagonal_elements = \
        np.ones([len(matrix_dimensions), len(matrix_dimensions)]) * off_diagonal_values
    diagonal_elements = \
        np.eye(len(matrix_dimensions)) * \
        (1. - len(matrix_dimensions) * off_diagonal_values)
    assortative_matrix = off_diagonal_elements + diagonal_elements

    for i_row in range(len(assortative_matrix)):
        assert sum(assortative_matrix[i_row, :]) == 1.
    for i_col in range(len(assortative_matrix)):
        assert sum(assortative_matrix[:, i_col]) == 1.

    return assortative_matrix