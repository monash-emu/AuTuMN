"""
Builds a mixing matrix for the Victorian multi-cluster model.
"""
import numpy as np
from summer.model import StratifiedModel


def build_victorian_mixing_matrix_func():
    def get_mixing_matrix(model: StratifiedModel, time: float):
        return np.ones([144, 144])

    return get_mixing_matrix