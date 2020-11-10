"""
Builds a mixing matrix for the Victorian multi-cluster model.
"""
from copy import deepcopy

import numpy as np
from summer.model import StratifiedModel

from autumn.constants import Region
from apps.covid_19.model.preprocess.mixing_matrix import build_dynamic_mixing_matrix


MOB_REGIONS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]


def build_victorian_mixing_matrix_func(
    static_mixing_matrix,
    mobility,
    country,
):
    cluster_age_mm_funcs = []
    for region in MOB_REGIONS:
        cluster_mobility = deepcopy(mobility)
        cluster_mobility.region = region.upper()
        cluster_age_mm_func = build_dynamic_mixing_matrix(
            static_mixing_matrix,
            cluster_mobility,
            country,
        )
        cluster_age_mm_funcs.append(cluster_age_mm_func)

    def get_mixing_matrix(self: StratifiedModel, time: float):
        mm = np.zeros([144, 144])
        cluster_age_mms = [f(time) for f in cluster_age_mm_funcs]
        for cluster_idx, age_mm in enumerate(cluster_age_mms):
            for age_i in range(16):
                for age_j in range(16):
                    i = age_i * cluster_idx
                    j = age_j * cluster_idx
                    mm[i, j] = age_mm[age_i, age_j]

        return mm

    return get_mixing_matrix
