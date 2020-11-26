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
        inter_cluster_mixing=0.01,
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
        """
        """

        # Create the inter-cluster mixing matrix
        assert 0. < inter_cluster_mixing <= 1. / len(MOB_REGIONS)
        inter_cluster_mixing_matrix = \
            np.ones([len(MOB_REGIONS), len(MOB_REGIONS)]) * inter_cluster_mixing
        within_cluster_matrix = \
            np.eye(len(MOB_REGIONS)) * \
            (1. - len(MOB_REGIONS) * inter_cluster_mixing)
        inter_cluster_mixing_matrix = inter_cluster_mixing_matrix + within_cluster_matrix

        # Get the individual cluster mixing matrix
        cluster_age_mms = [f(time) for f in cluster_age_mm_funcs]

        # Pre-allocate
        combined_matrix_size = len(static_mixing_matrix) * len(MOB_REGIONS)
        mm = np.zeros((combined_matrix_size, combined_matrix_size))

        # Loop over clusters being infected
        for infectee_cluster, age_mm in enumerate(cluster_age_mms):

            # Populate out across the age group matrix by cluster
            for infector_cluster in range(len(cluster_age_mms)):

                # Find the starting points for the age-based mixing matrix
                for age_i in range(len(static_mixing_matrix)):
                    start_i = age_i * len(MOB_REGIONS)
                    for age_j in range(len(static_mixing_matrix)):
                        start_j = age_j * len(MOB_REGIONS)

                        # Populate the final super-matrix
                        mm[start_i + infectee_cluster,
                           start_j + infector_cluster] = \
                            age_mm[age_i, age_j] * \
                            inter_cluster_mixing_matrix[infectee_cluster, infector_cluster]

        return mm

    return get_mixing_matrix
