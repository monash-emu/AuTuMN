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
        """
        Construct a 144 x 144 matrix that is the same shape as the Kronecker product
        of our age-based mixing matrix and the identity matrix *except*, we swap in
        region specific values. The matrix should have shape (for ages a and clutsers c)

        a00c0         a01c0
         a00c1         a01c1
          a00c2         a01c2
            a00c3        a01c3
             a00c4        a01c4      ... 16 times
              a00c5        a01c5
               a00c6        a01c6
                a00c7        a01c7
                 a00c8        a01c8
        a10c0
         a10c1         .
          a10c2         .
            a10c3        .
             a10c4
              a10c5
               a10c6
                a10c7
                 a10c8
            .
            .
            .
            16 times

        """

        # Pre-allocate
        combined_matrix_size = len(static_mixing_matrix) * len(MOB_REGIONS)
        mm = np.zeros((combined_matrix_size, combined_matrix_size))

        # Get the individual cluster mixing matrix
        cluster_age_mms = [f(time) for f in cluster_age_mm_funcs]

        # Define inter-cluster mixing matrix
        inter_cluster_mixing_matrix = np.ones([len(MOB_REGIONS), len(MOB_REGIONS)])

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

        print(mm)

        return mm

    return get_mixing_matrix
