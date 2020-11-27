"""
Builds a mixing matrix for the Victorian multi-cluster model.
"""
from copy import deepcopy

import numpy as np
from summer.model import StratifiedModel

from autumn.constants import Region
from autumn.mixing.mixing import create_assortative_matrix
from apps.covid_19.model.preprocess.mixing_matrix import build_dynamic_mixing_matrix


MOB_REGIONS = [Region.to_filename(r) for r in Region.VICTORIA_SUBREGIONS]


def build_victorian_mixing_matrix_func(
        static_mixing_matrix,
        metro_mobility,
        regional_mobility,
        metro_clusters,
        regional_clusters,
        country,
        inter_cluster_mixing,
):

    # Collate the cluster-specific mixing matrices
    cluster_age_mm_funcs = []
    for region in MOB_REGIONS:

        # Get the mobility parameters out for the cluster of interest
        if region in metro_clusters:
            cluster_mobility = deepcopy(metro_mobility)
        elif region in regional_clusters:
            cluster_mobility = deepcopy(regional_mobility)
        else:
            raise ValueError("Mobility region not found")
        cluster_mobility.region = region.upper()

        # Build the cluster-specific dyanmic mixing matrix
        cluster_age_mm_func = build_dynamic_mixing_matrix(
            static_mixing_matrix,
            cluster_mobility,
            country,
        )
        cluster_age_mm_funcs.append(cluster_age_mm_func)

    def get_mixing_matrix(self: StratifiedModel, time: float):
        """
        """

        # Get the inter-cluster mixing matrix
        inter_cluster_mixing_matrix = \
            create_assortative_matrix(inter_cluster_mixing, MOB_REGIONS)

        # Get the within-cluster mixing matrix
        cluster_age_mms = [f(time) for f in cluster_age_mm_funcs]

        # Pre-allocate
        combined_matrix_size = len(static_mixing_matrix) * len(MOB_REGIONS)
        super_matrix = np.zeros((combined_matrix_size, combined_matrix_size))

        # Find the starting points for the age-based mixing matrix
        for age_i in range(len(static_mixing_matrix)):
            start_i = age_i * len(MOB_REGIONS)
            for age_j in range(len(static_mixing_matrix)):
                start_j = age_j * len(MOB_REGIONS)

                # Loop over clusters being infected
                for infectee_cluster, age_mm in enumerate(cluster_age_mms):
                    i_row = start_i + infectee_cluster

                    # Populate out across the age group matrix by cluster
                    for infector_cluster in range(len(cluster_age_mms)):
                        i_col = start_j + infector_cluster

                        intercluster_modifier = \
                            inter_cluster_mixing_matrix[infectee_cluster, infector_cluster]

                        # Populate the final super-matrix
                        super_matrix[i_row, i_col] = \
                            age_mm[age_i, age_j] * intercluster_modifier

        return super_matrix

    return get_mixing_matrix
