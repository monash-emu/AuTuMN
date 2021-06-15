"""
Builds a mixing matrix for the Victorian multi-cluster model.
"""
from copy import deepcopy

import numpy as np
from numba import jit
from summer.legacy.model import StratifiedModel

from autumn.models.covid_19.preprocess.mixing_matrix import build_dynamic_mixing_matrix
from autumn.settings import Region


def build_victorian_mixing_matrix_func(
    static_mixing_matrix,
    metro_mobility,
    regional_mobility,
    country,
    intercluster_mixing_matrix,
):
    # Having a consistent ordering of cluster names when building the matrix is critical, or else
    # we will mix up our clutsers and people in Gippsland will be telepathically infecting people in the Grampians.
    metro_clusters = [r.replace("-", "_") for r in Region.VICTORIA_METRO]
    regional_clusters = [region.replace("-", "_") for region in Region.VICTORIA_RURAL]
    all_clusters = [region.replace("-", "_") for region in Region.VICTORIA_SUBREGIONS]

    # Collate the cluster-specific mixing matrices
    cluster_age_mm_funcs = []
    for region in all_clusters:

        # Get the mobility parameters out for the cluster of interest
        if region in metro_clusters:
            cluster_mobility = deepcopy(metro_mobility)
        elif region in regional_clusters:
            cluster_mobility = deepcopy(regional_mobility)
        else:
            raise ValueError("Mobility region not found")
        cluster_mobility.region = region.upper()

        # Build the cluster-specific dynamic mixing matrix
        cluster_age_mm_func = build_dynamic_mixing_matrix(
            static_mixing_matrix,
            cluster_mobility,
            country,
        )
        cluster_age_mm_funcs.append(cluster_age_mm_func)

    def get_mixing_matrix(time: float):

        # Collate the within-cluster mixing matrices
        cluster_age_mms = np.zeros((len(cluster_age_mm_funcs), *static_mixing_matrix.shape))
        for idx, func in enumerate(cluster_age_mm_funcs):
            cluster_age_mms[idx] = func(time)

        # Pre-allocate
        static_matrix_size = len(static_mixing_matrix)
        num_clusters = len(all_clusters)
        combined_matrix_size = static_matrix_size * num_clusters
        super_matrix = np.zeros((combined_matrix_size, combined_matrix_size))
        _set_mixing_matrix(
            super_matrix,
            intercluster_mixing_matrix,
            cluster_age_mms,
            static_matrix_size,
            num_clusters,
        )
        return super_matrix

    return get_mixing_matrix


# Use Numba's "just in time" (JIT) runtime compiler to optimize this code.
# Cannot use all Python language features in this code block.
@jit(nopython=True)
def _set_mixing_matrix(
    super_matrix: np.ndarray,
    intercluster_mixing_matrix: np.ndarray,
    cluster_age_mms: np.ndarray,
    static_matrix_size: int,
    num_clusters: int,
) -> np.ndarray:
    # Find the starting points for the age-based mixing matrix
    for age_i in range(static_matrix_size):
        start_i = age_i * num_clusters
        for age_j in range(static_matrix_size):
            start_j = age_j * num_clusters

            # Loop over clusters being infected
            for infectee_cluster in range(num_clusters):
                age_mm = cluster_age_mms[infectee_cluster]
                i_row = start_i + infectee_cluster

                # Populate out across the age group matrix by cluster
                for infector_cluster in range(num_clusters):
                    i_col = start_j + infector_cluster

                    intercluster_modifier = intercluster_mixing_matrix[
                        infectee_cluster, infector_cluster
                    ]

                    # Populate the final super-matrix
                    super_matrix[i_row, i_col] = age_mm[age_i, age_j] * intercluster_modifier
