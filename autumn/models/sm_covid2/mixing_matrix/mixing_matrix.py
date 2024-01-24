from typing import Callable, Dict
import numpy as np

from jax import numpy as jnp

from .macrodistancing import get_mobility_funcs

from computegraph.types import Function
from summer2.parameters import Time
from summer2.functions.util import capture_dict


def build_dynamic_mixing_matrix(
    base_matrices: Dict[str, np.ndarray], mobility, country,
    additional_mobility: dict = None, random_process_func=None, hh_contact_increase=0., 
    rp_affected_locations=["home", "school", "work", "other_locations"]
) -> Callable[[float], np.ndarray]:
    """
    Master function that builds the time-varying mixing matrix.

    Args:
        base_matrices: Empiric matrices of contact rates by location for the country of interest
        mobility: Mobility parameters
        country: The country being considered

    Returns:
        A function of time which returns a 16x16 mixing matrix

    """

    #microdistancing_funcs = get_microdistancing_funcs(
    #    mobility.microdistancing, mobility.square_mobility_effect, country.iso3
    #)

    square, smooth, locs = (
        mobility.square_mobility_effect,
        mobility.smooth_google_data,
        mobility.google_mobility_locations,
    )

    additional_mobility = additional_mobility or {}

    # A graphobject dict of timeseries for locations
    location_ts = get_mobility_funcs(
        country, mobility.region, additional_mobility, locs, square, smooth, random_process_func, hh_contact_increase, rp_affected_locations
    )
    location_ts.node_name = "mm_location_adj"

    # Get adjusters
    #location_adjuster = LocationMixingAdjuster(base_matrices, mobility_funcs, microdistancing_funcs)
    #age_adjuster = AgeMixingAdjuster(mobility.age_mixing) if mobility.age_mixing else None
    base_matrices = capture_dict(**base_matrices)
    base_matrices.node_name = "mm_base_matrices"

    def mixing_matrix_adjust(base_matrices: dict, location_ts: dict) -> np.ndarray:
        """
        The function to be called at model run-time to get the matrix.
        """

        mixing_matrix = base_matrices["all_locations"]
        # Base matrix
        # mixing_matrix = np.copy(base_matrices["all_locations"])

        # Apply adjustments - *** note that the order of these adjustments can't be reversed ***

        for loc_key, adj_ts in location_ts.items():
            adj_amount = adj_ts - 1.0
            mixing_matrix += adj_amount * base_matrices[loc_key]
        return mixing_matrix

    return Function(mixing_matrix_adjust, [base_matrices, location_ts])
