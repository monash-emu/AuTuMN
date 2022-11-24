from typing import Callable, Dict
import numpy as np

from autumn.models.sm_sir.parameters import Country, Mobility
from .mixing_adjusters import AgeMixingAdjuster, LocationMixingAdjuster
from .microdistancing import get_microdistancing_funcs
from .macrodistancing import get_mobility_funcs


def build_dynamic_mixing_matrix(
    base_matrices: Dict[str, np.ndarray], mobility: Mobility, country: Country
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

    microdistancing_funcs = get_microdistancing_funcs(mobility.microdistancing, mobility.square_mobility_effect, country.iso3)

    square, smooth, locs = mobility.square_mobility_effect, mobility.smooth_google_data, mobility.google_mobility_locations
    mobility_funcs = get_mobility_funcs(country, mobility.region, mobility.mixing, locs, square, smooth,
                                        mobility.lockdown_1_mobility, mobility.lockdown_2_mobility, mobility.constant_mobility)
    # Get adjusters
    location_adjuster = LocationMixingAdjuster(base_matrices, mobility_funcs, microdistancing_funcs)
    age_adjuster = AgeMixingAdjuster(mobility.age_mixing) if mobility.age_mixing else None

    def mixing_matrix_function(time: float, computed_values: dict=None) -> np.ndarray:
        """
        The function to be called at model run-time to get the matrix.
        """

        # Base matrix
        mixing_matrix = np.copy(base_matrices["all_locations"])

        # Apply adjustments - *** note that the order of these adjustments can't be reversed ***
        mixing_matrix = location_adjuster.get_adjustment(time, mixing_matrix)
        return age_adjuster.get_adjustment(time, mixing_matrix) if age_adjuster else mixing_matrix

    return mixing_matrix_function
