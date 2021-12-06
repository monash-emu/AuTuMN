from typing import Callable, Dict
import numpy as np

from autumn.models.covid_19.parameters import Country, Mobility
from .microdistancing import get_microdistancing_funcs
from .mixing_adjusters import LocationMixingAdjuster
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

    microdistancing_funcs = get_microdistancing_funcs(mobility.microdistancing, mobility.square_mobility_effect)

    locs = mobility.google_mobility_locations
    square, smooth = mobility.square_mobility_effect, mobility.smooth_google_data
    mobility_funcs = get_mobility_funcs(country, mobility.region, mobility.mixing, locs, square, smooth)

    # *** Do not apply additional mixing adjusters, because this one depends on the base matrix values ***
    location_adjuster = LocationMixingAdjuster(base_matrices, mobility_funcs, microdistancing_funcs)

    def mixing_matrix_function(time: float) -> np.ndarray:
        return location_adjuster.get_adjustment(time, np.copy(base_matrices["all_locations"]))

    return mixing_matrix_function
