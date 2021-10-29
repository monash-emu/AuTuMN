from typing import Callable, Dict

import numpy as np

from autumn.models.covid_19.parameters import Country, Mobility

from .microdistancing import get_microdistancing_funcs
from .mixing_adjusters import AgeMixingAdjuster, LocationMixingAdjuster
from .mobility import get_mobility_funcs


def build_dynamic_mixing_matrix(
    base_matrices: Dict[str, np.ndarray], mobility: Mobility, country: Country
) -> Callable[[float], np.ndarray]:
    """
    Build a time-varying mixing matrix
    Returns a function of time which returns a 16x16 mixing matrix.
    """

    microdistancing_funcs = get_microdistancing_funcs(mobility.microdistancing, mobility.square_mobility_effect)

    google_mobility_locations = {
        "work": {"workplaces": 1.},
        "other_locations": {"retail_and_recreation": 0.25, "grocery_and_pharmacy": 0.25, "parks": 0.25, "transit_stations": 0.25},
        "home": {"residential": 1.},
    }

    mobility_funcs = get_mobility_funcs(
        country, mobility.region, mobility.mixing, google_mobility_locations, mobility.npi_effectiveness,
        mobility.square_mobility_effect, mobility.smooth_google_data,
    )

    # Get adjusters
    location_adjuster = LocationMixingAdjuster(base_matrices, mobility_funcs, microdistancing_funcs)
    age_adjuster = AgeMixingAdjuster(mobility.age_mixing) if mobility.age_mixing else None

    def mixing_matrix_function(time: float) -> np.ndarray:
        """
        Returns a 16x16 mixing matrix for a given time.
        """

        # Base matrix
        mixing_matrix = np.copy(base_matrices["all_locations"])

        # Apply adjustments
        mixing_matrix = location_adjuster.get_adjustment(time, mixing_matrix)
        if age_adjuster:
            mixing_matrix = age_adjuster.get_adjustment(time, mixing_matrix)

        return mixing_matrix

    return mixing_matrix_function
