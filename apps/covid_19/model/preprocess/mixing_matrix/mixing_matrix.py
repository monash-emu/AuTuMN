from apps.covid_19.model.preprocess.mixing_matrix.mixing_adjusters import age_adjuster
from typing import Callable, List

import numpy as np

from apps.covid_19.model.parameters import Mobility, Country

from .mixing_adjusters import AgeMixingAdjuster, LocationMixingAdjuster
from .microdistancing import get_microdistancing_funcs
from .mobility import get_mobility_funcs


def build_dynamic_mixing_matrix(
    base_matrix: np.ndarray, mobility: Mobility, country: Country
) -> Callable[[float], np.ndarray]:
    """
    Build a time-varing mixing matrix
    Returns a function of time which returns a 16x16 mixing matrix.
    """
    microdistancing_funcs = get_microdistancing_funcs(
        mobility.microdistancing,
        mobility.microdistancing_locations,
        mobility.square_mobility_effect,
    )
    mobility_funcs = get_mobility_funcs(
        country,
        mobility.region,
        mobility.mixing,
        mobility.google_mobility_locations,
        mobility.npi_effectiveness,
        mobility.square_mobility_effect,
        mobility.smooth_google_data,
    )

    location_adjuster = LocationMixingAdjuster(country, mobility_funcs, microdistancing_funcs)
    if mobility.age_mixing:
        age_adjuster = AgeMixingAdjuster(mobility.age_mixing)
    else:
        age_adjuster = None

    def mixing_matrix_function(time: float) -> np.ndarray:
        """
        Returns a 16x16 mixing matrix for a given time.
        """
        mixing_matrix = np.copy(base_matrix)
        mixing_matrix = location_adjuster.get_adjustment(time, mixing_matrix)
        if age_adjuster:
            mixing_matrix = age_adjuster.get_adjustment(time, mixing_matrix)

        return mixing_matrix

    return mixing_matrix_function
