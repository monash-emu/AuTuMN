from typing import Callable

import numpy as np

from apps.covid_19.model.parameters import Country, Mobility


from .static import build_static
from .adjust_location import LocationMixingAdjustment


def build_dynamic(country: Country, mobility: Mobility) -> Callable[[float], np.ndarray]:
    """
    Build a time-varing mixing matrix
    Returns a function of time which returns a 16x16 mixing matrix.
    """
    adjuster = LocationMixingAdjustment(country, mobility)
    static_mixing_matrix = build_static(country.iso3)

    # Create mixing matrix function
    def mixing_matrix_function(time: float):
        """
        Returns a 16x16 mixing matrix for a given time.
        """
        mixing_matrix = np.copy(static_mixing_matrix)
        adjusted_mixing_matrix = adjuster.get_adjustment(
            time, mixing_matrix, mobility.microdistancing_locations
        )
        return adjusted_mixing_matrix

    return mixing_matrix_function
