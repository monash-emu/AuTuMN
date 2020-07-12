from typing import Callable

import numpy as np

from .static import build_static
from .adjust_age import AgeMixingAdjustment
from .adjust_location import LocationMixingAdjustment


def build_dynamic(
    country_iso3: str,
    region: str,
    mixing: dict,
    mixing_age_adjust: dict,
    npi_effectiveness_params: dict,
    google_mobility_locations: dict,
    is_periodic_intervention: bool,
    periodic_int_params: dict,
    periodic_end_time: float,
    microdistancing_params: dict,
) -> Callable[[float], np.ndarray]:
    """
    Build a time-varing mixing matrix
    Returns a function of time which returns a 16x16 mixing matrix.
    """
    adjuster = None
    if mixing_age_adjust:
        adjuster = AgeMixingAdjustment(mixing_age_adjust)
    else:
        adjuster = LocationMixingAdjustment(
            country_iso3,
            region,
            mixing,
            npi_effectiveness_params,
            google_mobility_locations,
            is_periodic_intervention,
            periodic_int_params,
            periodic_end_time,
            microdistancing_params,
        )

    static_mixing_matrix = build_static(country_iso3)

    # Create mixing matrix function
    def mixing_matrix_function(time: float):
        """
        Returns a 16x16 mixing matrix for a given time.
        """
        mixing_matrix = np.copy(static_mixing_matrix)
        adjusted_mixing_matrix = adjuster.get_adjustment(time, mixing_matrix)
        return adjusted_mixing_matrix

    return mixing_matrix_function

