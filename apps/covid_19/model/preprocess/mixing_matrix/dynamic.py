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
    microdistancing_params: dict,
    smooth_google_data: bool,
    microdistancing_locations: list,
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
            microdistancing_params,
            smooth_google_data,
            microdistancing_locations,
        )

    static_mixing_matrix = build_static(country_iso3)

    # Create mixing matrix function
    def mixing_matrix_function(time: float):
        """
        Returns a 16x16 mixing matrix for a given time.
        """
        mixing_matrix = np.copy(static_mixing_matrix)
        adjusted_mixing_matrix = adjuster.get_adjustment(time, mixing_matrix, microdistancing_locations)
        return adjusted_mixing_matrix

    return mixing_matrix_function
