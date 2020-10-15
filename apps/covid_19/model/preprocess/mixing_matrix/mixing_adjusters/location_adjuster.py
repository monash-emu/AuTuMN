from typing import Callable, Dict

import numpy as np

from apps.covid_19.model.parameters import Country
from autumn.inputs import get_country_mixing_matrix
from .base_adjuster import BaseMixingAdjuster

# Locations that can be used for mixing
LOCATIONS = ["home", "other_locations", "school", "work"]


class LocationMixingAdjuster(BaseMixingAdjuster):
    """
    Applies location based mixing adjustments (micro and macro) to a mixing matrix.
    """

    def __init__(
        self,
        country: Country,
        mobility_funcs: Dict[str, Callable[[float], float]],
        microdistancing_funcs: Dict[str, Callable[[float], float]],
    ):
        """Build the time variant location adjustment functions"""
        self.mobility_funcs = mobility_funcs
        self.microdistancing_funcs = microdistancing_funcs

        # Load all the location-specific mixing matrices.
        self.matrix_components = {}
        for sheet_type in LOCATIONS:
            # Loads a 16x16 ndarray
            self.matrix_components[sheet_type] = get_country_mixing_matrix(sheet_type, country.iso3)

    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Apply time-varying location adjustments.
        Returns a new mixing matrix, modified to adjust for dynamic mixing changes for a given point in time.
        """
        for loc_key in LOCATIONS:
            # Start the adjustment value for each location from a value of 1 for "no adjustment".
            loc_adjustment = 1

            # Adjust for Google Mobility data.
            mobility_func = self.mobility_funcs.get(loc_key)
            if mobility_func:
                loc_adjustment *= mobility_func(time)

            # Adjust for microdistancing
            microdistancing_func = self.microdistancing_funcs.get(loc_key)
            if microdistancing_func:
                loc_adjustment *= microdistancing_func(time)

            # Apply adjustment by subtracting the contacts that need to come off
            mixing_matrix += (loc_adjustment - 1) * self.matrix_components[loc_key]

        return mixing_matrix
