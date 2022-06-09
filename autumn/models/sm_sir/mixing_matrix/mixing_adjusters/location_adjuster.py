from typing import Callable, Dict
import numpy as np

from .base_adjuster import BaseMixingAdjuster
from autumn.models.sm_sir.constants import LOCATIONS


class LocationMixingAdjuster(BaseMixingAdjuster):
    """
    Applies location based mixing adjustments (micro and macro) to a mixing matrix.
    The functions themselves are created in the files in the directory one above this one.

    *** Note that this object works quite differently from age_adjuster, which is potentially confusing
    because the both inherit from the same parent class ***

    Specifically, this one subtracts absolute values of the starting matrices from the current values,
    whereas age adjuster adjusts the cells of the matrix according to ratios applied to their working values.

    Attributes:
        mobility_funcs: The macrodistancing/mobility functions to be applied to each of the locations
        microdistancing_funcs: The microdistancing functions to be applied to each of the locations
        base_matrices: The mixing matrices unadjusted for the location functions

    """

    def __init__(
        self, base_matrices: Dict[str, np.ndarray], mobility_funcs: Dict[str, Callable[[float], float]],
            microdistancing_funcs: Dict[str, Callable[[float], float]],
    ):
        """
        Create the attributes to this object, as described in attributes above.

        """

        self.mobility_funcs = mobility_funcs
        self.microdistancing_funcs = microdistancing_funcs
        self.base_matrices = base_matrices

    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Apply time-varying location adjustments.
        Returns a new mixing matrix, modified to adjust for dynamic mixing changes for a given point in time.
        """

        # Start the adjustment value for each location from a value of one, representing no adjustment
        for loc_key in LOCATIONS:
            loc_relative_mobility = 1.

            # Adjust for macrodistancing/mobility
            mobility_func = self.mobility_funcs.get(loc_key)
            loc_relative_mobility = mobility_func(time) if mobility_func else 1.

            # Adjust for microdistancing
            microdistancing_func = self.microdistancing_funcs.get(loc_key)
            if microdistancing_func:
                loc_relative_mobility *= microdistancing_func(time)

            # Apply the adjustment by subtracting the contacts that need to come off
            mobility_change = 1. - loc_relative_mobility
            mixing_matrix -= mobility_change * self.base_matrices[loc_key]

        return mixing_matrix
