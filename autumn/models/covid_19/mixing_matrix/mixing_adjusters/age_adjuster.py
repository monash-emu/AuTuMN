from typing import Dict
import numpy as np

from autumn.models.covid_19.parameters import TimeSeries
from autumn.models.covid_19.constants import AGEGROUP_STRATA
from autumn.tools.curve import scale_up_function

from .base_adjuster import BaseMixingAdjuster


class AgeMixingAdjuster(BaseMixingAdjuster):
    """
    Applies age-based mixing adjustments to a mixing matrix.
    The mixing matrix is expected to be 16x16.
    This is currently unused.

    *** Note that this object works quite differently from location_adjuster, which is potentially confusing
    because the both inherit from the same parent class ***

    Specifically, this one adjusts the cells of the matrix according to ratios applied to their working values,
    whereas location adjuster subtracts absolute values of the starting matrices from the current values.

    Attributes:
        adjustment_funcs: The adjustment functions to be applied for each age group

    """

    def __init__(self, age_mixing: Dict[str, TimeSeries]):
        """
        Build the time-variant age adjustment functions.
        """

        self.adjustment_funcs = {}
        for age_idx, timeseries in age_mixing.items():
            func = scale_up_function(timeseries.times, timeseries.values, method=4)
            self.adjustment_funcs[str(age_idx)] = func

    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Apply time-varying age adjustments during model run-time.

        Args:
            time: Model time
            mixing_matrix: The mixing matrix previously adjusted for location effects

        Returns:
            Returns the new mixing matrix based on the previous

        """

        # Iterate over matrix rows and columns
        adjusted_matrix = mixing_matrix.copy()
        for i_row_agegroup, row_agegroup in enumerate(AGEGROUP_STRATA):
            row_adjust_func = self.adjustment_funcs.get(row_agegroup)
            row_multiplier = row_adjust_func(time) if row_adjust_func else 1.

            for j_col_agegroup, col_agegroup in enumerate(AGEGROUP_STRATA):
                col_adjust_func = self.adjustment_funcs.get(col_agegroup)
                col_multiplier = col_adjust_func(time) if col_adjust_func else 1.

                adjusted_matrix[i_row_agegroup, j_col_agegroup] *= row_multiplier * col_multiplier

        return adjusted_matrix
