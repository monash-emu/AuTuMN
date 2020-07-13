import numpy as np

from autumn.curve import scale_up_function

from .adjust_base import BaseMixingAdjustment
from .utils import BASE_DATE

AGE_INDICES = list(range(16))


class AgeMixingAdjustment(BaseMixingAdjustment):
    def __init__(self, mixing_age_adjust: dict):
        """Build the time variant age adjustment functions"""
        self.age_adjustment_functions = {}
        affected_age_indices = [i for i in AGE_INDICES if f"age_{i}" in mixing_age_adjust]
        for age_idx in affected_age_indices:
            key = f"age_{age_idx}"
            mixing_age_adjust[key]["times"] = [
                (time_date - BASE_DATE).days for time_date in mixing_age_adjust[key]["times"]
            ]
            age_times = mixing_age_adjust[key]["times"]
            age_vals = mixing_age_adjust[key]["values"]
            func = scale_up_function(age_times, age_vals, method=4)
            self.age_adjustment_functions[age_idx] = func

    def get_adjustment(self, time: float, mixing_matrix: np.ndarray) -> np.ndarray:
        """
        Apply time-varying age adjustments.
        Returns a new mixing matrix, modified to adjust for dynamic mixing changes for a given point in time.
        """
        for row_index in range(len(AGE_INDICES)):
            if row_index in self.age_adjustment_functions:
                row_multiplier = self.age_adjustment_functions[row_index](time)
            else:
                row_multiplier = 1

            for col_index in range(len(AGE_INDICES)):
                if col_index in self.age_adjustment_functions:
                    col_multiplier = self.age_adjustment_functions[col_index](time)
                else:
                    col_multiplier = 1

                mixing_matrix[row_index, col_index] *= row_multiplier * col_multiplier

        return mixing_matrix

