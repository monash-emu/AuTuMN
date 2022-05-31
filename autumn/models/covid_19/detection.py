from typing import Callable

from summer.compute import ComputedValueProcessor


class CdrProc(ComputedValueProcessor):
    """
    Calculate prevalence from the active disease compartments.
    """

    def __init__(self, detected_proportion_func):
        self.detected_proportion_func = detected_proportion_func

    def process(self, compartment_values, computed_values, time):
        """
        Calculate the actual prevalence during run-time.
        """

        return self.detected_proportion_func(time)
