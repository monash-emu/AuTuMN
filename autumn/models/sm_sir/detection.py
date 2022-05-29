from typing import Tuple

from autumn.models.covid_19.detection import find_cdr_function_from_test_data
from .parameters import TestingToDetection, Population


def get_cdr_func(
        detect_prop: float,
        testing_params: TestingToDetection,
        pop_params: Population,
        iso3: str,
) -> Tuple[callable, callable]:
    """
    The master function that can call various approaches to calculating the proportion of cases detected over time.
    Currently just supporting two approaches, but would be the entry point if more approaches were added:
        - Testing-based case detection
        - Constant case detection fraction

    Args:
        detect_prop: Back-up single value to set a constant case detection rate over time
        testing_params: Parameters to specify the relationship between CDR and testing, if requested
        pop_params: Population-related parameters
        iso3: Country code

    Returns:
        The case detection rate function of time

    """

    if testing_params:
        cdr_func = find_cdr_function_from_test_data(testing_params, iso3, pop_params.region, pop_params.year)
    else:

        def cdr_func(time, computed_values):
            return detect_prop

    def non_detect_func(time, computed_values):
        return 1.0 - computed_values["cdr"]

    return cdr_func, non_detect_func