from typing import Tuple, Callable, Optional
import numpy as np

from autumn.core.inputs.testing.testing_data import get_testing_numbers_for_region
from .parameters import TestingToDetection, Population
from autumn.core.inputs import get_population_by_agegroup
from autumn.settings import COVID_BASE_AGEGROUPS
from autumn.model_features.curve import scale_up_function


def create_cdr_function(assumed_tests: int, assumed_cdr: float, floor_value: float=0.) -> Callable:
    """
    Factory function for finding CDRs from number of tests done in setting modelled
    To work out the function, only one parameter is needed, so this can be estimated from one known point on the curve,
    being a value of the CDR that is associated with a certain testing rate

    Args:
        assumed_tests: Value of CDR associated with the testing coverage
        assumed_cdr: Number of tests needed to result in this CDR
        floor_cdr: Floor value for the case detection rate

    Returns:
        Callable: Function to provide CDR for a certain number of tests

    """

    # Find the single unknown parameter to the function - i.e. for minus b, where CDR = 1 - (1 - f) * exp(-b * t)
    exponent_multiplier = np.log((1.0 - assumed_cdr) / (1.0 - floor_value)) / assumed_tests

    # Construct the function based on this parameter
    def cdr_function(tests_per_capita):
        return 1.0 - np.exp(exponent_multiplier * tests_per_capita) * (1.0 - floor_value)

    return cdr_function


def find_cdr_function_from_test_data(
    cdr_params: TestingToDetection,
    iso3: str, 
    region: Optional[str], 
    year: int,
    smoothing_period=1,
) -> Callable:
    """
    Sort out case detection rate from testing numbers, sequentially calling the functions above as required.
    
    Args:
        cdr_params: The user-requests re the testing process
        iso3: The country
        region: The subregion of the country being simulated, if any
        year: The year from which the population data should come
        smooth_period: The period in days over which testing data should be smoothed
    Return:
        The function that takes time as its input and returns the CDR
    """

    # Get the numbers of tests performed
    test_df = get_testing_numbers_for_region(iso3, region)
    smoothed_test_df = test_df.rolling(window=smoothing_period).mean().dropna()

    # Convert to per capita testing rates
    total_pop = sum(get_population_by_agegroup(COVID_BASE_AGEGROUPS, iso3, region, year=year))
    per_capita_tests_df = smoothed_test_df / total_pop

    # Calculate CDRs and the resulting CDR function
    cdr_test_params = cdr_params.assumed_tests_parameter, cdr_params.assumed_cdr_parameter, cdr_params.floor_value
    cdr_from_tests_func = create_cdr_function(*cdr_test_params)

    # Get the final CDR function
    times = per_capita_tests_df.index
    values = per_capita_tests_df.apply(cdr_from_tests_func)
    return scale_up_function(times, values, smoothness=0.2, method=4, bound_low=0.0)


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
        cdr_func = find_cdr_function_from_test_data(
            testing_params, 
            iso3, 
            pop_params.region, 
            pop_params.year,
            testing_params.smoothing_period,
        )
    else:

        def cdr_func(time, computed_values):
            return detect_prop

    def non_detect_func(time, computed_values):
        return 1.0 - computed_values["cdr"]

    return cdr_func, non_detect_func
