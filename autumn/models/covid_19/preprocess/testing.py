from typing import Callable, Optional, Tuple, Any
import numpy as np

from autumn.tools.inputs.testing.eur_testing_data import get_uk_testing_numbers, get_eu_testing_numbers
from autumn.tools.inputs.covid_au.queries import get_vic_testing_numbers
from autumn.tools.inputs.covid_phl.queries import get_phl_subregion_testing_numbers
from autumn.tools.inputs.covid_lka.queries import get_lka_testing_numbers
from autumn.tools.inputs.owid.queries import get_international_testing_numbers
from autumn.tools.inputs import get_population_by_agegroup
from autumn.tools.utils.utils import apply_moving_average
from autumn.tools.curve import scale_up_function
from autumn.models.covid_19.stratifications.agegroup import AGEGROUP_STRATA


def get_testing_numbers_for_region(country_iso3: str, subregion: Optional[str]) -> Tuple[list, list]:
    """
    Use the appropriate function to retrieve the testing numbers applicable to the region being modelled.
    Functions are taken from the autumn input tools module, as above.
    """

    subregion = subregion if subregion else False

    if country_iso3 == "AUS":
        test_dates, test_values = get_vic_testing_numbers()
    elif country_iso3 == "PHL":
        phl_region = subregion.lower() if subregion else "philippines"
        test_dates, test_values = get_phl_subregion_testing_numbers(phl_region)
    elif subregion == "Sabah":
        test_dates, test_values = get_international_testing_numbers(country_iso3)
    elif country_iso3 == "GBR":
        test_dates, test_values = get_uk_testing_numbers()
    elif country_iso3 in ["BEL", "ITA", "SWE", "FRA", "ESP"]:
        test_dates, test_values = get_eu_testing_numbers(country_iso3)
    elif country_iso3 == "LKA":
        test_dates, test_values = get_lka_testing_numbers()
    # elif subregion is not None and country_iso3 == "VNM":
    #     test_dates, test_values = get_vnm_testing_numbers(subregion)
    else:
        test_dates, test_values = get_international_testing_numbers(country_iso3)

    return test_dates, test_values


def create_cdr_function(assumed_tests: int, assumed_cdr: float) -> Callable:
    """
    Factory function for finding CDRs from number of tests done in setting modelled
    To work out the function, only one parameter is needed, so this can be estimated from one known point on the curve,
    being a value of the CDR that is associated with a certain testing rate

    :param assumed_cdr: float
    Value of CDR associated with the testing coverage
    :param assumed_tests: int
    Number of tests needed to result in this CDR
    :return: callable
    Function to provide CDR for a certain number of tests
    """

    # Find the single unknown parameter to the function - i.e. for minus b, where CDR = 1 - exp(-b * t)
    exponent_multiplier = np.log(1. - assumed_cdr) / assumed_tests

    # Construct the function based on this parameter
    def cdr_function(tests_per_capita):
        return 1. - np.exp(exponent_multiplier * tests_per_capita)

    return cdr_function


def inflate_test_data(test_multiplier: float, smoothed_per_capita_tests: list) -> list:
    """
    Apply inflation factor to test numbers if requested.
    Used in the Philippines applications only.
    """

    # Add test datapoints to original series so we can scale-up
    for time, value in zip(test_multiplier.times, test_multiplier.values):
        latest_per_capita_tests = smoothed_per_capita_tests[-1]
        if time not in test_dates:
            test_dates = np.append(test_dates, [time])
            smoothed_per_capita_tests.append(latest_per_capita_tests)

    # Reorder the data
    zipped_lists = zip(test_dates, smoothed_per_capita_tests)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    test_dates, smoothed_per_capita_tests = [list(tup) for tup in tuples]

    testing_scale_up = scale_up_function(test_multiplier.times, test_multiplier.values, method=4)
    new_per_capita_tests = [smoothed_per_capita_tests[i] * testing_scale_up(t) for i, t in enumerate(test_dates)]
    smoothed_per_capita_tests = new_per_capita_tests

    return smoothed_per_capita_tests


def find_cdr_function_from_test_data(
        test_detect_params, iso3: str, region: str, year: int
) -> Callable:
    """
    Sort out case detection rate from testing numbers, sequentially calling the functions above as required.
    """

    # Get the testing population denominator
    testing_pops = get_population_by_agegroup(AGEGROUP_STRATA, iso3, region, year=year)

    # Get the numbers of tests performed
    test_dates, test_values = get_testing_numbers_for_region(iso3, region)

    # Convert test numbers to per capita testing rates
    per_capita_tests = [i_tests / sum(testing_pops) for i_tests in test_values]

    # Smooth the testing data if requested
    if test_detect_params.smoothing_period:
        smoothed_per_capita_tests = apply_moving_average(per_capita_tests, test_detect_params.smoothing_period)
    else:
        smoothed_per_capita_tests = per_capita_tests

    # Scale testing with a time-variant request parameter
    if test_detect_params.test_multiplier:
        smoothed_inflated_per_capita_tests = inflate_test_data(
            test_detect_params.test_multiplier, smoothed_per_capita_tests
        )
    else:
        smoothed_inflated_per_capita_tests = smoothed_per_capita_tests

    # Calculate CDRs and the resulting CDR function over time
    cdr_from_tests_func: Callable[[Any], float] = create_cdr_function(
        test_detect_params.assumed_tests_parameter,
        test_detect_params.assumed_cdr_parameter,
    )

    # Get the final CDR function
    cdr_function = scale_up_function(
        test_dates,
        [cdr_from_tests_func(i_test_rate) for i_test_rate in smoothed_inflated_per_capita_tests],
        smoothness=0.2, method=4, bound_low=0.,
    )

    return cdr_function
