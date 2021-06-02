import numpy as np

from autumn.tools.inputs.testing.eur_testing_data import (
    get_uk_testing_numbers,
    get_eu_testing_numbers,
)
from autumn.tools.curve import scale_up_function
from autumn.tools.inputs.covid_au.queries import get_dhhs_testing_numbers
from autumn.tools.inputs.covid_phl.queries import get_phl_subregion_testing_numbers
from autumn.tools.inputs.covid_lka.queries import get_lka_testing_numbers
from autumn.tools.inputs.owid.queries import get_international_testing_numbers
from autumn.tools.utils.utils import apply_moving_average


def create_cdr_function(assumed_tests: int, assumed_cdr: float):
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

    assert assumed_tests >= 0, "Number of tests at certain CDR must be positive"
    assert 1.0 >= assumed_cdr >= 0.0, "CDR for given number of tests must be between zero and one"

    # Find the single unknown parameter to the function - i.e. for minus b, where CDR = 1 - exp(-b * t)
    exponent_multiplier = np.log(1.0 - assumed_cdr) / assumed_tests

    # Construct the function based on this parameter
    return lambda tests_per_capita: 1.0 - np.exp(exponent_multiplier * tests_per_capita)


def find_cdr_function_from_test_data(
    assumed_tests_parameter,
    assumed_cdr_parameter,
    smoothing_period,
    country_iso3,
    total_pops,
    subregion=None,
):

    # Get the appropriate testing data
    if country_iso3 == "AUS":
        test_dates, test_values = get_dhhs_testing_numbers()
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

    else:
        test_dates, test_values = get_international_testing_numbers(country_iso3)

    # Convert test numbers to per capita testing rates
    per_capita_tests = [i_tests / sum(total_pops) for i_tests in test_values]

    # Smooth the testing data if requested
    smoothed_per_capita_tests = (
        apply_moving_average(per_capita_tests, smoothing_period)
        if smoothing_period > 1
        else per_capita_tests
    )

    # Calculate CDRs and the resulting CDR function over time
    cdr_from_tests_func = create_cdr_function(assumed_tests_parameter, assumed_cdr_parameter)
    return scale_up_function(
        test_dates,
        [cdr_from_tests_func(i_test_rate) for i_test_rate in smoothed_per_capita_tests],
        smoothness=0.2,
        method=4,
        bound_low=0.0,
    )
