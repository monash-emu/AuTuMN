import numpy as np
from autumn.inputs.owid.queries import get_international_testing_numbers
from autumn.curve import scale_up_function
from autumn import inputs


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
    assumed_tests_parameter, assumed_cdr_parameter, country_iso3, total_pops,
):
    # Tests data
    test_dates, test_values = (
        inputs.get_vic_testing_numbers()
        if country_iso3 == "AUS"
        else get_international_testing_numbers(country_iso3)
    )

    # Convert test numbers to per capita testing rates
    per_capita_tests = [i_tests / sum(total_pops) for i_tests in test_values]

    # Calculate CDRs and the resulting CDR function over time
    cdr_from_tests_func = create_cdr_function(assumed_tests_parameter, assumed_cdr_parameter)
    return scale_up_function(
        test_dates,
        [cdr_from_tests_func(i_test_rate) for i_test_rate in per_capita_tests],
        smoothness=0.2,
        method=4,
        bound_low=0.0,
    )


