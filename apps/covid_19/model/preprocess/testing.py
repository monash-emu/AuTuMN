import numpy as np


def create_cdr_function(assumed_cdr: float, assumed_tests: int):
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

    # Find the single unknown parameter to the function
    exponent_multiplier = assumed_tests * np.log(1. - assumed_cdr)

    # Construct the function based on this parameter
    return lambda tests_per_capita: 1. - np.exp(exponent_multiplier * tests_per_capita)
