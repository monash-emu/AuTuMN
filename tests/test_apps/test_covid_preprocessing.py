import numpy as np

from autumn.models.sm_sir.detection import create_cdr_function


def test_cdr_intercept():
    """
    Test that there is zero case detection when zero tests are performed
    """

    for cdr_at_1000_tests in np.linspace(0.05, 0.5, 10):
        cdr_function = create_cdr_function(1000.0, cdr_at_1000_tests)
        assert cdr_function(0.0) == 0.0


def test_cdr_values():
    """
    Test that CDR is always a proportion, bounded by zero and one
    """

    for cdr_at_1000_tests in np.linspace(0.05, 0.5, 10):
        cdr_function = create_cdr_function(1000.0, cdr_at_1000_tests)
        for i_tests in list(np.linspace(0.0, 1e3, 11)) + list(np.linspace(0.0, 1e5, 11)):
            assert cdr_function(i_tests) >= 0.0
            assert cdr_function(i_tests) <= 1.0
