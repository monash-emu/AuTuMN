import numpy as np


from apps import covid_19
from apps.covid_19.model.preprocess.seasonality import get_seasonal_forcing


def test_cdr_intercept():
    """
    Test that there is zero case detection when zero tests are performed
    """

    for cdr_at_1000_tests in np.linspace(0.05, 0.5, 10):
        cdr_function = covid_19.model.preprocess.testing.create_cdr_function(
            1000.0, cdr_at_1000_tests
        )
        assert cdr_function(0.0) == 0.0


def test_cdr_values():
    """
    Test that CDR is always a proportion, bounded by zero and one
    """

    for cdr_at_1000_tests in np.linspace(0.05, 0.5, 10):
        cdr_function = covid_19.model.preprocess.testing.create_cdr_function(
            1000.0, cdr_at_1000_tests
        )
        for i_tests in list(np.linspace(0.0, 1e3, 11)) + list(np.linspace(0.0, 1e5, 11)):
            assert cdr_function(i_tests) >= 0.0
            assert cdr_function(i_tests) <= 1.0


def test_no_seasonal_forcing():
    """
    Test seasonal forcing function returns the average value when the magnitude is zero
    """

    seasonal_forcing_function = get_seasonal_forcing(365.0, 0.0, 0.0, 1.0)
    for i_time in np.linspace(-100.0, 100.0, 50):
        assert seasonal_forcing_function(i_time) == 1.0


def test_peak_trough_seasonal_forcing():
    """
    Test seasonal forcing returns the peak and trough values appropriately
    """

    seasonal_forcing_function = get_seasonal_forcing(365.0, 0.0, 2.0, 1.0)
    assert seasonal_forcing_function(0.0) == 2.0
    assert seasonal_forcing_function(365.0) == 2.0
    assert seasonal_forcing_function(365.0 / 2.0) == 0.0
