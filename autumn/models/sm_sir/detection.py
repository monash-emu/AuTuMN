from typing import Tuple, Callable, Any, Optional
import numpy as np
import pandas as pd

from .parameters import TestingToDetection, Population
from autumn.core.inputs import get_population_by_agegroup
from autumn.settings import COVID_BASE_AGEGROUPS
from autumn.core.utils.utils import apply_moving_average
from autumn.model_features.curve import scale_up_function
from autumn.core.inputs.testing.eur_testing_data import get_uk_testing_numbers, get_eu_testing_numbers
from autumn.core.inputs.covid_au.queries import get_vic_testing_numbers
from autumn.core.inputs.covid_phl.queries import get_phl_subregion_testing_numbers
from autumn.core.inputs.covid_lka.queries import get_lka_testing_numbers
from autumn.core.inputs.covid_mmr.queries import get_mmr_testing_numbers
from autumn.core.inputs.covid_bgd.queries import get_coxs_bazar_testing_numbers
from autumn.core.inputs.owid.queries import get_international_testing_numbers
from autumn.core.inputs.covid_btn.queries import get_btn_testing_numbers


def get_testing_numbers_for_region(
    country_iso3: str, 
    subregion: Optional[str]
) -> Tuple[list, list]:
    """
    Use the appropriate function to retrieve the testing numbers applicable to the region being modelled.
    Functions are taken from the autumn input tools module, as above.

    Args:
        country_iso3: ISO3 code for the country being simulated
        subregion: Name of the country's subregion being considered or None if it is a national level
    Return:
        The testing data for the country
    """

    subregion = subregion or False

    if country_iso3 == "AUS":
        test_dates, test_values = get_vic_testing_numbers()
    elif country_iso3 == "PHL":
        phl_region = subregion.lower() if subregion else "philippines"
        test_dates, test_values = get_phl_subregion_testing_numbers(phl_region)
    elif subregion == "Sabah":
        test_dates, test_values = get_international_testing_numbers(country_iso3)
    elif country_iso3 == "GBR":
        test_dates, test_values = get_uk_testing_numbers()
    elif country_iso3 in {"BEL", "ITA", "SWE", "FRA", "ESP"}:
        test_dates, test_values = get_eu_testing_numbers(country_iso3)
    elif country_iso3 == "LKA":
        test_dates, test_values = get_lka_testing_numbers()
    elif country_iso3 == "MMR":
        test_dates, test_values = get_mmr_testing_numbers()
    elif country_iso3 == "BGD" and subregion == "FDMN":
        test_dates, test_values = get_coxs_bazar_testing_numbers()
    elif country_iso3 == "BTN":
        test_dates, test_values = get_btn_testing_numbers(subregion)

    else:
        test_dates, test_values = get_international_testing_numbers(country_iso3)

    msg = "Length of test dates and test values are not equal"
    assert len(test_dates) == len(test_values), msg

    return pd.Series(test_values, index=test_dates)


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
    test_detect_params, 
    iso3: str, 
    region: str, 
    year: int,
    smoothing_period=1,
) -> Callable:
    """
    Sort out case detection rate from testing numbers, sequentially calling the functions above as required.
    """

    # Get the testing population denominator
    testing_pops = get_population_by_agegroup(COVID_BASE_AGEGROUPS, iso3, region, year=year)

    # Get the numbers of tests performed
    test_df = get_testing_numbers_for_region(iso3, region)
    smoothed_test_df = test_df.rolling(window=smoothing_period).mean().dropna()

    # Convert to per capita testing rates
    per_capita_tests_df = smoothed_test_df / sum(testing_pops)

    # Check data
    msg = "Negative test values present"
    assert (per_capita_tests_df > 0).all()

    # Calculate CDRs and the resulting CDR function
    cdr_from_tests_func: Callable[[Any], float] = create_cdr_function(
        test_detect_params.assumed_tests_parameter,
        test_detect_params.assumed_cdr_parameter,
        test_detect_params.floor_value,
    )

    # Get the final CDR function
    cdr_function = scale_up_function(
        per_capita_tests_df.index,
        [cdr_from_tests_func(i_test_rate) for i_test_rate in per_capita_tests_df],
        smoothness=0.2,
        method=4,
        bound_low=0.0,
    )

    return cdr_function


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
