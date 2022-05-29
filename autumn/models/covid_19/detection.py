from typing import Callable, Optional, Tuple, List
import numpy as np

from summer.compute import ComputedValueProcessor
from autumn.core.inputs.covid_btn.queries import get_btn_testing_numbers

from autumn.core.inputs.testing.eur_testing_data import (
    get_uk_testing_numbers,
    get_eu_testing_numbers,
)
from autumn.core.inputs.covid_au.queries import get_vic_testing_numbers
from autumn.core.inputs.covid_phl.queries import get_phl_subregion_testing_numbers
from autumn.core.inputs.covid_lka.queries import get_lka_testing_numbers
from autumn.core.inputs.covid_mmr.queries import get_mmr_testing_numbers
from autumn.core.inputs.covid_bgd.queries import get_coxs_bazar_testing_numbers
from autumn.core.inputs.owid.queries import get_international_testing_numbers
from autumn.model_features.curve import scale_up_function


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


def get_testing_numbers_for_region(
    country_iso3: str, subregion: Optional[str]
) -> Tuple[list, list]:
    """
    Use the appropriate function to retrieve the testing numbers applicable to the region being modelled.
    Functions are taken from the autumn input tools module, as above.
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

    assert len(test_dates) == len(
        test_values
    ), "Length of test dates and test values are not equal"

    return test_dates, test_values


def create_cdr_function(assumed_tests: int, assumed_cdr: float, floor: float=0.) -> Callable:
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

    # Find the single unknown parameter to the function - i.e. for minus b, where CDR = 1 - exp(-b * t)
    exponent_multiplier = np.log((1.0 - assumed_cdr) / (1.0 - floor)) / assumed_tests

    # Construct the function based on this parameter
    def cdr_function(tests_per_capita):
        return 1.0 - np.exp(exponent_multiplier * tests_per_capita) * (1.0 - floor)

    return cdr_function


def inflate_test_data(
    test_multiplier: float, test_dates: list, test_values: list
) -> List[float]:
    """
    Apply inflation factor to test numbers if requested.
    Used in the Philippines applications only.
    """

    # Add future test datapoints to original series so we can scale-up
    latest_per_capita_tests = test_values[-1]
    for time, value in zip(test_multiplier.times, test_multiplier.values):
        if time not in test_dates:
            test_dates = np.append(test_dates, [time])
            test_values.append(latest_per_capita_tests)

    # Reorder the data
    sorted_pairs = sorted(zip(test_dates, test_values))
    tuples = zip(*sorted_pairs)
    test_dates, test_values = [list(tup) for tup in tuples]

    # Create scale-up function
    testing_scale_up = scale_up_function(
        test_multiplier.times, test_multiplier.values, method=4
    )

    # Scale up added tests
    return [
        test_values[val] * testing_scale_up(time) for val, time in enumerate(test_dates)
    ]
