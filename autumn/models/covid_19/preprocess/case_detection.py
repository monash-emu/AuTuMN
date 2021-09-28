from typing import List, Tuple

from summer.compute import ComputedValueProcessor

from autumn.models.covid_19.parameters import Country, Population, TestingToDetection
from autumn.models.covid_19.preprocess.testing import find_cdr_function_from_test_data
from autumn.tools import inputs
from autumn.models.covid_19.constants import AGEGROUP_STRATA


def get_testing_pop(country: Country, pop: Population) -> Tuple[list, str]:
    """
    Returns the age-stratified population used for case detection testing.
    Use state denominator for testing rates for the Victorian health cluster models and temporarily use
    Philippines regional pops for all the Philippines sub-regions
    """

    testing_region = "Victoria" if country.iso3 == "AUS" else pop.region
    testing_year = 2020 if country.iso3 == "AUS" else pop.year
    testing_pop = inputs.get_population_by_agegroup(
        AGEGROUP_STRATA, country.iso3, testing_region, year=testing_year
    )
    return testing_pop, testing_region


def build_detected_proportion_func(
        agegroup_strata: List[str], country: Country, pop: Population, testing: TestingToDetection,
):
    """
    Returns a time varying function that gives us the proportion of cases detected.
    There is a previous version that uses a hyperbolic tan function that can be pulled out of earlier versions of the
    repo.
    """

    # Process the parameters
    assumed_tests_parameter = testing.assumed_tests_parameter
    assumed_cdr_parameter = testing.assumed_cdr_parameter
    smoothing_period = testing.smoothing_period
    test_multiplier = testing.test_multiplier if testing.test_multiplier else None

    # Get the empiric data
    testing_pop, testing_region = get_testing_pop(country, pop)

    # Create the function
    detected_proportion = find_cdr_function_from_test_data(
        assumed_tests_parameter, assumed_cdr_parameter, smoothing_period, country.iso3, testing_pop,
        subregion=testing_region, test_multiplier=test_multiplier
    )

    return detected_proportion


class CdrProc(ComputedValueProcessor):
    """
    Calculate prevalence from the active disease compartments.
    """

    def __init__(self, detected_proportion_func):
        self.detected_proportion_func = detected_proportion_func

    def process(self, compartment_values, computed_values, time):
        """
        Calculate the actual prevalence during run-time
        """

        return self.detected_proportion_func(time)
