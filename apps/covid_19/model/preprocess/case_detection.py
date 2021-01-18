from typing import List

from autumn import inputs
from autumn.curve import tanh_based_scaleup
from apps.covid_19.model.parameters import TestingToDetection, CaseDetection, Country, Population
from apps.covid_19.model.preprocess.testing import find_cdr_function_from_test_data


def get_testing_pop(agegroup_strata: List[str], country: Country, pop: Population):
    """
    Returns the age stratified population used for case detection testing.
    Use state denominator for testing rates for the Victorian health cluster models and temporarily use
    Philippines regional pops for all the Philippines sub-regions
    """
    testing_region = "Victoria" if country.iso3 == "AUS" else pop.region
    testing_year = 2020 if country.iso3 == "AUS" else pop.year
    testing_pop = inputs.get_population_by_agegroup(
        agegroup_strata, country.iso3, testing_region, year=testing_year
    )
    return testing_pop, testing_region


def build_detected_proportion_func(
    agegroup_strata: List[str],
    country: Country,
    pop: Population,
    testing: TestingToDetection,
    case_detection: CaseDetection,
):
    """
    Returns a time varying function that gives us the proportion of cases detected.
    """
    if testing is not None:
        # More empiric approach based on per capita testing rates
        assumed_tests_parameter = testing.assumed_tests_parameter
        assumed_cdr_parameter = testing.assumed_cdr_parameter
        smoothing_period = testing.smoothing_period

        testing_pop, testing_region = get_testing_pop(agegroup_strata, country, pop)
        detected_proportion = find_cdr_function_from_test_data(
            assumed_tests_parameter,
            assumed_cdr_parameter,
            smoothing_period,
            country.iso3,
            testing_pop,
            subregion=testing_region,
        )
    else:
        # Approach based on a hyperbolic tan function
        detected_proportion = tanh_based_scaleup(
            case_detection.maximum_gradient,
            case_detection.max_change_time,
            case_detection.start_value,
            case_detection.end_value,
        )

    return detected_proportion
