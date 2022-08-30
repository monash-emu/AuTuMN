from typing import Tuple, Callable, Optional

from computegraph.types import Function
from summer.parameters import Time, Data

from autumn.core.model_builder import ModelBuilder
from .parameters import TestingToDetection, Population
from autumn.core.inputs import get_population_by_agegroup
from autumn.settings import COVID_BASE_AGEGROUPS
from autumn.model_features.curve.interpolate import build_sigmoidal_multicurve, get_scale_data
from computegraph import jaxify

fnp = jaxify.get_modules()["numpy"]


def find_cdr_function_from_test_data(
    builder: ModelBuilder,
    cdr_params: TestingToDetection,
    iso3: str,
    region: Optional[str],
    year: int,
    smoothing_period=1,
) -> Callable:
    """
    Sort out case detection rate from testing numbers, sequentially calling the functions above as
    required.
    Args:
        cdr_params: The user-requests re the testing process
        iso3: The country
        region: The subregion of the country being simulated, if any
        year: The year from which the population data should come
        smooth_period: The period in days over which testing data should be smoothed
    Return:
        The function that takes time as its input and returns the CDR
    """

    # Get the numbers of tests performed
    from autumn.core.inputs.testing.testing_data import get_testing_numbers_for_region

    test_df = get_testing_numbers_for_region(iso3, region)
    smoothed_test_df = test_df.rolling(window=smoothing_period).mean().dropna()

    # Convert to per capita testing rates
    total_pop = sum(get_population_by_agegroup(COVID_BASE_AGEGROUPS, iso3, region, year=year))
    per_capita_tests_df = smoothed_test_df / total_pop

    def cdr_from_tests_func(
        tests_per_capita, assumed_cdr_parameter, assumed_tests_parameter, floor_value
    ):
        # Find the single unknown parameter to the function - i.e. for minus b,
        # where CDR = 1 - (1 - f) * exp(-b * t)
        exponent_multiplier = (
            fnp.log((1.0 - assumed_cdr_parameter) / (1.0 - floor_value)) / assumed_tests_parameter
        )
        cdr = 1.0 - fnp.exp(exponent_multiplier * tests_per_capita) * (1.0 - floor_value)
        return cdr

    # Construct the function based on this parameter
    # def cdr_function(tests_per_capita, exponent_multipler, floor_value):
    #    return 1.0 - fnp.exp(exponent_multiplier * tests_per_capita) * (1.0 - floor_value)

    # return cdr_function

    # cdr_from_tests_func = create_cdr_function(*cdr_test_params)

    # Get the final CDR function
    times = per_capita_tests_df.index
    # values = per_capita_tests_df.apply(cdr_from_tests_func)

    # Add the smoothed per capita testing rates
    per_capita_test_data = Data(fnp.array(per_capita_tests_df))

    # Add our function that computes CDR for these rates (and parameters)
    cdr_test_data = builder.get_mapped_func(
        cdr_from_tests_func, cdr_params, {"tests_per_capita": per_capita_test_data}
    )

    # Add an output that transforms this into scale_data for the sigmoidal curve interpolator
    cdr_ydata = Function(get_scale_data, [cdr_test_data])

    # Define a smoothed sigmoidal curve function
    cdr_smoothed_func = build_sigmoidal_multicurve(times)

    # Return the final Function object that will be used inside the model
    return Function(cdr_smoothed_func, [Time, cdr_ydata])


def get_cdr_func(
    builder: ModelBuilder,
    detect_prop: float,
    testing_params: TestingToDetection,
    pop_params: Population,
    iso3: str,
) -> Tuple[callable, callable]:
    """
    The master function that can call various approaches to calculating the proportion of cases
    detected over time.
    Currently just supporting two approaches, but would be the entry point if more approaches
    were added:
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
            builder,
            testing_params,
            iso3,
            pop_params.region,
            pop_params.year,
            testing_params.smoothing_period,
        )
    else:
        cdr_func = detect_prop

    non_detect_func = 1.0 - cdr_func

    return cdr_func, non_detect_func
