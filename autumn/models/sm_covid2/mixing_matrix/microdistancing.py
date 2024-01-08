from typing import Callable, Dict

import numpy as np

from autumn.models.sm_covid2.constants import LOCATIONS
from autumn.model_features.curve import scale_up_function, tanh_based_scaleup
from autumn.core.utils.utils import return_constant_value, get_product_two_functions
from autumn.core.inputs.covid_survey.queries import get_percent_mc


ADJUSTER_SUFFIX = "_adjuster"


def get_microdistancing_funcs(
    params, square_mobility_effect: bool, iso3: str
) -> Dict[str, Callable[[float], float]]:
    """
    Returns a dictionary of time-varying functions.
    Each key is a location and each function represents microdistancing effects at that location.
    `params` can specify one or more microdistancing functions, which may also have an "adjuster" function.

    Args:
        params: Microdistancing function requests
        square_mobility_effect: Whether to square the effect on transmission (to account for both infector and infectee)
        iso3: ISO3 code of the modelled country

    Returns:
        The microdistancing functions that should be applied to each of the mobility locations

    """

    # Supports any number of microdistancing functions contributing to contact rates, with any user-defined names
    final_adjustments = {}
    power = 2 if square_mobility_effect else 1

    # For each mobility location ...
    for loc in LOCATIONS:
        microdist_component_funcs = []

        # ... work through the microdistancing functions and apply them if relevant
        for key, func_params in params.items():

            # Ignore 'adjuster' functions - we'll use these later
            if key.endswith(ADJUSTER_SUFFIX):
                continue

            # Build the raw function of time according to user requests
            microdist_func = get_microdist_func_component(func_params, iso3)

            # Adjust an existing microdistancing function with another function if specified
            adjustment_key = f"{key}_adjuster"
            adjustment_func_params = params.get(adjustment_key)
            if adjustment_func_params:

                # Ensure the adjuster applies to the same locations as the original
                assert params[f"{key}_adjuster"].locations == params[f"{key}"].locations

                # An adjustment function is applied to the original function
                waning_adjustment = get_microdist_func_component(adjustment_func_params, iso3)

            # Otherwise no adjustments
            else:
                waning_adjustment = return_constant_value(1.0)

            if loc in params[key].locations:
                microdist_component_funcs.append(
                    get_product_two_functions(microdist_func, waning_adjustment)
                )

        # Generate the overall composite contact adjustment function as the product of the reciprocal all the effects
        if len(microdist_component_funcs) > 0:

            def microdist_composite_func(time: float) -> float:
                effects = [(1.0 - func(time)) ** power for func in microdist_component_funcs]
                return np.product(effects)

            final_adjustments[loc] = microdist_composite_func

        # Get the final location-specific microdistancing functions

    return final_adjustments


def get_microdist_func_component(func_params, iso3: str):
    """
    Get a single function of time using the standard parameter request structure for any microdistancing function, or
    adjustment to a microdistancing function.
    In future, this could use more general code for requesting functions of time.

    Args:
        func_params: The parameters used to define the microdistancing function
        iso3: ISO3 code of the modelled country

    Returns:
        Function of time returning a scalar

    """

    if func_params.function_type == "tanh":
        return tanh_based_scaleup(**func_params.parameters.dict())

    elif func_params.function_type == "empiric":
        micro_times = func_params.parameters.times
        multiplier = func_params.parameters.max_effect
        micro_vals = [multiplier * value for value in func_params.parameters.values]
        return scale_up_function(micro_times, micro_vals, method=4)

    elif func_params.function_type == "constant":
        return return_constant_value(func_params.parameters.effect)

    elif func_params.function_type == "survey":
        survey_data = get_percent_mc(iso3)
        micro_times = list(survey_data[0])
        multiplier = func_params.parameters.effect
        micro_vals = [multiplier * value for value in survey_data[1]]
        return scale_up_function(micro_times, micro_vals, method=4)
