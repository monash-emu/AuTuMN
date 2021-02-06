import numpy as np

from typing import List, Callable, Dict

from autumn.curve import scale_up_function, tanh_based_scaleup
from apps.covid_19.model.parameters import MicroDistancingFunc
from apps.covid_19.model.preprocess.mixing_matrix.mobility import LOCATIONS

ADJUSTER_SUFFIX = "_adjuster"


def get_microdistancing_funcs(
    params: Dict[str, MicroDistancingFunc], square_mobility_effect: bool
) -> Dict[str, Callable[[float], float]]:
    """
    Returns a dictionary of time-varying functions.
    Each key is a location and each function represents microdistancing effects at that location.
    `params` can specify one or more microdistancing functions, which may also have an "adjuster" function.
    """

    # Supports any number of microdistancing functions, with any user-defined names
    microdist_component_funcs = []
    final_adjustments = {}

    # For each Prem location ...
    for loc in LOCATIONS:

        # ... work through the microdistancing functions and apply them if relevant
        for key, func_params in params.items():
            if key.endswith(ADJUSTER_SUFFIX):
                # Ignore 'adjuster' functions - we'll use these later
                continue

            # Build the raw function of time according to user requests
            microdist_func = get_microdist_func_component(func_params)

            # Adjust an existing microdistancing function with another function if specified
            adjustment_key = f"{key}_adjuster"
            adjustment_func_params = params.get(adjustment_key)
            if adjustment_func_params:
                # An adjustment function is applied to the original function
                waning_adjustment = get_microdist_func_component(adjustment_func_params)
                microdistancing_func = lambda t: microdist_func(t) * waning_adjustment(t)
            else:
                # Just use the original function
                microdistancing_func = microdist_func

            if loc in params[key].locations:
                microdist_component_funcs.append(microdistancing_func)

        # Generate the overall composite contact adjustment function as the product of the reciprocal all the components
        if len(microdist_component_funcs) > 0:
            def microdist_composite_func(time: float) -> float:
                power = 2 if square_mobility_effect else 1
                return np.product([(1.0 - func(time)) ** power for func in microdist_component_funcs])
        else:
            def microdist_composite_func(time: float) -> float:
                return 1.

        # Get the final location-specific microdistancing functions
        final_adjustments[loc] = microdist_composite_func

    return final_adjustments


def get_microdist_func_component(func_params: MicroDistancingFunc):
    """
    Get one function of time using the standard parameter request structure for any microdistancing function or
    adjustment to a microdistancing function.
    """
    if func_params.function_type == "tanh":
        return tanh_based_scaleup(**func_params.parameters.dict())
    elif func_params.function_type == "empiric":
        micro_times = func_params.parameters.times
        multiplier = func_params.parameters.max_effect
        micro_vals = [multiplier * value for value in func_params.parameters.values]
        return scale_up_function(micro_times, micro_vals, method=4)
