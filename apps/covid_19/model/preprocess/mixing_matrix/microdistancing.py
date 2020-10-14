import numpy as np

from typing import List, Callable, Dict

from autumn.curve import scale_up_function, tanh_based_scaleup
from apps.covid_19.model.parameters import MicroDistancingFunc


def get_microdistancing_funcs(
    params: MicroDistancingFunc, locations: List[str], square_mobility_effect: bool
) -> Dict[str, Callable[[float], float]]:
    """
    Work out microdistancing function to be applied as a multiplier to some or all of the Prem locations as requested in
    the microdistancing_locations.
    """

    # Collate the components to the microdistancing function
    microdist_component_funcs = []
    for microdist_type in [param for param in params if "_adjuster" not in param]:
        adjustment_string = f"{microdist_type}_adjuster"
        unadjusted_microdist_func = get_microdist_func_component(params, microdist_type)
        microdist_component_funcs.append(
            get_microdist_adjustment(params, adjustment_string, unadjusted_microdist_func)
        )

    # Generate the overall composite contact adjustment function as the product of the reciprocal all the components
    def microdist_composite_func(time: float) -> float:
        power = 2 if square_mobility_effect else 1
        return np.product(
            [
                (1.0 - microdist_component_funcs[i_func](time)) ** power
                for i_func in range(len(microdist_component_funcs))
            ]
        )

    return {loc: microdist_composite_func for loc in locations}


def get_empiric_microdist_func(params, microdist_type):
    """
    Construct a microdistancing function according to the "empiric" request format.
    """

    micro_times = params[microdist_type].parameters.times
    multiplier = params[microdist_type].parameters.max_effect
    micro_vals = [multiplier * value for value in params[microdist_type].parameters.values]
    return scale_up_function(micro_times, micro_vals, method=4)


def get_microdist_adjustment(params, adjustment_string, unadjusted_microdist_func):
    """
    Adjust an existing microdistancing function according to another function that modifies this function (if one
    exists / has been requested).
    """

    if adjustment_string in params:
        waning_adjustment = get_microdist_func_component(params, adjustment_string)
        return lambda time: unadjusted_microdist_func(time) * waning_adjustment(time)
    else:
        return unadjusted_microdist_func


def get_microdist_func_component(params, adjustment_string):
    """
    Get one function of time using the standard parameter request structure for any microdistancing function or
    adjustment to a microdistancing function.
    """

    if params[adjustment_string].function_type == "tanh":
        return tanh_based_scaleup(**params[adjustment_string].parameters.dict())
    elif params[adjustment_string].function_type == "empiric":
        return get_empiric_microdist_func(params, adjustment_string)
