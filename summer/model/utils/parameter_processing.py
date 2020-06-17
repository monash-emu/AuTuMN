import numpy
from scipy.integrate import quad
from summer.model import order_dict_by_keys, add_zero_to_age_breakpoints
from copy import copy


"""
pallettes of functions that may be useful for creating parameter values to submit to the SUMMER module
"""


def change_parameter_unit(parameter_dict, multiplier):
    """
    used to adapt the parameters according their unit - for example, could be used for models that are running in time
        steps that are different from the time step assumed by the input parameter

    :param parameter_dict: dict
        dictionary whose values need to be adjusted
    :param multiplier: float
        multiplier
    :return: dict
        dictionary with values multiplied by the multiplier argument
    """
    return {
        param_key: param_value * multiplier for param_key, param_value in parameter_dict.items()
    }


def add_w_to_param_names(parameter_dict):
    """
    add a "W" string to the end of the parameter name to indicate that the parameter should over-write up the chain of
        stratification, rather than being a multiplier or adjustment function for the upstream parameters

    :param parameter_dict: dict
        the dictionary before the adjustments
    :return: dict
        same dictionary but with the "W" string added to each of the keys
    """
    return {str(key) + "W": value for key, value in parameter_dict.items()}


def get_average_value_of_function(input_function, start_value, end_value):
    """
    use numeric integration to find the average value of a function between two extremes

    :param input_function: function
        function to be interrogated
    :param start_value: float
        lower limit of the independent variable over which to integrate the function
    :param end_value: float
        upper limit of the independent variable over which to integrate the function
    """
    return quad(input_function, start_value, end_value)[0] / (end_value - start_value)


def get_parameter_dict_from_function(input_function, breakpoints, upper_value=100.0):
    """
    create a dictionary of parameter values from a continuous function, an arbitrary upper value and some breakpoints
    within which to evaluate the function
    """
    revised_breakpoints = copy.copy(add_zero_to_age_breakpoints(breakpoints))
    revised_breakpoints.append(upper_value)
    param_values = []
    for n_breakpoint in range(len(revised_breakpoints) - 1):
        param_values.append(
            get_average_value_of_function(
                input_function,
                revised_breakpoints[n_breakpoint],
                revised_breakpoints[n_breakpoint + 1],
            )
        )
    return {str(key): value for key, value in zip(revised_breakpoints, param_values)}


def substratify_parameter(
    parameter_to_stratify, stratum_to_split, param_value_dict, breakpoints, stratification="age",
):
    """
    produce dictionary revise a stratum of a parameter that has been split at a higher level from dictionary of the
        values for each stratum of the higher level of the split

    :param parameter_to_stratify: str
        name of the parameter that was split at the higher level
    :param stratum_to_split: str
        stratum whose values should be revised
    :param param_value_dict: dict
        dictionary with keys age breakpoints and values parameter values
    :param breakpoints: list
        list of age breakpoints submitted as integer
    :param stratification: str
        name of the stratification this is being applied to
    :return: dict
        dictionary with keys being upstream stratified parameter to split and keys dictionaries with their keys the
            current stratum of interest and values the parameter multiplier
    """
    return {
        parameter_to_stratify
        + "X"
        + stratification
        + "_"
        + str(age_group): {stratum_to_split: param_value_dict[str(age_group)]}
        for age_group in add_zero_to_age_breakpoints(breakpoints)
    }


"""
functions that return a function of an independent variable
expected that this will often be a model quantity, such as time or age
"""


def get_parameter_dict_from_function(input_function, breakpoints, upper_value=100.0):
    """
    create a dictionary of parameter values from a continuous function, an arbitrary upper value and some breakpoints
        within which to evaluate the function
    """
    revised_breakpoints = copy(add_zero_to_age_breakpoints(breakpoints))
    revised_breakpoints.append(upper_value)
    param_values = []
    for n_breakpoint in range(len(revised_breakpoints) - 1):
        param_values.append(
            get_average_value_of_function(
                input_function,
                revised_breakpoints[n_breakpoint],
                revised_breakpoints[n_breakpoint + 1],
            )
        )
    return {str(key): value for key, value in zip(revised_breakpoints, param_values)}


def create_step_function_from_dict(input_dict):
    """
    create a step function out of dictionary with numeric keys and values, where the keys determine the values of the
        independent variable at which the steps between the output values occur

    :param input_dict: dict
        dictionary in standard format with numeric keys for the points at which the steps occur and numeric values for
            the values to be returned from these points onwards
    :return: function
        the function constructed from input data
    """
    dict_keys, dict_values = order_dict_by_keys(input_dict)

    def step_function(input_value):
        if input_value >= dict_keys[-1]:
            return dict_values[-1]
        else:
            for key in range(len(dict_keys)):
                if input_value < dict_keys[key + 1]:
                    return dict_values[key]

    return step_function


def sinusoidal_scaling_function(start_time, baseline_value, end_time, final_value):
    """
    in order to implement scale-up functions over time, use the cosine function to produce smooth scale-up functions
        from one point to another, returning the starting value before the starting point and the final value after the
        end point

    :param start_time: float
        starting value of the independent variable
    :param baseline_value: float
        starting value of the dependent variable
    :param end_time: float
        final value of the independent variable
    :param final_value: float
        final value of the dependent variable
    :return:
        function scaling from the starting value to the final value
    """

    def sinusoidal_function(time):
        if not isinstance(time, float):
            raise ValueError("value provided to scaling function not a float")
        elif start_time > end_time:
            raise ValueError("start time is later than end time")
        elif time < start_time:
            return baseline_value
        elif start_time <= time <= end_time:
            return baseline_value + (final_value - baseline_value) * (
                0.5 - 0.5 * numpy.cos((time - start_time) * numpy.pi / (end_time - start_time))
            )
        else:
            return final_value

    return sinusoidal_function


def logistic_scaling_function(parameter):
    """
    a specific sigmoidal form of function that scales up from zero to one around the point of the value of parameter

    :param parameter: float
        the single parameter to the function
    :return: function
        the logistic function
    """
    return lambda x: 1.0 - 1.0 / (1.0 + numpy.exp(-(parameter - x)))
