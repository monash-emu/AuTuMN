"""
Miscellanious utility functions.
If several functions here develop a theme, consider reorganising them into a module.
"""
import subprocess as sp

import numpy
from summer_py.summer_model import find_name_components


def get_git_hash():
    """
    Return the current commit hash, or an empty string.
    """
    return run_command("git rev-parse HEAD").strip()


def get_git_branch():
    """
    Return the current git branch, or an empty string
    """
    return run_command("git rev-parse --abbrev-ref HEAD").strip()


def run_command(cmds):
    """
    Run a process and retun the stdout.
    """
    try:
        result = sp.run(cmds, shell=True, check=True, stdout=sp.PIPE, encoding="utf-8")
        return result.stdout
    except sp.CalledProcessError:
        return ""


def return_function_of_function(inner_function, outer_function):
    """
    Returns a chained function from two functions
    """
    return lambda value: outer_function(inner_function(value))


def step_function_maker(start_time, end_time, value):
    def my_function(time):
        if start_time <= time <= end_time:
            y = value
        else:
            y = 0.0
        return y

    return my_function


def progressive_step_function_maker(start_time, end_time, average_value, scaling_time_fraction=0.2):
    """
    Make a step_function with linear increasing and decreasing slopes to simulate more progressive changes
    :param average_value: targeted average value (auc)
    :param scaling_time_fraction: fraction of (end_time - start_time) used for scaling up the value (classic step
    function obtained with scaling_time_fraction=0, triangle function obtained with scaling_time_fraction=.5)
    :return: function
    """
    assert scaling_time_fraction <= 0.5, "scaling_time_fraction must be <=.5"

    def my_function(time):
        if time <= start_time or time >= end_time:
            y = 0
        else:
            total_time = end_time - start_time
            plateau_height = average_value / (1.0 - scaling_time_fraction)
            slope = plateau_height / (scaling_time_fraction * total_time)
            intercept_left = -slope * start_time
            intercept_right = slope * end_time
            if (
                start_time + scaling_time_fraction * total_time
                <= time
                <= end_time - scaling_time_fraction * total_time
            ):
                y = plateau_height
            elif time < start_time + scaling_time_fraction * total_time:
                y = slope * time + intercept_left
            else:
                y = -slope * time + intercept_right
        return y

    return my_function


def change_parameter_unit(parameter_dict, multiplier):
    """
    used to adapt the latency parameters from the earlier functions according to whether they are needed as by year
        rather than by day
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
    add a "W" string to the end of the parameter name to indicate that we should over-write up the chain
    :param parameter_dict: dict
        the dictionary before the adjustments
    :return: dict
        same dictionary but with the "W" string added to each of the keys
    """
    return {str(age_group) + "W": value for age_group, value in parameter_dict.items()}


def find_stratum_index_from_string(compartment, stratification, remove_stratification_name=True):
    """
    finds the stratum which the compartment (or parameter) name falls in when provided with the compartment name and the
        name of the stratification of interest
    for example, if the compartment name was infectiousXhiv_positiveXdiabetes_none and the stratification of interest
        provided through the stratification argument was hiv, then
    :param compartment: str
        name of the compartment or parameter to be interrogated
    :param stratification: str
        the stratification of interest
    :param remove_stratification_name: bool
        whether to remove the stratification name and its trailing _ from the string to return
    :return: str
        the name of the stratum within which the compartment falls
    """
    stratum_name = [
        name
        for n_name, name in enumerate(find_name_components(compartment))
        if stratification in name
    ][0]
    return (
        stratum_name[stratum_name.find("_") + 1 :] if remove_stratification_name else stratum_name
    )


def find_first_list_element_above(list, value):
    """
    Simple method to return the index of the first element of a list that is greater than a specified value.

    Args:
        list: List of floats
        value: The value that the element must be greater than
    """
    return next(x[0] for x in enumerate(list) if x[1] > value)


def get_integration_times(start_year: int, end_year: int, time_step: int):
    """
    Get a list of timesteps from start_year to end_year, spaced by time_step.
    """
    n_iter = int(round((end_year - start_year) / time_step)) + 1
    return numpy.linspace(start_year, end_year, n_iter).tolist()


def element_wise_list_summation(list_1, list_2):
    """
    Element-wise summation of two lists of the same length.
    """
    return [value_1 + value_2 for value_1, value_2 in zip(list_1, list_2)]
