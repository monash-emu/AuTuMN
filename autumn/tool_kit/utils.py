"""
Miscellaneous utility functions.
If several functions here develop a theme, consider reorganising them into a module.
"""
import json, types, hashlib, itertools
import subprocess as sp
from datetime import date

import numpy


def get_data_hash(*args):
    """
    Get a hash of a bunch of JSON serializable data.
    Returns first 8 characters of the MD5 hash of the data.
    Eg. args of ("foo", {"a": 1}, [1, 2, 3]) --> "34d333dw"
    """
    data_str = ""
    for arg in args:
        try:
            data_str += json.dumps(arg)
        except TypeError:
            continue  # Fail silently :(

    hash_str = hashlib.md5(data_str.encode()).hexdigest()
    return hash_str[:8]


def merge_dicts(src: dict, dest: dict):
    """
    Merge src dict into dest dict.
    """
    for key, value in src.items():
        if isinstance(value, dict):
            # Get node or create one
            node = dest.setdefault(key, {})
            if node is None:
                dest[key] = value
            else:
                merge_dicts(value, node)
        else:
            dest[key] = value

    return dest


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
    return numpy.linspace(start_year, end_year, n_iter)


def element_wise_list_summation(list_1, list_2):
    """
    Element-wise summation of two lists of the same length.
    """
    return [value_1 + value_2 for value_1, value_2 in zip(list_1, list_2)]


def repeat_list_elements(repetitions, list_to_repeat):
    return list(
        itertools.chain.from_iterable(
            itertools.repeat(i_element, repetitions) for i_element in list_to_repeat
        )
    )


def repeat_list_elements_average_last_two(raw_props, prop_over_80):
    """
    Repeat 5-year age-specific proportions, but with 75+s taking the weighted average of the last two groups.
    prop_over_80 is the proportion of 80+ individuals among the 75+ population.
    """
    repeated_props = repeat_list_elements(2, raw_props[:-1])
    repeated_props[-1] = (1.0 - prop_over_80) * raw_props[-2] + prop_over_80 * raw_props[-1]
    return repeated_props


def find_series_compartment_parameter(proportion_to_split, n_compartments, original_parameter):
    return (1.0 - (1.0 - proportion_to_split) ** (1.0 / n_compartments)) * original_parameter


def find_rates_and_complements_from_ifr(cfrs, n_compartment_repeats, overall_rates):
    """
    Given a list of proportions (CFRs) to be applied to a set of n compartments in series with equal flow rates through
    them, work out the death rates and their complements
    """
    death_rates = [
        find_series_compartment_parameter(i_cfr, n_compartment_repeats, i_rate)
        for i_cfr, i_rate in zip(cfrs, overall_rates)
    ]
    complements = [
        i_overall_rate - i_death_rate
        for i_overall_rate, i_death_rate in zip(overall_rates, death_rates)
    ]
    return death_rates, complements


def find_first_index_reaching_cumulative_sum(a_list, threshold):
    """
    Returns the index at which the cumulative sum of a list has reached a given value
    :param a_list: list object containing floats or integers
    :param threshold: a float or integer
    :return: an index (integer)
    """
    cumsum_list = numpy.cumsum(a_list).tolist()

    if cumsum_list[-1] < threshold:
        raise ValueError("The cumulative sum of the entire list is smaller than the threshold")

    return next(i for i, val in enumerate(cumsum_list) if val >= threshold)


def get_date_from_tuple(date_as_tuple):
    return date(date_as_tuple[0], date_as_tuple[1], date_as_tuple[2])


def get_date_from_string(date_as_string):
    return date(int(date_as_string[:4]), int(date_as_string[4:6]), int(date_as_string[6:]))


def find_relative_date_from_string_or_tuple(requested_date, base_date=(2019, 12, 31)):
    requested_date = (
        get_date_from_string(requested_date)
        if type(requested_date) == str
        else get_date_from_tuple(requested_date)
    )
    difference = requested_date - get_date_from_tuple(base_date)
    return difference.days


def normalise_sequence(input_sequence):
    """
    Normalise a list or tuple to produce a tuple with values representing the proportion of each to the total of the
    input sequence.
    """
    return (i_value / sum(input_sequence) for i_value in input_sequence)


def convert_list_contents_to_int(input_list):
    return [int(i_element) for i_element in input_list]


def element_wise_list_division(numerator, denominator, must_be_prop=False):
    """
    Simple function to find the quotients of two lists.
    """
    if must_be_prop:
        assert all(
            [num <= 1.01 * den for num, den in zip(numerator, denominator)]
        ), f"Numerators: {numerator} \n Denominators: {denominator}"
    return [num / den for num, den in zip(numerator, denominator)]


def copy_function(f, name=None):
    """
    return a function with same code, globals, defaults, closure, and
    name (or provide a new name)
    """
    fn = types.FunctionType(
        f.__code__, f.__globals__, name or f.__name__, f.__defaults__, f.__closure__
    )
    # in case f was given attrs (note this dict is a shallow copy):
    fn.__dict__.update(f.__dict__)
    return fn


def print_target_to_plots_from_calibration(target_outputs):
    print("outputs_to_plot:")
    for i in range(len(target_outputs)):
        target_outputs[i]["years"] = [str(t) for t in target_outputs[i]["years"]]
        print("  - name: " + target_outputs[i]["output_key"])
        print("    target_times: [" + ", ".join(target_outputs[i]["years"]) + "]")
        to_print = "    target_values: ["
        for j, val in enumerate(target_outputs[i]["values"]):
            if j > 0:
                to_print += ", "
            to_print += "[" + str(val) + "]"
        to_print += "]"
        print(to_print)
        print()


def apply_moving_average(data, period):
    """
    Smooth the data by applying moving average with a specified period
    :param data: a list
    :param period: an integer
    """
    smooth_data = []
    for i, d in enumerate(data):
        n_backwards_timepoints = period - 1 if i >= period - 1 else i
        smooth_data.append(float(numpy.mean(data[i - n_backwards_timepoints : i + 1])))
    return smooth_data


def apply_odds_ratio_to_proportion(proportion, odds_ratio):
    """
    Use an odds ratio to adjust a proportion.

    Starts from the premise that the odds associated with the original proportion (p1) = p1 / (1 - p1)
    and similarly, that the odds associated with the adjusted proportion (p2) = p2 / (1 - p2)
    We want to multiply the odds associated with p1 by a certain odds ratio.
    That, is we need to solve the following equation for p2:
        p1 / (1 - p1) * OR = p2 / (1 - p2)
    By simple algebra, the solution to this is:
        p2 = p1 * OR / (p1 * (OR - 1) + 1)

    Args:
        proportion: The original proportion (p1 in the description above)
        odds_ratio: The odds ratio to adjust by
    Returns:
        The adjusted proportion
    """

    # Check inputs
    assert 0.0 <= odds_ratio
    assert 0.0 <= proportion <= 1.0

    # Transform and return
    modified_proportion = proportion * odds_ratio / (proportion * (odds_ratio - 1.0) + 1.0)

    return modified_proportion
