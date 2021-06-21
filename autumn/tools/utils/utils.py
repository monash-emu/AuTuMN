"""
Miscellaneous utility functions.
If several functions here develop a theme, consider reorganising them into a module.
"""
import itertools
import subprocess as sp

import numpy


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


def get_integration_times(start_year: int, end_year: int, time_step: int):
    """
    Get a list of timesteps from start_year to end_year, spaced by time_step.
    """
    n_iter = int(round((end_year - start_year) / time_step)) + 1
    return numpy.linspace(start_year, end_year, n_iter)


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


def normalise_sequence(input_sequence):
    """
    Normalise a list or tuple to produce a tuple with values representing the proportion of each to the total of the
    input sequence.
    """
    return (i_value / sum(input_sequence) for i_value in input_sequence)


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


def list_element_wise_division(a, b):
    """
    Performs element-wise division between two lists and returns zeros where denominator is zero.
    """
    return numpy.divide(a, b, out=numpy.zeros_like(a), where=b != 0.)
