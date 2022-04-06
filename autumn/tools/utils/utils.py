"""
Miscellaneous utility functions.
If several functions here develop a theme, consider reorganising them into a module.
"""
import itertools
import json
import subprocess as sp
import os
import numpy
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Union, Callable, Dict, Optional

from summer.compute import ComputedValueProcessor

from autumn.tools.utils.s3 import download_from_s3, list_s3, get_s3_client
from autumn.tools import registry
from autumn.settings.folders import PROJECTS_PATH
from autumn.tools.utils import secrets

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


def flatten_list(x):
    """
    Transform a list of lists into a single flat list.

    Args:
        x: a nested list
    Returns:
        A flat list

    """
    return [v for sublist in x for v in sublist]


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


def get_git_modified() -> bool:
    """
    Return True if there are (tracked and uncommited) modifications
    """

    status = run_command("git status --porcelain").split("\n")
    return any([s.startswith(" M") for s in status])


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
    This should be considered deprecated; timeseries should be expressed as Pandas Series (not lists), and the appropriate pandas methods used
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


def get_apply_odds_ratio_to_prop(odds_ratio):
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
        odds_ratio: The odds ratio to adjust by
    Returns:
        The adjusted proportion

    """

    assert 0.0 <= odds_ratio

    def or_to_prop_func(proportion):

        # Check input
        assert 0.0 <= proportion <= 1.0

        # Transform and return
        modified_proportion = proportion * odds_ratio / (proportion * (odds_ratio - 1.0) + 1.0)

        return modified_proportion

    return or_to_prop_func


def apply_odds_ratio_to_props(props, adjuster):
    """
    Very simple, but just because it is used a few times.
    """

    or_to_prop_func = get_apply_odds_ratio_to_prop(adjuster)
    return [or_to_prop_func(i_prop) for i_prop in props]


def subdivide_props(
    base_props: numpy.ndarray, split_props: Union[numpy.ndarray, float]
) -> numpy.ndarray:
    """
    Split an array (base_props) of proportions into two arrays (split_arr, complement_arr) according to the split
    proportions provided (split_props).
    """

    split_arr = base_props * split_props
    complement_arr = base_props * (1.0 - split_props)
    return split_arr, complement_arr


def list_element_wise_division(a, b):
    """
    Performs element-wise division between two lists and returns zeros where denominator is zero.
    """

    return numpy.divide(a, b, out=numpy.zeros_like(a), where=b != 0.0)


def update_mle_from_remote_calibration(model, region, run_id=None):
    """
    Download MLE parameters from s3 and update the relevant project automatically.
    :param model: model name (e.g. 'covid_19')
    :param region: region name
    :param run_id: Optional run identifier such as 'covid_19/calabarzon/1626941419/5495a75'. If None, the latest run is
    considered.
    """

    s3_client = get_s3_client()

    if run_id is None:
        run_id = find_latest_run_id(model, region, s3_client)

    msg = f'Could not read run ID {run_id}, use format "{{app}}/{{region}}/{{timestamp}}/{{commit}}" - exiting'
    assert len(run_id.split("/")) == 4, msg

    # Define the destination folder
    project_registry_name = registry._PROJECTS[model][region]
    region_subfolder_names = (
        project_registry_name.split(f"autumn.projects.{model}.")[1].split(".project")[0].split(".")
    )
    destination_dir_path = os.path.join(PROJECTS_PATH, model)
    for region_subfolder_name in region_subfolder_names:
        destination_dir_path = os.path.join(destination_dir_path, region_subfolder_name)
    destination_dir_path = os.path.join(destination_dir_path, "params")
    dest_path = os.path.join(destination_dir_path, "mle-params.yml")

    # Download the mle file
    key_prefix = os.path.join(run_id, "data", "calibration_outputs").replace("\\", "/")
    all_mle_files = list_s3(s3_client, key_prefix, key_suffix="mle-params.yml")

    if len(all_mle_files) == 0:
        print(f"WARNING: No MLE file found for {run_id}")
    else:
        s3_key = all_mle_files[0]
        download_from_s3(s3_client, s3_key, dest_path, quiet=True)
        print(f"Updated MLE file for {region}'s {model} model, using {run_id}")


def find_latest_run_id(model, region, s3_client):

    # Workout latest timestamp
    key_prefix = f"{model}/{region}/"
    all_files = list_s3(s3_client, key_prefix)
    all_timestamps = [int(f.split("/")[2]) for f in all_files]
    latest_timestamp = max(all_timestamps)

    # Identify commit id
    key_prefix = f"{model}/{region}/{latest_timestamp}/"
    all_files = list_s3(s3_client, key_prefix)
    all_commits = [f.split("/")[3] for f in all_files]

    n_commits = len(list(numpy.unique(all_commits)))
    msg = f"There must be exactly one run on the server associated with the latest timestamp, found {n_commits}."
    msg += f"Please enter the run_id manually for model {model}, region {region}."
    assert n_commits == 1, msg

    commit = all_commits[0]

    return f"{model}/{region}/{latest_timestamp}/{commit}"


def update_timeseries(TARGETS_MAPPING, df, file_path, *args):
    """
    Simple function to update timeseries.json
    """
    with open(file_path, mode="r") as f:
        targets = json.load(f)

    df.sort_values(by=["date_index"], inplace=True)

    for key, val in TARGETS_MAPPING.items():

        if val in df.columns and key in targets.keys():

            # Drop the NaN value rows from df before writing data.
            temp_df = df[["date_index", val]].dropna(0, subset=[val])

            targets[key]["times"] = list(temp_df["date_index"])
            targets[key]["values"] = list(temp_df[val])
    with open(file_path, "w") as f:
        json.dump(targets, f, indent=2)

    if args:
        secrets.write(file_path, *args)


def create_date_index(base_datetime, df, datecol):
    df.rename(columns=lambda x: x.lower().strip().replace(" ", "_"), inplace=True)
    df.rename(columns={datecol.lower().strip().replace(" ", "_"): "date"}, inplace=True)

    formats = ["%Y-%m-%d", "%d/%m/%Y"]

    for fmt in formats:

        try:
            df.date = pd.to_datetime(
                df["date"], errors="raise", format=fmt, infer_datetime_format=False
            ).dt.date

        except:
            continue

        else:
            print("Success")

    df["date_index"] = (df.date - base_datetime.date()).dt.days

    return df


def find_closest_value_in_list(list_request: List, value_request: int) -> int:
    """
    Find the closest value within one list to the value of interest.
    """

    return min(list_request, key=lambda list_value: abs(list_value - value_request))


def check_list_increasing(list_to_check):
    assert all(list_to_check[i] <= list_to_check[i + 1] for i in range(len(list_to_check) - 1))


def get_prop_two_numerators(numerator_1, numerator_2, denominator):
    return (numerator_1 + numerator_2) / denominator


def get_complement_prop(numerator, denominator):
    return 1.0 - numerator / denominator


def return_constant_value(value: float) -> Callable:
    """
    For situations below in which we just need a function that returns a constant value.

    Args:
        value: The value that the function will return

    Returns:
        The function that returns the value, ignoring the time input

    """

    def constant_value_func(time):
        return value

    return constant_value_func


def get_product_two_functions(function_1, function_2):
    """
    For the situation where we want a function that returns the product of two other functions.

    Args:
        function_1: First function of time
        function_2: Second function of time

    Returns:
        New function of time that returns a scalar, being the product of the two other functions at that point in time

    """

    def product_function(time):
        return function_1(time) * function_2(time)

    return product_function


def multiply_function_or_constant(
        function_or_constant: Union[callable, float],
        multiplier: float,
) -> Union[callable, float]:
    """
    Multiply a function that returns a single value and takes inputs in the standard format of a summer time-varying
    process by a multiplier - or do the same if the value is just a simple float.

    Args:
        function_or_constant: The function or constant to be multiplied
        multiplier: The value for this thing to be multiplied by

    Returns:
        The same type of object as the function_or_constant input, but multiplied by the desired number

    """

    if callable(function_or_constant):
        def revised_function(t, c):
            return function_or_constant(t, c) * multiplier

        return revised_function
    else:
        return function_or_constant * multiplier


def weighted_average(
        distribution: Dict[str, float],
        weights: Dict[str, float],
        rounding: Optional[int] = None,
) -> float:
    """
    Calculate a weighted average from dictionaries with the same keys, representing the values and the weights.

    Args:
        distribution: The values
        weights: The weights
        rounding: Whether to round off the result

    Returns:
        The weighted average

    """

    msg = "Attempting to calculate weighted average over two dictionaries that do not share keys"
    assert distribution.keys() == weights.keys(), msg
    numerator = sum([distribution[i] * weights[i] for i in distribution.keys()])
    denominator = sum(weights.values())
    fraction = numerator / denominator
    result = round(fraction, rounding) if rounding else fraction
    return result


class FunctionWrapper(ComputedValueProcessor):
    """
    Very basic processor that wraps a time/computed values function
    of the type used in flow and adjusters

    FIXME:
    This is such a basic use case, it probably belongs in summer

    """

    def __init__(self, function_to_wrap: callable):
        """
        Initialise with just the param function
        Args:
            function_to_wrap: The function
        """

        self.wrapped_function = function_to_wrap

    def process(self, compartment_values, computed_values, time):
        return self.wrapped_function(time, computed_values)


def wrap_series_transform_for_ndarray(
        process_to_apply: callable
) -> callable:
    """
    Return a function that converts an ndarray to a Series, applies a transform, and returns the result as a new ndarray

    Args:
        process_to_apply: A function that can be applied to a pandas series

    Returns:
        Function that applies transform to ndarray
    """

    def apply_series_transform_to_ndarray(
            in_data: np.ndarray
    ) -> np.ndarray:
        """
        The function that can be applied directly to ndarrays

        Args:
            in_data: numpy array which will be transformed

        Returns:
        The processed ndarray
        """
        #Return in the appropriate format
        return process_to_apply(pd.Series(in_data)).to_numpy()
    
    return apply_series_transform_to_ndarray
