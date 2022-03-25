import scipy
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from typing import List

from autumn.tools.curve import scale_up_function
from autumn.tools.utils.utils import flatten_list


def calculate_transition_rates_from_dynamic_props(props_df: pd.DataFrame, active_flows: List[str]) -> dict:
    """
    Calculate the transition rates associated with each inter-stratum flow that will produce the requested population
    proportions over time.

    To see a working example showing the format expected for the user requests, please see the following notebook:
    'notebooks/user/rragonnet/dynamic_stratum_props.ipynb'

    Args:
        props_df: User-requested stratum proportions over time (pandas data frame indexed using time points)
        active_flows: list of strings representing the flows driving the inter-stratum transitions

    Returns:
        A dictionary of time-variant functions

    """
    # Check that the user requested sensible proportions
    check_requested_proportions(props_df, active_flows)

    # Determine some basic characteristics
    strata = props_df.columns.to_list()
    times = props_df.index.to_list()

    # Work out the transition rates associated with each flow and each time interval
    tv_rates = {flow: [] for flow in active_flows}
    for i in range(len(times) - 1):
        delta_t = times[i + 1] - times[i]
        # Requested proportions at beginning and end of time interval
        start_props = props_df.loc[times[i]]
        end_props = props_df.loc[times[i + 1]]

        # Main calculations requiring numerical solving
        rates = calculate_rates_for_interval(start_props, end_props, delta_t, strata, active_flows)
        for i_flow, flow in enumerate(active_flows):
            tv_rates[flow].append(rates[i_flow])

    # Create time-variant functions
    # First create the list of time points such that parameter changes occur at each requested time
    scaleup_times = flatten_list([[t, t+1] for t in times])

    # Then create the list of values using the rates estimated previously
    scaleup_param_functions = {}
    for flow in active_flows:
        values = flatten_list([[0.]] + [[v]*2 for v in tv_rates[flow]] + [[0.]])
        scaleup_param_functions[flow] = scale_up_function(scaleup_times, values, method=4)

    return scaleup_param_functions


def calculate_rates_for_interval(
        start_props: pd.core.series.Series, end_props: pd.core.series.Series, delta_t: float, strata: List[str],
        active_flows: List[str]
) -> List[float]:
    """
    Calculate the transition rates associated with each inter-stratum flow for a given time interval.

    The system can be described using a linear ordinary differential equations such as:
    X'(t) = M.X(t) ,
    where M is the transition matrix and X is a column vector representing the proportions over time

    The solution of this equation is X(t) = exp(M.t).X_0,
    where X_0 represents the proportions at the start of the time interval.

    The transition parameters informing M must then verify the following equation:
    X(t_end) = exp(M.delta_t).X_0,
    where t_end represents the end of the time interval.

    Args:
        start_props: user-requested stratum proportions at the start of the time interval
        end_props: user-requested stratum proportions at the end of the time interval
        delta_t: width of the time interval
        strata: list of strata
        active_flows: list of strings representing the flows driving the inter-stratum transitions

    Returns:
        The estimated transition rates stored in a list using the same order as active_flows

    """
    # Determine some basic characteristics
    n_strata = len(strata)
    n_params = len(active_flows)

    # Create the function that we need to find the root of
    def function_to_zero(params):
        # params is a list ordered in the same order as active_flows

        # Create the transition matrix associated with a given set of transition parameters
        m = np.zeros((n_strata, n_strata))
        for i_row, stratum_row in enumerate(strata):
            for i_col, stratum_col in enumerate(strata):
                if i_row == i_col:
                    # Diagonal components capture flows starting from the associated stratum
                    relevant_flows = [f for f in active_flows if f.startswith(f"{stratum_row}_to_")]
                    for f in relevant_flows:
                        m[i_row, i_col] -= params[active_flows.index(f)]
                else:
                    # Off-diagonal components capture flows from stratum_col to stratum_row
                    potential_flow = f"{stratum_col}_to_{stratum_row}"
                    if potential_flow in active_flows:
                        m[i_row, i_col] = params[active_flows.index(potential_flow)]

        # Calculate the matrix exponential, accounting for the time interval width
        exp_mt = scipy.linalg.expm(m * delta_t)

        # Calculate the difference between the left and right terms of the equation
        diff = np.matmul(exp_mt, start_props) - end_props

        # Return the norm of the vector to make the minimised function a scalar function
        return scipy.linalg.norm(diff)

    # Define bounds to force the parameters to be positive
    bounds = [(0., None)] * n_params

    # Numerical solving
    solution = minimize(function_to_zero, x0=np.zeros(n_params), bounds=bounds, method="TNC")

    return solution.x


def check_requested_proportions(props_df: pd.DataFrame, active_flows: List[str], rel_diff_tol: float = 0.01):
    """
    Check that:
     - the sum of the requested proportions remains (approximately) constant over time.
     - strata with increasing proportions must have inflows
     - strata with decreasing proportions must have outflows
    Args:
        props_df: User-requested stratum proportions over time (pandas data frame indexed using time points)
        active_flows: list of strings representing the flows driving the inter-stratum transitions
        rel_diff_tol: Maximum accepted relative difference between smallest and largest sum.
    """
    # Check that the sum of the requested proportions remains constant over time.
    row_sums = props_df.sum(axis=1)
    smallest_sum, largest_sum = row_sums.min(), row_sums.max()
    rel_perc_diff = 100. * (largest_sum - smallest_sum) / smallest_sum

    msg = f"Relative difference between smaller and larger proportion sums is {int(rel_perc_diff)}%.\n"
    msg += f"This is greater than the maximum accepted value of {int(100. * rel_diff_tol)}%."
    assert rel_perc_diff <= 100. * rel_diff_tol, msg

    # Check that strata with increasing proportions have inflows
    ever_increasing_strata = props_df.loc[:, props_df.diff().max() > 0.].columns.to_list()
    test = all([any([flow.endswith(f"_to_{stratum}") for flow in active_flows]) for stratum in ever_increasing_strata])
    assert test, "Found at least one stratum with no inflows for which increasing proportions were requested."

    # Check that strata with decreasing proportions have outflows
    ever_decreasing_strata = props_df.loc[:, props_df.diff().min() < 0.].columns.to_list()
    test = all([any([flow.startswith(f"{stratum}_to_") for flow in active_flows]) for stratum in ever_decreasing_strata])
    assert test, "Found at least one stratum with no outflows for which decreasing proportions were requested."
