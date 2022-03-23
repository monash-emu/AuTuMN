import scipy
from scipy.optimize import minimize
import numpy as np

from autumn.tools.curve import scale_up_function
from autumn.tools.utils.utils import flatten_list


# props_df = pd.DataFrame(
#     data={
#         "A": [1., .2, .2, .2],
#         "B": [0., .8, .6, .7],
#         "C": [0., .0, .2, .1]
#     },
#     index=[0, 100, 150, 175]
# )
#
# active_flows = [
#     "A_to_B",
#     "B_to_C",
#     "C_to_B"
# ]


def calculate_transition_rates_from_dynamic_props(props_df, active_flows):
    """
    Calculate the transition rates associated with each inter-stratum flow that will produce the requested population
    proportions over time.

    Args:
        props_df: User-requested stratum proportions over time (pandas data frame indexed using time points)
        active flows: list of strings representing the flows driving the inter-stratum transitions

    Returns:
        A dictionary of time-variant functions

    """

    # Determine basic characteristics
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


def calculate_rates_for_interval(start_props, end_props, delta_t, strata, active_flows):
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
        active flows: list of strings representing the flows driving the inter-stratum transitions

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


# sc_functions = calculate_transition_rates_from_dynamic_props(props_df, active_flows)
