import logging

from numpy import linspace

from autumn.calibration import (
    Calibration,
    run_full_models_for_mcmc as _run_full_models_for_mcmc,
)
from autumn.tool_kit.utils import find_first_index_reaching_cumulative_sum
from autumn.inputs import get_john_hopkins_data
from autumn.tool_kit.params import load_params

from .model import build_model

N_ITERS = 100000
N_BURNED = 0
N_CHAINS = 1

BASE_CALIBRATION_PARAMS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.015, 0.07],},
    {"param_name": "start_time", "distribution": "uniform", "distri_params": [0.0, 40.0],},
    {
        "param_name": "compartment_periods_calculated.exposed.total_period",
        "distribution": "uniform",
        "distri_params": [3.0, 6.0],
    },
    {
        "param_name": "compartment_periods_calculated.active.total_period",
        "distribution": "uniform",
        "distri_params": [4.5, 9.5],
    },
]


def provide_default_calibration_params(excluded_params=()):
    """
    Provide the standard default parameters as listed above, unless requested not to include any.

    :param excluded_params: tuple
        strings of the parameters that are not to be returned
    :return: list
        calibration parameters
    """

    return [
        BASE_CALIBRATION_PARAMS[param]
        for param in range(len(BASE_CALIBRATION_PARAMS))
        if BASE_CALIBRATION_PARAMS[param]["param_name"] not in excluded_params
    ]


def add_standard_dispersion_parameter(params, target_outputs, output_name):
    """
    Add standard dispersion parameter for negative binomial distribution

    :param params: list
        Parameter priors to be updated by this function
    :param target_outputs: list
        Target outputs, to see whether the quantity of interest is an output
    :param output_name: str
        Name of the output of interest
    :return: list
        Updated version of the parameter priors
    """

    if any([i["output_key"] == output_name for i in target_outputs]):
        params += [
            {
                "param_name": output_name + "_dispersion_param",
                "distribution": "uniform",
                "distri_params": [0.1, 5.0],
            },
        ]
    return params


def add_standard_philippines_params(params):
    """
    Add standard set of parameters to vary case detection for the Philippines
    """

    return params + [
        {
            "param_name": "time_variant_detection.maximum_gradient",
            "distribution": "uniform",
            "distri_params": [0.05, 0.1],
        },
        {
            "param_name": "time_variant_detection.max_change_time",
            "distribution": "uniform",
            "distri_params": [70.0, 110.0],
        },
        {
            "param_name": "time_variant_detection.end_value",
            "distribution": "uniform",
            "distri_params": [0.10, 0.70],
        },
        {"param_name": "ifr_multiplier", "distribution": "uniform", "distri_params": [1.0, 2.28],},
    ]


def assign_trailing_weights_to_halves(end_weights, calibration_target):
    """
    Create a list of (float) halves and ones, of the length of the calibration target, with the last few values being
    ones and the earlier values being halves.

    :param end_weights: int
        How many values at the end should be ones
    :param calibration_target: list
        List of calibration targets to determine the length of the weights to be returned
    :return: list
        List of the weights as described above
    """

    time_weights = [0.5] * (len(calibration_target) - end_weights)
    time_weights += [1.0] * end_weights
    return time_weights


logger = logging.getLogger(__name__)


def run_full_models_for_mcmc(region: str, burn_in: int, src_db_path: str, dest_db_path: str):
    """
    Run the full baseline model and all scenarios for all accepted MCMC runs in src db.
    """
    params = load_params("covid_19", region)
    _run_full_models_for_mcmc(burn_in, src_db_path, dest_db_path, build_model, params)


def run_calibration_chain(
    max_seconds: int,
    run_id: int,
    num_chains: int,
    region: str,
    par_priors,
    target_outputs,
    mode="autumn_mcmc",
):
    """
    Run a calibration chain for the covid model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    mode is either 'lsm' or 'autumn_mcmc'
    """
    logger.info(f"Preparing to run covid model calibration for region {region}")
    params = load_params("covid_19", region)
    calib = Calibration(
        "covid_19",
        build_model,
        params,
        par_priors,
        target_outputs,
        run_id,
        num_chains,
        param_set_name=region,
    )
    logger.info("Starting calibration.")
    calib.run_fitting_algorithm(
        run_mode=mode,
        n_iterations=N_ITERS,
        n_burned=N_BURNED,
        n_chains=N_CHAINS,
        available_time=max_seconds,
    )
    logger.info(f"Finished calibration for run {run_id}.")


def get_priors_and_targets(region, data_type="confirmed", start_after_n_cases=1):
    """
    Automatically build prior distributions and calibration targets using John Hopkins data
    :param region: the region name
    :param data_type: either "confirmed" or "deaths"
    :return:
    """

    # for JH data, day_1 is '1/22/20', that is 22 Jan 2020
    n_daily_cases = get_john_hopkins_data(data_type, country=region.title(), latest=True)

    # get the subset of data points starting after 1st case detected
    index_start = find_first_index_reaching_cumulative_sum(n_daily_cases, start_after_n_cases)
    data_of_interest = n_daily_cases[index_start:]

    start_day = index_start + 22  # because JH data starts 22/1

    PAR_PRIORS = [
        {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.1, 4.0],},
        {
            "param_name": "start_time",
            "distribution": "uniform",
            "distri_params": [-30, start_day - 1],
        },
    ]

    output_key = {"confirmed": "notifications", "deaths": "infection_deathsXall"}

    assert data_type in output_key

    TARGET_OUTPUTS = [
        {
            "output_key": output_key[data_type],
            "years": linspace(
                start_day, start_day + len(data_of_interest) - 1, num=len(data_of_interest),
            ),
            "values": data_of_interest,
            "loglikelihood_distri": "poisson",
        }
    ]

    return PAR_PRIORS, TARGET_OUTPUTS
