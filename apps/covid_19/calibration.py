import logging

from numpy import linspace
import numpy as np

from autumn.calibration import (
    Calibration,
    run_full_models_for_mcmc as _run_full_models_for_mcmc,
)
from autumn.tool_kit.utils import find_first_index_reaching_cumulative_sum
from autumn.inputs import get_john_hopkins_data
from autumn.tool_kit.params import load_params
from autumn.calibration.utils import ignore_calibration_target_before_date

from .model import build_model

N_ITERS = 100000
N_BURNED = 0
N_CHAINS = 1

BASE_CALIBRATION_PARAMS = [

    # Arbitrary, but always required and this range should span the range of values that would be needed
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [0.015, 0.07],
    },

    # Arbitrary, but useful to allow epidemic to take off from a flexible time
    {
        "param_name": "start_time",
        "distribution": "uniform",
        "distri_params": [0.0, 40.0],
    },

    # Rationale described in parameters table of the methods Gdoc at:
    # https://docs.google.com/document/d/1Uhzqm1CbIlNXjowbpTlJpIphxOm34pbx8au2PeqpRXs/edit#
    {
        "param_name": "compartment_periods_calculated.exposed.total_period",
        "distribution": "trunc_normal",
        "distri_params": [5.5, 0.97],
        "trunc_range": [1.0, np.inf],
    },

    {
        "param_name": "compartment_periods_calculated.active.total_period",
        "distribution": "uniform",
        "distri_params": [4.5, 9.5],
    },
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
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.1, 4.0],
        },
        {
            "param_name": "start_time",
            "distribution": "uniform",
            "distri_params": [-30, start_day - 1],
        },
    ]

    output_key = {
        "confirmed": "notifications",
        "deaths": "infection_deathsXall"
    }

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

    if any([i_target["output_key"] == output_name for i_target in target_outputs]):
        params += [
            {
                "param_name": output_name + "_dispersion_param",
                "distribution": "uniform",
                "distri_params": [0.1, 5.0],
            },
        ]
    return params


"""
Application-specific methods
"""


def add_standard_philippines_params(params):
    """
    Add standard set of parameters to vary case detection for the Philippines
    """

    return params + [
        {
            "param_name": "ifr_multiplier",
            "distribution": "uniform",
            "distri_params": [1.5, 2.28]
        },
        {
            "param_name": "testing_to_detection.assumed_cdr_parameter",
            "distribution": "uniform",
            "distri_params": [0.3, 0.5],
        },
        {
            "param_name": "start_time",
            "distribution": "uniform",
            "distri_params": [40.0, 60.0],
        },
        {
            "param_name": "microdistancing.parameters.max_effect",
            "distribution": "uniform",
            "distri_params": [0.25, 0.75],
        },
    ]


def add_standard_philippines_targets(targets):

    # Ignore notification values before day 100
    notifications = \
        ignore_calibration_target_before_date(targets["notifications"], 100)

    return [
        {
            "output_key": "notifications",
            "years": notifications["times"],
            "values": notifications["values"],
            "loglikelihood_distri": "normal",
            "time_weights": assign_trailing_weights_to_halves(14, notifications["times"]),
        },
        {
            "output_key": "icu_occupancy",
            "years": [targets["icu_occupancy"]["times"][-1]],
            "values": [targets["icu_occupancy"]["values"][-1]],
            "loglikelihood_distri": "normal",

        },
        {
            "output_key": "accum_deaths",
            "years": [targets["total_infection_deaths"]["times"][-1]],
            "values": [float(sum(targets["total_infection_deaths"]["values"]))],
            "loglikelihood_distri": "normal",
        },
    ]


def add_standard_victoria_params(params):

    return params + [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.025, 0.05],
        },
        {
            "param_name": "seasonal_force",
            "distribution": "uniform",
            "distri_params": [0.0, 0.4],
        },
        {
            "param_name": "compartment_periods_calculated.exposed.total_period",
            "distribution": "trunc_normal",
            "distri_params": [5.5, 0.97],
            "trunc_range": [1.0, np.inf],
        },
        {
            "param_name": "compartment_periods_calculated.active.total_period",
            "distribution": "trunc_normal",
            "distri_params": [7.0, 0.7],
            "trunc_range": [1.0, np.inf],
        },
        {
            "param_name": "symptomatic_props_multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.1],
            "trunc_range": [0.5, np.inf],
        },
        {
            "param_name": "hospital_props_multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.25],
            "trunc_range": [0.1, np.inf],
        },
        {
            "param_name": "ifr_multiplier",
            "distribution": "trunc_normal",
            "distri_params": [1.0, 0.25],
            "trunc_range": [0.1, np.inf],
        },
        {
            "param_name": "icu_prop",
            "distribution": "uniform",
            "distri_params": [0.08, 0.2],
        },
        {
            "param_name": "compartment_periods.icu_early",
            "distribution": "uniform",
            "distri_params": [5.0, 17.0],
        },
        {
            "param_name": "compartment_periods.icu_late",
            "distribution": "uniform",
            "distri_params": [5.0, 15.0],
        },
        {
            "param_name": "microdistancing.parameters.max_effect",
            "distribution": "uniform",
            "distri_params": [0.25, 0.6],
        },
    ]


def add_standard_victoria_targets(target_outputs, targets):
    notifications = targets["notifications"]
    hospital_occupancy = targets["hospital_occupancy"]
    icu_occupancy = targets["icu_occupancy"]
    total_infection_deaths = targets["total_infection_deaths"]

    return target_outputs + [
        {
            "output_key": notifications["output_key"],
            "years": notifications["times"],
            "values": notifications["values"],
            "loglikelihood_distri": "normal",
            "time_weights": list(range(1, len(notifications["times"]) + 1)),
        },
        {
            "output_key": hospital_occupancy["output_key"],
            "years": hospital_occupancy["times"],
            "values": hospital_occupancy["values"],
            "loglikelihood_distri": "normal",
            "time_weights": list(range(1, len(hospital_occupancy["times"]) + 1)),
        },
        {
            "output_key": icu_occupancy["output_key"],
            "years": icu_occupancy["times"],
            "values": icu_occupancy["values"],
            "loglikelihood_distri": "normal",
            "time_weights": list(range(1, len(icu_occupancy["times"]) + 1)),
        },
        {
            "output_key": total_infection_deaths["output_key"],
            "years": total_infection_deaths["times"],
            "values": total_infection_deaths["values"],
            "loglikelihood_distri": "normal",
            "time_weights": list(range(1, len(total_infection_deaths["times"]) + 1)),
        },
    ]
