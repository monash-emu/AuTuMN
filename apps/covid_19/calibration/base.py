from autumn.calibration import (
    Calibration,
    run_full_models_for_mcmc as _run_full_models_for_mcmc,
)
from autumn.tool_kit.utils import find_first_index_reaching_cumulative_sum

from ..app import RegionApp
from ..john_hopkins import read_john_hopkins_data_from_csv

from numpy import linspace

N_ITERS = 100000
N_BURNED = 0
N_CHAINS = 1


def run_full_models_for_mcmc(
    region: str, burn_in: int, src_db_path: str, dest_db_path: str
):
    """
    Run the full baseline model and all scenarios for all accepted MCMC runs in src db.
    """
    region_model = RegionApp(region)
    build_model = region_model.build_model
    params = region_model.params
    _run_full_models_for_mcmc(burn_in, src_db_path, dest_db_path, build_model, params)


def run_calibration_chain(
    max_seconds: int,
    run_id: int,
    region: str,
    par_priors,
    target_outputs,
    mode="autumn_mcmc",
    _grid_info=None,
    _start_time_range=None,
    _multipliers={},
):
    """
    Run a calibration chain for the covid model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    mode is either 'lsm' or 'autumn_mcmc'
    """
    print(f"Preparing to run covid model calibration for region {region}")

    region_model = RegionApp(region)
    build_model = region_model.build_model
    params = region_model.params
    calib = Calibration(
        f"covid_{region}",
        build_model,
        par_priors,
        target_outputs,
        _multipliers,
        run_id,
        model_parameters=params,
        start_time_range=_start_time_range,
        run_extra_scenarios=False,
    )
    print("Starting calibration.")
    calib.run_fitting_algorithm(
        run_mode=mode,
        n_iterations=N_ITERS,
        n_burned=N_BURNED,
        n_chains=N_CHAINS,
        available_time=max_seconds,
        grid_info=_grid_info,
    )
    print(f"Finished calibration for run {run_id}.")


def get_priors_and_targets(region, data_type="confirmed", start_after_n_cases=1):
    """
    Automatically build prior distributions and calibration targets using John Hopkins data
    :param region: the region name
    :param data_type: either "confirmed" or "deaths"
    :return:
    """

    # for JH data, day_1 is '1/22/20', that is 22 Jan 2020
    n_daily_cases = read_john_hopkins_data_from_csv(data_type, country=region.title())

    # get the subset of data points starting after 1st case detected
    index_start = find_first_index_reaching_cumulative_sum(
        n_daily_cases, start_after_n_cases
    )
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

    output_key = {"confirmed": "notifications", "deaths": "infection_deathsXall"}

    assert data_type in output_key

    TARGET_OUTPUTS = [
        {
            "output_key": output_key[data_type],
            "years": linspace(
                start_day,
                start_day + len(data_of_interest) - 1,
                num=len(data_of_interest),
            ),
            "values": data_of_interest,
            "loglikelihood_distri": "poisson",
        }
    ]

    return PAR_PRIORS, TARGET_OUTPUTS
