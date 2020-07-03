from autumn.calibration import (
    Calibration,
    run_full_models_for_mcmc as _run_full_models_for_mcmc,
)
from autumn.tool_kit.utils import find_first_index_reaching_cumulative_sum

from ..app import RegionApp

from numpy import linspace

N_ITERS = 100000
N_BURNED = 0
N_CHAINS = 1


def run_full_models_for_mcmc(region: str, burn_in: int, src_db_path: str, dest_db_path: str):
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
    _multipliers={},
):
    """
    Run a calibration chain for the covid model

    num_iters: Maximum number of iterations to run.
    available_time: Maximum time, in seconds, to run the calibration.
    mode is either 'lsm' or 'autumn_mcmc'
    """
    print(f"Preparing to run DR-TB model calibration for region {region}")

    region_model = RegionApp(region)
    build_model = region_model.build_model
    params = region_model.params
    calib = Calibration(
        "dr_tb_malancha",
        region,
        build_model,
        par_priors,
        target_outputs,
        _multipliers,
        run_id,
        model_parameters=params,
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
