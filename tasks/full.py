import logging
import os
import shutil

import pandas as pd

from autumn import db
from autumn.db.database import get_database
from autumn.tool_kit.params import update_params
from autumn.tool_kit.scenarios import Scenario, calculate_differential_outputs
from utils.s3 import list_s3, download_from_run_s3, upload_to_run_s3
from utils.parallel import run_parallel_tasks
from utils.timer import Timer
from settings import REMOTE_BASE_DIR
from tasks.utils import get_app_region
from tasks.calibrate import CALIBRATE_DATA_DIR


logger = logging.getLogger(__name__)


FULL_RUN_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "data", "full_model_runs")


def full_model_run_task(run_id: str, burn_in: int, quiet: bool):

    # Set up directories for output data.
    with Timer(f"Creating calibration directories"):
        if os.path.exists(FULL_RUN_DATA_DIR):
            shutil.rmtree(FULL_RUN_DATA_DIR)

        os.makedirs(FULL_RUN_DATA_DIR)

    # Find the calibration chain databases in AWS S3.
    key_prefix = os.path.join(run_id, os.path.relpath(CALIBRATE_DATA_DIR, REMOTE_BASE_DIR))
    chain_db_keys = list_s3(key_prefix, key_suffix=".feather")

    # Download the calibration chain databases.
    with Timer(f"Downloading calibration data"):
        args_list = [(run_id, src_key, quiet) for src_key in chain_db_keys]
        run_parallel_tasks(download_from_run_s3, args_list)

    # Run the models for the full time period plus all scenarios for each accepted parameter
    # set, while also applying burn-in.
    db_paths = db.load.find_db_paths(CALIBRATE_DATA_DIR)
    chain_ids = [int(p.split("/")[-1].split("-")[-1]) for p in db_paths]
    num_chains = len(chain_ids)
    with Timer(f"Running full models for {num_chains} chains: {chain_ids}"):
        args_list = [
            (run_id, db_path, chain_id, burn_in, quiet)
            for chain_id, db_path in zip(chain_ids, db_paths)
        ]
        chain_ids = run_parallel_tasks(run_full_model_for_chain, args_list)

    # Upload the full model run outputs of AWS S3.
    db_paths = db.load.find_db_paths(FULL_RUN_DATA_DIR)
    with Timer(f"Uploading full model run data to AWS S3"):
        for db_path in db_paths:
            upload_to_run_s3(run_id, db_path, quiet)


def run_full_model_for_chain(
    run_id: str, src_db_path: str, chain_id: int, burn_in: int, quiet: bool
):
    if quiet:
        logging.disable(logging.INFO)

    app_region = get_app_region(run_id)
    dest_db_path = os.path.join(FULL_RUN_DATA_DIR, f"chain-{chain_id}")
    logger.info(
        f"Running {app_region.app_name} {app_region.region_name} full model with burn-in of {burn_in}s"
    )
    src_db = get_database(src_db_path)
    dest_db = get_database(dest_db_path)
    db.process.apply_burn_in(src_db, dest_db, burn_in)
    mcmc_run_df = dest_db.query("mcmc_run")
    outputs = []
    derived_outputs = []

    for _, mcmc_run in mcmc_run_df.iterrows():
        run_id = mcmc_run["run"]
        chain_id = mcmc_run["chain"]
        if not mcmc_run["accept"]:
            logger.info("Ignoring non-accepted MCMC run %s", run_id)
            continue

        logger.info("Running full model for MCMC run %s", run_id)
        param_updates = db.load.load_mcmc_params(dest_db, run_id)
        update_func = lambda ps: update_params(ps, param_updates)
        with Timer("Running model scenarios"):
            num_scenarios = 1 + len(app_region.params["scenarios"].keys())
            scenarios = []
            for scenario_idx in range(num_scenarios):
                scenario = Scenario(app_region.build_model, scenario_idx, app_region.params)
                scenarios.append(scenario)

            # Run the baseline scenario.
            baseline_scenario = scenarios[0]
            baseline_scenario.run(update_func=update_func)
            baseline_model = baseline_scenario.model

            # Run all the other scenarios
            for scenario in scenarios[1:]:
                scenario.run(base_model=baseline_model, update_func=update_func)

        run_id = int(run_id)
        chain_id = int(chain_id)

        with Timer("Processing model outputs"):
            models = [s.model for s in scenarios]
            models = calculate_differential_outputs(models, app_region.targets)
            outputs_df = db.store.build_outputs_table(models, run_id, chain_id)
            derived_outputs_df = db.store.build_derived_outputs_table(models, run_id, chain_id)
            outputs.append(outputs_df)
            derived_outputs.append(derived_outputs_df)

    with Timer("Saving model outputs to the database"):
        outputs_df = pd.concat(outputs, copy=False, ignore_index=True)
        derived_outputs_df = pd.concat(derived_outputs, copy=False, ignore_index=True)
        dest_db.dump_df(db.store.Table.OUTPUTS, outputs_df)
        dest_db.dump_df(db.store.Table.DERIVED, derived_outputs_df)

    logger.info("Finished running full models for all accepted MCMC runs.")
    return chain_id
