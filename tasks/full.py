import logging
import os

import pandas as pd

from autumn import db
from autumn.db.store import Table
from autumn.db.database import get_database
from autumn.tool_kit.params import update_params
from autumn.tool_kit.scenarios import Scenario, calculate_differential_outputs
from utils.s3 import list_s3, download_from_run_s3, upload_to_run_s3
from utils.parallel import run_parallel_tasks
from utils.timer import Timer
from utils.fs import recreate_dir
from tasks.utils import get_app_region, set_logging_config
from tasks.calibrate import CALIBRATE_DATA_DIR
from settings import REMOTE_BASE_DIR


logger = logging.getLogger(__name__)


FULL_RUN_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "data", "full_model_runs")
TABLES_TO_DOWNLOAD = [Table.MCMC, Table.PARAMS]


def full_model_run_task(run_id: str, burn_in: int, sample_size: int, quiet: bool):
    # Set up directories for output data.
    recreate_dir(FULL_RUN_DATA_DIR)

    # Find the calibration chain databases in AWS S3.
    key_prefix = os.path.join(run_id, os.path.relpath(CALIBRATE_DATA_DIR, REMOTE_BASE_DIR))
    chain_db_keys = list_s3(key_prefix, key_suffix=".parquet")
    chain_db_keys = [k for k in chain_db_keys if any([t in k for t in TABLES_TO_DOWNLOAD])]

    # Download the calibration chain databases.
    with Timer(f"Downloading calibration data"):
        for src_key in chain_db_keys:
            download_from_run_s3(run_id, src_key, quiet)

    # Run the models for the full time period plus all scenarios.
    db_paths = db.load.find_db_paths(CALIBRATE_DATA_DIR)
    chain_ids = [int(p.split("/")[-1].split("-")[-1]) for p in db_paths]
    num_chains = len(chain_ids)
    with Timer(f"Running full models for {num_chains} chains: {chain_ids}"):
        args_list = [
            (run_id, db_path, chain_id, burn_in, sample_size, quiet)
            for chain_id, db_path in zip(chain_ids, db_paths)
        ]
        chain_ids = run_parallel_tasks(run_full_model_for_chain, args_list)

    # Upload the full model run outputs of AWS S3.
    db_paths = db.load.find_db_paths(FULL_RUN_DATA_DIR)
    with Timer(f"Uploading full model run data to AWS S3"):
        for db_path in db_paths:
            upload_to_run_s3(run_id, db_path, quiet)


def run_full_model_for_chain(
    run_id: str, src_db_path: str, chain_id: int, burn_in: int, sample_size: int, quiet: bool
):
    """
    Run the full model (all time steps, all scenarios) for a subset of accepted calibration runs.
    It works like this:
        - We start off with a calibration chain of length C
        - We apply "burn in" by throwing away the first B iterations of the chain, leaving us with C - B iterations
        - We then sample runs from the chain using a "sample size" parameter S by calculating N = floor(C - B / S)
          once we know N, we then start from the end of the chain, working backwards, and select every Nth run
              if a run is accepted then we select it
              if a run is not accepted, we select the first accepted run that precedes it

    Once we've sampled all the runs we need, then we re-run them in full, including all their scenarios.
    """
    set_logging_config(not quiet, chain_id)
    msg = "Running full models for chain %s with burn-in of %s and sample size of %s."
    logger.info(msg, chain_id, burn_in, sample_size)
    try:
        app_region = get_app_region(run_id)
        msg = f"Running the {app_region.app_name} {app_region.region_name} model"
        logger.info(msg)

        dest_db_path = os.path.join(FULL_RUN_DATA_DIR, f"chain-{chain_id}")
        src_db = get_database(src_db_path)
        dest_db = get_database(dest_db_path)

        # Burn in MCMC parameter history and copy it across so it can be used in visualizations downstream.
        # Don't apply sampling to it - we want to see the whole parameter space that was explored.
        mcmc_params_df = src_db.query(Table.PARAMS)
        burn_mask = mcmc_params_df["run"] >= burn_in
        dest_db.dump_df(Table.PARAMS, mcmc_params_df[burn_mask])

        # Add some extra columns to MCMC run history to track sampling.
        mcmc_run_df = src_db.query(Table.MCMC)
        num_runs = len(mcmc_run_df)
        msg = f"Tried to burn {burn_in} runs with sample size {sample_size}, but there are only {num_runs}"
        assert num_runs > (burn_in + sample_size), msg

        # Sampled column tells us whether a run will be sampled.
        sampled = []
        sample_step = (num_runs - burn_in) // sample_size
        for idx, mcmc_run in mcmc_run_df.iterrows():
            should_sample = 1 if (num_runs - idx - 1) % sample_step == 0 else 0
            sampled.append(should_sample)

        mcmc_run_df["sampled"] = sampled

        # Parent column tells us which accepted run precedes this run
        parents = []
        for _, mcmc_run in mcmc_run_df.iterrows():
            if mcmc_run["accept"]:
                parent = int(mcmc_run["run"])

            parents.append(parent)

        mcmc_run_df["parent"] = parents

        # Burn in MCMC run history.
        burn_mask = mcmc_run_df["run"] >= burn_in
        burned_runs_str = ", ".join([str(i) for i in mcmc_run_df[~burn_mask].run])
        logger.info("Burned MCMC runs %s", burned_runs_str)
        mcmc_run_df = mcmc_run_df[burn_mask].copy()
        dest_db.dump_df(Table.MCMC, mcmc_run_df)

        # Figure out which model runs to actually re-run.
        sampled_run_ids = mcmc_run_df[mcmc_run_df["sampled"] == 1].parent.unique().tolist()
        logger.info(
            "Running full model for %s sampled runs %s", len(sampled_run_ids), sampled_run_ids
        )

        outputs = []
        derived_outputs = []
        for sampled_run_id in sampled_run_ids:
            try:
                mcmc_run = mcmc_run_df.loc[mcmc_run_df["run"] == sampled_run_id].iloc[0]
            except IndexError:
                # This happens when we try to sample a parent run that has been burned, we log this and ignore it.
                logger.warn("Skipping (probably) burned parent run id %s", sampled_run_id)
                continue

            run_id = mcmc_run["run"]
            chain_id = mcmc_run["chain"]
            assert mcmc_run["accept"]
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
            dest_db.dump_df(Table.OUTPUTS, outputs_df)
            dest_db.dump_df(Table.DERIVED, derived_outputs_df)
            dest_db.dump_df(Table.MCMC, mcmc_run_df)

    except Exception:
        logger.exception("Full model run for chain %s failed", chain_id)
        raise

    logger.info("Finished running full models for chain %s.", chain_id)
    return chain_id
