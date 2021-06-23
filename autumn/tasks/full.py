import logging
import os

import pandas as pd

from autumn.tools import db, plots
from autumn.tools.db.database import get_database
from autumn.tools.db.store import Table
from autumn.settings import REMOTE_BASE_DIR, Region
from autumn.tasks.calibrate import CALIBRATE_DATA_DIR
from autumn.tasks.utils import get_project_from_run_id, set_logging_config
from autumn.tools.utils.fs import recreate_dir
from autumn.tools.utils.parallel import run_parallel_tasks, report_errors, gather_exc_plus
from autumn.tools.utils.s3 import download_from_run_s3, list_s3, upload_to_run_s3, get_s3_client
from autumn.tools.utils.timer import Timer
from autumn.tools.project import post_process_scenario_outputs

logger = logging.getLogger(__name__)

N_CANDIDATES = 15

FULL_RUN_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "data", "full_model_runs")
FULL_RUN_PLOTS_DIR = os.path.join(REMOTE_BASE_DIR, "plots")
FULL_RUN_LOG_DIR = os.path.join(REMOTE_BASE_DIR, "logs")
FULL_RUN_DIRS = [FULL_RUN_DATA_DIR, FULL_RUN_PLOTS_DIR, FULL_RUN_LOG_DIR]
TABLES_TO_DOWNLOAD = [Table.MCMC, Table.PARAMS]


def full_model_run_task(run_id: str, burn_in: int, sample_size: int, quiet: bool):
    project = get_project_from_run_id(run_id)

    # Set up directories for output data.
    # Set up directories for plots and output data.
    with Timer(f"Creating full run directories"):
        for dirpath in FULL_RUN_DIRS:
            recreate_dir(dirpath)

    s3_client = get_s3_client()

    # Find the calibration chain databases in AWS S3.
    key_prefix = os.path.join(run_id, os.path.relpath(CALIBRATE_DATA_DIR, REMOTE_BASE_DIR))
    chain_db_keys = list_s3(s3_client, key_prefix, key_suffix=".parquet")
    chain_db_keys = [k for k in chain_db_keys if any([t in k for t in TABLES_TO_DOWNLOAD])]

    # Download the calibration chain databases.
    with Timer(f"Downloading calibration data"):
        for src_key in chain_db_keys:
            download_from_run_s3(s3_client, run_id, src_key, quiet)

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

    # Create candidate plots from full run outputs
    if project.region_name in Region.PHILIPPINES_REGIONS:
        with Timer(f"Creating candidate selection plots"):
            candidates_df = db.process.select_pruning_candidates(FULL_RUN_DATA_DIR, N_CANDIDATES)
            plots.model.plot_post_full_run(
                project.plots, FULL_RUN_DATA_DIR, FULL_RUN_PLOTS_DIR, candidates_df
            )

    # Upload the plots to AWS S3.
    with Timer(f"Uploading plots to AWS S3"):
        upload_to_run_s3(s3_client, run_id, FULL_RUN_PLOTS_DIR, quiet)

    # Upload the full model run outputs of AWS S3.
    db_paths = db.load.find_db_paths(FULL_RUN_DATA_DIR)
    with Timer(f"Uploading full model run data to AWS S3"):
        for db_path in db_paths:
            upload_to_run_s3(s3_client, run_id, db_path, quiet)


@report_errors
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
    set_logging_config(not quiet, chain_id, FULL_RUN_LOG_DIR)
    msg = "Running full models for chain %s with burn-in of %s and sample size of %s."
    logger.info(msg, chain_id, burn_in, sample_size)
    try:
        project = get_project_from_run_id(run_id)
        msg = f"Running the {project.model_name} {project.region_name} model"
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
        sample_step = max(1, (num_runs - burn_in) // sample_size)
        logger.info("Using a sample step of %s", sample_step)
        for idx, mcmc_run in mcmc_run_df.iterrows():
            should_sample = 1 if (num_runs - idx - 1) % sample_step == 0 else 0
            sampled.append(should_sample)

        mcmc_run_df["sampled"] = sampled

        # Parent column tells us which accepted run precedes this run
        parents = []
        i_row = 0  # FIXME: This is a temporary patch.
        for _, mcmc_run in mcmc_run_df.iterrows():
            if mcmc_run["accept"] or i_row == 0:
                parent = int(mcmc_run["run"])

            parents.append(parent)
            i_row += 1

        mcmc_run_df["parent"] = parents

        # Burn in MCMC run history.
        burn_mask = mcmc_run_df["run"] >= burn_in
        burned_runs_str = ", ".join([str(i) for i in mcmc_run_df[~burn_mask].run])
        mcmc_run_df = mcmc_run_df[burn_mask].copy()
        num_remaining = len(mcmc_run_df)
        logger.info(
            "Burned %s of %s MCMC runs leaving %s remaining.", burn_in, num_runs, num_remaining
        )

        logger.info("Burned MCMC runs %s", burned_runs_str)

        # Figure out which model runs to actually re-run.
        sampled_run_ids = mcmc_run_df[mcmc_run_df["sampled"] == 1].parent.unique().tolist()

        # Also include the MLE
        mle_df = db.process.find_mle_run(mcmc_run_df)
        mle_run_id = mle_df["run"].iloc[0]

        logger.info("Including MLE run %s", mle_run_id)

        # Update sampled column to reflect inclusion of MLE run
        mle_run_loc = mcmc_run_df.index[mcmc_run_df["run"] == mle_run_id][0]
        mcmc_run_df.loc[mle_run_loc, "sampled"] = 1

        sampled_run_ids.append(mle_run_id)
        sampled_run_ids = sorted(list(set(sampled_run_ids)))
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

            with Timer("Running all model scenarios"):
                baseline_params = project.param_set.baseline.update(
                    param_updates, calibration_format=True
                )
                scenario_params = [
                    p.update(param_updates, calibration_format=True)
                    for p in project.param_set.scenarios
                ]
                start_times = [
                    sc_params.to_dict()["time"]["start"] for sc_params in scenario_params
                ]

                baseline_model = project.run_baseline_model(baseline_params)
                sc_models = project.run_scenario_models(
                    baseline_model, scenario_params, start_times=start_times
                )

            models = [baseline_model, *sc_models]

            run_id = int(run_id)
            chain_id = int(chain_id)
            with Timer("Processing model outputs"):
                processed_outputs = post_process_scenario_outputs(
                    models, project, run_id=run_id, chain_id=chain_id
                )
                outputs.append(processed_outputs[Table.OUTPUTS])
                derived_outputs.append(processed_outputs[Table.DERIVED])

        with Timer("Saving model outputs to the database"):
            final_outputs = {}
            final_outputs[Table.OUTPUTS] = pd.concat(outputs, copy=False, ignore_index=True)
            final_outputs[Table.DERIVED] = pd.concat(derived_outputs, copy=False, ignore_index=True)
            final_outputs[Table.MCMC] = mcmc_run_df
            db.store.save_model_outputs(dest_db, **final_outputs)

    except Exception:
        logger.exception("Full model run for chain %s failed", chain_id)
        gather_exc_plus(os.path.join(FULL_RUN_LOG_DIR, f"crash-full-{chain_id}.log"))
        raise

    logger.info("Finished running full models for chain %s.", chain_id)
    return chain_id
