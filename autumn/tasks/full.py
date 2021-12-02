import logging
import os
import sys

import pandas as pd
import numpy as np

from autumn.tools import db, plots
from autumn.tools.db.database import get_database
from autumn.tools.db.store import Table
from autumn.tools.db.process import find_mle_run
from autumn.tools.utils.pandas import pdfilt
from autumn.settings import REMOTE_BASE_DIR, Region
from autumn.tasks.calibrate import CALIBRATE_DATA_DIR
from autumn.tasks.utils import get_project_from_run_id, set_logging_config
from autumn.tools.utils.fs import recreate_dir
from autumn.tools.utils.parallel import run_parallel_tasks, report_errors, gather_exc_plus
from autumn.tools.utils.s3 import download_from_run_s3, list_s3, upload_to_run_s3, get_s3_client
from autumn.tools.utils.timer import Timer
from autumn.tools.project import post_process_scenario_outputs
from autumn.tools.runs import ManagedRun

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

    mr = ManagedRun(run_id, s3_client=s3_client)

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

    # Select the runs to sample for each chain.  Do this before
    # we enter the parallel loop, so we can filter candidate outputs before
    # the run (and only store data for those outputs)
    mcmc_runs = mr.calibration.get_mcmc_runs()
    mcmc_params = mr.calibration.get_mcmc_params()
    
    # Note that sample_size in the task argument is per-chain, whereas here it is all samples
    sampled_runs_df = select_full_run_samples(mcmc_runs, num_chains * sample_size, burn_in)
    candidates_df = db.process.select_pruning_candidates(sampled_runs_df, N_CANDIDATES)


    with Timer(f"Running full models for {num_chains} chains: {chain_ids}"):
        args_list = [
            (run_id, chain_id, sampled_runs_df[sampled_runs_df['chain'] == chain_id],
            mcmc_params.loc[(sampled_runs_df["chain"] == chain_id).index],
            candidates_df, quiet)
            for chain_id in chain_ids
        ]
        try:
            chain_ids = run_parallel_tasks(run_full_model_for_chain, args_list, False)
        except Exception as e:
            # Run failed but we still want to capture the logs
            upload_to_run_s3(s3_client, run_id, FULL_RUN_LOG_DIR, quiet)
            # +++ FIXME Do we always want sys.exit here? (Presumably need it to stop remote tasks hanging)
            sys.exit(-1)

    # Upload the full model run outputs of AWS S3.
    db_paths = db.load.find_db_paths(FULL_RUN_DATA_DIR)
    with Timer(f"Uploading full model run data to AWS S3"):
        for db_path in db_paths:
            upload_to_run_s3(s3_client, run_id, db_path, quiet)

    # Create candidate plots from full run outputs

    try:
        with Timer(f"Creating candidate selection plots"):
            plots.model.plot_post_full_run(
                project.plots, FULL_RUN_DATA_DIR, FULL_RUN_PLOTS_DIR, candidates_df
            )

        # Upload the plots to AWS S3.
        with Timer(f"Uploading plots to AWS S3"):
            upload_to_run_s3(s3_client, run_id, FULL_RUN_PLOTS_DIR, quiet)

    except Exception as e:
        logger.warning("Candidate plots failed, resuming task")



@report_errors
def run_full_model_for_chain(
    run_id: str, chain_id: int, sampled_runs_df: pd.DataFrame, 
    mcmc_params_df: pd.DataFrame, candidates_df: pd.DataFrame, quiet: bool
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
    set_logging_config(not quiet, chain_id, FULL_RUN_LOG_DIR, task='full')
    #msg = "Running full models for chain %s with burn-in of %s and sample size of %s."
    #logger.info(msg, chain_id, burn_in, sample_size)
    try:
        project = get_project_from_run_id(run_id)
        msg = f"Running the {project.model_name} {project.region_name} model"
        logger.info(msg)

        dest_db_path = os.path.join(FULL_RUN_DATA_DIR, f"chain-{chain_id}")
        #src_db = get_database(src_db_path)
        dest_db = get_database(dest_db_path)

        # Burn in MCMC parameter history and copy it across so it can be used in visualizations downstream.
        # Don't apply sampling to it - we want to see the whole parameter space that was explored.
        # OK, so this is pre-filtered for sampling - but maybe we shouldn't be storing this here anyway?
        # ie just get it from the calibration run rather than having weird duplicates everywhere...
        dest_db.dump_df(Table.PARAMS, mcmc_params_df)

        outputs = []
        derived_outputs = []
        for urun in sampled_runs_df.index:
            
            mcmc_run = sampled_runs_df.loc[urun]

            run_id = mcmc_run["run"]
            chain_id = mcmc_run["chain"]
            assert mcmc_run["accept"]
            logger.info("Running full model for MCMC run %s", run_id)

            param_updates = mcmc_params_df.loc[urun].to_dict()

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
                # Only store outputs that are required for candidate runs
                if urun in candidates_df.index:
                    outputs.append(processed_outputs[Table.OUTPUTS])
                derived_outputs.append(processed_outputs[Table.DERIVED])

        with Timer("Saving model outputs to the database"):
            final_outputs = {}
            final_outputs[Table.OUTPUTS] = pd.concat(outputs, copy=False, ignore_index=True)
            final_outputs[Table.DERIVED] = pd.concat(derived_outputs, copy=False, ignore_index=True)
            final_outputs[Table.MCMC] = sampled_runs_df
            db.store.save_model_outputs(dest_db, **final_outputs)

    except Exception:
        logger.exception("Full model run for chain %s failed", chain_id)
        gather_exc_plus(os.path.join(FULL_RUN_LOG_DIR, f"crash-full-{chain_id}.log"))
        raise

    logger.info("Finished running full models for chain %s.", chain_id)
    return chain_id

def select_full_run_samples(mcmc_runs_df: pd.DataFrame, n_samples: int, burn_in: int) -> pd.DataFrame:
    
    accept_mask = mcmc_runs_df['accept'] == 1
    df_accepted = mcmc_runs_df[accept_mask]
    mle_idx = find_mle_run(df_accepted).index[0]
    df_accepted = df_accepted.drop(index=mle_idx)
    
    post_burn = df_accepted[df_accepted['run'] >= burn_in]
    
    weights = post_burn['weight'] / post_burn['weight'].sum()
    
    run_indices = [mle_idx] + list(np.random.choice(post_burn.index, n_samples-1, False, p = weights))
    
    return mcmc_runs_df.loc[run_indices]