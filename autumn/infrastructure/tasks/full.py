import logging
import os
import sys
from pathlib import Path, PurePosixPath
import tempfile
import shutil

import pandas as pd
import numpy as np
import psutil

from autumn.core import db, plots
from autumn.core.db.database import get_database
from autumn.core.db.store import Table
from autumn.core.db.process import find_mle_run
from autumn.infrastructure.tasks.utils import get_project_from_run_id, set_logging_config
from autumn.core.utils.fs import recreate_dir
from autumn.core.utils.parallel import run_parallel_tasks, gather_exc_plus
from autumn.core.utils.s3 import download_from_run_s3, list_s3, upload_to_run_s3, get_s3_client
from autumn.core.utils.timer import Timer
from autumn.core.project import post_process_scenario_outputs
from autumn.core.runs import ManagedRun

from .storage import StorageMode, MockStorage, S3Storage, LocalStorage

logger = logging.getLogger(__name__)

N_CANDIDATES = 15


def full_model_run_task(
    run_id: str, burn_in: int, sample_size: int, quiet: bool, store: str = "s3"
):
    project = get_project_from_run_id(run_id)

    REMOTE_BASE_DIR = Path(tempfile.mkdtemp())
    FULL_RUN_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "data", "full_model_runs")
    FULL_RUN_PLOTS_DIR = os.path.join(REMOTE_BASE_DIR, "plots")
    FULL_RUN_LOG_DIR = os.path.join(REMOTE_BASE_DIR, "logs")
    FULL_RUN_DIRS = [FULL_RUN_DATA_DIR, FULL_RUN_PLOTS_DIR, FULL_RUN_LOG_DIR]

    paths = {"FULL_RUN_DATA_DIR": FULL_RUN_DATA_DIR, "FULL_RUN_LOG_DIR": FULL_RUN_LOG_DIR}

    # Set up directories for output data.
    # Set up directories for plots and output data.
    with Timer(f"Creating full run directories"):
        for dirpath in FULL_RUN_DIRS:
            recreate_dir(dirpath)

    mr = ManagedRun(run_id)

    if store == StorageMode.MOCK:
        storage = MockStorage()
    elif store == StorageMode.S3:
        s3_client = get_s3_client()
        storage = S3Storage(s3_client, run_id, REMOTE_BASE_DIR, not quiet)
    elif store == StorageMode.LOCAL:
        storage = LocalStorage(run_id, REMOTE_BASE_DIR)

    # Select the runs to sample for each chain.  Do this before
    # we enter the parallel loop, so we can filter candidate outputs before
    # the run (and only store data for those outputs)
    mcmc_runs = mr.calibration.get_mcmc_runs()
    mcmc_params = mr.calibration.get_mcmc_params()

    # total_runs = num_chains * sample_size
    total_runs = sample_size

    # Note that sample_size in the task argument is per-chain, whereas here it is all samples
    sampled_runs_df = select_full_run_samples(mcmc_runs, total_runs, burn_in)
    candidates_df = db.process.select_pruning_candidates(sampled_runs_df, N_CANDIDATES)

    # Allocate the runs to available number of cores as evenly as possible
    # Get the number of _physical_ cores (hyperthreading will only muddy the waters...)
    n_cores = psutil.cpu_count(logical=False)

    # Really shouldn't happen, but will come up in testing...
    if total_runs < n_cores:
        base_runs_per_core = 1
        n_base = total_runs
        n_additional = 0
    else:
        # The usual case, of having more runs than we have cores
        base_runs_per_core = total_runs // n_cores
        # Number of cores that need to run 1 extra run
        n_additional = total_runs % n_cores
        n_base = n_cores - n_additional

    subset_runs = []

    brpc = base_runs_per_core
    arpc = base_runs_per_core + 1

    cur_start_idx = 0

    for i in range(n_additional):
        subset_runs.append(sampled_runs_df.iloc[cur_start_idx : cur_start_idx + arpc])
        # This will be correct for the next iteration, hence computing at the end
        cur_start_idx += arpc
    for i in range(n_base):
        subset_runs.append(sampled_runs_df.iloc[cur_start_idx : cur_start_idx + brpc])
        cur_start_idx += brpc

    # Check that we ended up with all the runs
    assert cur_start_idx == total_runs, "Invalid run index calculation"

    # Check we're using our cores correctly..
    if len(subset_runs) < n_cores:
        logger.warning(f"{n_cores} CPU cores, but only {len(subset_runs)} being used")

    assert len(subset_runs) <= n_cores, "Invalid CPU oversubscription"

    # OK, actually run the thing...
    with Timer(f"Running {total_runs} full models over {len(subset_runs)} subsets"):
        args_list = [
            (
                run_id,
                subset_id,
                subset_runs[subset_id],
                mcmc_params.loc[subset_runs[subset_id].index],
                candidates_df,
                quiet,
                paths,
            )
            for subset_id in range(len(subset_runs))
        ]
        try:
            subset_ids = run_parallel_tasks(run_full_model_for_subset, args_list, False)
            success = True
        except Exception as e:
            # Run failed but we still want to capture the logs
            success = False

    with Timer("Uploading logs"):
        storage.store(FULL_RUN_LOG_DIR)

    if not success:
        logger.info("Terminating early from failure")
        sys.exit(-1)

    # Upload the full model run outputs of AWS S3.
    # db_paths = db.load.find_db_paths(FULL_RUN_DATA_DIR)
    with Timer(f"Uploading full model run data to AWS S3"):
        storage.store(FULL_RUN_DATA_DIR)
    #    for db_path in db_paths:
    #        upload_to_run_s3(s3_client, run_id, db_path, quiet)

    # Create candidate plots from full run outputs

    try:
        with Timer(f"Creating candidate selection plots"):
            plots.model.plot_post_full_run(
                project.plots, FULL_RUN_DATA_DIR, FULL_RUN_PLOTS_DIR, candidates_df
            )
    except Exception as e:
        logger.warning("Candidate plots failed, resuming task")

    # May we may still have some valid plots - upload them to AWS S3.
    with Timer(f"Uploading plots to AWS S3"):
        # upload_to_run_s3(s3_client, run_id, FULL_RUN_PLOTS_DIR, quiet)
        storage.store(FULL_RUN_PLOTS_DIR)

    shutil.rmtree(REMOTE_BASE_DIR)


def run_full_model_for_subset(
    run_id: str,
    subset_id: int,
    sampled_runs_df: pd.DataFrame,
    mcmc_params_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    quiet: bool,
    paths: dict,
) -> int:
    """
    Run the full model (all time steps, all scenarios) for a subset of accepted calibration runs.
    The sampling for these runs occurs in the main process, and all arguments here are specific
    to the current chain

    Args:
        run_id (str): Model run id
        subset_id (int): The current subset
        sampled_runs_df (pd.DataFrame): mcmc_runs (samples for this chain only)
        mcmc_params_df (pd.DataFrame): mcmc_params (samples for this chain only)
        candidates_df (pd.DataFrame): Candidates for all chains
        quiet (bool): Inverse of logging verbosity

    Returns:
        subset_id (int): The current subset
    """

    FULL_RUN_DATA_DIR = paths["FULL_RUN_DATA_DIR"]
    FULL_RUN_LOG_DIR = paths["FULL_RUN_LOG_DIR"]

    set_logging_config(not quiet, subset_id, FULL_RUN_LOG_DIR, task="full")
    # msg = "Running full models for chain %s with burn-in of %s and sample size of %s."
    # logger.info(msg, chain_id, burn_in, sample_size)
    try:
        project = get_project_from_run_id(run_id)
        msg = f"Running the {project.model_name} {project.region_name} model"
        logger.info(msg)

        dest_db_path = os.path.join(FULL_RUN_DATA_DIR, f"chain-{subset_id}")
        # src_db = get_database(src_db_path)
        dest_db = get_database(dest_db_path)

        # Burn in MCMC parameter history and copy it across so it can be used in visualizations downstream.
        # Don't apply sampling to it - we want to see the whole parameter space that was explored.
        # OK, so this is pre-filtered for sampling - but maybe we shouldn't be storing this here anyway?
        # ie just get it from the calibration run rather than having weird duplicates everywhere...
        dest_db.dump_df(Table.PARAMS, mcmc_params_df)

        outputs = []
        derived_outputs = []

        # Set up build options variables; empty for first run, these will be filled in later
        bl_build_opts = None
        sc_build_opts = None

        for urun in sampled_runs_df.index:

            mcmc_run = sampled_runs_df.loc[urun]

            run_id = int(mcmc_run["run"])
            chain_id = int(mcmc_run["chain"])
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

                baseline_model = project.run_baseline_model(
                    baseline_params, build_options=bl_build_opts
                )
                sc_models = project.run_scenario_models(
                    baseline_model, scenario_params, build_options=sc_build_opts
                )

            models = [baseline_model, *sc_models]

            # Get cache info etc for build_options dict
            # This improves performance for subsequent runs

            bl_build_opts = {
                "enable_validation": False,
                "derived_outputs_idx_cache": baseline_model._derived_outputs_idx_cache,
            }

            sc_build_opts = []
            for scm in sc_models:
                sc_build_opts.append(
                    {
                        "enable_validation": False,
                        "derived_outputs_idx_cache": scm._derived_outputs_idx_cache,
                    }
                )

            run_id = int(run_id)
            chain_id = int(chain_id)
            with Timer("Processing model outputs"):
                processed_outputs = post_process_scenario_outputs(
                    models, project, run_id=run_id, chain_id=chain_id
                )
                derived_outputs.append(processed_outputs[Table.DERIVED])

        with Timer("Saving model outputs to the database"):
            final_outputs = {}
            # We may not have anything in outputs
            final_outputs[Table.DERIVED] = pd.concat(derived_outputs, copy=False, ignore_index=True)
            final_outputs[Table.MCMC] = sampled_runs_df
            db.store.save_model_outputs(dest_db, **final_outputs)

    except Exception:
        logger.exception("Full model run for subset %s failed", subset_id)
        gather_exc_plus(os.path.join(FULL_RUN_LOG_DIR, f"crash-full-{subset_id}.log"))
        raise

    logger.info("Finished running full models for subset %s.", subset_id)
    return subset_id


def select_full_run_samples(
    mcmc_runs_df: pd.DataFrame, n_samples: int, burn_in: int
) -> pd.DataFrame:

    accept_mask = mcmc_runs_df["accept"] == 1
    df_accepted = mcmc_runs_df[accept_mask]
    mle_idx = find_mle_run(df_accepted).index[0]
    df_accepted = df_accepted.drop(index=mle_idx)

    post_burn = df_accepted[df_accepted["run"] >= burn_in]

    weights = post_burn["weight"] / post_burn["weight"].sum()

    run_indices = [mle_idx] + list(
        np.random.choice(post_burn.index, n_samples - 1, False, p=weights)
    )

    return mcmc_runs_df.loc[run_indices]
