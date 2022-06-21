import logging
import os
import sys
from tempfile import TemporaryDirectory

from autumn.core import db, plots
from autumn.settings import REMOTE_BASE_DIR
from autumn.core.utils.parallel import run_parallel_tasks, gather_exc_plus
from autumn.core.utils.fs import recreate_dir
from autumn.core.utils.s3 import upload_to_run_s3, get_s3_client, list_s3, download_from_run_s3
from autumn.core.utils.timer import Timer
from autumn.calibration
from .utils import get_project_from_run_id, set_logging_config

logger = logging.getLogger(__name__)


os.makedirs(REMOTE_BASE_DIR, exist_ok=True)

CALIBRATE_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "data", "calibration_outputs")
CALIBRATE_PLOTS_DIR = os.path.join(REMOTE_BASE_DIR, "plots")
CALIBRATE_LOG_DIR = os.path.join(REMOTE_BASE_DIR, "logs")
CALIBRATE_DIRS = [CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR, CALIBRATE_LOG_DIR]
MLE_PARAMS_PATH = os.path.join(CALIBRATE_DATA_DIR, "mle-params.yml")


def resume_calibration_task(run_id: str, base_run_id: str, runtime: float, num_chains: int, verbose: bool = False):
    s3_client = get_s3_client()

    # Set up directories for plots and output data.
    with Timer(f"Creating calibration directories"):
        for dirpath in CALIBRATE_DIRS:
            recreate_dir(dirpath)

    # Download data for existing calibration chain
    # We need the pickled Calibration objects, and the previous MCMC run/parameter data

    key_prefix = os.path.join(base_run_id, os.path.relpath(CALIBRATE_DATA_DIR, REMOTE_BASE_DIR))
    all_cal_data_keys = list_s3(s3_client, key_prefix)

    keys_to_dl = [k for k in all_cal_data_keys if any([t in k for t in ['.pkl', 'mcmc_run.parquet', 'mcmc_params.parquet']])]

    with Timer(f"Downloading existing calibration data"):
        for src_key in keys_to_dl:
            download_from_run_s3(s3_client, base_run_id, src_key, not verbose)

    # Run the actual calibrations
    with Timer(f"Resuming {num_chains} calibration chains"):
        args_list = [
            (runtime, chain_id, verbose) for chain_id in range(num_chains)
        ]
        try:
            chain_ids = run_parallel_tasks(resume_calibration_chain, args_list, False)
            cal_success = True
        except Exception as e:
            # Calibration failed, but we still want to store some results
            cal_success = False
    
    with Timer("Uploading logs"):
        upload_to_run_s3(s3_client, run_id, CALIBRATE_LOG_DIR, quiet=not verbose)

    with Timer("Uploading run data"):
        upload_to_run_s3(s3_client, run_id, CALIBRATE_DATA_DIR, quiet=not verbose)
        
    if not cal_success:
        logger.info("Terminating early from failure")
        sys.exit(-1)

    # Upload the calibration outputs of AWS S3.
    #with Timer(f"Uploading calibration data to AWS S3"):
    #    for chain_id in chain_ids:
    #        with Timer(f"Uploading data for chain {chain_id} to AWS S3"):
    #            src_dir = os.path.join(CALIBRATE_DATA_DIR, f"chain-{chain_id}")
    #            upload_to_run_s3(s3_client, run_id, src_dir, quiet=not verbose)

    # Create plots from the calibration outputs.
    with Timer(f"Creating post-calibration plots"):
        project = get_project_from_run_id(run_id)
        plots.calibration.plot_post_calibration(
            project.plots, CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR, priors=[]
        )

    # Upload the plots to AWS S3.
    with Timer(f"Uploading plots to AWS S3"):
        upload_to_run_s3(s3_client, run_id, CALIBRATE_PLOTS_DIR, quiet=not verbose)

    # Find the MLE parameter set from all the chains.
    with Timer(f"Finding max likelihood estimate params"):
        database_paths = db.load.find_db_paths(CALIBRATE_DATA_DIR)
        with TemporaryDirectory() as tmp_dir_path:
            collated_db_path = os.path.join(tmp_dir_path, "collated.db")
            db.process.collate_databases(
                database_paths, collated_db_path, tables=["mcmc_run", "mcmc_params"]
            )
            db.store.save_mle_params(collated_db_path, MLE_PARAMS_PATH)

    # Upload the MLE parameter set to AWS S3.
    with Timer(f"Uploading max likelihood estimate params to AWS S3"):
        upload_to_run_s3(s3_client, run_id, MLE_PARAMS_PATH, quiet=not verbose)

    with Timer(f"Uploading final logs to AWS S3"):
        upload_to_run_s3(s3_client, run_id, 'log', quiet=not verbose)


def resume_calibration_chain(
    runtime: float, chain_id: int, verbose: bool
):
    """
    Run a single calibration chain.
    """
    set_logging_config(verbose, chain_id, CALIBRATE_LOG_DIR, task='calibration')
    logging.info("Running calibration chain %s", chain_id)
    os.environ["AUTUMN_CALIBRATE_DIR"] = CALIBRATE_DATA_DIR

    try:
        #project = get_project_from_run_id(run_id)
        #project.calibrate(runtime, chain_id, num_chains)
        rcal = Calibration.from_existing(os.path.join(CALIBRATE_DATA_DIR, f"calstate-{chain_id}.pkl"), CALIBRATE_DATA_DIR)
        rcal.resume_autumn_mcmc(runtime)
    except Exception:
        logger.exception("Calibration chain %s failed", chain_id)
        gather_exc_plus(os.path.join(CALIBRATE_LOG_DIR, f"crash-resume-{chain_id}.log"))
        raise
    logging.info("Finished running calibration chain %s", chain_id)
    return chain_id