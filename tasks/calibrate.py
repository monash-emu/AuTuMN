import logging
import os
import shutil
from tempfile import TemporaryDirectory

from autumn import db, plots
from settings import REMOTE_BASE_DIR
from tasks.utils import get_app_region, set_logging_config
from utils.fs import recreate_dir
from utils.parallel import run_parallel_tasks
from utils.s3 import upload_to_run_s3
from utils.timer import Timer

logger = logging.getLogger(__name__)


os.makedirs(REMOTE_BASE_DIR, exist_ok=True)

CALIBRATE_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "data", "calibration_outputs")
CALIBRATE_PLOTS_DIR = os.path.join(REMOTE_BASE_DIR, "plots")
CALIBRATE_DIRS = [CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR]
MLE_PARAMS_PATH = os.path.join(CALIBRATE_DATA_DIR, "mle-params.yml")


def calibrate_task(run_id: str, runtime: float, num_chains: int, verbose: bool):

    # Set up directories for plots and output data.
    with Timer(f"Creating calibration directories"):
        for dirpath in CALIBRATE_DIRS:
            recreate_dir(dirpath)

    # Run the actual calibrations
    with Timer(f"Running {num_chains} calibration chains"):
        args_list = [
            (run_id, runtime, chain_id, num_chains, verbose) for chain_id in range(num_chains)
        ]
        chain_ids = run_parallel_tasks(run_calibration_chain, args_list)

    # Upload the calibration outputs of AWS S3.
    with Timer(f"Uploading calibration data to AWS S3"):
        for chain_id in chain_ids:
            with Timer(f"Uploading data for chain {chain_id} to AWS S3"):
                src_dir = os.path.join(CALIBRATE_DATA_DIR, f"chain-{chain_id}")
                upload_to_run_s3(run_id, src_dir, quiet=not verbose)

    # Create plots from the calibration outputs.
    with Timer(f"Creating post-calibration plots"):
        app_region = get_app_region(run_id)
        plots.calibration.plot_post_calibration(
            app_region.targets, CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR, priors=[None]
        )

    # Upload the plots to AWS S3.
    with Timer(f"Uploading plots to AWS S3"):
        upload_to_run_s3(run_id, CALIBRATE_PLOTS_DIR, quiet=not verbose)

    # Find the MLE parameter set from all the chains.
    with Timer(f"Finding max likelihood esitmate params"):
        database_paths = db.load.find_db_paths(CALIBRATE_DATA_DIR)
        with TemporaryDirectory() as tmp_dir_path:
            collated_db_path = os.path.join(tmp_dir_path, "collated.db")
            db.process.collate_databases(
                database_paths, collated_db_path, tables=["mcmc_run", "mcmc_params"]
            )
            db.store.save_mle_params(collated_db_path, MLE_PARAMS_PATH)

    # Upload the MLE parameter set to AWS S3.
    with Timer(f"Uploading max likelihood esitmate params to AWS S3"):
        upload_to_run_s3(run_id, MLE_PARAMS_PATH, quiet=not verbose)


def run_calibration_chain(
    run_id: str, runtime: float, chain_id: int, num_chains: int, verbose: bool
):
    """
    Run a single calibration chain.
    """
    set_logging_config(verbose, chain_id)
    logging.info("Running calibration chain %s", chain_id)
    os.environ["AUTUMN_CALIBRATE_DIR"] = CALIBRATE_DATA_DIR
    try:
        app_region = get_app_region(run_id)
        app_region.calibrate_model(runtime, chain_id, num_chains)
    except Exception:
        logger.exception("Calibration chain %s failed", chain_id)
        raise
    logging.info("Finished running calibration chain %s", chain_id)
    return chain_id
