import logging
import os
import sys
from pathlib import Path, PurePosixPath

from autumn.core import db, plots
from autumn.settings import REMOTE_BASE_DIR
from autumn.core.utils.parallel import run_parallel_tasks, gather_exc_plus
from autumn.core.utils.fs import recreate_dir
from autumn.core.utils.s3 import get_s3_client
from autumn.core.utils.timer import Timer
from .utils import get_project_from_run_id, set_logging_config
from .storage import StorageMode, MockStorage, S3Storage, LocalStorage

logger = logging.getLogger(__name__)


os.makedirs(REMOTE_BASE_DIR, exist_ok=True)

CALIBRATE_DATA_DIR = REMOTE_BASE_DIR / "data/calibration_outputs"
CALIBRATE_PLOTS_DIR = REMOTE_BASE_DIR / "plots"
CALIBRATE_LOG_DIR = REMOTE_BASE_DIR / "logs"
CALIBRATE_DIRS = [CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR, CALIBRATE_LOG_DIR]
MLE_PARAMS_PATH = CALIBRATE_DATA_DIR / "mle-params.yml"


def calibrate_task(run_id: str, runtime: float, num_chains: int, verbose: bool, store="s3"):

    if store == StorageMode.MOCK:
        storage = MockStorage()
    elif store == StorageMode.S3:
        s3_client = get_s3_client()
        storage = S3Storage(s3_client, run_id, REMOTE_BASE_DIR, verbose)
    elif store == StorageMode.LOCAL:
        storage = LocalStorage(run_id, REMOTE_BASE_DIR)

    # Set up directories for plots and output data.
    with Timer(f"Creating calibration directories"):
        for dirpath in CALIBRATE_DIRS:
            recreate_dir(dirpath)

    # Run the actual calibrations
    with Timer(f"Running {num_chains} calibration chains"):
        args_list = [
            (run_id, runtime, chain_id, num_chains, verbose) for chain_id in range(num_chains)
        ]
        try:
            chain_ids = run_parallel_tasks(run_calibration_chain, args_list, False)
            cal_success = True
        except Exception as e:
            # Calibration failed, but we still want to store some results
            cal_success = False

    with Timer("Persisting logs"):
        # store_run(s3_client, run_id, CALIBRATE_LOG_DIR, quiet=not verbose)
        storage.store(CALIBRATE_LOG_DIR)

    with Timer("Persisting run data"):
        # store_run(s3_client, run_id, CALIBRATE_DATA_DIR, quiet=not verbose)
        storage.store(CALIBRATE_DATA_DIR)

    if not cal_success:
        logger.info("Terminating early from failure")
        sys.exit(-1)

    # Create plots from the calibration outputs.
    with Timer(f"Creating post-calibration plots"):
        project = get_project_from_run_id(run_id)
        plots.calibration.plot_post_calibration(
            project.plots, CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR, priors=[]
        )

    # Upload the plots to AWS S3.
    with Timer(f"Persisting plots"):
        storage.store(CALIBRATE_PLOTS_DIR)

    # Find the MLE parameter set from all the chains.
    with Timer(f"Finding max likelihood estimate params"):
        database_paths = db.load.find_db_paths(CALIBRATE_DATA_DIR)
        collated_db_path = CALIBRATE_DATA_DIR / "mcmc_collated.db"
        db.process.collate_databases(
            database_paths, collated_db_path, tables=["mcmc_run", "mcmc_params"]
        )
        db.store.save_mle_params(collated_db_path, MLE_PARAMS_PATH)
        storage.store(collated_db_path)
        storage.store(MLE_PARAMS_PATH)

    with Timer(f"Persisting final logs"):
        storage.store(CALIBRATE_LOG_DIR)


def run_calibration_chain(
    run_id: str, runtime: float, chain_id: int, num_chains: int, verbose: bool
):
    """
    Run a single calibration chain.
    """
    set_logging_config(verbose, chain_id, CALIBRATE_LOG_DIR, task="calibration")
    logging.info("Running calibration chain %s", chain_id)
    os.environ["AUTUMN_CALIBRATE_DIR"] = str(CALIBRATE_DATA_DIR)

    import numpy as np

    np.seterr(divide="raise", over="raise", under="ignore", invalid="raise")

    try:
        project = get_project_from_run_id(run_id)
        project._calibrate(runtime, chain_id, num_chains)
    except Exception:
        logger.exception("Calibration chain %s failed", chain_id)
        gather_exc_plus(os.path.join(CALIBRATE_LOG_DIR, f"crash-calibration-{chain_id}.log"))
        raise
    logging.info("Finished running calibration chain %s", chain_id)
    return chain_id
