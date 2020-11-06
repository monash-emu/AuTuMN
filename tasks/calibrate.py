import logging
import os
import shutil
from tempfile import TemporaryDirectory

from autumn.inputs import build_input_database
from autumn.tool_kit import Timer
from autumn import plots, db

from tasks import utils, settings

logger = logging.getLogger(__name__)


os.makedirs(settings.BASE_DIR, exist_ok=True)

CALIBRATE_DATA_DIR = os.path.join(settings.BASE_DIR, "data", "calibration_outputs")
CALIBRATE_PLOTS_DIR = os.path.join(settings.BASE_DIR, "plots")
CALIBRATE_DIRS = [CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR]
MLE_PARAMS_PATH = os.path.join(CALIBRATE_DATA_DIR, "mle-params.yml")


def calibrate_task(run_id: str, runtime: float, num_chains: int, quiet: bool):
    # Prepare inputs for running the model
    build_input_database()

    # Set up directories for plots and output data.
    with Timer(f"Creating calibration directories"):
        for dirpath in CALIBRATE_DIRS:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)

            os.makedirs(dirpath)

    # Run the actual calibrations
    with Timer(f"Running {num_chains} calibration chains"):
        args_list = [
            (run_id, runtime, chain_id, num_chains, quiet) for chain_id in range(num_chains)
        ]
        chain_ids = utils.run_parallel_tasks(run_calibration_chain, args_list)

    # Upload the calibration outputs of AWS S3.
    with Timer(f"Uploading calibration data to AWS S3"):
        args_list = [
            (run_id, os.path.join(CALIBRATE_DATA_DIR, f"chain-{chain_id}"), quiet)
            for chain_id in chain_ids
        ]
        utils.run_parallel_tasks(utils.upload_to_run_s3, args_list)

    # Create plots from the calibration outputs.
    with Timer(f"Creating post-calibration plots"):
        app_region = utils.get_app_region(run_id)
        plots.calibration.plot_post_calibration(
            app_region.targets, CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR
        )

    # Upload the plots to AWS S3.
    with Timer(f"Uploading plots to AWS S3"):
        utils.upload_to_run_s3(run_id, CALIBRATE_PLOTS_DIR, quiet)

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
        utils.upload_to_run_s3(run_id, MLE_PARAMS_PATH, quiet)


def run_calibration_chain(run_id: str, runtime: float, chain_id: int, num_chains: int, quiet: bool):
    """
    Run a single calibration chain.
    """
    if quiet:
        logging.disable(logging.INFO)

    os.environ["AUTUMN_CALIBRATE_DIR"] = CALIBRATE_DATA_DIR
    app_region = utils.get_app_region(run_id)
    app_region.calibrate_model(runtime, chain_id, num_chains)

    logging.info("Running calibration chain %s", chain_id)
    return chain_id
