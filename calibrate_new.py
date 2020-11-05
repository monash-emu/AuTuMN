import os
import logging
import traceback
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, List, Any
import shutil
from contextlib import contextmanager


# Configure command logging
logging.basicConfig(format="%(asctime)s %(module)s:%(levelname)s: %(message)s", level=logging.INFO)

# Configure logging for the Boto3 library
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("nose").setLevel(logging.WARNING)

from autumn.inputs import build_input_database
from autumn.tool_kit import Timer
from autumn import plots

from tasks import utils
from tasks import settings

logger = logging.getLogger(__name__)


MAX_WORKERS = mp.cpu_count() - 1
os.makedirs(settings.BASE_DIR, exist_ok=True)

CALIBRATE_DATA_DIR = os.path.join(settings.BASE_DIR, "data", "calibrate")
CALIBRATE_PLOTS_DIR = os.path.join(settings.BASE_DIR, "plots", "calibrate")
CALIBRATE_DIRS = [CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR]


def calibrate_task(run_id: str, runtime: float, num_chains: int, verbose: bool):
    build_input_database()

    with Timer(f"Creating calibration directories"):
        for dirpath in CALIBRATE_DIRS:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)

            os.makedirs(dirpath)

    with Timer(f"Running {num_chains} calibration chains"):
        args_list = [
            (run_id, runtime, chain_id, num_chains, verbose) for chain_id in range(num_chains)
        ]
        chain_ids = run_parallel_tasks(run_calibration_chain, args_list)

    with Timer(f"Uploading calibration data to AWS S3"):
        args_list = [(run_id, chain_id, verbose) for chain_id in chain_ids]
        run_parallel_tasks(upload_calibration_data, args_list)

    with Timer(f"Creating post-calibration plots"):
        app_region = utils.get_app_region(run_id)
        plots.calibration.plot_post_calibration(
            app_region.targets, CALIBRATE_DATA_DIR, CALIBRATE_PLOTS_DIR
        )

    with Timer(f"Uploading plots to AWS S3"):
        if not verbose:
            logging.disable(logging.INFO)

        src_path = CALIBRATE_PLOTS_DIR
        relpath = os.path.relpath(src_path, settings.BASE_DIR)
        dest_key = os.path.join(run_id, relpath)
        utils.upload_s3(src_path, dest_key)

        if not verbose:
            logging.disable(logging.NOTSET)


def upload_calibration_data(run_id: str, chain_id, verbose: bool):
    if not verbose:
        previous_level = logging.root.manager.disable
        logging.disable(logging.INFO)

    src_path = os.path.join(CALIBRATE_DATA_DIR, f"chain-{chain_id}")
    relpath = os.path.relpath(src_path, settings.BASE_DIR)
    dest_key = os.path.join(run_id, relpath)
    utils.upload_s3(src_path, dest_key)
    return chain_id


def run_calibration_chain(
    run_id: str, runtime: float, chain_id: int, num_chains: int, verbose: bool
):
    """
    Run a single calibration chain.
    """
    if not verbose:
        logging.disable(logging.INFO)

    os.environ["AUTUMN_CALIBRATE_DIR"] = CALIBRATE_DATA_DIR
    app_region = utils.get_app_region(run_id)
    app_region.calibrate_model(runtime, chain_id, num_chains)

    logging.info("Running calibration chain %s", chain_id)
    return chain_id


def run_parallel_tasks(func: Callable, arg_list: List[Any]):
    excecutor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    futures = [excecutor.submit(func, *args) for args in arg_list]
    success_results = []
    failure_exceptions = []
    for future in as_completed(futures):
        exception = future.exception()
        if exception:
            logger.info("Parallel task failed.")
            failure_exceptions.append(exception)
            continue

        result = future.result()
        logger.info("Parallel task completed: %s", result)
        success_results.append(result)

    logger.info("Successfully ran %s parallel tasks: %s", len(success_results), success_results)
    if failure_exceptions:
        logger.info("Failed to run ran %s parallel tasks", len(failure_exceptions))

    for e in failure_exceptions:
        start = "\n\n===== Exception when running a parallel task =====\n"
        end = "\n================ End of error message ================\n"
        error_message = "".join(traceback.format_exception(e.__class__, e, e.__traceback__))
        logger.error(start + error_message + end)

    if failure_exceptions:
        logger.error(
            "%s / %s parallel tasks failed - exiting.", len(failure_exceptions), len(arg_list)
        )
        sys.exit(-1)

    return success_results


if __name__ == "__main__":
    calibrate_task("covid_19/manila/111111111/bbbbbbb", 10, 2, False)
