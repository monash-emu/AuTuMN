import os
import logging
import sys
import traceback
from typing import Callable, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from apps import covid_19, tuberculosis, tuberculosis_strains
from autumn.remote import read_run_id
from .s3 import (
    download_from_run_s3,
    download_from_s3,
    upload_to_run_s3,
    list_s3,
    download_s3,
    upload_s3,
    upload_folder_s3,
    upload_file_s3,
)

logger = logging.getLogger(__name__)


SENTRY_DSN = os.environ.get("SENTRY_DSN")
MAX_WORKERS = mp.cpu_count() - 1


def get_app_region(run_id: str):
    app_map = {
        "covid_19": covid_19,
        "tuberculosis": tuberculosis,
        "tuberculosis_strains": tuberculosis_strains,
    }
    app_name, region_name, _, _ = read_run_id(run_id)
    app_module = app_map[app_name]
    return app_module.app.get_region(region_name)


def run_parallel_tasks(func: Callable, arg_list: List[Any]):
    if len(arg_list) == 1:
        return [func(*arg_list[0])]

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
        logger.info("Failed to run %s parallel tasks", len(failure_exceptions))

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
