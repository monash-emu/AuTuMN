import sys
import time
import os
import logging
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any

import sentry_sdk

logger = logging.getLogger(__name__)


MAX_WORKERS = mp.cpu_count() - 1
SENTRY_ERROR_DELAY = 10  # seconds
SENTRY_DSN = os.environ.get("SENTRY_DSN")


def report_errors(func):
    """
    Decorator that ensures that errors found inside parallel tasks
    are captured by Sentry, rather than being buried by the parent process.
    """

    @functools.wraps(func)
    def error_handler_wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            if SENTRY_DSN:
                # Sentry uses an async model to send exceptions to their server,
                # so we need to wait for Sentry to send the message or else the parent process
                # will kill the child before it can log the error
                logger.info("Waiting %s seconds for Sentry to upload an error.", SENTRY_ERROR_DELAY)
                sentry_sdk.capture_exception(e)
                time.sleep(SENTRY_ERROR_DELAY)

            raise e

    return error_handler_wrapper


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

    if failure_exceptions:
        logger.error(
            "%s / %s parallel tasks failed - exiting.", len(failure_exceptions), len(arg_list)
        )
        sys.exit(-1)

    return success_results
