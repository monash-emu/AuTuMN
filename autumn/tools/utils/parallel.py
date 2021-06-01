import logging
import multiprocessing as mp
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


MAX_WORKERS = mp.cpu_count() - 1


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
