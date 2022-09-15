import sys
import traceback
import time
import os
import logging
import functools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any

logger = logging.getLogger(__name__)


MAX_WORKERS = mp.cpu_count()


def gather_exc_plus(filename="crash.log"):
    """
    Dump tracebacks and locals to a file
    Borrowed from: https://www.oreilly.com/library/view/python-cookbook/0596001673/ch14s05.html
    """

    out_f = open(filename, "w")

    tb = sys.exc_info()[2]
    while 1:
        if not tb.tb_next:
            break
        tb = tb.tb_next
    stack = []
    f = tb.tb_frame
    while f:
        stack.append(f)
        f = f.f_back
    stack.reverse()
    traceback.print_exc(file=out_f)
    out_f.write("\n")
    out_f.write("Showing stack for all frames:\n\n")
    for frame in stack:
        out_f.write(
            f"Frame {frame.f_code.co_name} in {frame.f_code.co_filename} at line {frame.f_lineno}\n"
        )
        for key, value in frame.f_locals.items():
            out_f.write(f"\t{key} = \n"),
            try:
                out_f.write(f"{value}\n")
            except:
                out_f.write("Could not represent as string\n")


def run_parallel_tasks(func: Callable, arg_list: List[Any], auto_exit=True):
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
        for f in failure_exceptions:
            logger.error(f)
        if auto_exit:
            logger.error(
                "%s / %s parallel tasks failed - exiting.",
                len(failure_exceptions),
                len(arg_list),
            )
            sys.exit(-1)
        else:
            raise Exception("Parallel tasks failed")

    return success_results
