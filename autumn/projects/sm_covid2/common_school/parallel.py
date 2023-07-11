from typing import Callable, Iterable

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_context

import cloudpickle


# initialize a worker in the process pool
def process_init_cloudpickle(custom_cpkl):
    """Initialize a mp.Process with custom cloudpickled data

    Args:
        custom_cpkl: The cloudpickle.dumps output of the custom data
    """
    # declare global variable
    global custom_data
    # assign the global variable
    custom_data = cloudpickle.loads(custom_cpkl)


def generic_cpkl_worker(*args):
    """Worker function to be used with multiprocessing pools, where
    custom_data is a global initialized to a function, usually via
    process_init_cloudpickle

    Returns:
        Return type of the custom global function
    """
    global custom_data
    run_func = custom_data

    return run_func(*args)


def map_parallel(run_func: Callable, input_iterator: Iterable, n_workers: int = None):
    """Map the values of input_iterator over a function run_func, using n_workers parallel workers

    Args:
        run_func: The function to call over the mapped inputs
        input_iterator: An iterable containing the values to map
        n_workers: Number of processes used by Pool

    Returns:
        A list of values return by run_func
    """

    if n_workers is None:
        n_workers = int(cpu_count() / 2)

    with ProcessPoolExecutor(
        n_workers,
        get_context("spawn"),
        initializer=process_init_cloudpickle,
        initargs=(cloudpickle.dumps(run_func),),
    ) as pool:
        pres = pool.map(generic_cpkl_worker, input_iterator)
        pres = list(pres)
    return pres
