import logging
import os
import sys
from pathlib import Path
import tempfile
import shutil

# from autumn.settings import REMOTE_BASE_DIR
from autumn.core.utils.fs import recreate_dir
from autumn.core.utils.s3 import get_s3_client
from autumn.core.utils.parallel import gather_exc_plus
from autumn.core.utils.timer import Timer
from .utils import get_project_from_run_id, set_logging_config
from .storage import StorageMode, MockStorage, S3Storage, LocalStorage

from autumn.calibration.optimisation.optimisation import pso_optimisation_task as _pso_optimisation_task

logger = logging.getLogger(__name__)

def _test_stub(run_id, out_path, **kwargs):
    logger.info(f"Running with run_id {run_id}, out path {out_path}, and kwargs {kwargs}")
    with open(out_path / "output.txt", 'w') as out_file:
        out_file.write(f"Running with run_id {run_id}, out path {out_path}, and kwargs {kwargs}")


TASK_CONFIG = {
    "test_stub": _test_stub,
    "pso_opti": _pso_optimisation_task,
}


def generic_task(run_id: str, task_key: str, task_kwargs: dict = None, verbose: bool = False, store="s3"):

    TEMP_BASE_DIR = Path(tempfile.mkdtemp())
    
    JOB_LOGDIR = TEMP_BASE_DIR / "logs"
    JOB_OUTDATA = TEMP_BASE_DIR / task_key

    JOB_DIRS = [JOB_LOGDIR, JOB_OUTDATA]

    task_kwargs = task_kwargs or {}
    
    if store == StorageMode.MOCK:
        storage = MockStorage()
    elif store == StorageMode.S3:
        s3_client = get_s3_client()
        storage = S3Storage(s3_client, run_id, TEMP_BASE_DIR, verbose)
    elif store == StorageMode.LOCAL:
        storage = LocalStorage(run_id, TEMP_BASE_DIR)

    # Set up directories for plots and output data.
    with Timer(f"Creating calibration directories"):
        for dirpath in JOB_DIRS:
            recreate_dir(dirpath)

    set_logging_config(False, 0, JOB_LOGDIR, task=task_key)

    try:
        # Run the actual calibrations
        with Timer(f"Running {task_key}"):
            f = TASK_CONFIG[task_key]
            f(run_id, JOB_OUTDATA, **task_kwargs)
    except Exception as e:
        logger.exception(f"Task {task_key} failed")
        gather_exc_plus(
            os.path.join(JOB_LOGDIR, f"crash-{task_key}.log")
        )
        raise e
    finally:
        storage.store(JOB_OUTDATA)
        storage.store(JOB_LOGDIR)
        logging.shutdown()
        shutil.rmtree(TEMP_BASE_DIR)
