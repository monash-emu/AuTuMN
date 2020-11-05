import logging
import os

from autumn.inputs import build_input_database
from autumn.tool_kit import Timer

from tasks import utils, settings
from tasks.full import FULL_RUN_DATA_DIR

logger = logging.getLogger(__name__)


POWERBI_DATA_DIR = os.path.join(settings.BASE_DIR, "data", "powerbi")


def powerbi_task(run_id: str, quiet: bool):
    build_input_database()
    key_prefix = os.path.join(run_id, os.path.relpath(FULL_RUN_DATA_DIR, settings.BASE_DIR))
    chain_db_keys = utils.list_s3(key_prefix, key_suffix=".feather")
    with Timer(f"Downloading full model run data"):
        args_list = [(run_id, src_key, quiet) for src_key in chain_db_keys]
        utils.run_parallel_tasks(utils.download_from_run_s3, args_list)

    with Timer(f"Pruning chain databases"):
        src_filename = utils.get_full_model_run_db_filename(self.chain_id)
        src_db_path = os.path.join(settings.BASE_DIR, "data", "full_model_runs", src_filename)
        dest_db_path = self.get_dest_path()
        db.process.prune_chain(src_db_path, dest_db_path)
