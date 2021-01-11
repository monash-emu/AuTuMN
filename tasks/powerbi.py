import logging
import os
import shutil

from autumn import db, plots
from autumn.inputs import build_input_database
from autumn.tool_kit import Timer

from tasks import utils, settings
from tasks.full import FULL_RUN_DATA_DIR

logger = logging.getLogger(__name__)

POWERBI_PLOT_DIR = os.path.join(settings.BASE_DIR, "plots", "uncertainty")
POWERBI_DATA_DIR = os.path.join(settings.BASE_DIR, "data", "powerbi")
POWERBI_DIRS = [POWERBI_DATA_DIR, POWERBI_PLOT_DIR]
POWERBI_PRUNED_DIR = os.path.join(POWERBI_DATA_DIR, "pruned")
POWERBI_COLLATED_PATH = os.path.join(POWERBI_DATA_DIR, "collated")
POWERBI_COLLATED_PRUNED_PATH = os.path.join(POWERBI_DATA_DIR, "collated-pruned")


def powerbi_task(run_id: str, quiet: bool):
    # Prepare inputs for running the model.
    build_input_database()

    # Set up directories for plots and output data.
    with Timer(f"Creating PowerBI directories"):
        for dirpath in POWERBI_DIRS:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)

            os.makedirs(dirpath)

    # Find the full model run databases in AWS S3.
    key_prefix = os.path.join(run_id, os.path.relpath(FULL_RUN_DATA_DIR, settings.BASE_DIR))
    chain_db_keys = utils.list_s3(key_prefix, key_suffix=".feather")

    # Download the full model run databases.
    with Timer(f"Downloading full model run data"):
        args_list = [(run_id, src_key, quiet) for src_key in chain_db_keys]
        utils.run_parallel_tasks(utils.download_from_run_s3, args_list)

    # Remove unnecessary data from each full model run database.
    full_db_paths = db.load.find_db_paths(FULL_RUN_DATA_DIR)
    with Timer(f"Pruning chain databases"):
        get_dest_path = lambda p: os.path.join(POWERBI_PRUNED_DIR, os.path.basename(p))
        args_list = [(full_db_path, get_dest_path(full_db_path)) for full_db_path in full_db_paths]
        utils.run_parallel_tasks(db.process.prune_chain, args_list)

    # Collate data from each pruned full model run database into a single database.
    pruned_db_paths = db.load.find_db_paths(POWERBI_PRUNED_DIR)
    with Timer(f"Collating pruned databases"):
        db.process.collate_databases(pruned_db_paths, POWERBI_COLLATED_PATH)

    # Calculate uncertainty for model outputs.
    app_region = utils.get_app_region(run_id)
    with Timer(f"Calculating uncertainty quartiles"):
        db.uncertainty.add_uncertainty_quantiles(POWERBI_COLLATED_PATH, app_region.targets)

    # Remove unnecessary data from the database.
    with Timer(f"Pruning final database"):
        db.process.prune_final(POWERBI_COLLATED_PATH, POWERBI_COLLATED_PRUNED_PATH)

    # Unpivot database tables so that they're easier to process in PowerBI.
    run_slug = run_id.replace("/", "-")
    dest_db_path = os.path.join(POWERBI_DATA_DIR, f"powerbi-{run_slug}.db")
    with Timer(f"Applying PowerBI specific post-processing final database"):
        db.process.powerbi_postprocess(POWERBI_COLLATED_PRUNED_PATH, dest_db_path, run_id)

    # Upload final database to AWS S3
    with Timer(f"Uploading PowerBI data to AWS S3"):
        utils.upload_to_run_s3(run_id, dest_db_path, quiet)

    # Create uncertainty plots
    with Timer(f"Creating uncertainty plots"):
        plots.uncertainty.plot_uncertainty(app_region.targets, dest_db_path, POWERBI_PLOT_DIR)

    # Upload the plots to AWS S3.
    with Timer(f"Uploading plots to AWS S3"):
        utils.upload_to_run_s3(run_id, POWERBI_PLOT_DIR, quiet)
