import logging
import os
import shutil

import pandas as pd

from autumn.core import db, plots
from autumn.core.db.load import load_mcmc_tables
from autumn.settings import REMOTE_BASE_DIR
from autumn.infrastructure.tasks.full import FULL_RUN_DATA_DIR
from autumn.infrastructure.tasks.utils import get_project_from_run_id
from autumn.core.utils.s3 import download_from_run_s3, list_s3, upload_to_run_s3, get_s3_client
from autumn.core.utils.timer import Timer

logger = logging.getLogger(__name__)

POWERBI_PLOT_DIR = os.path.join(REMOTE_BASE_DIR, "plots", "uncertainty")
POWERBI_DATA_DIR = os.path.join(REMOTE_BASE_DIR, "data", "powerbi")
POWERBI_DIRS = [POWERBI_DATA_DIR, POWERBI_PLOT_DIR]
POWERBI_PRUNED_DIR = os.path.join(POWERBI_DATA_DIR, "pruned")
POWERBI_COLLATED_PATH = os.path.join(POWERBI_DATA_DIR, "collated")
POWERBI_COLLATED_PRUNED_PATH = os.path.join(POWERBI_DATA_DIR, "collated-pruned")


def powerbi_task(run_id: str, urunid: str, quiet: bool):
    s3_client = get_s3_client()
    project = get_project_from_run_id(run_id)

    # Set up directories for plots and output data.
    with Timer(f"Creating PowerBI directories"):
        for dirpath in POWERBI_DIRS:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)

            os.makedirs(dirpath)

    # Find the full model run databases in AWS S3.
    key_prefix = os.path.join(run_id, os.path.relpath(FULL_RUN_DATA_DIR, REMOTE_BASE_DIR))
    chain_db_keys = []
    for filename_base in ["mcmc_run", "mcmc_params", "derived_outputs"]:
        chain_db_keys += list_s3(s3_client, key_prefix, key_suffix=f"{filename_base}.feather")

    # Download the full model run databases.
    with Timer(f"Downloading full model run data"):
        for src_key in chain_db_keys:
            download_from_run_s3(s3_client, run_id, src_key, quiet)

    # No urunid supplied; get a single candidate dataframe (ie the MLE run)
    if urunid == "mle":
        all_mcmc_df = pd.concat(load_mcmc_tables(FULL_RUN_DATA_DIR), ignore_index=True)
        candidates_df = db.process.select_pruning_candidates(all_mcmc_df, 1)
    else:
        c, r = (int(x) for x in urunid.split("_"))
        candidates_df = pd.DataFrame(columns=["chain", "run"])
        candidates_df.loc[0] = dict(chain=c, run=r)

    # Remove unnecessary data from each full model run database.
    full_db_paths = db.load.find_db_paths(FULL_RUN_DATA_DIR)
    with Timer(f"Pruning chain databases"):
        get_dest_path = lambda p: os.path.join(POWERBI_PRUNED_DIR, os.path.basename(p))
        for full_db_path in full_db_paths:
            chain_id = int(full_db_path.split("-")[-1])
            chain_candidates = candidates_df[candidates_df["chain"] == chain_id]
            db.process.prune_chain(full_db_path, get_dest_path(full_db_path), chain_candidates)

    # Collate data from each pruned full model run database into a single database.
    pruned_db_paths = db.load.find_db_paths(POWERBI_PRUNED_DIR)
    with Timer(f"Collating pruned databases"):
        db.process.collate_databases(pruned_db_paths, POWERBI_COLLATED_PATH)

    # Calculate uncertainty for model outputs.
    with Timer(f"Calculating uncertainty quartiles"):
        db.uncertainty.add_uncertainty_quantiles(POWERBI_COLLATED_PATH, project.plots)

    # Remove unnecessary data from the database.
    with Timer(f"Pruning final database"):
        db.process.prune_final(POWERBI_COLLATED_PATH, POWERBI_COLLATED_PRUNED_PATH, candidates_df)

    # Unpivot database tables so that they're easier to process in PowerBI.
    run_slug = run_id.replace("/", "-")
    dest_db_path = os.path.join(POWERBI_DATA_DIR, f"powerbi-{run_slug}.db")
    with Timer(f"Applying PowerBI specific post-processing final database"):
        db.process.powerbi_postprocess(POWERBI_COLLATED_PRUNED_PATH, dest_db_path, run_id)

    # Upload final database to AWS S3
    with Timer(f"Uploading PowerBI data to AWS S3"):
        upload_to_run_s3(s3_client, run_id, dest_db_path, quiet)

    # Create uncertainty plots
    with Timer(f"Creating uncertainty plots"):
        plots.uncertainty.plot_uncertainty(project.plots, dest_db_path, POWERBI_PLOT_DIR)

    # Upload the plots to AWS S3.
    with Timer(f"Uploading plots to AWS S3"):
        upload_to_run_s3(s3_client, run_id, POWERBI_PLOT_DIR, quiet)
