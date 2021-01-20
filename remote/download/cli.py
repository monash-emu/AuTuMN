"""
Utilities for downloading data from autumn-data.com
"""
import os
from datetime import datetime
from shutil import rmtree

import click

from tasks.utils import download_from_s3, list_s3
from autumn import constants
from autumn.tool_kit import Timer

from autumn.db.process import read_run_id


@click.group()
def download_cli():
    """Download run data from AWS S3"""


@download_cli.command("calibration")
@click.argument("run_id", type=str)
def download_calibration(run_id: str):
    """
    Downloads data for a calibration. Requires a run ID in format "covid_19/gippsland/1607552563/3cbe11d"
    """
    _download_run(run_id, "data/calibration_outputs", "calibrate")


@download_cli.command("full")
@click.argument("run_id", type=str)
def download_full_model(run_id: str):
    """
    Downloads data for a full model run. Requires a run ID in format "covid_19/gippsland/1607552563/3cbe11d"
    """
    _download_run(run_id, "data/full_model_runs", "full")


def _download_run(run_id: str, src_dir_key: str, dest_dir_key: str):
    msg = f'Could not read run ID {run_id}, use format "{{app}}/{{region}}/{{timestamp}}/{{commit}}" - exiting'
    assert len(run_id.split("/")) == 4, msg
    key_prefix = os.path.join(run_id, src_dir_key)
    with Timer(f"Finding data for run {run_id}"):
        chain_db_keys = list_s3(key_prefix, key_suffix=".feather")

    app_name, region_name, timestamp, _ = read_run_id(run_id)
    datestamp = datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d")
    base_dest_path = os.path.join(
        constants.OUTPUT_DATA_PATH, dest_dir_key, app_name, region_name, datestamp
    )

    if os.path.exists(base_dest_path):
        rmtree(base_dest_path)

    num_files = len(chain_db_keys)
    with Timer(f"Downloading {num_files} files to {base_dest_path}"):
        for key in chain_db_keys:
            chain_name, file_name = key.split("/")[-2:]
            dest_dir = os.path.join(base_dest_path, chain_name)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, file_name)
            with Timer(f"Downloading {chain_name} table {file_name}"):
                download_from_s3(key, dest_path, quiet=True)
