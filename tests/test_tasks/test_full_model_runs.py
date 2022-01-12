import os

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from moto import mock_s3

from autumn.tools.db import ParquetDatabase, FeatherDatabase
from autumn.tools.db.store import Table
from autumn.tasks import full
from autumn.tasks.full import full_model_run_task
from autumn import settings as s3_settings
from autumn.tools.utils.s3 import get_s3_client, upload_to_run_s3, list_s3, download_from_run_s3, sanitise_path
from autumn.tools.utils.fs import recreate_dir

from tests.test_tasks.project import get_test_project

BUCKET_NAME = "autumn-test-bucket"
TEST_RUN_ID = "test_app/test_region/111111111/zzzzzzz"
MCMC_RUN_PATH = "data/calibration_outputs/chain-0/mcmc_run.parquet"
MCMC_PARAMS_PATH = "data/calibration_outputs/chain-1/mcmc_params.parquet"

# Increasingly unfit for purpose; marking as skip but leaving here as a reminder
# to rewrite tasks in a way that doesn't require convoluted monkeypatching and mockery
# in order to test

@pytest.mark.skip
@mock_s3
def test_full_model_run_task(monkeypatch, tmpdir):
    """
    Test the full model run task.  Mostly a smoke test at the moment
    """
    # Ensure data is read/written to a transient test directory
    test_full_data_dir = os.path.join(tmpdir, "data", "full_model_runs")
    test_calibration_data_dir = os.path.join(tmpdir, "data", "calibration_outputs")
    monkeypatch.setattr(full, "REMOTE_BASE_DIR", tmpdir)
    monkeypatch.setattr(full, "FULL_RUN_DATA_DIR", test_full_data_dir)
    monkeypatch.setattr(full, "CALIBRATE_DATA_DIR", test_calibration_data_dir)
    monkeypatch.setattr(s3_settings, "REMOTE_BASE_DIR", tmpdir)
    monkeypatch.setattr(s3_settings, "S3_BUCKET", BUCKET_NAME)

    # Ignore logging config for now
    monkeypatch.setattr(full, "set_logging_config", lambda *args, **kwargs: None)

    # Create a calibration database as input to the full model run
    test_db_path = os.path.join(test_calibration_data_dir, "chain-0")
    calib_db = ParquetDatabase(test_db_path)
    mcmc_run_columns = ["accept", "ap_loglikelihood", "chain", "loglikelihood", "run", "weight"]
    mcmc_run_rows = [
        # NB: ap_loglikelihood not used so we can ignore.
        [1, 0.0, 0, -110.0, 0, 1],
        [1, 0.0, 0, -101.0, 1, 2],
        [0, 0.0, 0, -102.0, 2, 0],
        [1, 0.0, 0, -103.2, 3, 4],
        [0, 0.0, 0, -102.1, 4, 0],
        [0, 0.0, 0, -101.4, 5, 0],
        [0, 0.0, 0, -101.6, 6, 0],
        [1, 0.0, 0, -100.0, 7, 2],  # Maximum likelihood run (MLE)
        [0, 0.0, 0, -103.1, 8, 0],
        [1, 0.0, 0, -100.1, 9, 1],
        [1, 0.0, 0, -100.2, 10, 1],
    ]
    mcmc_run_df = pd.DataFrame(mcmc_run_rows, columns=mcmc_run_columns)
    calib_db.dump_df(Table.MCMC, mcmc_run_df)

    mcmc_param_columns = ["chain", "name", "run", "value"]
    mcmc_param_rows = [
        [0, "recovery_rate", 0, 0.0],
        [0, "recovery_rate", 1, 0.1],
        [0, "recovery_rate", 2, 0.2],
        [0, "recovery_rate", 3, 0.3],
        [0, "recovery_rate", 4, 0.4],
        [0, "recovery_rate", 5, 0.5],
        [0, "recovery_rate", 6, 0.6],
        [0, "recovery_rate", 7, 0.7],  # Maximum likelihood run (MLE)
        [0, "recovery_rate", 8, 0.8],
        [0, "recovery_rate", 9, 0.9],
        [0, "recovery_rate", 10, 1.0],
    ]
    mcmc_param_df = pd.DataFrame(mcmc_param_rows, columns=mcmc_param_columns)
    calib_db.dump_df(Table.PARAMS, mcmc_param_df)

    # Upload calibration database to mock AWS S3, then delete local copy
    s3 = get_s3_client()
    s3.create_bucket(
        Bucket=BUCKET_NAME, CreateBucketConfiguration={"LocationConstraint": s3_settings.AWS_REGION}
    )
    upload_to_run_s3(s3, TEST_RUN_ID, test_db_path, quiet=True)
    recreate_dir(test_calibration_data_dir)

    # Ensure our test model is being run.
    def get_project_from_run_id(run_id):
        assert run_id == TEST_RUN_ID
        return get_test_project()

    monkeypatch.setattr(full, "get_project_from_run_id", get_project_from_run_id)

    # Run the full model task
    full_model_run_task(run_id=TEST_RUN_ID, burn_in=2, sample_size=3, quiet=True)

    # Delete local data, download AWS S3 data and check the results
    recreate_dir(test_full_data_dir)
    key_prefix = os.path.join(TEST_RUN_ID, "data", "full_model_runs")
    chain_db_keys = list_s3(s3, key_prefix, key_suffix=".feather")
    for src_key in chain_db_keys:
        download_from_run_s3(s3, TEST_RUN_ID, src_key, quiet=True)

    full_db_path = os.path.join(test_full_data_dir, "chain-0")
    full_db = FeatherDatabase(full_db_path)
    assert set(full_db.table_names()) == {"outputs", "mcmc_run", "derived_outputs", "mcmc_params"}

