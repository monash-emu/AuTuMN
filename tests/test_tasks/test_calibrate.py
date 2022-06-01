import os

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from moto import mock_s3

from autumn.core.db import ParquetDatabase, ParquetDatabase
from autumn.core.db.store import Table
from autumn.infrastructure.tasks import calibrate
from autumn.infrastructure.tasks.calibrate import calibrate_task
from autumn import settings as s3_settings
from autumn.core.utils.s3 import get_s3_client, upload_to_run_s3, list_s3, download_from_run_s3
from autumn.core.utils.fs import recreate_dir

from tests.test_tasks.project import get_test_project

BUCKET_NAME = "autumn-test-bucket"
TEST_RUN_ID = "test_app/test_region/111111111/zzzzzzz"
MCMC_RUN_PATH = "data/calibration_outputs/chain-0/mcmc_run.parquet"
MCMC_PARAMS_PATH = "data/calibration_outputs/chain-1/mcmc_params.parquet"


# Fails randomly in Actions only
@pytest.mark.skip
@pytest.mark.local_only
@mock_s3
def test_calibration_task(monkeypatch, tmpdir):
    """
    Test the calibration task.
    """
    # Create S3 bucket to upload data to
    s3 = get_s3_client()
    s3.create_bucket(
        Bucket=BUCKET_NAME, CreateBucketConfiguration={"LocationConstraint": s3_settings.AWS_REGION}
    )

    # Ensure data is read/written to a transient test directory
    test_calibration_data_dir = os.path.join(tmpdir, "data", "calibration_outputs")
    monkeypatch.setattr(calibrate, "REMOTE_BASE_DIR", tmpdir)
    monkeypatch.setattr(calibrate, "CALIBRATE_DATA_DIR", test_calibration_data_dir)
    monkeypatch.setattr(s3_settings, "REMOTE_BASE_DIR", tmpdir)
    monkeypatch.setattr(s3_settings, "S3_BUCKET", BUCKET_NAME)

    # Ignore logging config for now
    monkeypatch.setattr(calibrate, "set_logging_config", lambda *args, **kwargs: None)

    # Ensure our test model is being run.
    def get_project_from_run_id(run_id):
        assert run_id == TEST_RUN_ID
        return get_test_project()

    monkeypatch.setattr(calibrate, "get_project_from_run_id", get_project_from_run_id)

    # Run the calibration
    calibrate_task(run_id=TEST_RUN_ID, runtime=1, num_chains=1, verbose=False)

    # Delete local data, download AWS S3 data and check the results
    recreate_dir(test_calibration_data_dir)
    key_prefix = os.path.join(TEST_RUN_ID, "data", "calibration_outputs")
    chain_db_keys = list_s3(s3, key_prefix, key_suffix=".parquet")
    for src_key in chain_db_keys:
        download_from_run_s3(s3, TEST_RUN_ID, src_key, quiet=True)

    # Do some very basic, cursory checks of the outputs.
    calib_db_path = os.path.join(test_calibration_data_dir, "chain-0")
    calib_db = ParquetDatabase(calib_db_path)
    assert set(calib_db.table_names()) == {"outputs", "mcmc_run", "derived_outputs", "mcmc_params"}
    assert (calib_db.query("mcmc_params").name == "recovery_rate").all()
    assert (calib_db.query("mcmc_params").chain == 0).all()
    assert calib_db.query("mcmc_params").run.max() > 10
