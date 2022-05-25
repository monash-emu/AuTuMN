# PyTest configuration file.
# See pytest fixtue docs: https://docs.pytest.org/en/latest/fixture.html
import os
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from autumn import settings
from autumn.runners.calibration import calibration
from autumn.tools.db import database

from .utils import in_memory_db_factory

APPROVAL_DIR = os.path.join(settings.DATA_PATH, "approvals")
IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)
IS_NIGHTLY = os.environ.get("NIGHTLY_TESTING", False)

get_in_memory_db_engine = in_memory_db_factory()


def pytest_configure(config):
    config.addinivalue_line("markers", "github_only: Mark test to run only in GitHub Actions")
    config.addinivalue_line("markers", "local_only: Mark test to never run in GitHub Actions")
    config.addinivalue_line("markers", "nightly_only: Mark test to run only in nightly testing")
    config.addinivalue_line("markers", "calibrate_models: A test which runs full calibrations")
    config.addinivalue_line("markers", "run_models: A test which runs the full models")
    config.addinivalue_line(
        "markers", "benchmark: A test which benchmarks the performance of some code"
    )
    config.addinivalue_line(
        "markers", "mixing_optimisation: A test which runs mixing optimisation checks"
    )


def pytest_runtest_setup(item):
    for marker in item.iter_markers(name="github_only"):
        if not IS_GITHUB_CI:
            pytest.skip("Long running test: run on GitHub only.")

    for marker in item.iter_markers(name="local_only"):
        if IS_GITHUB_CI:
            pytest.skip("Local test: never run on GitHub.")

    for marker in item.iter_markers(name="nightly_only"):
        if not IS_NIGHTLY:
            pytest.skip("Long running non-essential test: run in nightly testing only.")


@pytest.fixture(autouse=True)
def memory_db(monkeypatch):
    """
    Replaces all SQLite on-disk databases with in-memory databases.
    Automatically run at the start of every test run.
    """
    monkeypatch.setattr(database, "get_sql_engine", get_in_memory_db_engine)


@pytest.fixture(autouse=True)
def temp_data_dir(monkeypatch, tmp_path):
    """
    Replaces DATA_PATH with a tempoary directory.
    Automatically run at the start of every test run.
    """
    path_str = tmp_path.as_posix()
    monkeypatch.setattr(settings, "DATA_PATH", path_str)
    monkeypatch.setattr(settings, "OUTPUT_DATA_PATH", os.path.join(path_str, "outputs"))
    return path_str


@pytest.fixture
def verify(*args, **kwargs):
    """
    Provides a verify function for approval tests.
    https://understandlegacycode.com/approval-tests/
    """
    os.makedirs(APPROVAL_DIR, exist_ok=True)

    def _verify(obj, key: str):
        fpath = os.path.join(APPROVAL_DIR, f"{key}.pickle")
        if os.path.exists(fpath):
            # Check against existing fixture
            with open(fpath, "rb") as f:
                target = pickle.load(f)

            if type(obj) is np.ndarray:

                err_msg = f"Approval fixture array mismatch for {key}"
                assert_allclose(obj, target, atol=1, rtol=0.02, err_msg=err_msg, verbose=True)
            else:
                assert obj == target, f"Approval fixture mismatch for {key}"

        else:
            # Silently write new fixture
            with open(fpath, "wb") as f:
                pickle.dump(obj, f)

    return _verify


@pytest.fixture
def verify_model(verify):
    """
    Approval test for a model that has already been run.
    """

    def _verify(model, region: str):
        assert model.outputs is not None, f"Model for {region} has not been run yet."
        verify(model.times, f"times-{region}")
        verify(model.outputs, f"outputs-{region}")
        for output, arr in model.derived_outputs.items():
            verify(arr, f"do-{output}-{region}")

    return _verify
