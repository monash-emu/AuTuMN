# PyTest configuration file.
# See pytest fixtue docs: https://docs.pytest.org/en/latest/fixture.html
import os
import pickle

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose


from autumn.db import database
from autumn import constants
from autumn.calibration import calibration

from .utils import in_memory_db_factory, get_deterministic_random_seed

get_in_memory_db = in_memory_db_factory()

APPROVAL_DIR = os.path.join(constants.OUTPUT_DATA_PATH, "approvals")
IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


def pytest_configure(config):
    config.addinivalue_line("markers", "github_only: Mark test to run only in GitHub Actions")
    config.addinivalue_line("markers", "local_only: Mark test to never run in GitHub Actions")
    config.addinivalue_line("markers", "run_models: A test which runs the full models")
    config.addinivalue_line("markers", "calibrate_models: A test which runs full calibrations")
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


@pytest.fixture(autouse=True)
def memory_db(monkeypatch):
    """
    Replaces all SQLite on-disk databases with in-memory databases.
    Automatically run at the start of every test run.
    """
    monkeypatch.setattr(database, "get_sql_engine", get_in_memory_db)


@pytest.fixture(autouse=True)
def deterministic_seed(monkeypatch):
    """
    Replaces all random seed non-deterministic seed.
    Automatically run at the start of every test run.
    """
    monkeypatch.setattr(calibration, "get_random_seed", get_deterministic_random_seed)


@pytest.fixture(autouse=True)
def temp_data_dir(monkeypatch, tmp_path):
    """
    Replaces DATA_PATH with a tempoary directory.
    Automatically run at the start of every test run.
    """
    path_str = tmp_path.as_posix()
    monkeypatch.setattr(constants, "DATA_PATH", path_str)
    monkeypatch.setattr(constants, "OUTPUT_DATA_PATH", os.path.join(path_str, "outputs"))
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
