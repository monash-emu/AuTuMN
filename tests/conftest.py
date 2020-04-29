# PyTest configuration file.
# See pytest fixtue docs: https://docs.pytest.org/en/latest/fixture.html
import os
import pytest

from autumn.db import database, models
from autumn import constants
from autumn.calibration import calibration

from .utils import in_memory_db_factory, get_deterministic_random_seed

get_in_memory_db = in_memory_db_factory()

IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


def pytest_runtest_setup(item):
    for marker in item.iter_markers(name="github_only"):
        if not IS_GITHUB_CI:
            pytest.skip("Long running test: run on GitHub only.")


@pytest.fixture(autouse=True)
def memory_db(monkeypatch):
    """
    Replaces all SQLite on-disk databases with in-memory databases.
    Automatically run at the start of every test run.
    """
    monkeypatch.setattr(database, "get_sql_engine", get_in_memory_db)
    monkeypatch.setattr(models, "get_sql_engine", get_in_memory_db)


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
    return path_str
