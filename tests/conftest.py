# PyTest configuration file.
# See pytest fixtue docs: https://docs.pytest.org/en/latest/fixture.html
import pytest

from autumn.db import database
from autumn.tb_model import outputs
from autumn import constants
from autumn.calibration import calibration

from .utils import in_memory_db_factory, get_deterministic_random_seed

get_in_memory_db = in_memory_db_factory()


@pytest.fixture(autouse=True)
def memory_db(monkeypatch):
    """
    Replaces all SQLite on-disk databases with in-memory databases.
    Automatically run at the start of the test run.
    """
    monkeypatch.setattr(database, "get_sql_engine", get_in_memory_db)
    monkeypatch.setattr(outputs, "get_sql_engine", get_in_memory_db)


@pytest.fixture(autouse=True)
def deterministic_seed(monkeypatch):
    """
    Replaces all random seed non-deterministic seed.
    Automatically run at the start of the test run.
    """
    monkeypatch.setattr(calibration, "get_random_seed", get_deterministic_random_seed)


@pytest.fixture
def temp_data_dir(monkeypatch, tmp_path):
    """
    Replaces DATA_PATH with a tempoary directory.
    Runs every time it is invoked by a test.
    """
    path_str = tmp_path.as_posix()
    monkeypatch.setattr(constants, "DATA_PATH", path_str)
    return path_str
