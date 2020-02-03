import os
from unittest import mock

import pytest

from autumn.db.input_data import build_input_database

IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


@pytest.mark.skipif(not IS_GITHUB_CI, reason="This takes way too long to run (~20s).")
@mock.patch("autumn.db.input_data.get_new_database_name")
def test_build_input_database(mock_get_db_name):
    """
    Ensure we can build the input database with nothing crashing
    """
    # We use this to force the SQLite connection string to be "sqlite:///",
    # which means we use an in-memory database rather than writing to a file.
    mock_get_db_name.return_value = ""
    db = build_input_database()
