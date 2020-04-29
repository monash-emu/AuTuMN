import pytest

from autumn.db.input_data import build_input_database


@pytest.mark.github_only
def test_build_input_database():
    """
    Ensure we can build the input database with nothing crashing
    """
    # We use this to force the SQLite connection string to be "sqlite:///",
    # which means we use an in-memory database rather than writing to a file.
    db = build_input_database()
