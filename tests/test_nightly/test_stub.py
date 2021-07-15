import pytest

@pytest.mark.nightly
@pytest.mark.github_only
def test_run_stub_test():
    """
    Empty test just to make sure our nightly job has something to do
    """
    pass
