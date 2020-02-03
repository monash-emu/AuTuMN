import pytest

from summer_py.summer_model import StratifiedModel

from applications.mongolia.mongolia_tb_model import build_mongolia_model


@pytest.mark.skip(reason="Need a consistent input db in source control.")
def test_build_mongolia_model():
    """
    Ensure we can build the Mongolia model with nothing crashing
    """
    model = build_mongolia_model({})
    assert type(model) is StratifiedModel
