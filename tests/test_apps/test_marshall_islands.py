import pytest
from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.marshall_islands.marshall_islands import build_rmi_model


@pytest.mark.skip(reason="Need a consistent input db in source control.")
@mock.patch('applications.marshall_islands.marshall_islands.write_model_data')
def test_build_marshall_model(mock_write_model_data):
    """
    Ensure we can build the Marshall Islands model with nothing crashing
    """
    model = build_rmi_model({})
    assert type(model) is StratifiedModel
    mock_write_model_data.assert_called_once()
