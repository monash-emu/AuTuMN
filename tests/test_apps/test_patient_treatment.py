import pytest

from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.tb_model_for_pt.tb_model_for_pt import build_model


@pytest.mark.skip(reason="Need a consistent input db in source control.")
@mock.patch('applications.tb_model_for_pt.tb_model_for_pt.write_model_data')
def test_build_pt_model(mock_write_model_data):
    """
    Ensure we can build the patient treatment model with nothing crashing
    """
    model = build_model({})
    assert type(model) is StratifiedModel
