import os
from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.marshall_islands.marshall_islands import build_rmi_model, run_model

IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


# FIXME: the order of these tests matter, STRATIFY_BY patch isn't working as expected.
@mock.patch("applications.marshall_islands.marshall_islands.STRATIFY_BY", ["age"])
@mock.patch("applications.marshall_islands.marshall_islands.write_model_data")
@mock.patch("autumn.tb_model.outputs.Outputs")
def test_run_marshall_model(mock_output_cls, mock_write_model_data):
    """
    Ensure Marshall Islands model runs.
    """
    run_model()


@mock.patch("applications.marshall_islands.marshall_islands.write_model_data")
def test_build_marshall_model(mock_write_model_data):
    """
    Ensure we can build the Marshall Islands model with nothing crashing
    """
    model = build_rmi_model({})
    assert type(model) is StratifiedModel
