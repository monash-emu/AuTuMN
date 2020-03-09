import os
from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.marshall_islands.rmi_model import build_rmi_model
from applications.marshall_islands.rmi_single_run import run_model

IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


# FIXME: the order of these tests matter, STRATIFY_BY patch isn't working as expected.
@mock.patch("applications.marshall_islands.rmi_model.STRATIFY_BY", ["age"])
@mock.patch("builtins.input", return_value="")
@mock.patch("autumn.tb_model.outputs.Outputs")
def test_run_marshall_model(mock_output_cls, mock_input):
    """
    Ensure Marshall Islands model runs.
    """
    run_model()


def test_build_marshall_model():
    """
    Ensure we can build the Marshall Islands model with nothing crashing
    """
    model = build_rmi_model({})
    assert type(model) is StratifiedModel
