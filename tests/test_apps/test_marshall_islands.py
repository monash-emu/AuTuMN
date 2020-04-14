import os
from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.marshall_islands.rmi_model import build_rmi_model
from applications.marshall_islands.runners import run_rmi_model


@mock.patch("autumn.model_runner.Outputs")
@mock.patch("autumn.model_runner.OutputPlotter")
def test_run_marshall_model(mock_1, mock_2):
    """
    Ensure Marshall Islands model runs.
    """
    run_rmi_model()


def test_build_marshall_model():
    """
    Ensure we can build the Marshall Islands model with nothing crashing
    """
    model = build_rmi_model()
    assert type(model) is StratifiedModel
