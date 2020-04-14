import os
from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.covid_19.covid_model import build_covid_model
from applications.run_single_application import run_model

IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


# FIXME: the order of these tests matter, STRATIFY_BY patch isn't working as expected.
@mock.patch("builtins.input", return_value="")
@mock.patch("autumn.tb_model.outputs.Outputs")
def test_run_covid_model(mock_output_cls, mock_input):
    """
    Ensure COVID model runs.
    """
    run_model("covid_19")


def test_build_covid_model():
    """
    Ensure we can build the COVID model with nothing crashing
    """
    model = build_covid_model({})
    assert type(model) is StratifiedModel
