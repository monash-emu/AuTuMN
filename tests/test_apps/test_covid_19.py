import os
from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.covid_19.covid_model import build_covid_model
from applications.run_single_application import run_model

IS_GITHUB_CI = os.environ.get("GITHUB_ACTION", False)


# FIXME: the order of these tests matter, STRATIFY_BY patch isn't working as expected.
@mock.patch("builtins.input", return_value="")
@mock.patch("autumn.tb_model.outputs.Outputs")
def test_run_marshall_model(mock_output_cls, mock_input):
    """
    Ensure Marshall Islands model runs.
    """
    run_model('covid_19')


def test_build_marshall_model():
    """
    Ensure we can build the Marshall Islands model with nothing crashing
    """
    model = build_covid_model({})
    assert type(model) is StratifiedModel
