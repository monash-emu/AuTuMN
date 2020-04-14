import os
from unittest import mock

from summer_py.summer_model import StratifiedModel

from applications.covid_19.covid_model import AUSTRALIA, PHILLIPINES, build_covid_model

from applications.covid_19.runners import run_covid_aus_model, run_covid_phl_model


@mock.patch("autumn.model_runner.Outputs")
@mock.patch("autumn.model_runner.OutputPlotter")
def test_run_covid_aus_model(mock_1, mock_2):
    """
    Ensure COVID Australian model runs.
    """
    run_covid_aus_model()


@mock.patch("autumn.model_runner.Outputs")
@mock.patch("autumn.model_runner.OutputPlotter")
def test_run_covid_phl_model(mock_1, mock_2):
    """
    Ensure COVID Phillipines model runs.
    """
    run_covid_aus_model()


def test_build_aus_covid_model():
    """
    Ensure we can build the Australian COVID model with nothing crashing
    """
    model = build_covid_model(AUSTRALIA, update_params={})
    assert type(model) is StratifiedModel


def test_build_phl_covid_model():
    """
    Ensure we can build the Phillipines COVID model with nothing crashing
    """
    model = build_covid_model(PHILLIPINES, update_params={})
    assert type(model) is StratifiedModel
