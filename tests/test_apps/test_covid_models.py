from copy import deepcopy
import numpy as np

import pytest
from summer.model import StratifiedModel
from summer.constants import Flow, IntegrationType

from apps import covid_19
from autumn.tool_kit.utils import merge_dicts
from autumn.environment.seasonality import get_seasonal_forcing


def test_cdr_intercept():
    """
    Test that there is zero case detection when zero tests are performed
    """

    for cdr_at_1000_tests in np.linspace(0.05, 0.5, 10):
        cdr_function = covid_19.model.preprocess.testing.create_cdr_function(
            1000.0, cdr_at_1000_tests
        )
        assert cdr_function(0.0) == 0.0


def test_cdr_values():
    """
    Test that CDR is always a proportion, bounded by zero and one
    """

    for cdr_at_1000_tests in np.linspace(0.05, 0.5, 10):
        cdr_function = covid_19.model.preprocess.testing.create_cdr_function(
            1000.0, cdr_at_1000_tests
        )
        for i_tests in list(np.linspace(0.0, 1e3, 11)) + list(np.linspace(0.0, 1e5, 11)):
            assert cdr_function(i_tests) >= 0.0
            assert cdr_function(i_tests) <= 1.0


def test_no_seasonal_forcing():
    """
    Test seasonal forcing function returns the average value when the magnitude is zero
    """

    seasonal_forcing_function = get_seasonal_forcing(365.0, 0.0, 0.0, 1.0)
    for i_time in np.linspace(-100.0, 100.0, 50):
        assert seasonal_forcing_function(i_time) == 1.0


def test_peak_trough_seasonal_forcing():
    """
    Test seasonal forcing returns the peak and trough values appropriately
    """

    seasonal_forcing_function = get_seasonal_forcing(365.0, 0.0, 2.0, 1.0)
    assert seasonal_forcing_function(0.0) == 2.0
    assert seasonal_forcing_function(365.0) == 2.0
    assert seasonal_forcing_function(365.0 / 2.0) == 0.0


@pytest.mark.local_only
@pytest.mark.parametrize("region", covid_19.app.region_names)
def test_run_models_partial(region):
    """
    Smoke test: ensure we can build and run each default model with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    region_app = covid_19.app.get_region(region)
    ps = deepcopy(region_app.params["default"])
    # Only run model for ~10 epochs.
    ps["time"]["end"] = ps["time"]["start"] + 10
    model = region_app.build_model(ps)
    model.run_model()


@pytest.mark.local_only
@pytest.mark.parametrize("region", covid_19.app.region_names)
def test_build_scenario_models(region):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    region_app = covid_19.app.get_region(region)
    for idx, scenario_params in enumerate(region_app.params["scenarios"].values()):
        default_params = deepcopy(region_app.params["default"])
        params = merge_dicts(scenario_params, default_params)
        model = region_app.build_model(params)
        assert type(model) is StratifiedModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", covid_19.app.region_names)
def test_run_models_full(region, verify):
    """
    Smoke test: ensure our models run to completion without crashing.
    This takes ~30s per model.
    """
    region_app = covid_19.app.get_region(region)
    model = region_app.build_model(region_app.params["default"])
    model.run_model()
