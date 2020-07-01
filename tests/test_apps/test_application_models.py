from copy import deepcopy

import pytest
from summer.model import StratifiedModel

from apps import mongolia, covid_19, marshall_islands
from autumn.tool_kit.utils import merge_dicts


@pytest.mark.local_only
@pytest.mark.parametrize("region", covid_19.REGION_APPS)
def test_run_models_partial(region):
    """
    Smoke test: ensure we can build and run each default model with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    region_app = covid_19.get_region_app(region)
    ps = deepcopy(region_app.params["default"])
    # Only run model for ~2 epochs.
    ps["end_time"] = ps["start_time"] + 2
    model = region_app.build_model(ps)
    model.run_model()


@pytest.mark.local_only
@pytest.mark.parametrize("region", covid_19.REGION_APPS)
def test_build_scenario_models(region):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    region_app = covid_19.get_region_app(region)
    for scenario_params in region_app.params["scenarios"].values():
        default_params = deepcopy(region_app.params["default"])
        params = merge_dicts(scenario_params, default_params)
        params = {**params, "start_time": region_app.params["scenario_start_time"]}
        model = region_app.build_model(params)
        assert type(model) is StratifiedModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", covid_19.REGION_APPS)
def test_run_models_full(region):
    """
    Smoke test: ensure our models run to completion without crashing.
    This takes ~30s per model.
    """
    region_app = covid_19.get_region_app(region)
    region_app.run_model()
