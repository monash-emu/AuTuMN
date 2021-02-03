from copy import deepcopy

import pytest
from summer.model import StratifiedModel
from summer2 import CompartmentalModel

from apps import covid_19
from autumn.tool_kit.utils import merge_dicts


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

    if type(model) is StratifiedModel:
        model.run_model()
    else:
        model.run()


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
        assert (type(model) is StratifiedModel) or (type(model) is CompartmentalModel)


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
    if type(model) is StratifiedModel:
        model.run_model()
    else:
        model.run()
