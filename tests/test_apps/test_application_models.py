from copy import deepcopy

import pytest
from summer.model import StratifiedModel

from autumn.tool_kit.serializer import serialize_model
from apps import mongolia, covid_19, marshall_islands


@pytest.mark.local_only
@pytest.mark.parametrize("region", covid_19.REGION_APPS)
def test_run_models_partial(region):
    """
    Smoke test: ensure we can build and run each model with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    region_app = covid_19.get_region_app(region)
    ps = deepcopy(region_app.params["default"])
    # Only run model for ~2 epochs.
    ps["end_time"] = ps["start_time"] + 2
    model = region_app.build_model(ps)
    model.run_model()


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", covid_19.REGION_APPS)
def test_build_models(region):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    region_app = covid_19.get_region_app(region)
    ps = deepcopy(region_app.params["default"])
    model = region_app.build_model(ps)
    assert type(model) is StratifiedModel
    model_data = serialize_model(model)


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", covid_19.REGION_APPS)
def test_run_models_full(region):
    """
    Smoke test: ensure our models run to completion without crashing.
    This can take 2+ minutes per model.
    """
    region_app = covid_19.get_region_app(region)
    region_app.run_model()
