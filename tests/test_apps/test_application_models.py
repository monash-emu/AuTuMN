from copy import deepcopy

import pytest
from summer.model import StratifiedModel

from autumn.tool_kit.serializer import serialize_model
from apps import mongolia, covid_19, marshall_islands


MODEL_BUILDERS_LOCAL = [
    ["COVID-19 AUS", covid_19.aus.build_model, covid_19.aus.params],
    ["COVID-19 VIC", covid_19.vic.build_model, covid_19.vic.params],
    ["COVID-19 MYS", covid_19.mys.build_model, covid_19.mys.params],
    ["COVID-19 PHL", covid_19.phl.build_model, covid_19.phl.params],
]


@pytest.mark.local_only
@pytest.mark.parametrize("name, build_model, params", MODEL_BUILDERS_LOCAL)
def test_run_models_partial(name, build_model, params):
    """
    Smoke test: ensure we can build and run each model with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    ps = deepcopy(params["default"])
    # Only run model for ~2 epochs.
    ps["end_time"] = ps["start_time"] + 2
    model = build_model(ps)
    model.run_model()


MODEL_BUILDERS_GITHUB = [
    *MODEL_BUILDERS_LOCAL,
    ["Mongolia", mongolia.build_model, mongolia.params],
    ["Marshall Islands", marshall_islands.build_model, marshall_islands.params],
]


@pytest.mark.github_only
@pytest.mark.parametrize("name, build_model, params", MODEL_BUILDERS_GITHUB)
def test_build_models(name, build_model, params):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    ps = deepcopy(params["default"])
    model = build_model(ps)
    assert type(model) is StratifiedModel
    model_data = serialize_model(model)


MODEL_RUNNERS_GITHUB = [
    ["COVID-19 AUS", covid_19.aus.run_model],
    ["COVID-19 VIC", covid_19.vic.run_model],
    ["COVID-19 MYS", covid_19.mys.run_model],
    ["COVID-19 PHL", covid_19.phl.run_model],
]


@pytest.mark.github_only
@pytest.mark.parametrize("name, run_model", MODEL_RUNNERS_GITHUB)
def test_run_models_full(name, run_model):
    """
    Smoke test: ensure our models run to completion without crashing.
    This can take 2+ minutes per model.
    """
    run_model()
