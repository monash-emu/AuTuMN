import pytest
from summer.model import StratifiedModel

from autumn.tool_kit.serializer import serialize_model
from apps import mongolia, covid_19, marshall_islands


MODEL_RUNNERS = [
    ["Mongolia", mongolia.run_model],
    ["Marshall Islands", marshall_islands.run_model],
    ["COVID-19 AUS", covid_19.aus.run_model],
]


MODEL_BUILDERS = [
    ["Mongolia", mongolia.build_model, mongolia.params],
    ["Marshall Islands", marshall_islands.build_model, marshall_islands.params],
    ["COVID-19 AUS", covid_19.aus.build_model, covid_19.aus.params],
]


@pytest.mark.github_only
@pytest.mark.parametrize("name, build_model, params", MODEL_BUILDERS)
def test_build_models(name, build_model, params):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    model = build_model(params["default"])
    assert type(model) is StratifiedModel
    model_data = serialize_model(model)
    # TODO: Actually check these


@pytest.mark.github_only
@pytest.mark.parametrize("name, run_model", MODEL_RUNNERS)
def test_run_models(name, run_model):
    """
    Smoke test: ensure our models run to completion without crashing.
    This can take up to 2 minutes per model.
    """
    run_model()
