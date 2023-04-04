import pytest
from summer2 import CompartmentalModel
import os
import yaml

from autumn.settings import Models, Region, PROJECTS_PATH
from autumn.core.project.project import get_project

REGIONS_FOR_TEST_RUNS = ["france", "australia"]

source = os.path.join(PROJECTS_PATH, "sm_covid2", "common_school", "included_countries.yml")
country_dict = yaml.load(open(source), Loader=yaml.UnsafeLoader)
ALL_SCHOOL_REGIONS = list(country_dict["all"].values())

@pytest.mark.local_only
@pytest.mark.parametrize("region", REGIONS_FOR_TEST_RUNS)
def test_run_models_partial(region):
    """
    Smoke test: ensure we can build and run each default model with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    project = get_project(Models.SM_COVID2, region)
    # Run model for complete run length.  Currently there seem to be hardcoded parameters
    # in the random process that prevent us running for only a certain number of timesteps
    # This probably needs to be addressed at some point...
    baseline_params = project.param_set.baseline
    baseline_params_dict = baseline_params.to_dict()
    baseline_params = baseline_params.update({"time": {"end": baseline_params_dict["time"]["end"]}})
    model = project.run_baseline_model(baseline_params)
    assert type(model) is CompartmentalModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", REGIONS_FOR_TEST_RUNS)  # ALL_SCHOOL_REGIONS)
def test_build_scenario_models(region):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    project = get_project(Models.SM_COVID2, region)
    for param in project.param_set.scenarios:
        model = project.build_model(param.to_dict())
        assert type(model) is CompartmentalModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", REGIONS_FOR_TEST_RUNS)
def test_run_models_full(region):
    """
    Smoke test: ensure our models run to completion without crashing.
    This takes ~30s per model.
    """
    project = get_project(Models.SM_COVID2, region)
    baseline_model = project.run_baseline_model(project.param_set.baseline)
    assert type(baseline_model) is CompartmentalModel
    assert baseline_model.outputs is not None

    sc_models = project.run_scenario_models(baseline_model, project.param_set.scenarios)
    for sc_model in sc_models:
        assert type(sc_model) is CompartmentalModel
        assert sc_model.outputs is not None
