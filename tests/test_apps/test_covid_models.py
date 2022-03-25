import pytest
from summer import CompartmentalModel

from autumn.settings import Models
from autumn.tools.project.project import _PROJECTS, get_project

COVID_PROJECTS = list(_PROJECTS[Models.COVID_19].keys())


@pytest.mark.local_only
@pytest.mark.parametrize("project_name", COVID_PROJECTS)
def test_run_models_partial(project_name):
    """
    Smoke test: ensure we can build and run each default model with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    project = get_project(Models.COVID_19, project_name)
    # Only run model for 5 timesteps.
    baseline_params = project.param_set.baseline
    baseline_params_dict = baseline_params.to_dict()
    baseline_params = baseline_params.update(
        {"time": {"end": baseline_params_dict["time"]["start"] + 5}}
    )
    model = project.run_baseline_model(baseline_params)
    assert type(model) is CompartmentalModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.nightly_only
@pytest.mark.parametrize("project_name", COVID_PROJECTS)
def test_build_scenario_models(project_name):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    project = get_project(Models.COVID_19, project_name)
    for param in project.param_set.scenarios:
        model = project.build_model(param.to_dict())
        assert type(model) is CompartmentalModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.nightly_only
@pytest.mark.parametrize("project_name", COVID_PROJECTS)
def test_run_models_full(project_name):
    """
    Smoke test: ensure our models run to completion without crashing.
    This takes ~30s per model.
    """
    project = get_project(Models.COVID_19, project_name)
    baseline_model = project.run_baseline_model(project.param_set.baseline)
    assert type(baseline_model) is CompartmentalModel
    assert baseline_model.outputs is not None

    start_times = [
        sc_params.to_dict()["time"]["start"] for sc_params in project.param_set.scenarios
    ]
    sc_models = project.run_scenario_models(
        baseline_model, project.param_set.scenarios, start_times=start_times
    )
    for sc_model in sc_models:
        assert type(sc_model) is CompartmentalModel
        assert sc_model.outputs is not None
