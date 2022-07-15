import pytest

from autumn.core.model_builder import RunnableModel

from autumn.settings import Models
from autumn.core.project.project import _PROJECTS, get_project

SM_SIR_PROJECTS = list(_PROJECTS[Models.SM_SIR].keys())


@pytest.mark.local_only
@pytest.mark.parametrize("project_name", SM_SIR_PROJECTS)
def test_run_models_partial(project_name):
    """
    Smoke test: ensure we can build and run each default model with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    project = get_project(Models.SM_SIR, project_name)
    # Run model for complete run length.  Currently there seem to be hardcoded parameters
    # in the random process that prevent us running for only a certain number of timesteps
    # This probably needs to be addressed at some point...
    baseline_params = project.param_set.baseline
    baseline_params_dict = baseline_params.to_dict()
    baseline_params = baseline_params.update({"time": {"end": baseline_params_dict["time"]["end"]}})
    model = project.run_baseline_model(baseline_params)
    assert isinstance(model, RunnableModel)


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("project_name", SM_SIR_PROJECTS)
def test_build_scenario_models(project_name):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    project = get_project(Models.SM_SIR, project_name)
    for param in project.param_set.scenarios:
        model = project.build_model(param.to_dict())
        assert isinstance(model, RunnableModel)


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("project_name", SM_SIR_PROJECTS)
def test_run_models_full(project_name):
    """
    Smoke test: ensure our models run to completion without crashing.
    This takes ~30s per model.
    """
    project = get_project(Models.SM_SIR, project_name)
    baseline_model = project.run_baseline_model(project.param_set.baseline)
    assert isinstance(baseline_model, RunnableModel)
    assert baseline_model.outputs is not None

    start_times = [
        sc_params.to_dict()["time"]["start"] for sc_params in project.param_set.scenarios
    ]
    sc_models = project.run_scenario_models(
        baseline_model, project.param_set.scenarios, start_times=start_times
    )
    for sc_model in sc_models:
        assert isinstance(sc_model, RunnableModel)
        assert sc_model.outputs is not None
