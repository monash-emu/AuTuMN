import pytest
from summer import CompartmentalModel

from autumn.settings import Models
from autumn.core.project.project import _PROJECTS, get_project

TB_PROJECTS = list(_PROJECTS[Models.TB].keys())


@pytest.mark.local_only
@pytest.mark.parametrize("project_name", TB_PROJECTS)
def test_tb_run_models_partial(project_name):
    """
    Smoke test: ensure we can build and run each default model with various stratification requests with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    project = get_project(Models.TB, project_name)
    # Only run model for 5 timesteps.
    baseline_params = project.param_set.baseline
    baseline_params_dict = baseline_params.to_dict()
    params = baseline_params.update(
        {
            "time": {
                "critical_ranges": [],
                "end": baseline_params_dict["time"]["start"] + 5,
            },
        }
    )
    model = project.run_baseline_model(params)
    assert type(model) is CompartmentalModel


@pytest.mark.run_models
@pytest.mark.nightly_only
@pytest.mark.github_only
@pytest.mark.parametrize("project_name", TB_PROJECTS)
def test_tb_build_scenario_models(project_name):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    project = get_project(Models.TB, project_name)
    for param in project.param_set.scenarios:
        model = project.build_model(param.to_dict())
        assert type(model) is CompartmentalModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.nightly_only
@pytest.mark.parametrize("project_name", TB_PROJECTS)
def test_tb_run_models_full(project_name):
    """
    Smoke test: ensure our models run to completion for any stratification request without crashing.
    This takes ~30s per model.
    """
    project = get_project(Models.TB, project_name)
    baseline_model = project.run_baseline_model(project.param_set.baseline)
    assert type(baseline_model) is CompartmentalModel
    assert baseline_model.outputs is not None

    sc_models = project.run_scenario_models(baseline_model, project.param_set.scenarios)
    for sc_model in sc_models:
        assert type(sc_model) is CompartmentalModel
        assert sc_model.outputs is not None
