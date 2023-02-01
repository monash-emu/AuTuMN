import pytest
from summer2 import CompartmentalModel

from autumn.settings import Models
from autumn.core.project.project import _PROJECTS, get_project

TB_PROJECTS = list(_PROJECTS[Models.TBD2].keys())


@pytest.mark.local_only
@pytest.mark.parametrize("project_name", TB_PROJECTS)
def test_tb_run_models_partial(project_name):
    """
    Smoke test: ensure we can build and run each default model with various stratification requests with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    project = get_project(Models.TBD2, project_name)
    # Only run model for 5 timesteps.
    baseline_params = project.param_set.baseline
    baseline_params_dict = baseline_params.to_dict()
    params = baseline_params.update(
        {
            "time": {
                "end": baseline_params_dict["time"]["start"] + 5,
            },
        }
    )
    model = project.run_baseline_model(params)
    assert type(model) is CompartmentalModel


# @pytest.mark.run_models
# @pytest.mark.github_only
# @pytest.mark.parametrize("project_name", TB_PROJECTS)
# def test_run_models_full(project_name):
#     """
#     Smoke test: ensure our models run to completion without crashing.
#     This takes ~30s per model.
#     """
#     project = get_project(Models.TBD2, project_name)
#     baseline_model = project.build_model(project.param_set.baseline.to_dict())
#     assert type(baseline_model) is CompartmentalModel
#     assert baseline_model.outputs is not None


# @pytest.mark.run_models
# @pytest.mark.github_only
# @pytest.mark.parametrize("project_name", TB_PROJECTS)
# def test_run_models_full(project_name):
#     """
#     Smoke test: ensure our models run to completion without crashing.
#     This takes ~30s per model.
#     """
#     project = get_project(Models.TBD2, project_name)
#     baseline_model = project.build_model(project.param_set.baseline.to_dict())
#     assert type(baseline_model) is CompartmentalModel
#     assert baseline_model.outputs is not None
