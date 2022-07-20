import pytest
from summer import CompartmentalModel

from autumn.settings import Models
from autumn.core.project.project import _PROJECTS, get_project

TB_PROJECTS = list(_PROJECTS[Models.TBD].keys())


@pytest.mark.local_only
@pytest.mark.parametrize("project_name", TB_PROJECTS)
def test_tb_run_models_partial(project_name):
    """
    Smoke test: ensure we can build and run each default model with various stratification requests with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    project = get_project(Models.TBD, project_name)
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

