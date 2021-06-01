import pytest

from autumn.settings import Models
from autumn.tools.project.project import _PROJECTS, get_project

COVID_PROJECTS = list(_PROJECTS[Models.COVID_19].keys())
COVID_CALIBS = list(zip(COVID_PROJECTS, [Models.COVID_19] * len(COVID_PROJECTS)))
TB_PROJECTS = list(_PROJECTS[Models.TB].keys())
TB_CALIBS = list(zip(TB_PROJECTS, [Models.TB] * len(TB_PROJECTS)))
CALIBS = COVID_CALIBS + TB_CALIBS


@pytest.mark.github_only
@pytest.mark.calibrate_models
@pytest.mark.parametrize("project_name, model_name", CALIBS)
def test_calibration(project_name, model_name):
    """
    Calibration smoke test - make sure everything can run for 10 seconds without exploding.
    """
    project = get_project(model_name, project_name)
    project.calibrate(max_seconds=10, chain_idx=1, num_chains=1)
