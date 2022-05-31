import pytest

from autumn.settings import Models
from autumn.core.project.project import _PROJECTS, get_project

COVID_PROJECTS = list(_PROJECTS[Models.COVID_19].keys())


@pytest.mark.benchmark
@pytest.mark.github_only
@pytest.mark.parametrize("project_name", COVID_PROJECTS)
def test_benchmark_covid_models(project_name, benchmark):
    """
    Performance benchmark: check how long our models take to run.
    See: https://pytest-benchmark.readthedocs.io/en/stable/
    Run these with pytest -vv -m benchmark --benchmark-json benchmark.json
    """
    project = get_project(Models.COVID_19, project_name)
    benchmark(_run_model, project=project)


def _run_model(project):
    project.run_baseline_model(project.param_set.baseline)
