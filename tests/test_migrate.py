import pytest

from autumn.tools.project import get_project
from autumn.settings import Models, Region


def test_migrate_marshall_islands(verify_model):
    project = get_project(Models.TB, Region.MARSHALL_ISLANDS)
    model = project.run_baseline_model(project.param_set.baseline)
    verify_model(model, "marshall-islands-baseline")
    start_times = [
        sc_params.to_dict()["time"]["start"] for sc_params in project.param_set.scenarios
    ]
    sc_models = project.run_scenario_models(
        model, project.param_set.scenarios, start_times=start_times
    )
    for idx, sc_model in enumerate(sc_models):
        verify_model(sc_model, f"marshall-islands-sc-{idx}")
