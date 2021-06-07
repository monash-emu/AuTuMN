from autumn.tools.project import get_project
from autumn.settings import Models, Region


def test_migrate_marshall_islands(verify_model):
    project = get_project(Models.TB, Region.MARSHALL_ISLANDS)
    baseline_params = project.param_set.baseline
    baseline_model = project.build_model(baseline_params.to_dict())
    baseline_model.run()
    verify_model(baseline_model, "marshall-islands-baseline")


def test_migrate_marshall_islands_scenario(verify_model):
    project = get_project(Models.TB, Region.MARSHALL_ISLANDS)
    baseline_params = project.param_set.baseline
    baseline_model = project.build_model(baseline_params.to_dict())
    baseline_model.run()
    verify_model(baseline_model, "marshall-islands-baseline")
