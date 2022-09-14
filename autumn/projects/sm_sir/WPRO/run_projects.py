from autumn.core.project import get_project, ParameterSet
iso3s = ["WPRO_MALAYSIA"]

for iso3 in iso3s:
    project = get_project("WPRO", iso3)
    # run baseline model
    #model_0 = project.run_baseline_model(project.param_set.baseline)