from autumn.tools.project import get_project


def get_calibration_object(model, region):
    project = get_project(model, region)
    calib = project.calibration

    calib.end_time = 300 + max([max(t.data.index) for t in calib.targets])
    calib.model_parameters = project.param_set.baseline
    calib._is_first_run = False
    calib.project = project
    calib.run_mode = "autumn_mcmc"
    target_names = [t.data.name for t in calib.targets]
    calib.derived_outputs_whitelist = list(set(target_names))
    calib.build_options = dict(enable_validation = False)

    calib.workout_unspecified_target_sds()
    calib.workout_unspecified_time_weights()
    return calib


def calculate_objective_to_minimize(calib, params_dict):
    loglikelihood = calib.loglikelihood(params_dict)
    logprior = calib.logprior(params_dict)

    return - (loglikelihood + logprior)


