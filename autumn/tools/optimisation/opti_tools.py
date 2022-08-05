from autumn.core.project import get_project


def get_calibration_object(model, region):
    project = get_project(model, region)
    calib = project.calibration

    calib.end_time = 2 + max([max(t.data.index) for t in calib.targets])
    calib.model_parameters = project.param_set.baseline

     # adjust random process values number if required
    if calib.model_parameters['random_process']:
        n_values = len(calib.model_parameters['random_process']['values'])
        n_updates = len(calib.random_process.values)
        if n_values >= n_updates:
            calib.model_parameters['random_process']['values'] = calib.model_parameters['random_process']['values'][:n_updates]
        else:
            rp_values = calib.model_parameters['random_process']['values'] + [0.] * (n_updates - n_values)
            calib.model_parameters = calib.model_parameters.update(
                {
                    "random_process.values": rp_values
                }, calibration_format=True
            )

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
    log_rp_likelihood = calib.random_process.evaluate_rp_loglikelihood()

    return - (loglikelihood + logprior + log_rp_likelihood)