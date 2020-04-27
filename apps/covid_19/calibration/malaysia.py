from apps.covid_19.calibration.base import run_calibration_chain, get_priors_and_targets

country = "malaysia"
PAR_PRIORS, TARGET_OUTPUTS = get_priors_and_targets(country, "deaths", 1)

PAR_PRIORS = [
    {'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [0.20, 0.4]},
    # {'param_name': 'infectious_seed', 'distribution': 'uniform', 'distri_params': [1, 1000]},
]


# target_to_plots = {
#     "infection_deathsXall": {
#         "times": TARGET_OUTPUTS[0]["years"],
#         "values": [[d] for d in TARGET_OUTPUTS[0]["values"]],
#     }
# }
# print(target_to_plots)


def run_mys_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc')
