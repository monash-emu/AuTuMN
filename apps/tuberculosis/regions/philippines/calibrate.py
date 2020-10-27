import logging

from autumn.constants import Region
from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params, load_targets

from apps.tuberculosis.model import build_model

targets = load_targets("tuberculosis", Region.PHILIPPINES)
prevalence_infectious = targets["prevalence_infectious"]
mortality = targets["mortality"]
incidence = targets["incidence"]
notifications = targets["notifications"]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    params = load_params("tuberculosis", Region.PHILIPPINES)
    calib = Calibration(
        "tuberculosis",
        build_model,
        params,
        PRIORS,
        TARGET_OUTPUTS,
        run_id,
        num_chains,
        param_set_name=Region.PHILIPPINES,
    )
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_iterations=1e6,
        n_burned=0,
        n_chains=1,
        available_time=max_seconds,
        haario_scaling_factor=params['default']['haario_scaling_factor'],
    )


PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.025, 0.05]},
]

TARGET_OUTPUTS = [
    {
        "output_key": prevalence_infectious["output_key"],
        "years": prevalence_infectious["times"],
        "values": prevalence_infectious["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(prevalence_infectious["times"]) + 1)),
    },
    {
        "output_key": mortality["output_key"],
        "years": mortality["times"],
        "values": mortality["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(mortality["times"]) + 1)),
    },
    {
        "output_key": incidence["output_key"],
        "years": incidence["times"],
        "values": incidence["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(incidence["times"]) + 1)),
    },
    {
        "output_key": notifications["output_key"],
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(notifications["times"]) + 1)),
    },
]
