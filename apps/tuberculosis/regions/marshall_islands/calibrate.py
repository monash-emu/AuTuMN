import logging

from autumn.constants import Region
from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params, load_targets

from apps.tuberculosis.model import build_model

targets = load_targets("tuberculosis", Region.MARSHALL_ISLANDS)
prevalence_infectious = targets["prevalence_infectious"]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    params = load_params("tuberculosis", Region.MARSHALL_ISLANDS)
    calib = Calibration(
        "tuberculosis",
        build_model,
        params,
        PRIORS,
        TARGET_OUTPUTS,
        run_id,
        num_chains,
        param_set_name=Region.MARSHALL_ISLANDS,
    )
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_iterations=1e6,
        n_burned=0,
        n_chains=1,
        available_time=max_seconds,
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
]
