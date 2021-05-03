import logging

from apps.tuberculosis.model import build_model
from autumn.calibration import Calibration
from autumn.region import Region
from autumn.utils.params import load_params, load_targets

targets = load_targets("tuberculosis", Region.PHILIPPINES)
prevalence_infectious = targets["prevalence_infectious"]


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
        region_name=Region.PHILIPPINES,
    )
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_chains=1,
        available_time=max_seconds,
        haario_scaling_factor=params["default"]["haario_scaling_factor"],
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
