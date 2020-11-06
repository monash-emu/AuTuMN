import logging

from autumn.constants import Region
from autumn.calibration import Calibration
from autumn.tool_kit.params import load_params, load_targets

from apps.sir_example.model import build_model

targets = load_targets("sir_example", Region.AUSTRALIA)
prevalence_infectious = targets["prevalence_infectious"]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    params = load_params("sir_example", Region.AUSTRALIA)
    calib = Calibration(
        "sir_example",
        build_model,
        params,
        PRIORS,
        TARGET_OUTPUTS,
        run_id,
        num_chains,
        region_name=Region.AUSTRALIA,
    )
    calib.run_fitting_algorithm(
        run_mode="autumn_mcmc",
        n_iterations=100000,
        n_burned=0,
        n_chains=1,
        available_time=max_seconds,
    )


PRIORS = [
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [0.025, 0.05],
    },
    {
        "param_name": "recovery_rate",
        "distribution": "uniform",
        "distri_params": [0.9, 1.2],
    },
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
