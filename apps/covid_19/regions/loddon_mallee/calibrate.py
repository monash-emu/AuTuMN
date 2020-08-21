import numpy as np

from autumn.constants import Region
from autumn.tool_kit.params import load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian

from apps.covid_19 import calibration as base

targets = load_targets("covid_19", Region.LODDON_MALLEE)
notifications = targets["notifications"]
hospital_occupancy = targets["hospital_occupancy"]
icu_occupancy = targets["icu_occupancy"]
total_infection_deaths = targets["total_infection_deaths"]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds, run_id, num_chains, Region.LODDON_MALLEE, PAR_PRIORS, TARGET_OUTPUTS,
    )


TARGET_OUTPUTS = [
    {
        "output_key": notifications["output_key"],
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(notifications["times"]) + 1)),
    },
    {
        "output_key": hospital_occupancy["output_key"],
        "years": hospital_occupancy["times"],
        "values": hospital_occupancy["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(hospital_occupancy["times"]) + 1)),
    },
    {
        "output_key": icu_occupancy["output_key"],
        "years": icu_occupancy["times"],
        "values": icu_occupancy["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(icu_occupancy["times"]) + 1)),
    },
    {
        "output_key": total_infection_deaths["output_key"],
        "years": total_infection_deaths["times"],
        "values": total_infection_deaths["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(total_infection_deaths["times"]) + 1)),
    },
]

PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.025, 0.05],},
    {"param_name": "seasonal_force", "distribution": "uniform", "distri_params": [0.0, 0.4],},
    {
        "param_name": "compartment_periods_calculated.exposed.total_period",
        "distribution": "trunc_normal",
        "distri_params": [5.0, 0.7],
        "trunc_range": [1.0, np.inf],
    },
    {
        "param_name": "compartment_periods_calculated.active.total_period",
        "distribution": "trunc_normal",
        "distri_params": [7.0, 0.7],
        "trunc_range": [1.0, np.inf],
    },
    {
        "param_name": "symptomatic_props_multiplier",
        "distribution": "trunc_normal",
        "distri_params": [1.0, 0.1],
        "trunc_range": [0.5, np.inf],
    },
    {
        "param_name": "testing_to_detection.maximum_detection",
        "distribution": "uniform",
        "distri_params": [0.6, 0.9],
    },
    {
        "param_name": "hospital_props_multiplier",
        "distribution": "trunc_normal",
        "distri_params": [1.0, 0.25],
        "trunc_range": [0.1, np.inf],
    },
    {
        "param_name": "ifr_multiplier",
        "distribution": "trunc_normal",
        "distri_params": [1.0, 0.25],
        "trunc_range": [0.1, np.inf],
    },
    {"param_name": "icu_prop", "distribution": "uniform", "distri_params": [0.08, 0.2],},
    {
        "param_name": "compartment_periods.icu_early",
        "distribution": "uniform",
        "distri_params": [5.0, 17.0],
    },
    {
        "param_name": "compartment_periods.icu_late",
        "distribution": "uniform",
        "distri_params": [5.0, 15.0],
    },
    {
        "param_name": "microdistancing.parameters.multiplier",
        "distribution": "uniform",
        "distri_params": [0.04, 0.08],
    },
]

PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)
