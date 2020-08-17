from autumn.constants import Region
from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    provide_default_calibration_params,
    add_standard_dispersion_parameter,
)
from apps.covid_19.mixing_optimisation.utils import add_dispersion_param_prior_for_gaussian
from autumn.tool_kit.params import load_targets

targets = load_targets("covid_19", Region.MALAYSIA)
notifications = targets["notifications"]
icu_occupancy = targets["icu_occupancy"]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.MALAYSIA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "negative_binomial",
    },
    {
        "output_key": "icu_occupancy",
        "years": icu_occupancy["times"],
        "values": icu_occupancy["values"],
        "loglikelihood_distri": "negative_binomial",
    },
]

PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "icu_occupancy")

# PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)

PAR_PRIORS += [
    # ICU-related
    {
        "param_name": "compartment_periods.icu_early",
        "distribution": "uniform",
        "distri_params": [5.0, 25.0],
    },
    {"param_name": "icu_prop", "distribution": "uniform", "distri_params": [0.12, 0.25],},
    # Detection-related
    {
        "param_name": "time_variant_detection.start_value",
        "distribution": "uniform",
        "distri_params": [0.1, 0.4],
    },
    {
        "param_name": "time_variant_detection.maximum_gradient",
        "distribution": "uniform",
        "distri_params": [0.05, 0.1],
    },
    {
        "param_name": "time_variant_detection.end_value",
        "distribution": "uniform",
        "distri_params": [0.6, 0.9],
    },
    # Microdistancing-related
    {
        "param_name": "microdistancing.parameters.sigma",
        "distribution": "uniform",
        "distri_params": [0.3, 0.7],
    },
    {
        "param_name": "microdistancing.parameters.c",
        "distribution": "uniform",
        "distri_params": [78.0, 124.0],
    },
]
