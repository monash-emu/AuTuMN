from autumn.region import Region
from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    provide_default_calibration_params,
    add_standard_philippines_params,
    add_standard_philippines_targets,
    add_standard_dispersion_parameter,
)
from autumn.tool_kit.params import load_targets

targets = load_targets("covid_19", Region.CENTRAL_VISAYAS)
TARGET_OUTPUTS = add_standard_philippines_targets(targets)
PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_philippines_params(PAR_PRIORS, Region.CENTRAL_VISAYAS)

PAR_PRIORS.append(
    {
        "param_name": "params.voc_emmergence.final_proportion",
        "distribution": "uniform",
        "distri_params": [0.3, 0.7],
    },
)

PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "icu_occupancy")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "accum_deaths")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.CENTRAL_VISAYAS,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )
