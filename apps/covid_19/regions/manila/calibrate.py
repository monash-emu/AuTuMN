from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    add_standard_dispersion_parameter,
    add_standard_philippines_params,
    add_standard_philippines_targets,
    provide_default_calibration_params,
)
from autumn.region import Region
from autumn.utils.params import load_targets

targets = load_targets("covid_19", Region.MANILA)
TARGET_OUTPUTS = add_standard_philippines_targets(targets)
PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_standard_philippines_params(PAR_PRIORS, Region.MANILA)
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "icu_occupancy")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "accum_deaths")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "notifications")


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.MANILA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )
