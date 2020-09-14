from autumn.constants import Region
from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    provide_default_calibration_params,
    add_standard_philippines_params,
    add_standard_philippines_targets,
    add_standard_dispersion_parameter,
)
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from autumn.tool_kit.params import load_targets

targets = load_targets("covid_19", Region.MANILA)
TARGET_OUTPUTS = add_standard_philippines_targets(targets)
PAR_PRIORS = provide_default_calibration_params(excluded_params=("start_time",))
PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)
PAR_PRIORS = add_standard_philippines_params(PAR_PRIORS)
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "icu_occupancy")
PAR_PRIORS = add_standard_dispersion_parameter(PAR_PRIORS, TARGET_OUTPUTS, "accum_deaths")


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
