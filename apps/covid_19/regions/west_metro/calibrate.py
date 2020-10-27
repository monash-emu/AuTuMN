from apps.covid_19.calibration import (
    add_standard_victoria_params,
    add_standard_victoria_targets,
    provide_default_calibration_params,
)
from autumn.constants import Region
from autumn.tool_kit.params import load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from apps.covid_19 import calibration as base

targets = load_targets("covid_19", Region.WEST_METRO)
TARGET_OUTPUTS = add_standard_victoria_targets([], targets, Region.WEST_METRO)

PAR_PRIORS = provide_default_calibration_params(("time.start", "contact_rate"))
PAR_PRIORS = add_standard_victoria_params(PAR_PRIORS, Region.WEST_METRO)
PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.WEST_METRO,
        PAR_PRIORS,
        TARGET_OUTPUTS,
    )
