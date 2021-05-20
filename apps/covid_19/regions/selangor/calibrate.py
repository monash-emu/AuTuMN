from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    provide_default_calibration_params,
    truncate_targets_from_time,
)
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from autumn.region import Region
from autumn.utils.params import load_targets
from apps.covid_19.regions.malaysia.calibrate import MALAYSIA_PARAMS

targets = load_targets("covid_19", Region.SELANGOR)
notifications = truncate_targets_from_time(targets["notifications"], 270.0)

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
    },

]

PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)

PAR_PRIORS += MALAYSIA_PARAMS


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.SELANGOR,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )
