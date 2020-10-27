from apps.covid_19 import calibration as base
from autumn.constants import Region

country = Region.SWEDEN
TARGET_OUTPUTS, PAR_PRIORS = base.get_targets_and_priors_for_opti(country)


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        country,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )
