from apps.covid_19.calibration import base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import get_prior_distributions_for_opti, get_target_outputs_for_opti


country = Region.UNITED_KINGDOM

PAR_PRIORS = get_prior_distributions_for_opti()
TARGET_OUTPUTS = get_target_outputs_for_opti(country, data_start_time=50, update_jh_data=False)
MULTIPLIERS = {}


def run_calibration_chain(max_seconds: int, run_id: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        country,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


if __name__ == "__main__":
    run_calibration_chain(15 * 60 * 60, 1)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
