from apps.covid_19.calibration.base import run_calibration_chain
from apps.covid_19.mixing_optimisation.utils import get_prior_distributions_for_opti, get_target_outputs_for_opti
country = "united-kingdom"

PAR_PRIORS = get_prior_distributions_for_opti()
TARGET_OUTPUTS = get_target_outputs_for_opti(country, data_start_time=50, update_jh_data=False)
MULTIPLIERS = {}


def run_gbr_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
                          _run_extra_scenarios=False, _multipliers=MULTIPLIERS)


if __name__ == "__main__":
    run_gbr_calibration_chain(15 * 60 * 60, 1)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
