from apps.covid_19.calibration import base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import (
    get_prior_distributions_for_opti,
    get_target_outputs_for_opti,
    get_weekly_summed_targets,
)


country = Region.UNITED_KINGDOM

PAR_PRIORS = get_prior_distributions_for_opti()
TARGET_OUTPUTS = get_target_outputs_for_opti(country, source='who', data_start_time=50)

# Use weekly counts
for target in TARGET_OUTPUTS:
    target["years"], target["values"] = get_weekly_summed_targets(target["years"], target["values"])

MULTIPLIERS = {}


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds, run_id, num_chains, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
    )


if __name__ == "__main__":
    for i in range(len(TARGET_OUTPUTS)):
        print(TARGET_OUTPUTS[i]["output_key"])
        print(TARGET_OUTPUTS[i]["years"])
        print([[v] for v in TARGET_OUTPUTS[i]["values"]])
        print()

    run_calibration_chain(
        30, 1
    )  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
