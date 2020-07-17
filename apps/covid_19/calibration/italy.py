from apps.covid_19.calibration import base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import (
    get_prior_distributions_for_opti,
    get_target_outputs_for_opti,
    get_weekly_summed_targets,
)
from autumn.tool_kit.utils import print_target_to_plots_from_calibration


country = Region.ITALY

PAR_PRIORS = get_prior_distributions_for_opti()

for i, par in enumerate(PAR_PRIORS):
    if par["param_name"] == "contact_rate":
        PAR_PRIORS[i]["distri_params"] = [0.025, 0.06]


TARGET_OUTPUTS = get_target_outputs_for_opti(country, source='who', data_start_time=61, data_end_time=182)

# Use weekly counts
for target in TARGET_OUTPUTS:
    target["years"], target["values"] = get_weekly_summed_targets(target["years"], target["values"])

MULTIPLIERS = {}


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds, run_id, num_chains, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
    )


if __name__ == "__main__":
    print_target_to_plots_from_calibration(TARGET_OUTPUTS)
