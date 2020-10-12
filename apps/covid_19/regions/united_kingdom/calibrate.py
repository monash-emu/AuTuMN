from apps.covid_19 import calibration as base
from autumn.constants import Region
from apps.covid_19.mixing_optimisation.utils import (
    get_prior_distributions_for_opti,
    get_weekly_summed_targets,
    add_dispersion_param_prior_for_gaussian,
)
from autumn.tool_kit.params import load_targets

country = Region.UNITED_KINGDOM

targets = load_targets("covid_19", country)
notifications = targets["notifications"]
deaths = targets["infection_deaths"]

PAR_PRIORS = get_prior_distributions_for_opti()
TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
    },
    {
        "output_key": "infection_deaths",
        "years": deaths["times"],
        "values": deaths["values"],
        "loglikelihood_distri": "normal",
    },
]

# Use weekly counts
for target in TARGET_OUTPUTS:
    target["years"], target["values"] = get_weekly_summed_targets(target["years"], target["values"])

PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)

TARGET_OUTPUTS = base.remove_early_points_to_prevent_crash(TARGET_OUTPUTS, PAR_PRIORS)


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds, run_id, num_chains, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
    )
