from apps.covid_19.calibration import add_standard_victoria_params

from autumn.constants import Region
from autumn.tool_kit.params import load_targets
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian

from apps.covid_19 import calibration as base

targets = load_targets("covid_19", Region.SOUTH_EAST_METRO)
notifications = targets["notifications"]
hospital_occupancy = targets["hospital_occupancy"]
icu_occupancy = targets["icu_occupancy"]
total_infection_deaths = targets["total_infection_deaths"]


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds, run_id, num_chains, Region.SOUTH_EAST_METRO, PAR_PRIORS, TARGET_OUTPUTS,
    )


TARGET_OUTPUTS = [
    {
        "output_key": notifications["output_key"],
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(notifications["times"]) + 1)),
    },
    {
        "output_key": hospital_occupancy["output_key"],
        "years": hospital_occupancy["times"],
        "values": hospital_occupancy["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(hospital_occupancy["times"]) + 1)),
    },
    {
        "output_key": icu_occupancy["output_key"],
        "years": icu_occupancy["times"],
        "values": icu_occupancy["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(icu_occupancy["times"]) + 1)),
    },
    {
        "output_key": total_infection_deaths["output_key"],
        "years": total_infection_deaths["times"],
        "values": total_infection_deaths["values"],
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(total_infection_deaths["times"]) + 1)),
    },
]

PAR_PRIORS = add_standard_victoria_params([])

PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)
