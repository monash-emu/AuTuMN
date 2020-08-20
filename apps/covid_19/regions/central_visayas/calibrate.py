from autumn.constants import Region
from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    provide_default_calibration_params,
    add_standard_dispersion_parameter,
    add_standard_philippines_params,
    assign_trailing_weights_to_halves,
)
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from autumn.tool_kit.params import load_targets

targets = load_targets("covid_19", Region.CENTRAL_VISAYAS)
notifications = targets["notifications"]
# icu_occupancy = targets["icu_occupancy"]
# total_infection_deaths = targets["total_infection_deaths"]

def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.CENTRAL_VISAYAS,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
        "time_weights": assign_trailing_weights_to_halves(14, notifications["times"]),
    },
#    {
#        "output_key": "icu_occupancy",
#        "years": icu_occupancy["times"],
#        "values": icu_occupancy["values"],
#        "loglikelihood_distri": "normal",
#    },
#    {
#        "output_key": total_infection_deaths["output_key"],
#        "years": total_infection_deaths["times"],
#        "values": total_infection_deaths["values"],
#        "loglikelihood_distri": "normal",
#    },
]

PAR_PRIORS = provide_default_calibration_params(excluded_params=("start_time"))
PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)
PAR_PRIORS = add_standard_philippines_params(PAR_PRIORS)

PAR_PRIORS += [
    {
        "param_name": "start_time",
        "distribution": "uniform",
        "distri_params": [40., 60.],
    },
    {
        "param_name": "microdistancing.parameters.multiplier",
        "distribution": "uniform",
        "distri_params": [0.04, 0.08],
    },
]
