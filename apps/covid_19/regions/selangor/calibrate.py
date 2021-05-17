from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    provide_default_calibration_params,
    truncate_targets_from_time,
)
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from autumn.region import Region
from autumn.utils.params import load_targets

targets = load_targets("covid_19", Region.SELANGOR)
notifications = truncate_targets_from_time(targets["notifications"], 270.0)
infection_deaths = truncate_targets_from_time(targets["infection_deaths"], 270)
deaths_times = infection_deaths["times"]
deaths_times = deaths_times[::7]

deaths_values = infection_deaths["values"]
deaths_values = deaths_values[::7]
TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
    },
    {
        "output_key": "infection_deaths",
        "years": deaths_times,
        "values": deaths_values,
        "loglikelihood_distri": "normal",
    },
]

PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)

PAR_PRIORS += [
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [0.018, 0.04],
    },
    {
        "param_name": "infectious_seed",
        "distribution": "uniform",
        "distri_params": [50.0, 200.0],
    },
    # Detection
    {
        "param_name": "testing_to_detection.assumed_cdr_parameter",
        "distribution": "uniform",
        "distri_params": [0.03, 0.12],
    },
    # Microdistancing
    {
        "param_name": "mobility.microdistancing.behaviour.parameters.upper_asymptote",
        "distribution": "uniform",
        "distri_params": [0.1, 0.32],
    },
    # Health system-related
    {
        "param_name": "clinical_stratification.props.hospital.multiplier",
        "distribution": "uniform",
        "distri_params": [0.7, 1.3],
    },
    {
        "param_name": "clinical_stratification.icu_prop",
        "distribution": "uniform",
        "distri_params": [0.12, 0.25],
    },
    {
        "param_name": "clinical_stratification.non_sympt_infect_multiplier",
        "distribution": "uniform",
        "distri_params": [0.15, 0.4],
    },
    {
        "param_name": "clinical_stratification.props.symptomatic.multiplier",
        "distribution": "uniform",
        "distri_params": [0.8, 2.0],
    },
    {
        "param_name": "vaccination.vacc_prop_prevent_infection",
        "distribution": "uniform",
        "distri_params": [0., 1.0],
        "sampling": "lhs",
    },
]


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
