from apps.covid_19 import calibration as base
from apps.covid_19.calibration import (
    provide_default_calibration_params,
    truncate_targets_from_time,
)
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from autumn.region import Region
from autumn.utils.params import load_targets


targets = load_targets("covid_19", Region.MALAYSIA)


# Truncate notifications from 1st August 2020
notifications = truncate_targets_from_time(targets["notifications"], 210)
icu_occupancy = truncate_targets_from_time(targets["icu_occupancy"], 210)
infection_deaths = truncate_targets_from_time(targets["infection_deaths"], 210)

icu_times = icu_occupancy["times"]
icu_times = icu_times[::7]

icu_values = icu_occupancy["values"]
icu_values = icu_values[::7]

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
        "output_key": "icu_occupancy",
        "years": icu_times,
        "values": icu_values,
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
        "distri_params": [0.015, 0.03]
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
        "distri_params": [0.03, 0.15],
    },
    # Microdistancing
    {
        "param_name": "mobility.microdistancing.behaviour.parameters.upper_asymptote",
        "distribution": "uniform",
        "distri_params": [0.05, 0.4],
    },
    # Health system-related
    {
        "param_name": "clinical_stratification.props.hospital.multiplier",
        "distribution": "uniform",
        "distri_params": [0.7, 1.5],
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
        "distri_params": [0.01, 1.5],
    },
    {
        "param_name": "vaccination.vacc_prop_prevent_infection",
        "distribution": "beta",
        "distri_mean": 0.7,
        "distri_ci": [0.5, 0.9],
        "sampling": "lhs",
    },
    {
        "param_name": "vaccination.overall_efficacy",
        "distribution": "uniform",
        "distri_params": [0., 1.0],
        "sampling": "lhs",
    },

    {
        "param_name": "vaccination.coverage",
        "distribution": "uniform",
        "distri_params": [0.4, 0.8],
        "sampling": "lhs",
    },

    {
        "param_name": "voc_emergence.contact_rate_multiplier",
        "distribution": "uniform",
        "distri_params": [1.2, 2.0],
    },

    {
        "param_name": "voc_emergence.start_time",
        "distribution": "uniform",
        "distri_params": [370, 400],
    },

]

def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.MALAYSIA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )
