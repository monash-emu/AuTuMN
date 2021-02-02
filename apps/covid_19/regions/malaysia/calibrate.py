from autumn.region import Region
from apps.covid_19 import calibration as base
from apps.covid_19.calibration import provide_default_calibration_params
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian
from autumn.tool_kit.params import load_targets
from apps.covid_19.calibration import truncate_targets_from_time


truncation_start_time = 210
targets = load_targets("covid_19", Region.MALAYSIA)
notifications = truncate_targets_from_time(targets["notifications"], truncation_start_time)
icu_occupancy = truncate_targets_from_time(targets["icu_occupancy"], truncation_start_time)

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notifications["times"],
        "values": notifications["values"],
        "loglikelihood_distri": "normal",
    },
    {
        "output_key": "icu_occupancy",
        "years": icu_occupancy["times"],
        "values": icu_occupancy["values"],
        "loglikelihood_distri": "normal",
    },
]

PAR_PRIORS = provide_default_calibration_params()
PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)

PAR_PRIORS += [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.01, 0.03]},
    {
        "param_name": "seasonal_force",
        "distribution": "uniform",
        "distri_params": [-0.3, 0.0],
    },
    # Detection
    {
        "param_name": "testing_to_detection.assumed_cdr_parameter",
        "distribution": "uniform",
        "distri_params": [0.12, 0.5],
    },
    # Microdistancing
    {
        "param_name": "mobility.microdistancing.behaviour.parameters.upper_asymptote",
        "distribution": "uniform",
        "distri_params": [0.05, 0.25],
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
        "param_name": "sojourn.compartment_periods.icu_early",
        "distribution": "uniform",
        "distri_params": [5.0, 25.0],
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
