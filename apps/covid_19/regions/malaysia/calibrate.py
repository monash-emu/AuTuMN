from autumn.constants import Region
from apps.covid_19 import calibration as base
from apps.covid_19.calibration import provide_default_calibration_params
from apps.covid_19.mixing_optimisation.utils import add_dispersion_param_prior_for_gaussian
from autumn.tool_kit.params import load_targets

targets = load_targets("covid_19", Region.MALAYSIA)
notifications = targets["notifications"]
icu_occupancy = targets["icu_occupancy"]

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

PAR_PRIORS = provide_default_calibration_params(
    excluded_params=("time.start",),
    override_params=[
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.015, 0.04]
        },
    ]
)
PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS)

PAR_PRIORS += [
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
        "param_name": "mobility.microdistancing.parameters.max_effect",
        "distribution": "uniform",
        "distri_params": [0.3, 0.9],
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
