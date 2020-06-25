from autumn.constants import Region
from apps.covid_19.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int):
    base.run_calibration_chain(
        max_seconds, run_id, Region.BICOL, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc", _multipliers=MULTIPLIERS
    )

MULTIPLIERS = {
    "prevXlateXclinical_icuXamong": 6133850.0
}  # to get absolute pop size instead of proportion


PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.010, 0.05],},
    {"param_name": "start_time", "distribution": "uniform", "distri_params": [0.0, 40.0],},
    # Add extra params for negative binomial likelihood
    {
        "param_name": "prevXlateXclinical_icuXamong_dispersion_param",
        "distribution": "uniform",
        "distri_params": [0.1, 5.0],
    },
    {
        "param_name": "compartment_periods_calculated.incubation.total_period",
        "distribution": "gamma",
        "distri_mean": 5.0,
        "distri_ci": [4.4, 5.6],
    },
    {
        "param_name": "compartment_periods_calculated.total_infectious.total_period",
        "distribution": "gamma",
        "distri_mean": 7.0,
        "distri_ci": [4.5, 9.5],
    },
    # parameters to derive age-specific IFRs
    {
        "param_name": "ifr_double_exp_model_params.k",
        "distribution": "uniform",
        "distri_params": [6., 14.],
    },
    {
        "param_name": "ifr_double_exp_model_params.last_representative_age",
        "distribution": "uniform",
        "distri_params": [75., 85.],
    },
]

# ICU data
icu_times = [
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    155,
    156,
    157,
    158,
    159,
    160,
    161,
    162,
    163,
    164,
    165,
    166,
    167,
]
icu_counts = [
    1,
    1,
    3,
    2,
    3,
    3,
    3,
    17,
    3,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    2,
    2,
    2,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]

TARGET_OUTPUTS = [
    {
        "output_key": "prevXlateXclinical_icuXamong",
        "years": icu_times,
        "values": icu_counts,
        "loglikelihood_distri": "negative_binomial",
    }
]
