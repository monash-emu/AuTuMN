from autumn.constants import Region
from apps.covid_19.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int):
    base.run_calibration_chain(
        max_seconds, run_id, Region.MANILA, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
    )


PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.010, 0.05],},
    {"param_name": "start_time", "distribution": "uniform", "distri_params": [0.0, 40.0],},
    # Add extra params for negative binomial likelihood
    {
        "param_name": "infection_deathsXall_dispersion_param",
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
]

# Death counts:
death_times = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 154, 155, 157, 158, 160,]
death_values = [1, 2, 3, 3, 3, 5, 9, 7, 12, 9, 11, 20, 20, 17, 14, 27, 34, 24, 21, 30, 17, 22, 21, 12, 18, 11, 17, 12, 14, 11, 11, 19, 14, 16, 10, 16, 2, 7, 10, 6, 13, 9, 8, 14, 7, 5, 4, 14, 1, 5, 6, 2, 7, 5, 5, 3, 9, 4, 1, 6, 3, 5, 2, 5, 10, 5, 5, 2, 2, 8, 2, 1, 4, 3, 1, 1, 1, 1, 1, 4, 2, 1, 1, 1, 2, 1,]

TARGET_OUTPUTS = [
    {
        "output_key": "infection_deathsXall",
        "years": death_times,
        "values": death_values,
        "loglikelihood_distri": "negative_binomial",
    }
]
