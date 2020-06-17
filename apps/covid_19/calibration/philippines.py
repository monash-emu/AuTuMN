from autumn.constants import Region
from apps.covid_19.calibration import base


def run_calibration_chain(max_seconds: int, run_id: int):
    base.run_calibration_chain(
        max_seconds, run_id, Region.PHILIPPINES, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
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
death_times = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 154, 155, 156, 157, 158, 160, 162,]
death_values = [3, 3, 4, 5, 6, 6, 12, 9, 14, 15, 13, 23, 30, 25, 21, 35, 40, 32, 30, 40, 26, 28, 25, 18, 26, 21, 18, 15, 18, 18, 18, 22, 15, 17, 12, 19, 3, 9, 13, 10, 14, 11, 12, 18, 7, 9, 6, 16, 3, 7, 9, 4, 8, 9, 8, 5, 14, 5, 2, 14, 3, 8, 5, 9, 11, 9, 7, 2, 3, 12, 3, 7, 9, 7, 1, 3, 4, 3, 4, 6, 6, 2, 4, 3, 2, 1, 1, 3, 2, 1,]

TARGET_OUTPUTS = [
    {
        "output_key": "infection_deathsXall",
        "years": death_times,
        "values": death_values,
        "loglikelihood_distri": "negative_binomial",
    }
]
