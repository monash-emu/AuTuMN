from autumn.constants import Region
from apps.covid_19.calibration import base
from apps.covid_19.mixing_optimisation.utils import add_dispersion_param_prior_for_gaussian
import numpy as np


def run_calibration_chain(max_seconds: int, run_id: int, num_chains: int):
    base.run_calibration_chain(
        max_seconds,
        run_id,
        num_chains,
        Region.VICTORIA,
        PAR_PRIORS,
        TARGET_OUTPUTS,
        mode="autumn_mcmc",
    )


case_times = [
      160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
      170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
      193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
      216, 217, 218, 219, 220, 221, 222, 223, 224
    ]
case_counts = [
    2, 0, 4, 8, 4, 8, 9, 12, 9, 21, 18, 13, 25, 19, 16, 17, 20, 33, 30, 41,
    49, 75, 64, 73, 77, 66, 108, 74, 127, 191, 134, 165, 288, 216, 273, 177, 270,
    238, 317, 428, 217, 363, 275, 374, 484, 403, 300, 357, 459, 532, 384, 295, 723,
    627, 397, 671, 429, 439, 725, 471, 450, 466, 394, 322, 331
]

hospital_times = [
    196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218,
    219, 220, 221, 222, 223
]
hospital_counts = [
    85, 111, 114, 126, 116, 135, 156, 183, 214, 211, 217, 239, 241, 259, 275, 323,
    330, 366, 401, 408, 439, 483, 538, 600, 632, 659, 658, 664
]

icu_times = [
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
    194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
    217, 218, 219, 220, 221, 222, 223
]
icu_counts = [
    2, 1, 2, 1, 1, 1, 2, 3, 2, 2, 2,
    2, 3, 2, 2, 3, 2, 2, 1, 1, 1, 1, 1, 2, 4, 6, 3, 3, 5, 9, 7, 9, 12,
    15, 16, 17, 26, 28, 30, 32, 26, 29, 33, 38, 42, 41, 44, 46, 46, 49, 47, 46,
    40, 44, 50, 46, 43, 44, 42, 52, 51, 53, 51, 54
]

death_times = [
    175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
    215, 216, 217, 218, 219, 220, 221, 222, 223, 224
]
death_counts = [
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 2, 1, 2, 3, 3, 3, 1, 3, 2, 5, 7, 5, 10, 6, 6, 9, 13,
    8, 3, 7, 13, 11, 15, 8, 11, 12, 16, 18, 19
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": case_times,
        "values": case_counts,
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(case_times) + 1)),
    },
    {
        "output_key": "hospital_occupancy",
        "years": hospital_times,
        "values": hospital_counts,
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(hospital_times) + 1)),
    },
    {
        "output_key": "icu_occupancy",
        "years": icu_times,
        "values": icu_counts,
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(icu_times) + 1)),
    },
    {
        "output_key": "total_infection_deaths",
        "years": death_times,
        "values": death_counts,
        "loglikelihood_distri": "normal",
        "time_weights": list(range(1, len(death_times) + 1)),
    },
]

PAR_PRIORS = [
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [0.025, 0.05],
    },
    {
        "param_name": "seasonal_force",
        "distribution": "uniform",
        "distri_params": [0., 0.4],
    },
    {
        "param_name": "compartment_periods_calculated.exposed.total_period",
        "distribution": "trunc_normal",
        "distri_params": [5., 0.7],
        "trunc_range": [1., np.inf],
    },
    {
        "param_name": "compartment_periods_calculated.active.total_period",
        "distribution": "trunc_normal",
        "distri_params": [7., 0.7],
        "trunc_range": [1., np.inf],
    },
    {
        "param_name": "symptomatic_props_multiplier",
        "distribution": "trunc_normal",
        "distri_params": [1., 0.1],
        "trunc_range": [0.5, np.inf],
    },
    {
        "param_name": "testing_to_detection.shape_parameter",
        "distribution": "uniform",
        "distri_params": [-5, -3.5]
    },
    {
        "param_name": "testing_to_detection.maximum_detection",
        "distribution": "uniform",
        "distri_params": [0.6, 0.9],
    },
    {
        "param_name": "hospital_props_multiplier",
        "distribution": "trunc_normal",
        "distri_params": [1., 0.25],
        "trunc_range": [0.1, np.inf],
    },
    {
        "param_name": "ifr_multiplier",
        "distribution": "trunc_normal",
        "distri_params": [1., 0.25],
        "trunc_range": [0.1, np.inf],
    },
    {
        "param_name": "icu_prop",
        "distribution": "uniform",
        "distri_params": [0.08, 0.2],
    },
    {
        "param_name": "compartment_periods.icu_early",
        "distribution": "uniform",
        "distri_params": [5., 17.],
    },
    {
        "param_name": "compartment_periods.icu_late",
        "distribution": "uniform",
        "distri_params": [5., 15.],
    },
    {
        "param_name": "microdistancing.parameters.multiplier",
        "distribution": "uniform",
        "distri_params": [0.04, 0.1],
    }
]

PAR_PRIORS = add_dispersion_param_prior_for_gaussian(PAR_PRIORS, TARGET_OUTPUTS, {})
