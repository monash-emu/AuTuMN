from apps.covid_19.calibration.base import run_calibration_chain

country = "united-kingdom"

PAR_PRIORS = [
    {
        'param_name': 'contact_rate',
        'distribution': 'uniform',
        'distri_params': [0.015, 0.040]
    },
    {
        'param_name': 'start_time',
        'distribution': 'uniform',
        'distri_params': [0., 40.]},
    {
        "param_name": "compartment_periods_calculated.incubation.total_period",
        "distribution": "gamma",
        "distri_mean": 5.,
        "distri_ci": [3., 7.]
    },
    {
        "param_name": "compartment_periods.icu_late",
        "distribution": "gamma",
        "distri_mean": 10.,
        "distri_ci": [5., 15.]
    },
    {
        "param_name": "compartment_periods.icu_early",
        "distribution": "gamma",
        "distri_mean": 10.,
        "distri_ci": [2., 25.]
    },
    {
        "param_name": "tv_detection_b",
        "distribution": "beta",
        "distri_mean": .075,
        "distri_ci": [.05, .1]
    },
    {
        "param_name": "prop_detected_among_symptomatic",
        "distribution": "beta",
        "distri_mean": .5,
        "distri_ci": [.2, .8]
    },
    {
        "param_name": "icu_prop",
        "distribution": "beta",
        "distri_mean": .25,
        "distri_ci": [.15, .35]
    },
    # Add negative binomial over-dispersion parameters
    {
        "param_name": "notifications_dispersion_param",
        "distribution": "uniform",
        "distri_params": [.1, 5.]
    }
]

# notification data, JH
notification_times = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166]
notification_counts = [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 2, 5, 3, 13, 4, 11, 34, 30, 48, 43, 67, 48, 61, 74, 0, 342, 342, 0, 403, 407, 676, 63, 1294, 1035, 665, 967, 1427, 1452, 2129, 2885, 2546, 2433, 2619, 3009, 4324, 4244, 4450, 3735, 5903, 3802, 3634, 5491, 4344, 8681, 5233, 5288, 4342, 5252, 4603, 4617, 5599, 5525, 5850, 4676, 4301, 4451, 4583, 5386, 4913, 4463, 4309, 3996, 4076, 6032, 6201, 4806, 4339, 3985, 4406, 6111, 5614, 4649, 3896, 3923, 3877, 3403, 3242, 3446, 3560, 3450, 3534, 2711, 2412, -525, 2615, 3287, 2959, 2405, 1625, 4043, 2013, 1887, 2095, 1604, 1936, 1570, 1653, 1871, 1805, 1650, 1557, 1326, 1205, 1741, 1003, 1266, 1541, 1425, 1514, 968]


# death data, JH
deaths_times = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166]
deaths_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 4, 0, 2, 1, 18, 15, 22, 16, 34, 43, 36, 56, 35, 74, 149, 186, 183, 284, 294, 214, 374, 382, 670, 652, 714, 760, 644, 568, 1038, 1034, 1103, 1152, 839, 686, 744, 1044, 842, 1029, 935, 1115, 498, 559, 1172, 837, 727, 1005, 843, 420, 338, 909, 795, 674, 739, 621, 315, 288, 693, 649, 539, 626, 346, 268, 210, 627, 494, 428, 384, 468, 170, 160, 545, 363, 338, 351, 282, 118, 121, 134, 412, 377, 324, 215, 113, 556, 324, 359, 176, 357, 204, 77, 55, 286, 245, 151, 202, 181, 36, 38]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_counts,
        "loglikelihood_distri": "negative_binomial",
    },
    {
        "output_key": "infection_deathsXall",
        "years": deaths_times,
        "values": deaths_counts,
        "loglikelihood_distri": "negative_binomial",
    }
]

MULTIPLIERS = {}

# __________  For the grid-based calibration approach
# define a grid of parameter values. The posterior probability will be evaluated at each node
par_grid = [
    {"param_name": "contact_rate", 'lower': .01, 'upper': .02, 'n': 6},
    {"param_name": "start_time", 'lower': 0., 'upper': 50., 'n': 11}
]


def run_gbr_calibration_chain(max_seconds: int, run_id: int):
    # run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='grid_based',
    #                       _grid_info=par_grid, _run_extra_scenarios=False, _multipliers=MULTIPLIERS)
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
                          _run_extra_scenarios=False, _multipliers=MULTIPLIERS)


if __name__ == "__main__":
    run_gbr_calibration_chain(15 * 60 * 60, 1)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
