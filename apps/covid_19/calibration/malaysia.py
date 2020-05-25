from apps.covid_19.calibration.base import run_calibration_chain

country = "malaysia"

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
        "param_name": "tv_detection_sigma",
        "distribution": "beta",
        "distri_mean": .25,
        "distri_ci": [.1, .4]
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
        "distri_mean": .7,
        "distri_ci": [.6, .9]
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
    },
    {
        "param_name": "prevXlateXclinical_icuXamong_dispersion_param",
        "distribution": "uniform",
        "distri_params": [.1, 5.]
    }
]

# notification data, provided by the country
notification_times = [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141]
notification_counts = [7, 14, 5, 28, 10, 6, 18, 12, 20, 9, 45, 35, 190, 125, 120, 117, 110, 130, 153, 123, 212, 106, 172, 235, 130, 159, 150, 156, 140, 142, 208, 217, 150, 179, 131, 170, 156, 109, 118, 184, 153, 134, 170, 85, 110, 69, 54, 84, 36, 57, 50, 71, 88, 51, 38, 40, 31, 94, 57, 69, 105, 122, 55, 30, 45, 39, 68, 54, 67, 70, 16, 37, 40, 36, 17, 22, 47, 37, 31]

# ICU data (prev / million pop), provided by the country
icu_times = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141]
icu_counts = [2.0, 3.0, 4.0, 5.0, 9.0, 12.0, 12.0, 15.0, 20.0, 26.0, 37.0, 46.0, 57.0, 64.0, 45.0, 45.0, 54.0, 73.0, 73.0, 94.0, 94.0, 102.0, 105.0, 108.0, 99.0, 99.0, 102.0, 92.0, 76.0, 72.0, 69.0, 72.0, 66.0, 66.0, 60.0, 56.0, 56.0, 51.0, 49.0, 46.0, 45.0, 43.0, 43.0, 42.0, 41.0, 36.0, 36.0, 37.0, 36.0, 40.0, 36.0, 37.0, 31.0, 27.0, 28.0, 24.0, 22.0, 19.0, 18.0, 18.0, 18.0, 20.0, 16.0, 16.0, 16.0, 14.0, 13.0, 13.0, 13.0, 11.0, 11.0]


TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": notification_times,
        "values": notification_counts,
        "loglikelihood_distri": "negative_binomial",
    },
    {
        "output_key": "prevXlateXclinical_icuXamong",
        "years": icu_times,
        "values": icu_counts,
        "loglikelihood_distri": "negative_binomial",
    }
]

MULTIPLIERS = {'prevXlateXclinical_icuXamong': 32364904.}  # to get absolute pop size instead of proportion

# __________  For the grid-based calibration approach
# define a grid of parameter values. The posterior probability will be evaluated at each node
par_grid = [
    {"param_name": "contact_rate", 'lower': .01, 'upper': .02, 'n': 6},
    {"param_name": "start_time", 'lower': 0., 'upper': 50., 'n': 11}
]


def run_mys_calibration_chain(max_seconds: int, run_id: int):
    # run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='grid_based',
    #                       _grid_info=par_grid, _run_extra_scenarios=False, _multipliers=MULTIPLIERS)
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
                          _run_extra_scenarios=False, _multipliers=MULTIPLIERS)


if __name__ == "__main__":
    run_mys_calibration_chain(15 * 60 * 60, 0)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
