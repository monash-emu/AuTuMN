from apps.covid_19.calibration.base import run_calibration_chain

country = "malaysia"

PAR_PRIORS = [
    {'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [0.015, 0.025]},
    {'param_name': 'start_time', 'distribution': 'uniform', 'distri_params': [15., 25.]},
]

# notification data, provided by the country
data_times = [63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0]
case_counts = [7, 14, 5, 28, 10, 6, 18, 12, 20, 9, 45, 35, 190, 125, 120, 117, 110, 130, 153, 123, 212, 106, 172, 235, 130, 159, 150, 156, 140, 142, 208, 217, 150, 179, 131, 170, 156, 109, 118, 184, 153, 134, 170, 85, 110, 69, 54, 84, 36, 57, 50, 71, 88, 51, 38, 40, 31, 94, 57, 69, 105, 122, 55, 30, 45, 39]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": data_times,
        "values": case_counts,
        "loglikelihood_distri": "poisson",
    }
]

# __________  For the grid-based calibration approach
# define a grid of parameter values. The posterior probability will be evaluated at each node
par_grid = [
    {"param_name": "contact_rate", 'lower': .015, 'upper': .025, 'n': 6},
    {"param_name": "start_time", 'lower': 15., 'upper': 25., 'n': 11},
]


def run_mys_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='grid_based',
                          _grid_info=par_grid, _run_extra_scenarios=False)
    # run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
    #                       _run_extra_scenarios=False)


if __name__ == "__main__":
    run_mys_calibration_chain(2 * 60 * 60, 0)  # first argument only relevant for autumn_mcmc mode
