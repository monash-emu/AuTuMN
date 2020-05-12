from apps.covid_19.calibration.base import run_calibration_chain

country = "malaysia"

PAR_PRIORS = [
    {'param_name': 'contact_rate', 'distribution': 'uniform', 'distri_params': [0.015, 0.025]},
    {'param_name': 'start_time', 'distribution': 'uniform', 'distri_params': [15., 25.]},
]

# notification data, provided by the country
data_times = [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
case_counts = [7, 14, 5, 28, 10, 6, 18, 12, 20, 9, 45, 35, 190, 125, 120, 117, 110, 130, 153, 123, 212, 106, 172, 235, 130, 159, 150, 156, 140, 142, 208, 217, 150, 179, 131, 170, 156, 109, 118, 184, 153, 134, 170, 85, 110, 69, 54, 84, 36, 57, 50, 71, 88, 51, 38, 40, 31, 94, 57, 69, 105, 122, 55, 30, 45, 39, 68, 54, 67, 70]

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
    {"param_name": "contact_rate", 'lower': .010, 'upper': .022, 'n': 7},
    {"param_name": "start_time", 'lower': 0., 'upper': 20., 'n': 11},
]


def run_mys_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='grid_based',
                          _grid_info=par_grid, _run_extra_scenarios=False)
    # run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
    #                       _run_extra_scenarios=False)


if __name__ == "__main__":
    run_mys_calibration_chain(2 * 60 * 60, 2)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
