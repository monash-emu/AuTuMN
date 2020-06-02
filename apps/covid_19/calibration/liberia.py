from apps.covid_19.calibration.base import run_calibration_chain, get_priors_and_targets

country = "liberia"

# CHANGE ALL THE VALUES BELOW

PAR_PRIORS = [
    {
        'param_name': 'contact_rate',
        'distribution': 'uniform',
        'distri_params': [0.010, 0.05]
    },
    {
        'param_name': 'start_time',
        'distribution': 'uniform',
        'distri_params': [0., 40.]
    },
    # Add extra params for negative binomial likelihood
    {
        'param_name': 'infection_deathsXall_dispersion_param',
        'distribution': 'uniform',
        'distri_params': [.1, 5.]
    },
    {
        "param_name": "compartment_periods_calculated.incubation.total_period",
        "distribution": "gamma",
        "distri_mean": 5.,
        "distri_ci": [4., 6.]
    },
]

# Death counts:
death_times = [30, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153]
death_values = [1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 3, 1, 0, 4, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 4, 4, 3, 3, 4, 5, 2, 9, 6, 10, 0, 1, 0, 3, 4, 7, 8, 8]

TARGET_OUTPUTS = [
    {
        "output_key": "infection_deathsXall",
        "years": death_times,
        "values": death_values,
        "loglikelihood_distri": "negative_binomial",
    }
]


def run_phl_calibration_chain(max_seconds: int, run_id: int):
    # run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='grid_based',
    #                       _grid_info=par_grid, _run_extra_scenarios=False, _multipliers=MULTIPLIERS)
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode='autumn_mcmc',
                          _run_extra_scenarios=False)


if __name__ == "__main__":
    run_phl_calibration_chain(15 * 60 * 60, 0)  # first argument only relevant for autumn_mcmc mode (time limit in seconds)
