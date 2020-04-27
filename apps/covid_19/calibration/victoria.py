from apps.covid_19.calibration.base import run_calibration_chain
from numpy import linspace

country = "victoria"

# #######  all cases
data_times = [
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    83,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
]

case_counts = [
    1,
    1,
    3,
    3,
    3,
    6,
    9,
    13,
    8,
    14,
    23,
    27,
    29,
    98,
    25,
    44,
    96,
    86,
    66,
    64,
    85,
    77,
    48,
    48,
    83,
    54,
    60,
    24,
    34,
    11,
    21,
    24,
    19,
    15,
    27,
    7,
]

target_to_plots = {"notifications": {"times": data_times, "values": [[d] for d in case_counts]}}
print(target_to_plots)
PAR_PRIORS = [
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.1, 0.5]},
    {"param_name": "prop_isolated_among_symptomatic", "distribution": "uniform", "distri_params": [0.8, .9]},
    {"param_name": "non_sympt_infect_multiplier", "distribution": "uniform", "distri_params": [0.4, 0.6]},

    {"param_name": "compartment_period_exposed", "distribution": "uniform", "distri_params": [2., 4.]},
    {"param_name": "compartment_period_late", "distribution": "uniform", "distri_params": [4., 7.]},

    {"param_name": "npi_effectiveness_school", "distribution": "uniform", "distri_params": [0.8, 1.]},
    {"param_name": "npi_effectiveness_work", "distribution": "uniform", "distri_params": [0.8, 1.]},
    {"param_name": "npi_effectiveness_other_locations", "distribution": "uniform", "distri_params": [0.8, 1.]},

    {"param_name": "n_imported_cases_final", "distribution": "uniform", "distri_params": [0., 2.]},
]

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": data_times,
        "values": case_counts,
        "loglikelihood_distri": "poisson",
    }
]


def run_vic_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc")


run_vic_calibration_chain(30, 1)
