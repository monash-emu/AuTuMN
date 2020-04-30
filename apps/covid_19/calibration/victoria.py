from apps.covid_19.calibration.base import run_calibration_chain
from numpy import linspace

country = "victoria"

# #######  all cases
# data_times = [
#     67,
#     68,
#     69,
#     70,
#     71,
#     72,
#     73,
#     74,
#     75,
#     76,
#     77,
#     78,
#     79,
#     80,
#     81,
#     82,
#     83,
#     84,
#     85,
#     86,
#     87,
#     88,
#     89,
#     90,
#     91,
#     92,
#     93,
#     94,
#     95,
#     96,
#     97,
#     98,
#     99,
#     100,
#     101,
#     102,
# ]
#
# case_counts = [
#     1,
#     1,
#     3,
#     3,
#     3,
#     6,
#     9,
#     13,
#     8,
#     14,
#     23,
#     27,
#     29,
#     98,
#     25,
#     44,
#     96,
#     86,
#     66,
#     64,
#     85,
#     77,
#     48,
#     48,
#     83,
#     54,
#     60,
#     24,
#     34,
#     11,
#     21,
#     24,
#     19,
#     15,
#     27,
#     7,
# ]


# #######  local cases only
data_times = [71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
case_counts = [1.0, 1.0, 3.0, 1.0, 0.0, 4.0, 8.0, 7.0, 9.0, 12.0, 13.0, 22.0, 19.0, 18.0, 16.0, 22.0, 40.0, 33.0, 33.0, 49.0, 34.0, 36.0, 43.0, 16.0, 23.0, 23.0, 18.0, 15.0, 10.0, 7.0, 15.0, 2.0, 1.0, 3.0, 4, 2, 4, 4, 5, 9, 1, 3, 0, 1, 4, 2, 1, 1, 1]

# target_to_plots = {"notifications": {"times": data_times, "values": [[d] for d in case_counts]}}
# print(target_to_plots)

PAR_PRIORS = [
    # Transmission parameter
    {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.1, 0.4]},
    # Parameters defining the natural history of COVID-19
    {
        "param_name": "non_sympt_infect_multiplier",
        "distribution": "uniform",
        "distri_params": [0.4, 0.6],
    },
    {
        "param_name": "compartment_periods_incubation",
        "distribution": "uniform",
        "distri_params": [2.0, 6.0],
    },
    {
        "param_name": "compartment_periods_late",
        "distribution": "uniform",
        "distri_params": [4.0, 7.0],
    },
    {
        "param_name": "young_reduced_susceptibility",
        "distribution": "uniform",
        "distri_params": [.4, .6],
    },
    # Programmatic parameters
    {
        "param_name": "prop_isolated_among_symptomatic",
        "distribution": "uniform",
        "distri_params": [0.8, 0.9],
    },
    # Parameter to vary the mixing adjustment in other_locations
    {
        "param_name": "npi_effectiveness_other_locations",
        "distribution": "uniform",
        "distri_params": [0.8, 1.0],
    },
    # Parameters related to case importation
    {
        "param_name": "n_imported_cases_final",
        "distribution": "uniform",
        "distri_params": [0.0, 2.0],
    },
    {
        "param_name": "self_isolation_effect",
        "distribution": "uniform",
        "distri_params": [.4, .8],
    },
    {
        "param_name": "enforced_isolation_effect",
        "distribution": "uniform",
        "distri_params": [.8, 1.],
    }

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
    run_calibration_chain(
        max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc"
    )
