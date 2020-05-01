from apps.covid_19.calibration.base import run_calibration_chain
from numpy import linspace

country = "victoria"

# _______ Define the priors
PAR_PRIORS = [
    # Transmission parameter
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [0.1, 0.4]
    },
    # Parameters defining the natural history of COVID-19
    {
        "param_name": "non_sympt_infect_multiplier",
        "distribution": "beta",
        "distri_mean": .5,
        "distri_ci": [.4, .6]
    },
    {
        "param_name": "compartment_periods_incubation",
        "distribution": "gamma",
        "distri_mean": 5.,
        "distri_ci": [3., 7.]
    },
    {
        "param_name": "compartment_periods_infectious",
        "distribution": "gamma",
        "distri_mean": 7.,
        "distri_ci": [5., 9.]
    },
    {
        "param_name": "young_reduced_susceptibility",
        "distribution": "beta",
        "distri_mean": .5,
        "distri_ci": [.4, .6]
    },
    # Programmatic parameters
    {
        "param_name": "prop_isolated_among_symptomatic",
        "distribution": "beta",
        "distri_mean": .85,
        "distri_ci": [.8, .9]
    },
    # Parameter to vary the mixing adjustment in other_locations
    {
        "param_name": "npi_effectiveness_other_locations",
        "distribution": "beta",
        "distri_mean": .9,
        "distri_ci": [.8, .99]
    },
    # Parameters related to case importation
    {
        "param_name": "n_imported_cases_final",
        "distribution": "gamma",
        "distri_mean": 1.,
        "distri_ci": [.1, 2.]
    },
    {
        "param_name": "self_isolation_effect",
        "distribution": "gamma",
        "distri_mean": .67,
        "distri_ci": [.55, .80],
        "distri_ci_width": .95
    },
    {
        "param_name": "enforced_isolation_effect",
        "distribution": "beta",
        "distri_mean": .90,
        "distri_ci": [.80, .99]
    }
]

# _______ Define the calibration targets
# Local transmission data
data_times = [71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]
case_counts = [1.0, 1.0, 3.0, 1.0, 0.0, 4.0, 8.0, 7.0, 9.0, 12.0, 13.0, 22.0, 19.0, 18.0, 16.0, 22.0, 40.0, 33.0, 33.0, 49.0, 34.0, 36.0, 43.0, 16.0, 23.0, 23.0, 18.0, 15.0, 10.0, 7.0, 15.0, 2.0, 1.0, 3.0, 4, 2, 4, 4, 5, 9, 1, 3, 0, 1, 4, 2, 1, 1, 1]

# _______ Print targets to plot to be added to plots.yml file
# target_to_plots = {"notifications": {"times": data_times, "values": [[d] for d in case_counts]}}
# print(target_to_plots)

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": data_times,
        "values": case_counts,
        "loglikelihood_distri": "poisson",
    }
]

# _______ Create the calibration function


def run_vic_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(
        max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc"
    )


run_vic_calibration_chain(30, 1)
