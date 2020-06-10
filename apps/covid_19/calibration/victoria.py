from apps.covid_19.calibration.base import run_calibration_chain
from numpy import linspace

country = "victoria"

# _______ Define the priors
PAR_PRIORS = [
    # Extra parameter for the negative binomial likelihood
    {
        "param_name": "notifications_dispersion_param",
        "distribution": "uniform",
        "distri_params": [.1, 5.]
    },
    # Transmission parameter
    {
        "param_name": "contact_rate",
        "distribution": "uniform",
        "distri_params": [.025, .08]
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
    # FIXME: the parameter value is changed automatically during model initialisation !!! Cant be included atm
    # {
    #     "param_name": "compartment_periods_infectious",
    #     "distribution": "gamma",
    #     "distri_mean": 7.,
    #     "distri_ci": [5., 9.]
    # },
    {
        "param_name": "young_reduced_susceptibility",
        "distribution": "beta",
        "distri_mean": .5,
        "distri_ci": [.4, .6]
    },
    # Programmatic parameters
    {
        "param_name": "prop_detected_among_symptomatic",
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
        "distribution": "beta",
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
data_times = [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161]
case_counts = [1, 1, 1, 3, 1, 0, 4, 8, 7, 9, 11, 11, 22, 17, 15, 14, 21, 34, 29, 31, 46, 34, 36, 39, 11, 20, 21, 16, 13, 10, 8, 14, 2, 0, 4, 2, 4, 4, 5, 8, 1, 3, 0, 1, 4, 3, 2, 1, 3, 2, 5, 1, 9, 10, 19, 13, 8, 15, 13, 12, 5, 1, 16, 5, 6, 12, 6, 2, 4, 5, 5, 5, 7, 6, 2, 0, 2, 6, 7, 3, 6, 5, 3, 8, 2, 2, 1, 0, 1, 0, 0]

# _______ Print targets to plot to be added to plots.yml file
# target_to_plots = {"notifications": {"times": data_times, "values": [[d] for d in case_counts]}}
# print(target_to_plots)

TARGET_OUTPUTS = [
    {
        "output_key": "notifications",
        "years": data_times,
        "values": case_counts,
        "loglikelihood_distri": "negative_binomial",
    }
]

# _______ Create the calibration function


def run_vic_calibration_chain(max_seconds: int, run_id: int):
    run_calibration_chain(
        max_seconds, run_id, country, PAR_PRIORS, TARGET_OUTPUTS, mode="autumn_mcmc",
        _run_extra_scenarios=False
    )


if __name__ == "__main__":
    run_vic_calibration_chain(30, 0)
