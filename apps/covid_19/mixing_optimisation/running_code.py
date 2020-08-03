from apps.covid_19.mixing_optimisation.utils import prepare_table_of_param_sets
from apps.covid_19.mixing_optimisation.mixing_opti import run_sensitivity_perturbations

if __name__ == "__main__":
    calib_paths = {
        # "france": "../../../data/outputs/calibrate/covid_19/france/UsingHosp-2020-07-30",
        # "belgium": "../../../data/outputs/calibrate/covid_19/belgium/Revised-2020-07-18",
        # "spain": "../../../data/outputs/calibrate/covid_19/spain/Revised-2020-07-18",
        # "italy": "../../../data/outputs/calibrate/covid_19/italy/Revised-2020-07-18",
        "sweden": "../../../data/outputs/calibrate/covid_19/sweden/UsingHosp-2020-07-30",
        # "united-kingdom": "../../../data/outputs/calibrate/covid_19/united-kingdom/Revised-2020-07-18",
    }

    burnin = {
        "france": 0,  
        "belgium": 1500,
        "spain": 1500,
        "italy": 2000,
        "sweden": 0,
        "united-kingdom": 1500,
    }

    # for country in calib_paths:
    #     prepare_table_of_param_sets(calib_paths[country],
    #                                 country,
    #                                 n_samples=2,
    #                                 burn_in=burnin[country])


    run_sensitivity_perturbations('optimisation_outputs/test_outputs/', 'sweden', 2, "by_age", "deaths", tol=.01)
