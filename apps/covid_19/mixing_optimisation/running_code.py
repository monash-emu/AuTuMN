from apps.covid_19.mixing_optimisation.utils import prepare_table_of_param_sets
from apps.covid_19.mixing_optimisation.mixing_opti import run_sensitivity_perturbations
from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS

if __name__ == "__main__":

    common_folder_name = 'Final-2020-08-04'

    burnin = {
        "france": 0,  
        "belgium": 0,
        "spain": 0,
        "italy": 0,
        "sweden": 0,
        "united-kingdom": 0,
    }

    for country in OPTI_REGIONS:
        path = "../../../data/outputs/calibrate/covid_19/" + country + "/" + common_folder_name

        prepare_table_of_param_sets(path,
                                    country,
                                    n_samples=2,
                                    burn_in=burnin[country])


    # run_sensitivity_perturbations('optimisation_outputs/test_outputs/', 'sweden', 2, "by_age", "deaths", tol=.01)
