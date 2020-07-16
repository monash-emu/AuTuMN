from apps.covid_19.mixing_optimisation.utils import prepare_table_of_param_sets


if __name__ == "__main__":
    calib_paths = {
        # "france": "../../../data/outputs/calibrate/covid_19/",
        "belgium": "../../../data/outputs/calibrate/covid_19/belgium/1768138e-2020-07-17",
        "spain": "../../../data/outputs/calibrate/covid_19/spain/2cfd3183-2020-07-17",
        # "italy": "../../../data/outputs/calibrate/covid_19/",
        # "sweden": "../../../data/outputs/calibrate/covid_19/",
        # "united-kingdom": "../../../data/outputs/calibrate/covid_19/",
    }

    for country in calib_paths:
        prepare_table_of_param_sets(calib_paths[country],
                                    country,
                                    n_samples=100,
                                    burn_in=100)
