from apps.covid_19.mixing_optimisation.utils import prepare_table_of_param_sets


if __name__ == "__main__":
    calib_paths = {
        "france": "../../../data/outputs/calibrate/covid_19/france/c555531b-2020-07-17",
        "belgium": "../../../data/outputs/calibrate/covid_19/belgium/ef2ee497-2020-07-17",
        # "spain": "../../../data/outputs/calibrate/covid_19/spain/2cfd3183-2020-07-17",
        "italy": "../../../data/outputs/calibrate/covid_19/italy/7e60b0ba-2020-07-17",
        # "sweden": "../../../data/outputs/calibrate/covid_19/",
        # "united-kingdom": "../../../data/outputs/calibrate/covid_19/",
    }

    burnin = {
        "france": 1505,
        "belgium": 1305,
        "spain":None,
        "italy":1995,
        "sweden":None ,
        "united-kingdom": None,
    }

    for country in calib_paths:
        prepare_table_of_param_sets(calib_paths[country],
                                    country,
                                    n_samples=100,
                                    burn_in=burnin[country])
