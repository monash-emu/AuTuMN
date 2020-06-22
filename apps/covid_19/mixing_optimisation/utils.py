from apps.covid_19.john_hopkins import download_jh_data, read_john_hopkins_data_from_csv
from autumn.plots.streamlit.run_mcmc_plots import load_mcmc_tables
from autumn.plots.plots import _overwrite_non_accepted_mcmc_runs
import pandas as pd
import os


def get_prior_distributions_for_opti():
    prior_list = [
        {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.015, 0.050],},
        {"param_name": "start_time", "distribution": "uniform", "distri_params": [0., 30.],},
        {
            "param_name": "compartment_periods_calculated.incubation.total_period",
            "distribution": "gamma",
            "distri_mean": 5.0,
            "distri_ci": [3.0, 7.0],
        },
        {
            "param_name": "compartment_periods_calculated.total_infectious.total_period",
            "distribution": "gamma",
            "distri_mean": 7.0,
            "distri_ci": [4.5, 9.5],
        },
        {
            "param_name": "tv_detection_b",  # shape parameter
            "distribution": "beta",
            "distri_mean": 0.075,
            "distri_ci": [0.05, 0.1],
        },
        {
            "param_name": "tv_detection_c",  # inflection point
            "distribution": "gamma",
            "distri_mean": 80.0,
            "distri_ci": [40.0, 120.0],
        },
        {
            "param_name": "prop_detected_among_symptomatic",  # upper asymptote
            "distribution": "beta",
            "distri_mean": 0.5,
            "distri_ci": [0.2, 0.8],
        },
        {
            "param_name": "icu_prop",
            "distribution": "beta",
            "distri_mean": 0.25,
            "distri_ci": [0.15, 0.35],
        },
        # parameters to derive age-specific IFRs
        {
            "param_name": "ifr_double_exp_model_params.a",
            "distribution": "uniform",
            "distri_params": [7., 14.],
        },
        {
            "param_name": "ifr_double_exp_model_params.last_representative_age",
            "distribution": "uniform",
            "distri_params": [75., 85.],
        },
        # Add negative binomial over-dispersion parameters
        {
            "param_name": "notifications_dispersion_param",
            "distribution": "uniform",
            "distri_params": [0.1, 5.0],
        },
        {
            "param_name": "infection_deathsXall_dispersion_param",
            "distribution": "uniform",
            "distri_params": [0.1, 5.0],
        },
    ]

    return prior_list


def get_target_outputs_for_opti(country, data_start_time=22, update_jh_data=False):
    """
    Automatically creates the calibration target list for a country in the context of the opti problem
    :param country: country name
    :param data_start_time: the desired starting point for the extracted data
    :param update_jh_data: whether to download the data from Johns Hopkins Github again
    :return:
    """
    jh_start_time = 22  # actual start time in JH csv files
    assert data_start_time >= jh_start_time

    if update_jh_data:
        download_jh_data()

    output_mapping = {"confirmed": "notifications", "deaths": "infection_deathsXall"}

    target_outputs = []
    for variable in ["confirmed", "deaths"]:
        data = read_john_hopkins_data_from_csv(variable, country)
        data = [max(d, 0) for d in data]
        times = [jh_start_time + i for i in range(len(data))]
        nb_elements_to_drop = data_start_time - jh_start_time
        target_outputs.append(
            {
                "output_key": output_mapping[variable],
                "years": times[nb_elements_to_drop:],
                "values": data[nb_elements_to_drop:],
                "loglikelihood_distri": "negative_binomial",
            }
        )

    return target_outputs


def extract_n_mcmc_samples(calibration_output_path, n_samples=100, burn_in=500):
    mcmc_tables = load_mcmc_tables(calibration_output_path)
    col_names = mcmc_tables[0].columns

    for col_name in [c for c in col_names if c not in ["accept"]]:
        _overwrite_non_accepted_mcmc_runs(mcmc_tables, col_name)

    for i, mcmc_table in enumerate(mcmc_tables):
        mcmc_tables[i] = mcmc_table.iloc[burn_in:]

    combined_burned_samples = pd.concat(mcmc_tables)
    nb_rows = combined_burned_samples.shape[0]
    thining_jump = int(nb_rows / n_samples)

    selected_indices = [i * thining_jump for i in range(n_samples)]

    thined_samples = combined_burned_samples.iloc[selected_indices, ]

    return thined_samples.drop(['Scenario', 'accept'], axis=1)


def prepare_table_of_param_sets(calibration_output_path, n_samples=100, burn_in=500):
    samples = extract_n_mcmc_samples(calibration_output_path, n_samples, burn_in)
    for i in range(16):
        samples["best_x" + str(i)] = ""
    samples["best_deaths"] = ""
    samples["all_vars_to_1_deaths"] = ""
    samples["best_p_immune"] = ""
    samples["all_vars_to_1_p_immune"] = ""

    output_file = os.path.join(calibration_output_path, "opti_sample.csv")
    samples.to_csv(output_file, index=False)


# prepare_table_of_param_sets("../../../data/covid_united-kingdom/calibration-covid_united-kingdom-c4c45836-20-06-2020")

