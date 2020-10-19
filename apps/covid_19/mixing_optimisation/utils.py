from autumn import db
from autumn.plots.calibration.plots import _overwrite_non_accepted_mcmc_runs
from autumn.calibration.utils import add_dispersion_param_prior_for_gaussian

import pandas as pd
import os
from matplotlib import pyplot as plt
from autumn.curve import scale_up_function, tanh_based_scaleup
import numpy as np
import copy
from autumn.db import Database

HOSPITAL_DATA_DIR = os.path.join("hospitalisation_data")
country_mapping = {"united-kingdom": "The United Kingdom"}


def get_prior_distributions_for_opti():
    prior_list = [
        {"param_name": "contact_rate", "distribution": "uniform", "distri_params": [0.02, 0.06],},
        {"param_name": "time.start", "distribution": "uniform", "distri_params": [0.0, 40.0],},
        {
            "param_name": "sojourn.compartment_periods_calculated.exposed.total_period",
            "distribution": "uniform",
            "distri_params": [3, 7],
        },
        {
            "param_name": "sojourn.compartment_periods_calculated.active.total_period",
            "distribution": "uniform",
            "distri_params": [4, 10],
        },
        {
            "param_name": "infection_fatality.multiplier",
            "distribution": "uniform",
            "distri_params": [.8, 1.2],
        },
        {
            "param_name": "case_detection.start_value",
            "distribution": "uniform",
            "distri_params": [0.0, 0.20],
        },
        {
            "param_name": "case_detection.maximum_gradient",
            "distribution": "uniform",
            "distri_params": [0.05, 0.1],
        },
        {
            "param_name": "case_detection.max_change_time",
            "distribution": "uniform",
            "distri_params": [100, 250],
        },
        {
            "param_name": "case_detection.end_value",
            "distribution": "uniform",
            "distri_params": [0.10, 0.90],
        },
        # {
        #     "param_name": "clinical_stratification.icu_prop",
        #     "distribution": "uniform",
        #     "distri_params": [0.15, 0.20],
        # },
        # vary hospital durations
        {
            "param_name": "sojourn.compartment_periods.hospital_late",
            "distribution": "uniform",
            "distri_params": [17.7, 20.4],
        },
        {
            "param_name": "sojourn.compartment_periods.icu_late",
            "distribution": "uniform",
            "distri_params": [9.0, 13.0],
        },
        # vary symptomatic and hospitalised proportions
        {
            "param_name": "clinical_stratification.props.symptomatic.multiplier",
            "distribution": "uniform",
            "distri_params": [0.6, 1.4],
        },
        {
            "param_name": "clinical_stratification.props.hospital.multiplier",
            "distribution": "uniform",
            "distri_params": [0.6, 1.4],
        },
        # Micro-distancing
        {
            "param_name": "mobility.microdistancing.behaviour.parameters.c",
            "distribution": "uniform",
            "distri_params": [80, 130],
        },
        {
            "param_name": "mobility.microdistancing.behaviour.parameters.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [.25, .70],
        },
        {
            "param_name": "mobility.microdistancing.behaviour_adjuster.parameters.c",
            "distribution": "uniform",
            "distri_params": [130, 250],
        },
        {
            "param_name": "mobility.microdistancing.behaviour_adjuster.parameters.sigma",
            "distribution": "uniform",
            "distri_params": [.4, 1.],
        },
    ]
    return prior_list


def get_weekly_summed_targets(times, values):
    assert len(times) == len(values), "times and values must have the same length"
    assert len(times) >= 7, "number of time points must be greater than 7 to compute weekly data"

    t_low = min(times)
    t_max = max(times)

    w_times = []
    w_values = []
    while t_low < t_max:
        this_week_indices = [i for i, t in enumerate(times) if t_low <= t < t_low + 7]
        this_week_times = [times[i] for i in this_week_indices]
        this_week_values = [values[i] for i in this_week_indices]
        w_times.append(round(np.mean(this_week_times)))
        w_values.append(np.mean(this_week_values))
        t_low += 7

    return w_times, w_values


def combine_and_burn_samples(calibration_output_path, burn_in=500):
    mcmc_tables = db.load.load_mcmc_tables(calibration_output_path)
    col_names = mcmc_tables[0].columns

    for col_name in [c for c in col_names if c not in ["accept"]]:
        _overwrite_non_accepted_mcmc_runs(mcmc_tables, col_name)

    for i, mcmc_table in enumerate(mcmc_tables):
        mcmc_tables[i] = mcmc_table.iloc[burn_in:]

    return pd.concat(mcmc_tables)


def extract_n_mcmc_samples(calibration_output_path, n_samples=100, burn_in=500, include_mle=True):
    combined_burned_samples = combine_and_burn_samples(calibration_output_path, burn_in)
    nb_rows = combined_burned_samples.shape[0]
    n_extracted = n_samples - 1 if include_mle else n_samples
    thining_jump = int(nb_rows / n_extracted)
    selected_indices = [i * thining_jump for i in range(n_extracted)]

    thined_samples = combined_burned_samples.iloc[
        selected_indices,
    ]

    if include_mle:
        # concatanate maximum likelihood row
        mle_row = combined_burned_samples.sort_values(by="loglikelihood", ascending=False).iloc[
            0, :
        ]
        thined_samples = thined_samples.append(mle_row)
    return thined_samples.drop(["Scenario", "accept"], axis=1)


def prepare_table_of_param_sets(calibration_output_path, country_name, n_samples=100, burn_in=500):
    samples = extract_n_mcmc_samples(calibration_output_path, n_samples, burn_in, include_mle=True)
    for i in range(16):
        samples["best_x" + str(i)] = ""
    samples["best_deaths"] = ""
    samples["all_vars_to_1_deaths"] = ""
    samples["best_yoll"] = ""
    samples["all_vars_to_1_yoll"] = ""
    samples["best_p_immune"] = ""
    samples["all_vars_to_1_p_immune"] = ""

    output_file = os.path.join(
        "calibrated_param_sets", country_name + "_calibrated_params" + ".csv"
    )
    samples.to_csv(output_file, index=False)


############# To create main table outputs
def read_percentile_from_pbi_table(calibration_output_path, scenario=0, quantile=0.5, time=0., type='notifications'):
    db_path = [
        os.path.join(calibration_output_path, f)
        for f in os.listdir(calibration_output_path)
        if f.startswith("powerbi")
    ][0]

    db = Database(db_path)
    unc_table = db.query("uncertainty")

    mask_1 = unc_table["Scenario"] == "S_" + str(scenario)
    mask_2 = unc_table["quantile"] == quantile
    mask_3 = unc_table["time"] == time
    mask_4 = unc_table["type"] == type
    mask = [m_1 and m_2 and m_3 and m_4 for (m_1, m_2, m_3, m_4) in zip(mask_1, mask_2, mask_3, mask_4)]
    value = float(unc_table[mask]["value"])

    return value


# FIXME: load_derived_output_tables is broken or not found
def read_cumulative_output_from_output_table(calibration_output_path, scenario, time_range, model_output):
    derived_output_tables = load_derived_output_tables(calibration_output_path)

    cumulative_values = []
    for d_t in derived_output_tables:
        mask_1 = d_t["Scenario"] == "S_" + str(scenario)
        mask_2 = d_t["times"] >= time_range[0]
        if time_range[1] == "end":
            mask = [m_1 and m_2 for (m_1, m_2) in zip(mask_1, mask_2)]
        else:
            mask_3 = d_t["times"] <= time_range[1]
            mask = [m_1 and m_2 and m_3 for (m_1, m_2, m_3) in zip(mask_1, mask_2, mask_3)]
        d_t = d_t[mask]  # a 2d array
        sum_by_run = d_t.groupby(['idx'])[model_output].sum()
        cumulative_values += list(sum_by_run)

    return cumulative_values


def get_uncertainty_cell_value(uncertainty_df, output, config, mode):
    # output is in ["deaths_before", "deaths_unmitigated", "deaths_opti_deaths", "deaths_opti_yoll",
    #                "yoll_before", "yoll_unmitigated", "yoll_opti_deaths", "yoll_opti_yoll"]

    if mode == 'by_location' and "unmitigated" in output:
        return ""

    if 'deaths_' in output:
        type = 'accum_deaths'
    else:
        type = 'accum_years_of_life_lost'
    mask_output = uncertainty_df["type"] == type
    output_df = uncertainty_df[mask_output]

    if "_yoll" in output:
        objective = "yoll"
    else:
        objective = "deaths"

    scenario_mapping = {
        1: mode + "_2_deaths",
        2: mode + "_2_yoll",
        3: mode + "_3_deaths",
        4: mode + "_3_yoll",
        5: "unmitigated",  # not used, just for completeness
    }

    full_tag = mode + "_" + str(config) + "_" + objective
    if "unmitigated" in output:
        scenario = 5
    elif "_before" in output:
        scenario = 0
    else:
        scenario = [key for key, val in scenario_mapping.items() if val == full_tag][0]

    mask_scenario = output_df["Scenario"] == "S_" + str(scenario)
    output_df = output_df[mask_scenario]

    mask_time = output_df["time"] == max(output_df['time'])
    output_df = output_df[mask_time]

    mask_025 = output_df["quantile"] == 0.025
    mask_50 = output_df["quantile"] == 0.5
    mask_975 = output_df["quantile"] == 0.975

    multiplier = {
        "accum_deaths": 1. / 1000.,
        "accum_years_of_life_lost": 1. / 1000.
    }
    rounding = {
        "accum_deaths": 1,
        "accum_years_of_life_lost": 0
    }

    # read the percentile
    median = round(multiplier[type] * float(output_df[mask_50]["value"]), rounding[type])
    lower = round(multiplier[type] * float(output_df[mask_025]["value"]), rounding[type])
    upper = round(multiplier[type] * float(output_df[mask_975]["value"]), rounding[type])

    cell_content = str(median) + " (" + str(lower) + "-" + str(upper) + ")"

    return cell_content


def make_main_outputs_tables(mode):
    countries = ['belgium', 'france', 'italy', 'spain', 'sweden', 'united-kingdom']
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    column_names = ["country",
                    "deaths_before", "deaths_unmitigated", "deaths_opti_deaths", "deaths_opti_yoll",
                     "yoll_before", "yoll_unmitigated", "yoll_opti_deaths", "yoll_opti_yoll"
                    ]

    for immunity in ["fully_immune"]: # , "partial_immune"]:
        table = pd.DataFrame(columns=column_names)
        i_row = -1
        for i, country in enumerate(countries):
            pbi_outputs_dir = "../../../data/pbi_outputs_for_opti/" + mode + "/" + immunity
            dir_content = os.listdir(pbi_outputs_dir)
            for f in dir_content:
                if country in f:
                    db_name = f

            db_path = os.path.join(pbi_outputs_dir, db_name)
            db = Database(db_path)
            uncertainty_df = db.query("uncertainty")

            for config in [2, 3]:
                i_row += 1
                row_as_list = [country]
                for output in [c for c in column_names if c != "country"]:
                    print(output)
                    row_as_list.append(
                        get_uncertainty_cell_value(uncertainty_df, output, config, mode)
                    )

                table.loc[i_row] = row_as_list

        table.to_csv("../../../data/pbi_outputs_for_opti/" + mode + "/" + immunity + "/output_table_" + immunity + "_" +
                     mode + ".csv")


###########################################
def plot_mixing_params_over_time(mixing_params, npi_effectiveness_range):

    titles = {
        "home": "Household",
        "work": "Workplace",
        "school": "School",
        "other_locations": "Other locations",
    }
    y_labs = {"home": "h", "work": "w", "school": "s", "other_locations": "l"}
    date_ticks = {
        32: "1/2",
        47: "16/2",
        61: "1/3",
        76: "16/3",
        92: "1/4",
        107: "16/4",
        122: "1/5",
        137: "16/5",
        152: "1/6",
    }
    # use italics for y_labs
    for key in y_labs:
        y_labs[key] = "$\it{" + y_labs[key] + "}$(t)"

    plt.style.use("ggplot")
    for i_loc, location in enumerate(
        [
            loc
            for loc in ["home", "other_locations", "school", "work"]
            if loc + "_times" in mixing_params
        ]
    ):
        plt.figure(i_loc)
        x = list(np.linspace(0.0, 152.0, num=10000))
        y = []
        for indice_npi_effect_range in [0, 1]:
            npi_effect = {
                key: val[indice_npi_effect_range] for key, val in npi_effectiveness_range.items()
            }

            modified_mixing_params = apply_npi_effectiveness(
                copy.deepcopy(mixing_params), npi_effect
            )

            location_adjustment = scale_up_function(
                modified_mixing_params[location + "_times"],
                modified_mixing_params[location + "_values"],
                method=4,
            )

            _y = [location_adjustment(t) for t in x]
            y.append(_y)
            plt.plot(x, _y, color="navy")

        plt.fill_between(x, y[0], y[1], color="cornflowerblue")
        plt.xlim((30.0, 152.0))
        plt.ylim((0, 1.1))

        plt.xticks(list(date_ticks.keys()), list(date_ticks.values()))
        plt.xlabel("Date in 2020")
        plt.ylabel(y_labs[location])
        plt.title(titles[location])
        plt.savefig("mixing_adjustment_" + location + ".png")


def apply_npi_effectiveness(mixing_params, npi_effectiveness):
    """
    Adjust the mixing parameters according by scaling them according to NPI effectiveness
    :param mixing_params: dict
        Instructions for how the mixing matrices should vary with time, including for the baseline
    :param npi_effectiveness: dict
        Instructions for how the input mixing parameters should be adjusted to account for the level of
        NPI effectiveness. mixing_params are unchanged if all NPI effectiveness values are 1.
    :return: dict
        Adjusted instructions
    """
    for location in [
        loc
        for loc in ["home", "other_locations", "school", "work"]
        if loc + "_times" in mixing_params
    ]:
        if location in npi_effectiveness:
            mixing_params[location + "_values"] = [
                1.0 - (1.0 - val) * npi_effectiveness[location]
                for val in mixing_params[location + "_values"]
            ]

    return mixing_params


def get_posterior_percentiles_time_variant_profile(
    calibration_path, function="detection", burn_in=0
):
    """
    :param calibration_path: string
    :param function: only 'detection' for now
    :param burn_in: integer
    """
    combined_burned_samples = combine_and_burn_samples(calibration_path, burn_in)
    calculated_times = range(200)
    store_matrix = np.zeros((len(calculated_times), combined_burned_samples.shape[0]))
    if function == "detection":
        i = 0
        for index, row in combined_burned_samples.iterrows():
            my_func = tanh_based_scaleup(
                row["case_detection.maximum_gradient"], row["case_detection.max_change_time"], 0.0,
            )
            detect_vals = [row["case_detection.end_value"] * my_func(t) for t in calculated_times]
            store_matrix[:, i] = detect_vals
            i += 1
    perc = np.percentile(store_matrix, [2.5, 25, 50, 75, 97.5], axis=1)
    calculated_times = np.array([calculated_times])
    perc = np.concatenate((calculated_times, perc))
    np.savetxt(function + ".csv", perc, delimiter=",")


# if __name__ == "__main__":
# get_posterior_percentiles_time_variant_profile("../../../data/covid_united-kingdom/calibration-covid_united-kingdom-c4c45836-20-06-2020", 'detection', 1000)

# mixing_pars = {'other_locations_times': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171], 'other_locations_values': [0.896666666666667, 0.933333333333333, 1.03, 1.01, 1, 0.993333333333334, 1.00333333333333, 0.973333333333333, 1, 0.98, 1.01333333333333, 1, 1.01, 1.00666666666667, 1, 1.04666666666667, 1.02666666666667, 1.01333333333333, 1.01, 0.993333333333333, 1.04, 1.00333333333333, 1.02666666666667, 1.00333333333333, 0.996666666666667, 0.996666666666667, 1.00666666666667, 1, 0.94, 0.933333333333333, 1.00333333333333, 0.93, 0.89, 0.886666666666667, 0.806666666666667, 0.683333333333333, 0.613333333333333, 0.646666666666667, 0.47, 0.443333333333333, 0.426666666666667, 0.413333333333333, 0.346666666666667, 0.313333333333333, 0.403333333333333, 0.413333333333333, 0.4, 0.4, 0.406666666666667, 0.37, 0.356666666666667, 0.413333333333333, 0.43, 0.433333333333333, 0.463333333333333, 0.38, 0.39, 0.256666666666667, 0.343333333333333, 0.446666666666667, 0.44, 0.433333333333333, 0.413333333333333, 0.37, 0.38, 0.436666666666667, 0.443333333333333, 0.443333333333333, 0.446666666666667, 0.45, 0.416666666666667, 0.403333333333333, 0.453333333333333, 0.406666666666667, 0.426666666666667, 0.43, 0.44, 0.43, 0.4, 0.466666666666667, 0.47, 0.49, 0.52, 0.43, 0.436666666666667, 0.383333333333333, 0.48, 0.49, 0.49, 0.503333333333333, 0.5, 0.476666666666667, 0.473333333333333, 0.526666666666667, 0.536666666666667, 0.546666666666667, 0.54, 0.523333333333333, 0.486666666666667, 0.506666666666667, 0.51, 0.57, 0.563333333333333, 0.576666666666667, 0.576666666666667, 0.553333333333333, 0.563333333333333, 0.6, 0.583333333333333, 0.54, 0.553333333333333, 0.546666666666667, 0.503333333333333, 0.53, 0.576666666666667, 0.573333333333333, 0.54, 0.553333333333333, 0.55, 0.563333333333333, 0.57, 0.633333333333333, 0.62, 0.606666666666667, 0.583333333333333, 0.613333333333333], 'work_times': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164], 'work_values': [0.96, 0.97, 0.86, 0.86, 0.86, 0.86, 0.85, 0.98, 0.99, 0.98, 1.01, 1.01, 1, 0.99, 1, 1.01, 1.01, 1.01, 1.01, 1, 1.01, 1.01, 1.01, 1, 1, 1, 0.99, 0.96, 0.98, 0.96, 0.91, 0.84, 0.77, 0.73, 0.71, 0.75, 0.7, 0.55, 0.42, 0.36, 0.34, 0.34, 0.43, 0.45, 0.31, 0.3, 0.31, 0.3, 0.31, 0.43, 0.46, 0.3, 0.3, 0.3, 0.3, 0.2, 0.43, 0.42, 0.17, 0.3, 0.31, 0.31, 0.32, 0.46, 0.51, 0.33, 0.33, 0.34, 0.34, 0.35, 0.49, 0.52, 0.35, 0.34, 0.34, 0.34, 0.36, 0.51, 0.54, 0.36, 0.36, 0.37, 0.37, 0.23, 0.52, 0.55, 0.38, 0.38, 0.39, 0.39, 0.4, 0.57, 0.61, 0.41, 0.41, 0.41, 0.41, 0.42, 0.59, 0.64, 0.23, 0.41, 0.42, 0.42, 0.44, 0.69, 0.74, 0.47, 0.46, 0.46, 0.46, 0.47, 0.67, 0.72, 0.48, 0.48, 0.48, 0.48, 0.49], 'school_times': [78, 80], 'school_values': [1.0, 0.1]}
# npi_effectiveness_range = {
#     'other_locations': [.4,.9],
#     'work': [1,1],
#     'school': [1,1],
# }
# plot_mixing_params_over_time(mixing_pars, npi_effectiveness_range)
#
# out_dir = "../../../data/outputs/calibrate/covid_19/france/Final-2020-08-04"
#

# if __name__ == "__main__":
#     make_main_outputs_tables("by_location")
