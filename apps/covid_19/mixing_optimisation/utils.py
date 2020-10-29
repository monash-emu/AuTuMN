from autumn import db, inputs
import pandas as pd
import os

import numpy as np
from autumn.db import Database

HOSPITAL_DATA_DIR = os.path.join("hospitalisation_data")
country_mapping = {"united-kingdom": "The United Kingdom"}


def get_prior_distributions_for_opti():
    prior_list = [
        {
            "param_name": "contact_rate",
            "distribution": "uniform",
            "distri_params": [0.02, 0.06],
        },
        {
            "param_name": "time.start",
            "distribution": "uniform",
            "distri_params": [0.0, 40.0],
        },
        {
            "param_name": "sojourn.compartment_periods_calculated.exposed.total_period",
            "distribution": "trunc_normal",
            "distri_params": [5.5, 0.97],
            "trunc_range": [1.0, np.inf],
        },
        {
            "param_name": "sojourn.compartment_periods_calculated.active.total_period",
            "distribution": "trunc_normal",
            "distri_params": [6.5, 0.77],
            "trunc_range": [1.0, np.inf],
        },
        {
            "param_name": "infection_fatality.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 3.8],  # 3.8 to match the highest value found in Levin et al.
        },
        {
            "param_name": "case_detection.start_value",
            "distribution": "uniform",
            "distri_params": [0.0, 0.30],
        },
        {
            "param_name": "case_detection.maximum_gradient",
            "distribution": "uniform",
            "distri_params": [0.03, 0.15],
        },
        {
            "param_name": "case_detection.max_change_time",
            "distribution": "uniform",
            "distri_params": [100, 250],
        },
        {
            "param_name": "case_detection.end_value",
            "distribution": "uniform",
            "distri_params": [0.10, 0.99],
        },
        # {
        #     "param_name": "clinical_stratification.icu_prop",
        #     "distribution": "uniform",
        #     "distri_params": [0.15, 0.20],
        # },
        # vary hospital durations
        # {
        #     "param_name": "sojourn.compartment_periods.hospital_late",
        #     "distribution": "trunc_normal",
        #     "distri_params": [18.4, 2.0],
        #     "trunc_range": [3.0, np.inf],
        # },
        # {
        #     "param_name": "sojourn.compartment_periods.icu_late",
        #     "distribution": "trunc_normal",
        #     "distri_params": [10.8, 4.0],
        #     "trunc_range": [3.0, np.inf],
        # },
        # vary symptomatic and hospitalised proportions
        {
            "param_name": "clinical_stratification.props.symptomatic.multiplier",
            "distribution": "uniform",
            "distri_params": [0.6, 1.4],
        },
        {
            "param_name": "clinical_stratification.props.hospital.multiplier",
            "distribution": "uniform",
            "distri_params": [0.5, 1.5],
        },
        # Micro-distancing
        {
            "param_name": "mobility.microdistancing.behaviour.parameters.c",
            "distribution": "uniform",
            "distri_params": [60, 130],
        },
        {
            "param_name": "mobility.microdistancing.behaviour.parameters.upper_asymptote",
            "distribution": "uniform",
            "distri_params": [0.25, 0.80],
        },
        {
            "param_name": "mobility.microdistancing.behaviour_adjuster.parameters.c",
            "distribution": "uniform",
            "distri_params": [130, 250],
        },
        {
            "param_name": "mobility.microdistancing.behaviour_adjuster.parameters.sigma",
            "distribution": "uniform",
            "distri_params": [0.4, 1.0],
        },
        {
            "param_name": "elderly_mixing_reduction.relative_reduction",
            "distribution": "uniform",
            "distri_params": [0., 0.5],
        }
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


def get_country_population_size(country):
    iso_3 = inputs.demography.queries.get_iso3_from_country_name(country)
    return sum(inputs.get_population_by_agegroup(
            ["0"], iso_3, None, year=2020
        ))


"""
To create main table outputs
"""


def read_percentile_from_pbi_table(
    calibration_output_path, scenario=0, quantile=0.5, time=0.0, type="notifications"
):
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
    mask = [
        m_1 and m_2 and m_3 and m_4 for (m_1, m_2, m_3, m_4) in zip(mask_1, mask_2, mask_3, mask_4)
    ]
    value = float(unc_table[mask]["value"])

    return value


# FIXME: load_derived_output_tables is broken or not found
def read_cumulative_output_from_output_table(
    calibration_output_path, scenario, time_range, model_output
):
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
        sum_by_run = d_t.groupby(["idx"])[model_output].sum()
        cumulative_values += list(sum_by_run)

    return cumulative_values


def get_uncertainty_cell_value(uncertainty_df, output, config, mode):
    # output is in ["deaths_before", "deaths_unmitigated", "deaths_opti_deaths", "deaths_opti_yoll",
    #                "yoll_before", "yoll_unmitigated", "yoll_opti_deaths", "yoll_opti_yoll"]

    if mode == "by_location" and "unmitigated" in output:
        return ""

    if "deaths_" in output:
        type = "accum_deaths"
    else:
        type = "accum_years_of_life_lost"
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

    mask_time = output_df["time"] == max(output_df["time"])
    output_df = output_df[mask_time]

    mask_025 = output_df["quantile"] == 0.025
    mask_50 = output_df["quantile"] == 0.5
    mask_975 = output_df["quantile"] == 0.975

    multiplier = {"accum_deaths": 1.0 / 1000.0, "accum_years_of_life_lost": 1.0 / 1000.0}
    rounding = {"accum_deaths": 1, "accum_years_of_life_lost": 0}

    # read the percentile
    median = round(multiplier[type] * float(output_df[mask_50]["value"]), rounding[type])
    lower = round(multiplier[type] * float(output_df[mask_025]["value"]), rounding[type])
    upper = round(multiplier[type] * float(output_df[mask_975]["value"]), rounding[type])

    cell_content = str(median) + " (" + str(lower) + "-" + str(upper) + ")"

    return cell_content


def make_main_outputs_tables(mode):
    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    column_names = [
        "country",
        "deaths_before",
        "deaths_unmitigated",
        "deaths_opti_deaths",
        "deaths_opti_yoll",
        "yoll_before",
        "yoll_unmitigated",
        "yoll_opti_deaths",
        "yoll_opti_yoll",
    ]

    for immunity in ["fully_immune"]:  # , "partial_immune"]:
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

        table.to_csv(
            "../../../data/pbi_outputs_for_opti/"
            + mode
            + "/"
            + immunity
            + "/output_table_"
            + immunity
            + "_"
            + mode
            + ".csv"
        )

