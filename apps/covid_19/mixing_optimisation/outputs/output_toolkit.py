import pandas as pd
import os
from autumn import db

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.mixing_opti import MODES, DURATIONS, OBJECTIVES, run_root_model, objective_function
from autumn.constants import BASE_PATH
from apps.covid_19.mixing_optimisation.utils import get_scenario_mapping, get_scenario_mapping_reverse


def load_derived_output(database_path, output_name):
    df = db.load.load_derived_output_tables(database_path, output_name)[0]
    return df


def read_percentile_from_pbi_table(
    database_path, scenario=0, quantile=0.5, time=0.0, output="notifications"
):
    unc_table = db.load.load_uncertainty_table(database_path)

    mask_1 = unc_table["Scenario"] == "S_" + str(scenario)
    mask_2 = unc_table["quantile"] == quantile
    mask_3 = unc_table["time"] == time
    mask_4 = unc_table["type"] == output
    mask = [
        m_1 and m_2 and m_3 and m_4 for (m_1, m_2, m_3, m_4) in zip(mask_1, mask_2, mask_3, mask_4)
    ]
    value = float(unc_table[mask]["value"])

    return value


def read_cumulative_output_from_output_table(
    calibration_output_path, scenario, time_range, model_output
):
    derived_output_tables = load_derived_output(calibration_output_path, model_output)

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


def get_uncertainty_cell_value(uncertainty_df, output, mode, duration, opti_objective):
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

    print("!!!!!   Need to check the code below this point  !!!!!!!!")
    exit()

    if "_yoll" in output:
        objective = "yoll"
    else:
        objective = "deaths"

    if "unmitigated" in output:
        scenario = get_scenario_mapping_reverse(None, None, None)
    elif "_before" in output:
        scenario = 0
    else:
        scenario = get_scenario_mapping_reverse(mode, duration, objective)

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
            database = db.Database(db_path)
            uncertainty_df = database.query("uncertainty")

            for duration in DURATIONS:
                i_row += 1
                row_as_list = [country]
                for output in [c for c in column_names if c != "country"]:
                    print(output)
                    row_as_list.append(
                        get_uncertainty_cell_value(uncertainty_df, output, mode, duration, objective)
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

