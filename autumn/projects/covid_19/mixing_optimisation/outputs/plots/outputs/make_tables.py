import os

import pandas as pd

from autumn.projects.covid_19.mixing_optimisation.constants import OPTI_REGIONS, PHASE_2_START_TIME
from autumn.projects.covid_19.mixing_optimisation.mixing_opti import DURATIONS, MODES
from autumn.projects.covid_19.mixing_optimisation.utils import (
    get_country_population_size,
    get_scenario_mapping_reverse,
)
from autumn.coreb.load import load_uncertainty_table
from autumn.settings import BASE_PATH

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "plots",
    "outputs",
    "figures",
    "tables",
)

DATA_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "covid_19",
    "mixing_optimisation",
    "outputs",
    "pbi_databases",
    "calibration_and_scenarios",
    "full_immunity",
)


who_deaths = {
    "by_october": {
        "belgium": 10.2,
        "france": 31.7,
        "italy": 35.9,
        "spain": 32.4,
        "sweden": 5.9,
        "united-kingdom": 42.1,
    },
    "by_january": {
        "belgium": 19.6,
        "france": 64.3,
        "italy": 74.2,
        "spain": 51.4,
        "sweden": 9.7,
        "united-kingdom": 73.5,
    },
}


def main():
    uncertainty_dfs = {}
    for country in OPTI_REGIONS:
        dir_path = os.path.join(DATA_PATH, country)
        uncertainty_dfs[country] = load_uncertainty_table(dir_path)

    for per_capita in [False, True]:
        make_main_outputs_tables_new_messaging(uncertainty_dfs, per_capita=per_capita)


def get_quantile(output_df, sc_idx, quantile):

    mask_scenario = output_df["scenario"] == sc_idx
    masked_output_df = output_df[mask_scenario]

    time_read = PHASE_2_START_TIME if sc_idx == 0 else max(masked_output_df["time"])

    mask_time = masked_output_df["time"] == time_read
    masked_output_df = masked_output_df[mask_time]

    mask_quantile = masked_output_df["quantile"] == quantile

    return float(masked_output_df[mask_quantile]["value"])


def get_uncertainty_cell_value(
    country, uncertainty_df, output, mode, duration, per_capita=False, population=None
):

    # output is in ["deaths_before", "deaths_unmitigated", "deaths_opti_deaths", "deaths_opti_yoll",
    #                "yoll_before", "yoll_unmitigated", "yoll_opti_deaths", "yoll_opti_yoll"]

    # return blank if repeat row
    if "_before" in output or "unmitigated" in output or "who_" in output:
        if mode != MODES[0] or duration != DURATIONS[0]:
            return ""

    # return WHO estimate if requested
    if "who_" in output:
        if "_before" in output:
            value = who_deaths["by_october"][country]
        else:
            value = who_deaths["by_january"][country]

        if per_capita:
            country_name = country.title() if country != "united-kingdom" else "United Kingdom"
            pop = get_country_population_size(country_name)
            value *= 1000 / pop * 1.0e6
            value = int(value)

        return value

    if "deaths_" in output:
        type = "accum_deaths"
    elif "yoll_" in output:
        type = "accum_years_of_life_lost"
    else:
        type = "proportion_seropositive"

    mask_output = uncertainty_df["type"] == type
    output_df = uncertainty_df[mask_output]

    if "opti_yoll" in output:
        objective = "yoll"
    else:
        objective = "deaths"

    if "unmitigated" in output:
        sc_idx = get_scenario_mapping_reverse(None, None, None)
    elif "_before" in output:
        sc_idx = 0
    else:
        sc_idx = get_scenario_mapping_reverse(mode, duration, objective)

    val_025 = get_quantile(output_df, sc_idx, 0.025)
    val_50 = get_quantile(output_df, sc_idx, 0.5)
    val_975 = get_quantile(output_df, sc_idx, 0.975)

    if output.startswith("total_"):
        val_025 += get_quantile(output_df, 0, 0.025)
        val_50 += get_quantile(output_df, 0, 0.5)
        val_975 += get_quantile(output_df, 0, 0.975)

    if per_capita:
        multiplier = {
            "accum_deaths": 1.0e6 / population,
            "accum_years_of_life_lost": 1.0e4 / population,
            "proportion_seropositive": 100,
        }
        rounding = {"accum_deaths": 0, "accum_years_of_life_lost": 0, "proportion_seropositive": 0}
    if not per_capita:
        multiplier = {
            "accum_deaths": 1.0 / 1000.0,
            "accum_years_of_life_lost": 1.0 / 1000.0,
            "proportion_seropositive": 100,
        }
        rounding = {"accum_deaths": 1, "accum_years_of_life_lost": 0, "proportion_seropositive": 0}

    # read the percentile
    median = round(multiplier[type] * val_50, rounding[type])
    lower = round(multiplier[type] * val_025, rounding[type])
    upper = round(multiplier[type] * val_975, rounding[type])

    if rounding[type] == 0:
        median = int(median)
        lower = int(lower)
        upper = int(upper)

    cell_content = f"{median} ({lower}-{upper})"
    return cell_content


def make_main_outputs_tables(uncertainty_dfs, per_capita=False):
    """
    This now combines Table 1 and Table 2
    """
    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    column_names = [
        "country",
        "deaths_before",
        "who_before" "deaths_unmitigated",
        "deaths_opti_deaths",
        "deaths_opti_yoll",
        "who_by_jan",
        "yoll_before",
        "yoll_unmitigated",
        "yoll_opti_deaths",
        "yoll_opti_yoll",
        "sero_before",
        "sero_unmitigated",
        "sero_opti_deaths",
        "sero_opti_yoll",
    ]

    table = pd.DataFrame(columns=column_names)
    i_row = -1
    for i, country in enumerate(countries):
        uncertainty_df = uncertainty_dfs[country]
        country_name = country.title() if country != "united-kingdom" else "United Kingdom"
        population = get_country_population_size(country_name) if per_capita else None

        for mode in MODES:
            for duration in DURATIONS:
                i_row += 1
                row_as_list = [country]
                for output in [c for c in column_names if c != "country"]:
                    print(output)
                    row_as_list.append(
                        get_uncertainty_cell_value(
                            country, uncertainty_df, output, mode, duration, per_capita, population
                        )
                    )
                table.loc[i_row] = row_as_list

    filename = f"output_table_per_capita.csv" if per_capita else f"output_table.csv"
    file_path = os.path.join(FIGURE_PATH, filename)
    table.to_csv(file_path)


def print_who_deaths_per_capita(by="october"):

    deaths_thousands = who_deaths[f"by_{by}"]

    for country in ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]:
        country_name = country.title() if country != "united-kingdom" else "United Kingdom"
        pop = get_country_population_size(country_name)
        print(int(deaths_thousands[country] * 1000 / pop * 1.0e6))


def make_main_outputs_tables_new_messaging(uncertainty_dfs, per_capita=False):
    """
    This now combines Table 1 and Table 2
    """
    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    country_names = [c.title() for c in countries]
    country_names[-1] = "United Kingdom"

    column_names = [
        "country",
        "deaths_before",
        "who_before",
        "total_deaths_unmitigated",
        "total_deaths_opti_deaths",
        "total_deaths_opti_yoll",
        "who_by_jan",
        "yoll_before",
        "total_yoll_unmitigated",
        "total_yoll_opti_deaths",
        "total_yoll_opti_yoll",
        "sero_before",
        "sero_unmitigated",
        "sero_opti_deaths",
        "sero_opti_yoll",
    ]

    table = pd.DataFrame(columns=column_names)
    i_row = -1
    for i, country in enumerate(countries):
        uncertainty_df = uncertainty_dfs[country]
        country_name = country.title() if country != "united-kingdom" else "United Kingdom"
        population = get_country_population_size(country_name) if per_capita else None

        for mode in MODES:
            for duration in DURATIONS:
                i_row += 1
                row_as_list = [country]
                for output in [c for c in column_names if c != "country"]:
                    print(output)
                    row_as_list.append(
                        get_uncertainty_cell_value(
                            country, uncertainty_df, output, mode, duration, per_capita, population
                        )
                    )
                table.loc[i_row] = row_as_list

    filename = (
        f"output_table_per_capita_new_messaging.csv"
        if per_capita
        else f"output_table_new_messaging.csv"
    )
    file_path = os.path.join(FIGURE_PATH, filename)
    table.to_csv(file_path)


if __name__ == "__main__":
    main()

    # print_who_deaths_per_capita()
