import os
import pandas as pd

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS, PHASE_2_START_TIME
from apps.covid_19.mixing_optimisation.mixing_opti import MODES, DURATIONS
from apps.covid_19.mixing_optimisation.utils import get_scenario_mapping_reverse
from autumn.constants import BASE_PATH
from autumn.db.load import load_uncertainty_table
from apps.covid_19.mixing_optimisation.utils import get_country_population_size


FIGURE_PATH = os.path.join(BASE_PATH, "apps", "covid_19", "mixing_optimisation",
                           "outputs", "plots", "outputs", "figures", "tables")

DATA_PATH = os.path.join(BASE_PATH, "apps", "covid_19", "mixing_optimisation",
                         "outputs", "pbi_databases", "calibration_and_scenarios", "full_immunity")


def main():
    uncertainty_dfs = {}
    for country in OPTI_REGIONS:
        dir_path = os.path.join(DATA_PATH, country)
        uncertainty_dfs[country] = load_uncertainty_table(dir_path)

    for mode in MODES:
        make_main_outputs_tables(mode, uncertainty_dfs, per_capita=True)


def get_uncertainty_cell_value(uncertainty_df, output, mode, duration, per_capita=False, population=None):
    # output is in ["deaths_before", "deaths_unmitigated", "deaths_opti_deaths", "deaths_opti_yoll",
    #                "yoll_before", "yoll_unmitigated", "yoll_opti_deaths", "yoll_opti_yoll"]

    if "deaths_" in output:
        type = "accum_deaths"
    elif "yoll_" in output:
        type = "accum_years_of_life_lost"
    else:
        type = "proportion_seropositive"

    mask_output = uncertainty_df["type"] == type
    output_df = uncertainty_df[mask_output]

    if "_yoll" in output:
        objective = "yoll"
    else:
        objective = "deaths"

    if "unmitigated" in output:
        sc_idx = get_scenario_mapping_reverse(None, None, None)
    elif "_before" in output:
        sc_idx = 0
    else:
        sc_idx = get_scenario_mapping_reverse(mode, duration, objective)

    mask_scenario = output_df["scenario"] == sc_idx
    output_df = output_df[mask_scenario]

    time_read = PHASE_2_START_TIME if sc_idx == 0 else max(output_df["time"])

    mask_time = output_df["time"] == time_read
    output_df = output_df[mask_time]

    mask_025 = output_df["quantile"] == 0.025
    mask_50 = output_df["quantile"] == 0.5
    mask_975 = output_df["quantile"] == 0.975

    if per_capita:
        multiplier = {"accum_deaths": 1.e6 / population, "accum_years_of_life_lost": 1.e4 / population,
                      "proportion_seropositive": 100}
        rounding = {"accum_deaths": 0, "accum_years_of_life_lost": 0,
                    "proportion_seropositive": 0}
    if not per_capita:
        multiplier = {"accum_deaths": 1.0 / 1000.0, "accum_years_of_life_lost": 1.0 / 1000.0,
                      "proportion_seropositive": 100}
        rounding = {"accum_deaths": 1, "accum_years_of_life_lost": 0,
                    "proportion_seropositive": 0}

    # read the percentile
    median = round(multiplier[type] * float(output_df[mask_50]["value"]), rounding[type])
    lower = round(multiplier[type] * float(output_df[mask_025]["value"]), rounding[type])
    upper = round(multiplier[type] * float(output_df[mask_975]["value"]), rounding[type])

    if rounding[type] == 0:
        median = int(median)
        lower = int(lower)
        upper = int(upper)

    cell_content = f"{median} ({lower}-{upper})"
    return cell_content


def make_main_outputs_tables(mode, uncertainty_dfs, per_capita=False):
    """
    This now combines Table 1 and Table 2
    """
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

        for duration in DURATIONS:
            i_row += 1
            row_as_list = [country]
            for output in [c for c in column_names if c != "country"]:
                print(output)
                row_as_list.append(
                    get_uncertainty_cell_value(uncertainty_df, output, mode, duration, per_capita, population)
                )
            table.loc[i_row] = row_as_list

    filename = f"output_table_{mode}_per_capita.csv" if per_capita else f"output_table_{mode}.csv"
    file_path = os.path.join(FIGURE_PATH, filename)
    table.to_csv(file_path)


def print_who_deaths_per_capita():
    deaths_thousands = {
        'belgium': 10.2,
        'france': 31.7,
        'italy': 35.9,
        'spain': 32.4,
        'sweden': 5.9,
        'united-kingdom': 42.1
    }
    for country in ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]:
        country_name = country.title() if country != "united-kingdom" else "United Kingdom"
        pop = get_country_population_size(country_name)
        print(int(deaths_thousands[country]*1000/pop*1.e6))


if __name__ == "__main__":
    main()

    # print_who_deaths_per_capita()
