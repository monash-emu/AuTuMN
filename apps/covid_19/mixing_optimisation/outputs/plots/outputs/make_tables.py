import os
import pandas as pd

from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.mixing_opti import MODES, DURATIONS
from apps.covid_19.mixing_optimisation.utils import get_scenario_mapping_reverse
from autumn.constants import BASE_PATH
from autumn.db.load import load_uncertainty_table


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
        make_main_outputs_tables(mode, uncertainty_dfs)


def get_uncertainty_cell_value(uncertainty_df, output, mode, duration):
    # output is in ["deaths_before", "deaths_unmitigated", "deaths_opti_deaths", "deaths_opti_yoll",
    #                "yoll_before", "yoll_unmitigated", "yoll_opti_deaths", "yoll_opti_yoll"]

    if mode == "by_location" and "unmitigated" in output:
        return ""

    if "deaths_" in output:
        type = "proportion_seropositive"  # FIXME "accum_deaths"
    elif "yoll_" in output:
        type = "proportion_seropositive"  # FIXME  "accum_years_of_life_lost"
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

    if sc_idx > 0:
        return "not ready yet"

    mask_scenario = output_df["scenario"] == sc_idx
    output_df = output_df[mask_scenario]

    mask_time = output_df["time"] == max(output_df["time"])
    output_df = output_df[mask_time]

    mask_025 = output_df["quantile"] == 0.025
    mask_50 = output_df["quantile"] == 0.5
    mask_975 = output_df["quantile"] == 0.975

    multiplier = {"accum_deaths": 1.0 / 1000.0, "accum_years_of_life_lost": 1.0 / 1000.0,
                  "proportion_seropositive": 100}
    rounding = {"accum_deaths": 1, "accum_years_of_life_lost": 0,
                "proportion_seropositive": 0}

    # read the percentile
    median = round(multiplier[type] * float(output_df[mask_50]["value"]), rounding[type])
    lower = round(multiplier[type] * float(output_df[mask_025]["value"]), rounding[type])
    upper = round(multiplier[type] * float(output_df[mask_975]["value"]), rounding[type])

    cell_content = f"{median} ({lower}-{upper})"
    return cell_content


def make_main_outputs_tables(mode, uncertainty_dfs):
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
        "sero_opti_deaths",
        "sero_opti_yoll",
    ]

    table = pd.DataFrame(columns=column_names)
    i_row = -1
    for i, country in enumerate(countries):
        uncertainty_df = uncertainty_dfs[country]
        for duration in DURATIONS:
            i_row += 1
            row_as_list = [country]
            for output in [c for c in column_names if c != "country"]:
                print(output)
                row_as_list.append(
                    get_uncertainty_cell_value(uncertainty_df, output, mode, duration)
                )

            table.loc[i_row] = row_as_list

    filename = f"output_table_{mode}.csv"
    file_path = os.path.join(FIGURE_PATH, filename)
    table.to_csv(file_path)


if __name__ == "__main__":
    main()
