import os

from autumn.projects.tuberculosis.marshall_islands.outputs.utils import (
    make_output_directories,
)
from autumn.tools.db.load import load_uncertainty_table
from autumn.settings import BASE_PATH

FIGURE_PATH = os.path.join(
    BASE_PATH,
    "apps",
    "tuberculosis",
    "regions",
    "marshall_islands",
    "outputs",
    "figures",
    "output_numbers",
)

DATA_PATH = os.path.join(
    BASE_PATH, "apps", "tuberculosis", "regions", "marshall_islands", "outputs", "pbi_databases"
)


def main():
    make_output_directories(FIGURE_PATH)
    uncertainty_df = load_uncertainty_table(DATA_PATH)

    # End TB Targets
    print("End TB Targets")
    for output in ["incidence", "mortality"]:
        print_median_and_ci(uncertainty_df, output, 2015, 0)

    # main outputs, national level
    for output in ["incidence", "mortality"]:
        for scenario in [1, 0]:
            print_median_and_ci(uncertainty_df, output, 2020, scenario)

    # regional level
    for region in ["majuro", "ebeye"]:
        for year in [2020, 2050]:
            for scenario in [1, 0]:
                print_median_and_ci(uncertainty_df, f"incidenceXlocation_{region}", year, scenario)

    # diabetes scenarios
    print()
    print("Diabetes scenarios")
    for scenario in [9, 10]:
        print_median_and_ci(uncertainty_df, "incidence", 2050, scenario)

    # PT in all contacts
    print()
    print("PT in all contacts")
    for output in ["incidence", "mortality"]:
        print_median_and_ci(uncertainty_df, output, 2050, 11)


def get_output_number(uncertainty_df, output, time, scenario_idx, quantile=0.5):
    mask = (
        (uncertainty_df["type"] == output)
        & (uncertainty_df["scenario"] == scenario_idx)
        & (uncertainty_df["time"] == time)
        & (uncertainty_df["quantile"] == quantile)
    )
    value = uncertainty_df[mask]["value"].tolist()
    return value[0]


def print_median_and_ci(uncertainty_df, output, time, scenario_idx):
    median_val = get_output_number(uncertainty_df, output, time, scenario_idx, 0.5)
    lowest_val = get_output_number(uncertainty_df, output, time, scenario_idx, 0.025)
    highest_val = get_output_number(uncertainty_df, output, time, scenario_idx, 0.975)

    out_str = f"{median_val} ({lowest_val} - {highest_val})"
    print("******************")
    print(f"{output} in {time} for Scenario {scenario_idx}:")
    print(out_str)


if __name__ == "__main__":
    main()
