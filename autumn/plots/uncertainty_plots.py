import os
from typing import List
from autumn.tool_kit import export_mcmc_quantiles

from autumn.plots import plots
from autumn.plots.plotter import FilePlotter


def plot_timeseries_with_uncertainty(
    path_to_percentile_outputs: str, output_names: List[str], scenario_list=[0], burn_in=0
):
    percentile_db_path = os.path.join(
        path_to_percentile_outputs, "mcmc_percentiles_burned_" + str(burn_in) + ".db"
    )
    if not os.path.exists(percentile_db_path):
        export_mcmc_quantiles(path_to_percentile_outputs, output_names, burn_in=burn_in)

    plotter = FilePlotter(path_to_percentile_outputs, {})
    for output_name in output_names:
        plots.plot_timeseries_with_uncertainty(
            plotter, path_to_percentile_outputs, output_name, scenario_list, burn_in=burn_in
        )


if __name__ == "__main__":
    plot_timeseries_with_uncertainty(
        "../../data/covid_malaysia/calibration-covid_malaysia-fake-18-05-2020",
        ["incidence", "cumulate_incidence"],
    )
