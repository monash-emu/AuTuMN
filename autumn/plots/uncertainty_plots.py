import os
import logging
from typing import List
from autumn.tool_kit import export_mcmc_quantiles

from autumn.plots import plots
from autumn.plots.plotter import FilePlotter
from autumn.db import Database
from apps.covid_19.plots import load_plot_config

logger = logging.getLogger(__name__)


def plot_timeseries_with_uncertainty_for_powerbi(
    region_name: str, powerbi_db_path: str, output_dir: str
):
    """
    works on powerbi version
    Assumes a COVID model.
    TODO: Unify PowerBI and local version
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_config = load_plot_config(region_name)
    db = Database(powerbi_db_path)
    uncertainty_df = db.query("uncertainty")
    outputs = uncertainty_df["type"].unique().tolist()
    quantile_vals = uncertainty_df["quantile"].unique().tolist()
    for output_name in outputs:
        this_output_dir = os.path.join(output_dir, output_name)
        os.makedirs(this_output_dir, exist_ok=True)
        plotter = FilePlotter(this_output_dir, plot_config["translations"])
        mask = uncertainty_df["type"] == output_name
        output_df = uncertainty_df[mask]
        scenarios = output_df.Scenario.unique().tolist()
        for scenario in scenarios:
            mask = output_df["Scenario"] == scenario
            scenario_df = output_df[mask]
            quantiles = {}
            for q in quantile_vals:
                mask = scenario_df["quantile"] == q
                quantiles[q] = scenario_df[mask]["value"].tolist()

            times = scenario_df.time.unique()
            logger.info("Plotting uncertainty for output %s, scenario %s", output_name, scenario)
            plots.plot_timeseries_with_uncertainty_for_powerbi(
                plotter, output_name, scenario, quantiles, times, plot_config
            )


def plot_timeseries_with_uncertainty(
    path_to_percentile_outputs: str, output_names: List[str], scenario_list=[0], burn_in=0,
):
    """
    works on local version
    TODO: Deprecate
    """
    percentile_db_path = os.path.join(
        path_to_percentile_outputs, "mcmc_percentiles_burned_" + str(burn_in) + ".db"
    )
    if not os.path.exists(percentile_db_path):
        export_mcmc_quantiles(path_to_percentile_outputs, output_names, burn_in=burn_in)

    plotter = FilePlotter(path_to_percentile_outputs, {})
    for output_name in output_names:
        plots.plot_timeseries_with_uncertainty(
            plotter, path_to_percentile_outputs, output_name, scenario_list, burn_in=burn_in,
        )


if __name__ == "__main__":
    plot_timeseries_with_uncertainty(
        "../../data/covid_malaysia/calibration-covid_malaysia-fake-18-05-2020",
        ["incidence", "cumulate_incidence"],
    )
