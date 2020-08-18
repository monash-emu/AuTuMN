import os
import logging
from typing import List
from autumn.tool_kit.params import load_targets

from autumn.plots import plots
from autumn.plots.plotter import FilePlotter
from autumn.db import Database

logger = logging.getLogger(__name__)


def plot_timeseries_with_uncertainty(region_name: str, powerbi_db_path: str, output_dir: str):
    """
    works on powerbi version
    Assumes a COVID model.
    """
    os.makedirs(output_dir, exist_ok=True)
    targets = load_targets("covid_19", region_name)
    db = Database(powerbi_db_path)
    uncertainty_df = db.query("uncertainty")
    outputs = uncertainty_df["type"].unique().tolist()
    quantile_vals = uncertainty_df["quantile"].unique().tolist()
    for output_name in outputs:
        this_output_dir = os.path.join(output_dir, output_name)
        os.makedirs(this_output_dir, exist_ok=True)
        plotter = FilePlotter(this_output_dir, targets)
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
            plots.plot_timeseries_with_uncertainty(
                plotter, output_name, scenario, quantiles, times, targets
            )
