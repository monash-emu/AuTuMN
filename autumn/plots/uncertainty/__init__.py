import os
import logging

from autumn.plots.plotter import FilePlotter
from autumn.db import Database

from . import plots

logger = logging.getLogger(__name__)


def plot_uncertainty(targets: dict, powerbi_db_path: str, output_dir: str):
    """
    works on powerbi version
    Assumes a COVID model.
    """
    os.makedirs(output_dir, exist_ok=True)
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
        scenarios = output_df["scenario"].unique().tolist()
        for scenario in scenarios:
            mask = output_df["scenario"] == scenario
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
