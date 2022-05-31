import logging
import os

from autumn.core.db import Database
from autumn.core.plots.plotter import FilePlotter

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
    for output_name in outputs:
        this_output_dir = os.path.join(output_dir, output_name)
        os.makedirs(this_output_dir, exist_ok=True)
        plotter = FilePlotter(this_output_dir, targets)
        scenario_idxs = uncertainty_df["scenario"].unique().tolist()
        for scenario_idx in scenario_idxs:
            logger.info(
                "Plotting uncertainty for output %s, scenario %s", output_name, scenario_idx
            )
            if scenario_idx == 0:
                # Just plot the baseline scenario for the full time period.
                scenario_idxs = [0]
                x_low = 0
            else:
                # Plot the baseline compared ot the scenario, but only for the time period
                # where the scenario is active.
                scenario_idxs = [0, scenario_idx]
                mask = uncertainty_df["scenario"] == scenario_idx
                x_low = uncertainty_df[mask]["time"].min()

            plots.plot_timeseries_with_uncertainty(
                plotter,
                uncertainty_df,
                output_name,
                scenario_idxs,
                targets,
                x_low=x_low,
            )
