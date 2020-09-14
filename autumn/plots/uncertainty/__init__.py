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
    for output_name in outputs:
        this_output_dir = os.path.join(output_dir, output_name)
        os.makedirs(this_output_dir, exist_ok=True)
        plotter = FilePlotter(this_output_dir, targets)
        scenarios = uncertainty_df["scenario"].unique().tolist()
        for scenario in scenarios:
            logger.info("Plotting uncertainty for output %s, scenario %s", output_name, scenario)
            plots.plot_timeseries_with_uncertainty(
                plotter, uncertainty_df, output_name, scenario, targets
            )
