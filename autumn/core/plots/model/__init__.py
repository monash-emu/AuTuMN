import logging
import os

import pandas as pd

from autumn.core import db
from autumn.core.plots.model import plots
from autumn.core.plots.plotter import FilePlotter
from autumn.core.db.process import select_outputs_from_candidates, target_to_series
from autumn.core.plots.utils import REF_DATE

from . import plots

logger = logging.getLogger(__name__)


def plot_post_full_run(targets: dict, mcmc_dir: str, plot_dir: str, candidates_df: pd.DataFrame):
    logger.info(f"Plotting {mcmc_dir} into {plot_dir}")
    plotter = FilePlotter(plot_dir, targets)
    mcmc_tables = db.load.load_mcmc_tables(mcmc_dir)

    derived_output_tables = db.load.load_derived_output_tables(mcmc_dir)

    logger.info("Plotting full-run candidates")
    subplotter = _get_sub_plotter(plot_dir, "full-run-candidates")
    for target in targets.values():
        output_name = target["output_key"]
        # need to bypass the differential output targets because these outputs are not computed yet.
        if output_name.startswith("rel_diff") or output_name.startswith("abs_diff"):
            continue
        logger.info("Plotting candidate selection for output %s", output_name)
        outputs = select_outputs_from_candidates(
            output_name, derived_output_tables, candidates_df, REF_DATE
        )
        target_series = target_to_series(target, REF_DATE)
        plots.plot_candidates_for_output(subplotter, output_name, outputs, target_series)


def _get_sub_plotter(plot_dir: str, subplot_dirname: str):
    subplot_dir = os.path.join(plot_dir, subplot_dirname)
    os.makedirs(subplot_dir, exist_ok=True)
    return FilePlotter(subplot_dir, {})
