import logging
import os

import pandas as pd

from autumn.tools import db
from autumn.tools.plots.model import plots
from autumn.tools.plots.plotter import FilePlotter
from autumn.tools.db.process import select_outputs_from_candidates, target_to_series
from autumn.tools.plots.utils import REF_DATE

from . import plots

logger = logging.getLogger(__name__)


def plot_post_full_run(targets: dict, mcmc_dir: str, plot_dir: str, candidates_df: pd.DataFrame):
    logger.info(f"Plotting {mcmc_dir} into {plot_dir}")
    plotter = FilePlotter(plot_dir, targets)
    mcmc_tables = db.load.load_mcmc_tables(mcmc_dir)

    pass


def _get_sub_plotter(plot_dir: str, subplot_dirname: str):
    subplot_dir = os.path.join(plot_dir, subplot_dirname)
    os.makedirs(subplot_dir, exist_ok=True)
    return FilePlotter(subplot_dir, {})
