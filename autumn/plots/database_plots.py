"""
Utilities to plot data from existing databases.
"""
import os
import yaml
import logging

from autumn.tool_kit.params import load_targets

from . import plots
from autumn.db.models import load_mcmc_tables, load_derived_output_tables
from .plotter import FilePlotter

APP_DIRNAMES = ["covid_", "marshall_islands", "mongolia", "dummy"]

logger = logging.getLogger(__name__)


def plot_from_mcmc_databases(app_name: str, region_name: str, mcmc_dir: str, plot_dir: str):
    logger.info(f"Plotting {mcmc_dir} into {plot_dir}")
    targets = load_targets(app_name, region_name)
    plotter = FilePlotter(plot_dir, targets)
    mcmc_tables = load_mcmc_tables(mcmc_dir)
    derived_output_tables = load_derived_output_tables(mcmc_dir)
    burn_in = 0
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_options = [c for c in mcmc_tables[0].columns if c not in non_param_cols]

    logger.info("Plotting loglikelihood traces")
    num_iters = len(mcmc_tables[0])
    plots.plot_loglikelihood_trace(plotter, mcmc_tables, burn_in)
    plots.plot_burn_in(plotter, num_iters, burn_in)

    logger.info("Plotting posterior distributions")
    num_bins = 16
    subplotter = _get_sub_plotter(plot_dir, "posteriors")
    for chosen_param in param_options:
        plots.plot_posterior(subplotter, mcmc_tables, chosen_param, num_bins)

    logger.info("Plotting loglikelihood vs params")
    subplotter = _get_sub_plotter(plot_dir, "params-vs-loglikelihood")
    for chosen_param in param_options:
        plots.plot_loglikelihood_vs_parameter(subplotter, mcmc_tables, chosen_param, burn_in)

    logger.info("Plotting parameter traces")
    subplotter = _get_sub_plotter(plot_dir, "params-traces")
    for chosen_param in param_options:
        plots.plot_mcmc_parameter_trace(subplotter, mcmc_tables, chosen_param)

    logger.info("Plotting calibration fits")
    subplotter = _get_sub_plotter(plot_dir, "calibration-fit")
    for target in targets.values():
        output_name = target["output_key"]
        logger.info("Plotting calibration fit for output %s", output_name)
        outputs, best_chain_index = plots.sample_outputs_for_calibration_fit(
            output_name, mcmc_tables, derived_output_tables
        )
        plots.plot_calibration_fit(
            subplotter, output_name, outputs, best_chain_index, targets, is_logscale=True
        )
        plots.plot_calibration_fit(
            subplotter, output_name, outputs, best_chain_index, targets, is_logscale=False
        )

    logger.info("MCMC plots complete")


def _get_sub_plotter(plot_dir: str, subplot_dirname: str):
    subplot_dir = os.path.join(plot_dir, subplot_dirname)
    os.makedirs(subplot_dir, exist_ok=True)
    return FilePlotter(subplot_dir, {})
