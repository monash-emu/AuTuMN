"""
Utilities to plot data from existing databases.
"""
import os
import yaml
import logging
from typing import List

from autumn.tool_kit.params import load_targets
from autumn import db
from autumn.plots.plotter import FilePlotter

from . import plots


logger = logging.getLogger(__name__)

PLOT_BURN_IN = 0


def plot_pre_calibration(priors: List[dict], directory: str):
    """
    Make graphs to display prior distributions used in calibration
    """
    logger.info("Plotting prior distributions")
    path = os.path.join(directory, "prior_plots")
    os.makedirs(path, exist_ok=True)
    for i, prior_dict in enumerate(priors):
        plots.plot_prior(i, prior_dict, path)


def plot_post_calibration(targets: dict, mcmc_dir: str, plot_dir: str):
    logger.info(f"Plotting {mcmc_dir} into {plot_dir}")
    plotter = FilePlotter(plot_dir, targets)
    mcmc_tables = db.load.load_mcmc_tables(mcmc_dir)
    mcmc_params = db.load.load_mcmc_params_tables(mcmc_dir)

    derived_output_tables = db.load.load_derived_output_tables(mcmc_dir)
    param_options = mcmc_params[0]["name"].unique().tolist()

    logger.info("Plotting calibration fits")
    subplotter = _get_sub_plotter(plot_dir, "calibration-fit")
    for target in targets.values():
        output_name = target["output_key"]
        logger.info("Plotting calibration fit for output %s", output_name)
        outputs = plots.sample_outputs_for_calibration_fit(
            output_name, mcmc_tables, derived_output_tables
        )
        plots.plot_calibration_fit(subplotter, output_name, outputs, targets, is_logscale=True)
        plots.plot_calibration_fit(subplotter, output_name, outputs, targets, is_logscale=False)

    logger.info("Plotting posterior distributions")
    num_bins = 16
    subplotter = _get_sub_plotter(plot_dir, "posteriors")
    for chosen_param in param_options:
        plots.plot_posterior(subplotter, mcmc_params, chosen_param, num_bins)

    logger.info("Plotting loglikelihood vs params")
    subplotter = _get_sub_plotter(plot_dir, "params-vs-loglikelihood")
    for chosen_param in param_options:
        plots.plot_single_param_loglike(subplotter, mcmc_tables, mcmc_params, chosen_param)

    logger.info("Plotting parameter traces")
    subplotter = _get_sub_plotter(plot_dir, "params-traces")
    for chosen_param in param_options:
        plots.plot_mcmc_parameter_trace(subplotter, mcmc_params, chosen_param)

    logger.info("Plotting acceptance ratios")
    plots.plot_acceptance_ratio(plotter, mcmc_tables)

    logger.info("Plotting loglikelihood traces")
    num_iters = len(mcmc_tables[0])
    plots.plot_burn_in(plotter, num_iters, PLOT_BURN_IN)
    plots.plot_loglikelihood_trace(plotter, mcmc_tables, PLOT_BURN_IN)

    logger.info("MCMC plots complete")


def _get_sub_plotter(plot_dir: str, subplot_dirname: str):
    subplot_dir = os.path.join(plot_dir, subplot_dirname)
    os.makedirs(subplot_dir, exist_ok=True)
    return FilePlotter(subplot_dir, {})

