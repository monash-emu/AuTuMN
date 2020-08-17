"""
Utilities to plot data from existing databases.
"""
import os
import yaml
import logging

from autumn.db.models import load_model_scenarios

from . import plots
from .scenario_plots import plot_scenarios

# from .streamlit.utils import try_find_app_code_path
from .streamlit.run_mcmc_plots import load_mcmc_tables, load_derived_output_tables
from .plotter import FilePlotter

APP_DIRNAMES = ["covid_", "marshall_islands", "mongolia", "dummy"]

logger = logging.getLogger(__name__)


def plot_from_mcmc_databases(app_name: str, param_set_name: str, mcmc_dir: str, plot_dir: str):
    logger.info(f"Plotting {mcmc_dir} into {plot_dir}")
    plotter = FilePlotter(plot_dir, {})
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
    plot_config = load_plot_config(app_name, param_set_name)
    outputs_to_plot = plot_config.get("outputs_to_plot", [])
    for output_to_plot in outputs_to_plot:
        output_name = output_to_plot["name"]
        logger.info("Plotting calibration fit for output %s", output_name)
        outputs, best_chain_index = plots.sample_outputs_for_calibration_fit(
            output_name, mcmc_tables, derived_output_tables
        )
        plots.plot_calibration_fit(
            subplotter, output_name, outputs, best_chain_index, plot_config, is_logscale=True
        )
        plots.plot_calibration_fit(
            subplotter, output_name, outputs, best_chain_index, plot_config, is_logscale=False
        )

    logger.info("MCMC plots complete")


def _get_sub_plotter(plot_dir: str, subplot_dirname: str):
    subplot_dir = os.path.join(plot_dir, subplot_dirname)
    os.makedirs(subplot_dir, exist_ok=True)
    return FilePlotter(subplot_dir, {})


def plot_from_database(run_path: str):
    """
    Reads data from an existing model run and re-plots the outputs.
    """
    output_db_path = os.path.join(run_path, "outputs.db")
    assert os.path.exists(output_db_path), "Folder does not contain outputs.db"
    app_dirname = [x for x in run_path.split("/") if x][-2]
    params_path = os.path.join(run_path, "params.yml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    app_code_path = try_find_app_code_path(app_dirname)

    # Load plot config from project dir
    plot_config_path = os.path.join(app_code_path, "plots.yml")
    with open(plot_config_path, "r") as f:
        plot_config = yaml.safe_load(f)

    plots.validate_plot_config(plot_config)

    # Get database from model data dir.
    db_path = os.path.join(run_path, "outputs.db")
    scenarios = load_model_scenarios(db_path, params)

    plot_scenarios(scenarios, run_path, plot_config)
