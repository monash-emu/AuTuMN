"""
Streamlit web UI for plotting MCMC outputs
"""
import os
from typing import List

import pandas as pd
import streamlit as st

from autumn.tool_kit.uncertainty import collect_all_mcmc_output_tables
from autumn.plots import plots
from autumn.plots.plotter import StreamlitPlotter

from . import selectors, utils


def run_mcmc_plots():
    app_dirname, app_dirpath = selectors.app()
    calib_dirname, calib_dirpath = selectors.calibration_run(app_dirpath)
    if not calib_dirname:
        st.write("No calibration folder found")
        return

    plot_config = utils.load_plot_config(app_dirname)

    # Load MCMC tables
    mcmc_tables, output_tables, derived_output_tables = collect_all_mcmc_output_tables(calib_dirpath)

    plotter = StreamlitPlotter({})
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    if plot_type == "Predictions":
        plot_func(plotter, calib_dirpath, mcmc_tables, plot_config)
    else:
        plot_func(plotter, mcmc_tables)


def plot_mcmc_parameter_trace(plotter: StreamlitPlotter, mcmc_tables: List[pd.DataFrame]):
    chosen_param = parameter_selector(mcmc_tables[0])
    plots.plot_mcmc_parameter_trace(plotter, mcmc_tables, chosen_param)


def plot_loglikelihood_trace(plotter: StreamlitPlotter, mcmc_tables: List[pd.DataFrame]):
    burn_in = st.sidebar.slider("Burn-in", 0, len(mcmc_tables[0]), 0)
    plots.plot_loglikelihood_trace(plotter, mcmc_tables, burn_in)
    num_iters = len(mcmc_tables[0])
    plots.plot_burn_in(plotter, num_iters, burn_in)


def plot_posterior(plotter: StreamlitPlotter, mcmc_tables: List[pd.DataFrame]):
    chosen_param = parameter_selector(mcmc_tables[0])
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 10)
    plots.plot_posterior(plotter, mcmc_tables, chosen_param, num_bins)


def plot_loglikelihood_vs_parameter(plotter: StreamlitPlotter, mcmc_tables: List[pd.DataFrame]):
    burn_in = st.sidebar.slider("Burn-in", 0, len(mcmc_tables[0]), 0)
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_options = [c for c in mcmc_tables[0].columns if c not in non_param_cols]
    chosen_param = st.sidebar.selectbox("Select parameter", param_options)
    plots.plot_loglikelihood_vs_parameter(plotter, mcmc_tables, chosen_param, burn_in)


def plot_timeseries_with_uncertainty(plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame],
                                     plot_config={}):
    burn_in = st.sidebar.slider("Burn-in", 0, min([len(mcmc_tables[i]) for i in range(len(mcmc_tables))]), 0)

    # FIXME
    chosen_output = 'notifications'  # to be replaced with selector options
    chosen_scenarios = [0]  # to be replaced with selector options

    plots.plot_timeseries_with_uncertainty(plotter, calib_dir_path, chosen_output, scenario_indices=chosen_scenarios,
                                           burn_in=burn_in, plot_config=plot_config
                                           )

PLOT_FUNCS = {
    "Posterior distributions": plot_posterior,
    "Loglikelihood trace": plot_loglikelihood_trace,
    "Loglikelihood vs param": plot_loglikelihood_vs_parameter,
    "Parameter trace": plot_mcmc_parameter_trace,
    "Predictions": plot_timeseries_with_uncertainty,
}


def parameter_selector(mcmc_table: pd.DataFrame):
    """
    Drop down for selecting parameters
    """
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_options = [c for c in mcmc_table.columns if c not in non_param_cols]
    return st.sidebar.selectbox("Select parameter", param_options)
