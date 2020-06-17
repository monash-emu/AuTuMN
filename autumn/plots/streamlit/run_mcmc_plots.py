"""
Streamlit web UI for plotting MCMC outputs
"""
import os
from typing import List

import pandas as pd
import streamlit as st

from autumn.db import Database
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
    mcmc_tables = load_mcmc_tables(calib_dirpath)

    plotter = StreamlitPlotter({})
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, calib_dirpath, mcmc_tables, plot_config)


def load_mcmc_tables(calib_dirpath: str):
    db_paths = [
        os.path.join(calib_dirpath, f)
        for f in os.listdir(calib_dirpath)
        if f.endswith(".db") and not f.startswith("mcmc_percentiles")
    ]
    mcmc_tables = []
    for db_path in db_paths:
        db = Database(db_path)
        mcmc_tables.append(db.db_query("mcmc_run"))

    return mcmc_tables


def load_derived_output_tables(calib_dirpath: str):
    db_paths = [
        os.path.join(calib_dirpath, f)
        for f in os.listdir(calib_dirpath)
        if f.endswith(".db") and not f.startswith("mcmc_percentiles")
    ]
    derived_output_tables = []
    for db_path in db_paths:
        db = Database(db_path)
        derived_output_tables.append(db.db_query("derived_outputs"))

    return derived_output_tables


def plot_mcmc_parameter_trace(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], plot_config={},
):
    chosen_param = parameter_selector(mcmc_tables[0])
    plots.plot_mcmc_parameter_trace(plotter, mcmc_tables, chosen_param)


def plot_loglikelihood_trace(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], plot_config={},
):
    burn_in = burn_in_selector(mcmc_tables)
    plots.plot_loglikelihood_trace(plotter, mcmc_tables, burn_in)
    num_iters = len(mcmc_tables[0])
    plots.plot_burn_in(plotter, num_iters, burn_in)


def plot_posterior(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], plot_config={},
):
    chosen_param = parameter_selector(mcmc_tables[0])
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 16)
    plots.plot_posterior(plotter, mcmc_tables, chosen_param, num_bins)


def plot_loglikelihood_vs_parameter(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], plot_config={},
):
    burn_in = burn_in_selector(mcmc_tables)
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_options = [c for c in mcmc_tables[0].columns if c not in non_param_cols]
    chosen_param = st.sidebar.selectbox("Select parameter", param_options)
    plots.plot_loglikelihood_vs_parameter(plotter, mcmc_tables, chosen_param, burn_in)


def plot_timeseries_with_uncertainty(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], plot_config={},
):
    derived_output_tables = load_derived_output_tables(calib_dir_path)
    # Choose one or more scenarios
    scenario_strs = list(derived_output_tables[0].Scenario.unique())
    scenario_idxs = [int(s.replace("S_", "")) for s in scenario_strs]
    scenario_names = ["Baseline" if i == 0 else f"Scenario {i}" for i in scenario_idxs]
    chosen_scenario_names = st.sidebar.multiselect("Select scenarios", scenario_names, "Baseline")
    chosen_scenarios = [scenario_idxs[scenario_names.index(n)] for n in chosen_scenario_names]

    # Choose which output to plot
    non_output_cols = ["idx", "Scenario", "times"]
    output_cols = list(set(derived_output_tables[0].columns) - set(non_output_cols))
    output_cols = [c for c in output_cols if "X" not in c or c.endswith("Xall")]
    chosen_output = st.sidebar.selectbox("Select derived output", output_cols)
    # Select burn in with a slider
    burn_in = burn_in_selector(mcmc_tables)
    plots.plot_timeseries_with_uncertainty(
        plotter,
        calib_dir_path,
        chosen_output,
        scenario_indices=chosen_scenarios,
        burn_in=burn_in,
        plot_config=plot_config,
    )


PLOT_FUNCS = {
    "Posterior distributions": plot_posterior,
    "Loglikelihood trace": plot_loglikelihood_trace,
    "Loglikelihood vs param": plot_loglikelihood_vs_parameter,
    "Parameter trace": plot_mcmc_parameter_trace,
    "Predictions": plot_timeseries_with_uncertainty,
}


def burn_in_selector(mcmc_tables: List[pd.DataFrame]):
    """
    Slider for selecting how much burn in we should apply to an MCMC trace.
    """
    min_length = min([len(t) for t in mcmc_tables])
    return st.sidebar.slider("Burn-in", 0, min_length, 0)


def parameter_selector(mcmc_table: pd.DataFrame):
    """
    Drop down for selecting parameters
    """
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_options = [c for c in mcmc_table.columns if c not in non_param_cols]
    return st.sidebar.selectbox("Select parameter", param_options)
