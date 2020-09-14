from typing import List

import pandas as pd
import streamlit as st

from autumn.tool_kit.params import load_targets
from autumn.plots.plotter import StreamlitPlotter
from autumn import db, plots

from dash import selectors

PLOT_FUNCS = {}


def plot_timeseries_with_uncertainty(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    chosen_target = next(t for t in targets.values() if t["output_key"] == chosen_output)
    targets = {k: v for k, v in targets.items() if v["output_key"] == chosen_output}
    derived_output_tables = db.load.load_derived_output_tables(calib_dir_path)
    mcmc_all_df = db.process.append_tables(mcmc_tables)
    do_all_df = db.process.append_tables(derived_output_tables)

    # Determine max chain length, throw away first half of that
    max_run = mcmc_all_df["run"].max()
    half_max = max_run // 2
    mcmc_all_df = mcmc_all_df[mcmc_all_df["run"] >= half_max]

    uncertainty_df = db.uncertainty.calculate_mcmc_uncertainty(mcmc_all_df, do_all_df, targets)
    is_logscale = st.sidebar.checkbox("Log scale")
    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter, uncertainty_df, chosen_output, 0, targets, is_logscale
    )


PLOT_FUNCS["Output uncertainty"] = plot_timeseries_with_uncertainty


def plot_calibration_fit(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    derived_output_tables = db.load.load_derived_output_tables(calib_dir_path, column=chosen_output)
    outputs = plots.calibration.plots.sample_outputs_for_calibration_fit(
        chosen_output, mcmc_tables, derived_output_tables
    )
    is_logscale = st.sidebar.checkbox("Log scale")
    plots.calibration.plots.plot_calibration_fit(
        plotter, chosen_output, outputs, targets, is_logscale
    )


PLOT_FUNCS["Output calibration fit"] = plot_calibration_fit


def plot_mcmc_parameter_trace(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    chosen_param = selectors.parameter(mcmc_params[0])
    plots.calibration.plots.plot_mcmc_parameter_trace(plotter, mcmc_params, chosen_param)


PLOT_FUNCS["Parameter trace"] = plot_mcmc_parameter_trace


def print_mle_parameters(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    df = db.process.append_tables(mcmc_tables)
    param_df = db.process.append_tables(mcmc_params)
    params = db.process.find_mle_params(df, param_df)
    st.write(params)


PLOT_FUNCS["Print MLE parameters"] = print_mle_parameters


def plot_loglikelihood_vs_parameter(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    chosen_param = selectors.parameter(mcmc_params[0])
    plots.calibration.plots.plot_loglikelihood_vs_parameter(
        plotter, mcmc_tables, mcmc_params, chosen_param
    )


PLOT_FUNCS["Loglikelihood vs param"] = plot_loglikelihood_vs_parameter


def plot_posterior(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    chosen_param = selectors.parameter(mcmc_params[0])
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 16)
    plots.calibration.plots.plot_posterior(plotter, mcmc_params, chosen_param, num_bins)


PLOT_FUNCS["Posterior distributions"] = plot_posterior


def plot_loglikelihood_trace(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    burn_in = selectors.burn_in(mcmc_tables)
    plots.calibration.plots.plot_loglikelihood_trace(plotter, mcmc_tables, burn_in)
    num_iters = len(mcmc_tables[0])
    plots.calibration.plots.plot_burn_in(plotter, num_iters, burn_in)


PLOT_FUNCS["Loglikelihood trace"] = plot_loglikelihood_trace

