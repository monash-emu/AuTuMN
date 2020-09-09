from typing import List

import pandas as pd
import streamlit as st

from autumn.tool_kit.params import load_targets
from autumn.plots.plotter import StreamlitPlotter
from autumn import db, plots

from dash import selectors

PLOT_FUNCS = {}


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


def print_mle_parameters(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    df = None
    for table_df in mcmc_tables:
        if df is not None:
            df = df.append(table_df)
        else:
            df = table_df

    param_df = None
    for table_df in mcmc_params:
        if param_df is not None:
            param_df = param_df.append(table_df)
        else:
            param_df = table_df

    params = db.process.find_mle_params(df, param_df)
    st.write(params)


PLOT_FUNCS["Print MLE parameters"] = print_mle_parameters


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
    mcmc_all_df = None
    do_all_df = None
    for mcmc_df, do_df in zip(mcmc_tables, derived_output_tables):
        if mcmc_all_df is None:
            mcmc_all_df = mcmc_df
            do_all_df = do_df
        else:
            mcmc_all_df = mcmc_all_df.append(mcmc_df)
            do_all_df = do_all_df.append(do_df)

    uncertainty_df = db.uncertainty.calculate_mcmc_uncertainty(mcmc_all_df, do_all_df, targets)

    times = uncertainty_df.time.unique()
    quantiles = {}
    for q in chosen_target["quantiles"]:
        mask = uncertainty_df["quantile"] == q
        quantiles[q] = uncertainty_df[mask]["value"].tolist()

    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter, chosen_output, 0, quantiles, times, targets
    )


PLOT_FUNCS["Output uncertainty"] = plot_timeseries_with_uncertainty


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

