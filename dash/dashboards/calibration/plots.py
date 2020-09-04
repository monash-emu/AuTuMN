from typing import List

import pandas as pd
import streamlit as st

from autumn.tool_kit.params import load_targets
from autumn.db.models import load_mcmc_tables
from autumn.plots.plotter import StreamlitPlotter
from autumn.calibration.utils import collect_map_estimate, print_reformated_map_parameters
from autumn.tool_kit.uncertainty import (
    calculate_uncertainty_weights,
    calculate_mcmc_uncertainty,
)


from autumn.plots import plots
from autumn.db.models import load_derived_output_tables

from dash import selectors

PLOT_FUNCS = {}


def plot_mcmc_parameter_trace(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], targets: dict,
):
    chosen_param = selectors.parameter(mcmc_tables[0])
    plots.plot_mcmc_parameter_trace(plotter, mcmc_tables, chosen_param)


PLOT_FUNCS["Parameter trace"] = plot_mcmc_parameter_trace


def plot_loglikelihood_trace(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], targets: dict,
):
    burn_in = selectors.burn_in(mcmc_tables)
    plots.plot_loglikelihood_trace(plotter, mcmc_tables, burn_in)
    num_iters = len(mcmc_tables[0])
    plots.plot_burn_in(plotter, num_iters, burn_in)


PLOT_FUNCS["Loglikelihood trace"] = plot_loglikelihood_trace


def plot_posterior(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], targets: dict,
):
    chosen_param = selectors.parameter(mcmc_tables[0])
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 16)
    plots.plot_posterior(plotter, mcmc_tables, chosen_param, num_bins)


PLOT_FUNCS["Posterior distributions"] = plot_posterior


def plot_loglikelihood_vs_parameter(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], targets: dict,
):
    burn_in = selectors.burn_in(mcmc_tables)
    non_param_cols = ["idx", "Scenario", "loglikelihood", "accept"]
    param_options = [c for c in mcmc_tables[0].columns if c not in non_param_cols]
    chosen_param = st.sidebar.selectbox("Select parameter", param_options)
    plots.plot_loglikelihood_vs_parameter(plotter, mcmc_tables, chosen_param, burn_in)


PLOT_FUNCS["Loglikelihood vs param"] = plot_loglikelihood_vs_parameter


def plot_timeseries_with_uncertainty(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], targets: dict,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    chosen_target = next(t for t in targets.values() if t["output_key"] == chosen_output)
    burn_in = selectors.burn_in(mcmc_tables)
    derived_output_tables = load_derived_output_tables(calib_dir_path, column=chosen_output)

    weights_df = None
    for raw_mcmc_df, derived_output_df in zip(mcmc_tables, derived_output_tables):
        mcmc_df = raw_mcmc_df[burn_in:]
        _weights_df = calculate_uncertainty_weights(chosen_output, mcmc_df, derived_output_df)
        if weights_df is None:
            weights_df = _weights_df
        else:
            weights_df = weights_df.append(_weights_df)

    uncertainty_df = calculate_mcmc_uncertainty(weights_df, targets)
    times = uncertainty_df.time.unique()
    quantiles = {}
    for q in chosen_target["quantiles"]:
        mask = uncertainty_df["quantile"] == q
        quantiles[q] = uncertainty_df[mask]["value"].tolist()

    plots.plot_timeseries_with_uncertainty(plotter, chosen_output, "S_0", quantiles, times, targets)


PLOT_FUNCS["Output uncertainty"] = plot_timeseries_with_uncertainty


def plot_calibration_fit(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], targets: dict,
):

    derived_output_tables = load_derived_output_tables(calib_dir_path)

    # Choose which output to plot
    non_output_cols = ["idx", "Scenario", "times"]
    output_cols = list(set(derived_output_tables[0].columns) - set(non_output_cols))
    output_cols = [c for c in output_cols if "X" not in c or c.endswith("Xall")]
    chosen_output = st.sidebar.selectbox("Select derived output", output_cols)

    outputs, best_chain_index = plots.sample_outputs_for_calibration_fit(
        chosen_output, mcmc_tables, derived_output_tables
    )
    is_logscale = st.sidebar.checkbox("Log scale")
    plots.plot_calibration_fit(
        plotter, chosen_output, outputs, best_chain_index, targets, is_logscale
    )


PLOT_FUNCS["Output calibration fit"] = plot_calibration_fit


def print_mle_parameters(
    plotter: StreamlitPlotter, calib_dir_path: str, mcmc_tables: List[pd.DataFrame], targets: dict,
):
    mle_params, _ = collect_map_estimate(calib_dir_path)
    print_reformated_map_parameters(mle_params)


PLOT_FUNCS["Print MLE parameters"] = print_mle_parameters
