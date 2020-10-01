from typing import List

import pandas as pd
import streamlit as st

from autumn.tool_kit.params import load_targets
from autumn.plots.plotter import StreamlitPlotter
from autumn import db, plots

from dash import selectors

PLOT_FUNCS = {}


def create_standard_plotting_sidebar():
    # Implement user request options
    title_font_size = \
        st.sidebar.slider("Title font size", 1, 15, 8)
    label_font_size = \
        st.sidebar.slider("Label font size", 1, 15, 8)
    dpi_request = \
        st.sidebar.slider("DPI", 50, 2000, 300)
    capitalise_first_letter = \
        st.sidebar.checkbox("Title start capital")
    return title_font_size, label_font_size, dpi_request, capitalise_first_letter


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


def plot_acceptance_ratio(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    label_font_size = \
        st.sidebar.slider("Label font size", 1, 15, 10)
    dpi_request = \
        st.sidebar.slider("DPI", 50, 2000, 300)
    plots.calibration.plots.plot_acceptance_ratio(
        plotter, mcmc_tables, label_font_size=label_font_size, dpi_request=dpi_request
    )


PLOT_FUNCS["Acceptance ratio"] = plot_acceptance_ratio


def plot_timeseries_with_uncertainty(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    targets = {k: v for k, v in targets.items() if v["output_key"] == chosen_output}

    try:  # if PBI processing has been performed already
        uncertainty_df = db.load.load_uncertainty_table(calib_dir_path)
    except:  # calculates percentiles
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


def plot_multi_output_fit(
        plotter: StreamlitPlotter,
        calib_dir_path: str,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        targets: dict,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    is_logscale = st.sidebar.checkbox("Log scale")

    outputs = {}
    for output in available_outputs:
        derived_output_tables = db.load.load_derived_output_tables(calib_dir_path, column=output)
        outputs[output] = plots.calibration.plots.sample_outputs_for_calibration_fit(
            output, mcmc_tables, derived_output_tables
        )
    plots.calibration.plots.plot_multi_fit(
        plotter, available_outputs, outputs, targets, is_logscale
    )


PLOT_FUNCS["Multi-output fit"] = plot_multi_output_fit


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


def plot_all_param_traces(
        plotter: StreamlitPlotter,
        calib_dir_path: str,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        targets: dict,
):

    title_font_size, label_font_size, dpi_request, capitalise_first_letter = \
        create_standard_plotting_sidebar()
    plots.calibration.plots.plot_multiple_param_traces(
        plotter, mcmc_params, title_font_size, label_font_size, capitalise_first_letter, dpi_request
    )


PLOT_FUNCS["All param traces"] = plot_all_param_traces


def plot_loglikelihood_vs_parameter(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    chosen_param = selectors.parameter(mcmc_params[0])
    plots.calibration.plots.plot_single_param_loglike(
        plotter, mcmc_tables, mcmc_params, chosen_param
    )


PLOT_FUNCS["Loglikelihood vs param"] = plot_loglikelihood_vs_parameter


def plot_loglike_vs_all_params(
        plotter: StreamlitPlotter,
        calib_dir_path: str,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        targets: dict,
):
    title_font_size, label_font_size, dpi_request, capitalise_first_letter = \
        create_standard_plotting_sidebar()
    plots.calibration.plots.plot_all_params_vs_loglike(
        plotter, mcmc_tables, mcmc_params, title_font_size, label_font_size, capitalise_first_letter, dpi_request
    )


PLOT_FUNCS["All loglike vs params"] = plot_loglike_vs_all_params


def plot_posterior(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    chosen_param = \
        selectors.parameter(mcmc_params[0])
    num_bins = \
        st.sidebar.slider("Number of bins", 1, 50, 16)
    plots.calibration.plots.plot_posterior(plotter, mcmc_params, chosen_param, num_bins)


PLOT_FUNCS["Posterior distributions"] = plot_posterior


def plot_all_posteriors(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):

    title_font_size, label_font_size, dpi_request, capitalise_first_letter = \
        create_standard_plotting_sidebar()
    num_bins = \
        st.sidebar.slider("Number of bins", 1, 50, 16)
    plots.calibration.plots.plot_multiple_posteriors(
        plotter, mcmc_params, num_bins, title_font_size, label_font_size, capitalise_first_letter, dpi_request
    )


PLOT_FUNCS["All posteriors"] = plot_all_posteriors


def plot_loglikelihood_trace(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    burn_in = selectors.burn_in(mcmc_tables)
    plots.calibration.plots.plot_loglikelihood_trace(plotter, mcmc_tables, burn_in)


PLOT_FUNCS["Loglikelihood trace"] = plot_loglikelihood_trace
