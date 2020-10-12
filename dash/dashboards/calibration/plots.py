from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from autumn.plots.plotter import StreamlitPlotter
from autumn.plots.calibration.plots import find_min_chain_length_from_mcmc_tables, get_posterior, get_epi_params
from autumn import db, plots

from dash import selectors

PLOT_FUNCS = {}


# FIXME: This is not in the right place - need to ask Matt where this should go
def write_mcmc_centiles(
        mcmc_params,
        burn_in,
        decimal_places,
        centiles,
):

    # Get parameter names
    parameters = get_epi_params(mcmc_params)

    # Create empty dataframe
    params_df = pd.DataFrame(index=parameters, columns=centiles)

    # Populate with data
    for param_name in parameters:
        param_values = get_posterior(mcmc_params, param_name, burn_in)
        centile_values = np.percentile(param_values, centiles)
        rounded_centile_values = [round(i_value, decimal_places) for i_value in centile_values]
        params_df.loc[param_name] = rounded_centile_values

    # Display
    st.write(params_df)


def create_standard_plotting_sidebar():
    title_font_size = st.sidebar.slider("Title font size", 1, 15, 8)
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 8)
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    capitalise_first_letter = st.sidebar.checkbox("Title start capital")
    return title_font_size, label_font_size, dpi_request, capitalise_first_letter


def create_xrange_selector(x_min, x_max):
    x_min = st.sidebar.slider("Plot start time", x_min, x_max, x_min)
    x_max = st.sidebar.slider("Plot end time", x_min, x_max, x_max)
    return x_min, x_max


def create_multi_scenario_selector(available_scenarios):
    return st.multiselect("Select scenarios", available_scenarios)


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
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 10)
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    plots.calibration.plots.plot_acceptance_ratio(
        plotter, mcmc_tables, burn_in, label_font_size=label_font_size, dpi_request=dpi_request
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

    x_min = round(min(uncertainty_df["time"]))
    x_max = round(max(uncertainty_df["time"]))
    x_low, x_up = create_xrange_selector(x_min, x_max)

    available_scenarios = uncertainty_df["scenario"].unique()
    selected_scenarios = create_multi_scenario_selector(available_scenarios)

    is_logscale = st.sidebar.checkbox("Log scale")
    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_output,
        selected_scenarios,
        targets,
        is_logscale,
        x_low,
        x_up,
    )


PLOT_FUNCS["Output uncertainty"] = plot_timeseries_with_uncertainty


def plot_multiple_timeseries_with_uncertainty(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_outputs = st.multiselect('Select outputs', available_outputs)
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

    x_min = round(min(uncertainty_df['time']))
    x_max = round(max(uncertainty_df['time']))
    x_low, x_up = create_xrange_selector(x_min, x_max)

    available_scenarios = uncertainty_df['scenario'].unique()
    selected_scenarios = create_multi_scenario_selector(available_scenarios)
    is_logscale = st.sidebar.checkbox("Log scale")
    n_xticks = st.sidebar.slider("Number of x ticks", 1, 10, 6)
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter, uncertainty_df, chosen_outputs, selected_scenarios, targets, is_logscale, x_low, x_up, n_xticks
    )


PLOT_FUNCS["Multi-output uncertainty"] = plot_multiple_timeseries_with_uncertainty


def plot_calibration_fit(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    available_outputs = [o["output_key"] for o in targets.values()]
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in (select 0 for default behaviour of discarding first half)", 0, chain_length, 0)
    chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
    derived_output_tables = db.load.load_derived_output_tables(calib_dir_path, column=chosen_output)
    outputs = plots.calibration.plots.sample_outputs_for_calibration_fit(
        chosen_output, mcmc_tables, derived_output_tables, burn_in=burn_in
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
    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = create_standard_plotting_sidebar()
    is_logscale = st.sidebar.checkbox("Log scale")
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in (select 0 for default behaviour of discarding first half)", 0, chain_length, 0)

    outputs = {}
    for output in available_outputs:
        derived_output_tables = db.load.load_derived_output_tables(calib_dir_path, column=output)
        outputs[output] = plots.calibration.plots.sample_outputs_for_calibration_fit(
            output, mcmc_tables, derived_output_tables, burn_in,
        )
    plots.calibration.plots.plot_multi_fit(
        plotter,
        available_outputs,
        outputs,
        targets,
        is_logscale,
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
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
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_mcmc_parameter_trace(plotter, mcmc_params, burn_in, chosen_param)


PLOT_FUNCS["Parameter trace"] = plot_mcmc_parameter_trace


def plot_all_param_traces(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):

    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = create_standard_plotting_sidebar()
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_multiple_param_traces(
        plotter, mcmc_params, burn_in, title_font_size, label_font_size, capitalise_first_letter, dpi_request
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
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_single_param_loglike(
        plotter, mcmc_tables, mcmc_params, burn_in, chosen_param
    )


PLOT_FUNCS["Loglikelihood vs param"] = plot_loglikelihood_vs_parameter


def plot_loglike_vs_all_params(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = create_standard_plotting_sidebar()
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    plots.calibration.plots.plot_all_params_vs_loglike(
        plotter,
        mcmc_tables,
        mcmc_params,
        burn_in,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
    )


PLOT_FUNCS["All loglike vs params"] = plot_loglike_vs_all_params


def plot_posterior(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):
    chosen_param = selectors.parameter(mcmc_params[0])
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 16)
    plots.calibration.plots.plot_posterior(plotter, mcmc_params, burn_in, chosen_param, num_bins)


PLOT_FUNCS["Posterior distributions"] = plot_posterior


def plot_all_posteriors(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
):

    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = create_standard_plotting_sidebar()
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    num_bins = st.sidebar.slider("Number of bins", 1, 50, 16)
    decimal_places = st.sidebar.slider("Decimal places", 0, 6, 3)
    plots.calibration.plots.plot_multiple_posteriors(
        plotter,
        mcmc_params,
        burn_in,
        num_bins,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
    )

    write_mcmc_centiles(mcmc_params, burn_in, decimal_places, [0.25, 0.5, 0.975])


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


def plot_param_matrix_by_chain(
        plotter: StreamlitPlotter,
        calib_dir_path: str,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        targets: dict,
):
    """
    Now unused because I prefer the version that isn't by chain.
    """
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 8)
    label_chars = st.sidebar.slider("Label characters", 1, 10, 2)
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    parameters = mcmc_params[0]["name"].unique().tolist()
    st.write(parameters)
    plots.calibration.plots.plot_param_vs_param_by_chain(
        plotter, mcmc_params, parameters, label_font_size, label_chars, dpi_request
    )


def plot_param_matrix(
        plotter: StreamlitPlotter,
        calib_dir_path: str,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        targets: dict,
):
    parameters = mcmc_params[0]["name"].unique().tolist()
    chain_length = find_min_chain_length_from_mcmc_tables(mcmc_tables)
    burn_in = st.sidebar.slider("Burn in", 0, chain_length, 0)
    label_font_size = st.sidebar.slider("Label font size", 1, 15, 8)
    label_chars = st.sidebar.slider("Label characters", 1, 10, 2)
    bins = st.sidebar.slider("Bins", 4, 50, 20)
    style = st.sidebar.selectbox("Style", ["Shade", "Scatter", "KDE"])
    dpi_request = st.sidebar.slider("DPI", 50, 2000, 300)
    plots.calibration.plots.plot_param_vs_param(
        plotter, mcmc_params, parameters, burn_in, style, bins, label_font_size, label_chars, dpi_request
    )
    st.write(parameters)


PLOT_FUNCS["Param versus param"] = plot_param_matrix


def plot_parallel_coordinates(
        plotter: StreamlitPlotter,
        calib_dir_path: str,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        targets: dict,
):
    plots.calibration.plots.plot_parallel_coordinates(
        plotter,
        mcmc_tables,
        mcmc_params,
    )


PLOT_FUNCS["Parallel Coordinates"] = plot_parallel_coordinates


def plot_loglikelihood_surface(
        plotter: StreamlitPlotter,
        calib_dir_path: str,
        mcmc_tables: List[pd.DataFrame],
        mcmc_params: List[pd.DataFrame],
        targets: dict,
):
    options = mcmc_params[0]["name"].unique().tolist()
    param_1 = st.sidebar.selectbox("Select parameter 1", options)
    param_2 = st.sidebar.selectbox("Select parameter 2", options)

    plots.calibration.plots.plot_loglikelihood_surface(
        plotter,
        mcmc_tables,
        mcmc_params,
        param_1,
        param_2,
    )


PLOT_FUNCS["Loglikelihood 3d scatter"] = plot_loglikelihood_surface
