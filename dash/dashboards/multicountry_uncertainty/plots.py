from math import ceil

import matplotlib.pyplot as pyplot
import streamlit as st

from autumn import plots
from dash import selectors
from dash.dashboards.calibration_results.plots import (
    get_uncertainty_data,
    get_uncertainty_df,
)

PLOT_FUNCS = {}


def multi_country_uncertainty(
    plotter, calib_dir_path, mcmc_tables, mcmc_params, targets, app_name, region_names
):
    """
    Code taken directly from the fit calibration file at this stage.
    """

    is_logscale = st.sidebar.checkbox("Log scale")
    n_xticks = st.sidebar.slider("Number of x ticks", 1, 10, 6)
    title_font_size = st.sidebar.slider("Title font size", 1, 30, 12)
    label_font_size = st.sidebar.slider("Label font size", 1, 30, 10)
    uncertainty_df = []

    for i_region in range(len(mcmc_tables)):
        uncertainty_df.append(
            get_uncertainty_df(calib_dir_path[i_region], mcmc_tables[i_region], targets)
        )

        if i_region == 0:
            available_outputs = [o["output_key"] for o in targets[i_region].values()]
            chosen_output = st.sidebar.selectbox("Select calibration target", available_outputs)
            x_min = round(min(uncertainty_df[0]["time"]))
            x_max = round(max(uncertainty_df[0]["time"]))
            x_low, x_up = selectors.create_xrange_selector(x_min, x_max)
            available_scenarios = uncertainty_df[0]["scenario"].unique()
            selected_scenarios = st.multiselect("Select scenarios", available_scenarios)

    plots.uncertainty.plots.plot_multicountry_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_output,
        selected_scenarios,
        targets,
        region_names,
        is_logscale,
        x_low,
        x_up,
        n_xticks,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
    )


PLOT_FUNCS["Multi-country uncertainty"] = multi_country_uncertainty
