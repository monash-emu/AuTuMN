import os
from typing import List

import pandas as pd
from matplotlib import pyplot

from autumn.settings import Region
from autumn.dashboards.calibration_results.plots import get_uncertainty_df
from autumn.tools.plots.plotter import StreamlitPlotter
from autumn.tools import plots
from autumn.tools.streamlit.utils import Dashboard
from autumn.tools.project import Project

from autumn.tools.streamlit import selectors
import streamlit as st


STANDARD_X_LIMITS = 153, 275
dash = Dashboard()


@dash.register("Model fits")
def plot_model_fits(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    chosen_outputs = ["notifications", "icu_occupancy", "accum_deaths"]

    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)

    x_min = round(min(uncertainty_df["time"]))
    x_max = round(max(uncertainty_df["time"]))
    x_low, x_up = selectors.create_xrange_selector(x_min, x_max)

    selected_scenarios = [0]
    show_uncertainty = st.sidebar.checkbox("Show uncertainty", value=True)
    n_xticks = st.sidebar.slider("Number of x ticks", 1, 10, 6)
    title_font_size = st.sidebar.slider("Title font size", 1, 30, 12)
    label_font_size = st.sidebar.slider("Label font size", 1, 30, 10)
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_outputs,
        selected_scenarios,
        targets,
        False,
        x_low,
        x_up,
        n_xticks,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
        overlay_uncertainty=show_uncertainty,
        is_legend=False,
    )


@dash.register("Multi-scenarios single")
def plot_multi_scenario_single_run(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    run_id = st.sidebar.text_input("run_id", value="MLE")

    available_outputs = [o["output_key"] for o in targets.values()]
    chosen_outputs = st.multiselect("Select outputs", available_outputs)

    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)

    x_min = round(min(uncertainty_df["time"]))
    x_max = round(max(uncertainty_df["time"]))
    x_low, x_up = selectors.create_xrange_selector(x_min, x_max)

    available_scenarios = uncertainty_df["scenario"].unique()
    selected_scenarios = st.multiselect("Select scenarios", available_scenarios)

    is_legend = st.sidebar.checkbox("Show legend")

    n_xticks = st.sidebar.slider("Number of x ticks", 1, 10, 6)

    title_font_size = st.sidebar.slider("Title font size", 1, 30, 12)
    label_font_size = st.sidebar.slider("Label font size", 1, 30, 10)


    plots.calibration.plots.plot_multi_output_single_run(
        plotter, mcmc_tables, calib_dir_path, chosen_outputs, selected_scenarios, run_id, x_low, x_up, is_legend, n_xticks, title_font_size, label_font_size
    )



# @dash.register("Seroprevalence by age")
# def plot_seroprevalence_by_age(
#     plotter: StreamlitPlotter,
#     calib_dir_path: str,
#     mcmc_tables: List[pd.DataFrame],
#     mcmc_params: List[pd.DataFrame],
#     project: Project,
# ):
#
#     n_columns = 2
#     n_rows = 2
#
#     fig = pyplot.figure(constrained_layout=True, figsize=(n_columns * 7, n_rows * 5))  # (w, h)
#     spec = fig.add_gridspec(ncols=n_columns, nrows=n_rows)
#     i_row = 0
#     i_col = 0
#     for region in Region.PHILIPPINES_REGIONS:
#         calib_dir_path = calib_dir_path.replace("philippines", region)
#         uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, project.plots)
#         # available_scenarios = uncertainty_df["scenario"].unique()
#         # selected_scenario = st.sidebar.selectbox("Select scenario", available_scenarios, key=str())
#         selected_scenario = 0
#
#         # min_time = int(min(uncertainty_df["time"]))
#         # max_time = int(max(uncertainty_df["time"]))
#         # time = st.sidebar.slider("time", min_time, max_time, max_time)
#         time = 397
#
#         with pyplot.style.context("ggplot"):
#             ax = fig.add_subplot(spec[i_row, i_col])
#             _, _, _ = plots.uncertainty.plots.plot_seroprevalence_by_age(
#                 plotter, uncertainty_df, selected_scenario, time, axis=ax, name=region.title()
#             )
#
#         i_col += 1
#         if i_col >= n_columns:
#             i_col = 0
#             i_row += 1
#     plotter.save_figure(fig, filename="sero_by_age", subdir="outputs", title_text="")
