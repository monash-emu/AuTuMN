import os
from typing import List

import pandas as pd
import streamlit as st
import yaml
from matplotlib import pyplot
from numpy import random

from autumn import plots
from autumn.plots.calibration.plots import get_epi_params
from autumn.plots.plotter import StreamlitPlotter
from autumn.plots.utils import get_plot_text_dict
from autumn.region import Region
from dash.dashboards.calibration_results.plots import (
    get_uncertainty_df,
    write_mcmc_centiles,
)
from dash.utils import create_downloadable_csv

STANDARD_X_LIMITS = 153, 275
PLOT_FUNCS = {}


def plot_seroprevalence_by_age(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    n_columns = 2
    n_rows = 2

    fig = pyplot.figure(constrained_layout=True, figsize=(n_columns * 7, n_rows * 5))  # (w, h)
    spec = fig.add_gridspec(ncols=n_columns, nrows=n_rows)
    i_row = 0
    i_col = 0
    for region in Region.PHILIPPINES_REGIONS:
        calib_dir_path = calib_dir_path.replace("philippines", region)
        uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
        # available_scenarios = uncertainty_df["scenario"].unique()
        # selected_scenario = st.sidebar.selectbox("Select scenario", available_scenarios, key=str())
        selected_scenario = 0

        # min_time = int(min(uncertainty_df["time"]))
        # max_time = int(max(uncertainty_df["time"]))
        # time = st.sidebar.slider("time", min_time, max_time, max_time)
        time = 397

        with pyplot.style.context("ggplot"):
            ax = fig.add_subplot(spec[i_row, i_col])
            _, _, _ = plots.uncertainty.plots.plot_seroprevalence_by_age(
                plotter, uncertainty_df, selected_scenario, time, axis=ax, name=region.title()
            )

        i_col += 1
        if i_col >= n_columns:
            i_col = 0
            i_row += 1
    plotter.save_figure(fig, filename="sero_by_age", subdir="outputs", title_text="")


PLOT_FUNCS["Seroprevalence by age"] = plot_seroprevalence_by_age
