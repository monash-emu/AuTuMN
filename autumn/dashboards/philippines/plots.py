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

STANDARD_X_LIMITS = 153, 275
dash = Dashboard()


@dash.register("Seroprevalence by age")
def plot_seroprevalence_by_age(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    project: Project,
):

    n_columns = 2
    n_rows = 2

    fig = pyplot.figure(constrained_layout=True, figsize=(n_columns * 7, n_rows * 5))  # (w, h)
    spec = fig.add_gridspec(ncols=n_columns, nrows=n_rows)
    i_row = 0
    i_col = 0
    for region in Region.PHILIPPINES_REGIONS:
        calib_dir_path = calib_dir_path.replace("philippines", region)
        uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, project.plots)
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
