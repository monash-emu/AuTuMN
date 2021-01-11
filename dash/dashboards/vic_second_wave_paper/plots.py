from typing import List
import pandas as pd
import streamlit as st

from autumn.plots.plotter import StreamlitPlotter
from autumn import plots
from dash.dashboards.calibration_results.plots import get_uncertainty_df

from dash import selectors

STANDARD_X_LIMITS = 153, 275
PLOT_FUNCS = {}


def plot_overall_notifications(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    chosen_output = "notifications"
    targets = {k: v for k, v in targets.items() if v["output_key"] == chosen_output}
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    x_low, x_up = STANDARD_X_LIMITS
    selected_scenarios = [0]

    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = selectors.create_standard_plotting_sidebar()
    is_logscale = False
    is_targets = True
    is_overlay_unceratinty = True
    is_legend = st.sidebar.checkbox("Add legend")
    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_output,
        selected_scenarios,
        targets,
        is_logscale,
        x_low,
        x_up,
        add_targets=is_targets,
        overlay_uncertainty=is_overlay_unceratinty,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
        dpi_request=dpi_request,
        capitalise_first_letter=capitalise_first_letter,
        legend=is_legend,
    )


PLOT_FUNCS["State-wide notifications"] = plot_overall_notifications