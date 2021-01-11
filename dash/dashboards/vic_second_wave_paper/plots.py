from typing import List
import pandas as pd
import streamlit as st

from autumn.plots.plotter import StreamlitPlotter
from autumn import plots
from dash.dashboards.calibration_results.plots import get_uncertainty_df

from autumn.constants import Region


STANDARD_X_LIMITS = 153, 275
PLOT_FUNCS = {}


def plot_overall_output(plotter, calib_dir_path, mcmc_tables, targets, chosen_output):
    targets = {k: v for k, v in targets.items() if v["output_key"] == chosen_output}
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    x_low, x_up = STANDARD_X_LIMITS
    title_font_size, label_font_size, dpi_request = 8, 8, 300
    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter, uncertainty_df, chosen_output, [0], targets, False, x_low, x_up, add_targets=True,
        overlay_uncertainty=True, title_font_size=title_font_size, label_font_size=label_font_size,
        dpi_request=dpi_request, capitalise_first_letter=False, legend=False,
    )


def plot_overall_notifications(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_overall_output(plotter, calib_dir_path, mcmc_tables, targets, "notifications")


PLOT_FUNCS["State-wide notifications"] = plot_overall_notifications


def plot_overall_hospitalisations(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_overall_output(plotter, calib_dir_path, mcmc_tables, targets, "hospital_admissions")


PLOT_FUNCS["State-wide hospitalisations"] = plot_overall_hospitalisations


def plot_overall_icu_admissions(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_overall_output(plotter, calib_dir_path, mcmc_tables, targets, "icu_admissions")


PLOT_FUNCS["State-wide ICU admissions"] = plot_overall_icu_admissions


def plot_overall_deaths(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_overall_output(plotter, calib_dir_path, mcmc_tables, targets, "infection_deaths")


PLOT_FUNCS["State-wide deaths"] = plot_overall_deaths


def plot_regional_outputs(plotter, calib_dir_path, mcmc_tables, targets, regions, indicator):
    chosen_outputs = [indicator + "_for_cluster_" + i_region.replace("-", "_") for i_region in regions]
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    x_low, x_up = STANDARD_X_LIMITS
    title_font_size, label_font_size, n_xticks = 12, 10, 6
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter, uncertainty_df, chosen_outputs, [0], targets, False, x_low, x_up, n_xticks,
        title_font_size=title_font_size, label_font_size=label_font_size,
    )


def metro_notifications(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_regional_outputs(plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_METRO, "notifications")


PLOT_FUNCS["Metro notifications"] = metro_notifications


def regional_notifications(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_regional_outputs(plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_RURAL, "notifications")


PLOT_FUNCS["Regional notifications"] = regional_notifications


def metro_hospitalisations(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_regional_outputs(plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_METRO, "hospital_admissions")


PLOT_FUNCS["Metro hospitalisations"] = metro_hospitalisations


def regional_hospitalisations(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_regional_outputs(plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_RURAL, "hospital_admissions")


PLOT_FUNCS["Regional hospitalisations"] = regional_hospitalisations


def metro_icu_admissions(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    plot_regional_outputs(plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_METRO, "icu_admissions")


PLOT_FUNCS["Metro ICU admissions"] = metro_icu_admissions
