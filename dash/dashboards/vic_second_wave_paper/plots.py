from typing import List
import pandas as pd
import streamlit as st
import os
import yaml
import matplotlib.pyplot as plt

from autumn.plots.plotter import StreamlitPlotter
from autumn import plots
from dash.dashboards.calibration_results.plots import get_uncertainty_df, write_mcmc_centiles
from autumn.plots.calibration.plots import get_epi_params
from dash.utils import create_downloadable_csv
from autumn.plots.utils import get_plot_text_dict

from autumn.region import Region


STANDARD_X_LIMITS = 153, 275
PLOT_FUNCS = {}
KEY_PARAMS = [
    "seasonal_force",
    "victorian_clusters.metro.mobility.microdistancing.behaviour_adjuster.parameters.effect",
    "victorian_clusters.metro.mobility.microdistancing.face_coverings_adjuster.parameters.effect",
]


def plot_multiple_timeseries_with_uncertainty(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    plt.style.use("ggplot")
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    chosen_outputs = ["notifications", "hospital_admissions", "icu_admissions", "infection_deaths"]
    x_low, x_up = STANDARD_X_LIMITS
    selected_scenarios, is_logscale, n_xticks, title_font_size, label_font_size = [0], False, 6, 20, 15
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter, uncertainty_df, chosen_outputs, selected_scenarios, targets, is_logscale, x_low, x_up, n_xticks,
        title_font_size=title_font_size, label_font_size=label_font_size,
    )


PLOT_FUNCS["Multi-output uncertainty"] = plot_multiple_timeseries_with_uncertainty


def plot_regional_outputs(plotter, calib_dir_path, mcmc_tables, targets, regions, indicator):
    chosen_outputs = [
        indicator + "_for_cluster_" + i_region.replace("-", "_") for i_region in regions
    ]
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    x_low, x_up = STANDARD_X_LIMITS
    title_font_size, label_font_size, n_xticks = 12, 10, 6
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_outputs,
        [0],
        targets,
        False,
        x_low,
        x_up,
        n_xticks,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
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
    plot_regional_outputs(
        plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_METRO, "notifications"
    )


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
    plot_regional_outputs(
        plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_RURAL, "notifications"
    )


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
    plot_regional_outputs(
        plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_METRO, "hospital_admissions"
    )


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
    plot_regional_outputs(
        plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_RURAL, "hospital_admissions"
    )


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
    plot_regional_outputs(
        plotter, calib_dir_path, mcmc_tables, targets, Region.VICTORIA_METRO, "icu_admissions"
    )


PLOT_FUNCS["Metro ICU admissions"] = metro_icu_admissions


def get_vic_epi_params(mcmc_params):
    strings_to_ignore = ["dispersion_param", "contact_rate_multiplier"] + KEY_PARAMS
    params = get_epi_params(mcmc_params, strings_to_ignore=strings_to_ignore)
    return params


def plot_posteriors(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    params: List,
):

    st.write(params)
    priors = []
    try:
        priors_path = os.path.join(calib_dir_path, "priors-1.yml")
        with open(priors_path) as file:
            priors = yaml.load(file, Loader=yaml.FullLoader)
    except:
        st.write("Check if priors-1.yml exists in the output folder")
    (
        burn_in,
        num_bins,
        sig_figs,
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
    ) = (0, 16, 3, 8, 8, 300, False)
    plots.calibration.plots.plot_multiple_posteriors(
        plotter,
        mcmc_params,
        mcmc_tables,
        burn_in,
        num_bins,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
        priors,
        parameters=params,
    )
    write_mcmc_centiles(mcmc_params, mcmc_tables, burn_in, sig_figs, [2.5, 50, 97.5])


def plot_epi_posteriors(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    plot_posteriors(
        plotter, calib_dir_path, mcmc_tables, mcmc_params, get_vic_epi_params(mcmc_params)
    )


PLOT_FUNCS["Epi posteriors"] = plot_epi_posteriors


def plot_contact_rate_modifiers(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    params = [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "contact_rate_multiplier" in param
    ]
    plot_posteriors(plotter, calib_dir_path, mcmc_tables, mcmc_params, params)


PLOT_FUNCS["Contact rate modifiers"] = plot_contact_rate_modifiers


def plot_key_params(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    plot_posteriors(plotter, calib_dir_path, mcmc_tables, mcmc_params, KEY_PARAMS)


PLOT_FUNCS["Key parameters"] = plot_key_params


def plot_param_matrix(
        plotter: StreamlitPlotter,
        mcmc_params: List[pd.DataFrame],
        parameters: List,
        label_param_string=False,
        show_ticks=False,
):

    burn_in, label_font_size, label_chars, bins, style, dpi_request = 0, 8, 2, 20, "Shade", 300
    plots.calibration.plots.plot_param_vs_param(
        plotter,
        mcmc_params,
        parameters,
        burn_in,
        style,
        bins,
        label_font_size,
        label_chars,
        dpi_request,
        label_param_string=label_param_string,
        show_ticks=show_ticks,
    )
    param_names = [get_plot_text_dict(param) for param in parameters]
    params_df = pd.DataFrame({"names": param_names})
    params_df["numbers"] = range(1, len(params_df) + 1)
    create_downloadable_csv(params_df, "parameter_indices")
    key_string = ""
    for i_param, param_name in enumerate(param_names):
        key_string += str(i_param + 1) + ", " + param_name + "; "
    st.write(key_string)
    st.dataframe(params_df)


def plot_all_param_matrix(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    plot_param_matrix(plotter, mcmc_params, mcmc_params[0]["name"].unique().tolist())


PLOT_FUNCS["All params matrix"] = plot_all_param_matrix


def plot_epi_param_matrix(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    plot_param_matrix(plotter, mcmc_params, get_vic_epi_params(mcmc_params))


PLOT_FUNCS["Epi params matrix"] = plot_epi_param_matrix


def plot_key_param_matrix(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    plot_param_matrix(plotter, mcmc_params, KEY_PARAMS, label_param_string=True, show_ticks=True)


PLOT_FUNCS["Key params matrix"] = plot_key_param_matrix

