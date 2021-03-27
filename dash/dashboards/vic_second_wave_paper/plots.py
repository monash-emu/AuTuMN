import os
import random
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from apps.covid_19.model.parameters import Country, Population
from apps.covid_19.model.preprocess.case_detection import get_testing_pop
from apps.covid_19.model.preprocess.testing import find_cdr_function_from_test_data
from autumn import plots
from autumn.plots.calibration.plots import get_epi_params
from autumn.plots.plotter import StreamlitPlotter
from autumn.plots.utils import get_plot_text_dict
from autumn.region import Region
from autumn.utils.params import load_params
from dash.dashboards.calibration_results.plots import (
    create_seroprev_csv,
    get_cdr_constants,
    get_uncertainty_db,
    get_uncertainty_df,
    write_mcmc_centiles,
)
from dash.utils import create_downloadable_csv

STANDARD_X_LIMITS = 153, 275
PLOT_FUNCS = {}
KEY_PARAMS = [
    "seasonal_force",
    "victorian_clusters.metro.mobility.microdistancing.behaviour_adjuster.parameters.effect",
    "victorian_clusters.metro.mobility.microdistancing.face_coverings_adjuster.parameters.effect",
]
STATEWIDE_OUTPUTS = ["notifications", "hospital_admissions", "icu_admissions", "infection_deaths"]
STANDARD_TITLE_FONTSIZE = 20
STANDARD_LABEL_FONTSIZE = 14
STANDARD_N_TICKS = 10


def get_contact_rate_multipliers(mcmc_params):
    return [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "contact_rate_multiplier" in param
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

    vlines = [150, None, None, None]
    plt.style.use("ggplot")
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    chosen_outputs = STATEWIDE_OUTPUTS
    x_low, x_up = STANDARD_X_LIMITS
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_outputs,
        [0],
        targets,
        False,
        x_low,
        x_up,
        STANDARD_N_TICKS,
        title_font_size=STANDARD_TITLE_FONTSIZE,
        label_font_size=STANDARD_LABEL_FONTSIZE,
        file_name="multi_output",
        vlines=vlines,
    )


PLOT_FUNCS["Multi-output uncertainty"] = plot_multiple_timeseries_with_uncertainty


def plot_regional_outputs(
    plotter, calib_dir_path, mcmc_tables, targets, regions, indicator, file_name, max_y_value=None
):
    chosen_outputs = [
        indicator + "_for_cluster_" + i_region.replace("-", "_") for i_region in regions
    ]
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    x_low, x_up = STANDARD_X_LIMITS
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_outputs,
        [0],
        targets,
        False,
        x_low,
        x_up,
        STANDARD_N_TICKS,
        title_font_size=STANDARD_TITLE_FONTSIZE,
        label_font_size=STANDARD_LABEL_FONTSIZE,
        file_name=file_name,
        share_yaxis=True,
        max_y_value=max_y_value,
        custom_titles=[i_region.replace("-", " ") for i_region in regions],
        custom_sup_title=indicator.replace("_", " "),
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
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_METRO,
        "notifications",
        "metro_notifications",
        max_y_value=370.0,
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
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_RURAL,
        "notifications",
        "regional_notifications",
        max_y_value=55.0,
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
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_METRO,
        "hospital_admissions",
        "metro_hospital",
        max_y_value=50.0,
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
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_RURAL,
        "hospital_admissions",
        "regional_hospital",
        max_y_value=5.0,
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
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_METRO,
        "icu_admissions",
        "metro_icu",
        max_y_value=9.0,
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
    file_name: str,
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
        file_name=file_name,
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
        plotter,
        calib_dir_path,
        mcmc_tables,
        mcmc_params,
        get_vic_epi_params(mcmc_params),
        "epi_posteriors",
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
    params = get_contact_rate_multipliers(mcmc_params)
    plot_posteriors(plotter, calib_dir_path, mcmc_tables, mcmc_params, params, "contact_posteriors")


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

    plot_posteriors(plotter, calib_dir_path, mcmc_tables, mcmc_params, KEY_PARAMS, "key_posteriors")


PLOT_FUNCS["Key parameters"] = plot_key_params


def plot_param_matrix(
    plotter: StreamlitPlotter,
    mcmc_params: List[pd.DataFrame],
    parameters: List,
    label_param_string=False,
    show_ticks=False,
    file_name="",
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
        file_name=file_name,
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

    plot_param_matrix(
        plotter,
        mcmc_params,
        mcmc_params[0]["name"].unique().tolist(),
        file_name="all_params_matrix",
    )


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

    plot_param_matrix(
        plotter, mcmc_params, get_vic_epi_params(mcmc_params), file_name="epi_param_matrix"
    )


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

    plot_param_matrix(
        plotter,
        mcmc_params,
        KEY_PARAMS,
        label_param_string=True,
        show_ticks=True,
        file_name="key_param_matrix",
    )


PLOT_FUNCS["Key params matrix"] = plot_key_param_matrix


def plot_key_param_traces(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    title_font_size, label_font_size, dpi_request, capitalise_first_letter, burn_in = (
        8,
        6,
        300,
        False,
        0,
    )
    plots.calibration.plots.plot_multiple_param_traces(
        plotter,
        mcmc_params,
        burn_in,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
        optional_param_request=KEY_PARAMS,
        file_name="key_traces",
        x_ticks_on=False,
    )


PLOT_FUNCS["Key param traces"] = plot_key_param_traces


def plot_epi_param_traces(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    title_font_size, label_font_size, dpi_request, capitalise_first_letter, burn_in = (
        8,
        6,
        300,
        False,
        0,
    )
    plots.calibration.plots.plot_multiple_param_traces(
        plotter,
        mcmc_params,
        burn_in,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
        optional_param_request=get_vic_epi_params(mcmc_params),
        file_name="epi_traces",
        x_ticks_on=False,
    )


PLOT_FUNCS["Epi param traces"] = plot_epi_param_traces


def plot_contact_param_traces(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    title_font_size, label_font_size, dpi_request, capitalise_first_letter, burn_in = (
        8,
        6,
        300,
        False,
        0,
    )
    plots.calibration.plots.plot_multiple_param_traces(
        plotter,
        mcmc_params,
        burn_in,
        title_font_size,
        label_font_size,
        capitalise_first_letter,
        dpi_request,
        optional_param_request=get_contact_rate_multipliers(mcmc_params),
        file_name="contact_traces",
        x_ticks_on=False,
    )


PLOT_FUNCS["Contact rate modifier traces"] = plot_contact_param_traces


def plot_seroprev_age_and_cluster(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    uncertainty_df = get_uncertainty_db(mcmc_tables, targets, calib_dir_path)
    selected_scenario, time = 0, 275
    _, seroprevalence_by_age, overall_seroprev = plots.uncertainty.plots.plot_vic_seroprevalences(
        plotter,
        uncertainty_df,
        selected_scenario,
        time,
        requested_quantiles=[0.025, 0.25, 0.5, 0.75, 0.975],
    )
    create_seroprev_csv(seroprevalence_by_age)
    st.write(overall_seroprev.to_dict())


PLOT_FUNCS["Seroprevalence by age and cluster"] = plot_seroprev_age_and_cluster


def plot_cdr_curves(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    start_date, end_date = STANDARD_X_LIMITS
    samples, label_rotation, region_name = 70, 90, "victoria"
    params = load_params(app_name, region_name)
    (
        iso3,
        testing_year,
        assumed_tests_parameter,
        smoothing_period,
        agegroup_params,
        time_params,
        times,
        agegroup_strata,
    ) = get_cdr_constants(params["default"])

    # Collate parameters into one structure
    testing_to_detection_values = []
    for i_chain in range(len(mcmc_params)):
        param_mask = mcmc_params[i_chain]["name"] == "testing_to_detection.assumed_cdr_parameter"
        testing_to_detection_values += mcmc_params[i_chain]["value"][param_mask].tolist()

    # Sample testing values from all the ones available, to avoid plotting too many curves
    if samples > len(testing_to_detection_values):
        st.write("Warning: Requested samples greater than detection values estimated")
        samples = len(testing_to_detection_values)
    sampled_test_to_detect_vals = random.sample(testing_to_detection_values, samples)

    pop = Population
    country = Country
    country.iso3 = "AUS"

    testing_pop, testing_region = get_testing_pop(agegroup_strata, country, pop)

    detected_proportion = []
    for assumed_cdr_parameter in sampled_test_to_detect_vals:
        detected_proportion.append(
            find_cdr_function_from_test_data(
                assumed_tests_parameter,
                assumed_cdr_parameter,
                smoothing_period,
                iso3,
                testing_pop,
            )
        )

    plots.calibration.plots.plot_cdr_curves(
        plotter,
        times,
        detected_proportion,
        end_date,
        label_rotation,
        start_date=start_date,
        alpha=0.1,
        line_width=1.5,
    )


PLOT_FUNCS["CDR curves"] = plot_cdr_curves


def plot_scenarios(
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
    selected_scenarios = uncertainty_df["scenario"].unique()
    (
        title_font_size,
        label_font_size,
        dpi_request,
        capitalise_first_letter,
        is_logscale,
        is_targets,
        is_overlay_uncertainty,
        is_legend,
    ) = (8, 8, 300, False, False, True, True, True)
    plots.uncertainty.plots.plot_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        chosen_output,
        selected_scenarios,
        targets,
        is_logscale,
        STANDARD_X_LIMITS[0],
        426,
        add_targets=is_targets,
        overlay_uncertainty=is_overlay_uncertainty,
        title_font_size=title_font_size,
        label_font_size=label_font_size,
        dpi_request=dpi_request,
        capitalise_first_letter=capitalise_first_letter,
        legend=is_legend,
    )


PLOT_FUNCS["Scenarios"] = plot_scenarios


def plot_scenarios_multioutput(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    uncertainty_df = get_uncertainty_df(calib_dir_path, mcmc_tables, targets)
    (
        dpi_request,
        capitalise_first_letter,
        is_logscale,
        is_targets,
        is_overlay_uncertainty,
        is_legend,
    ) = (300, False, False, True, True, True)

    scenario_outputs = ["notifications", "infection_deaths", "icu_occupancy", "hospital_occupancy"]
    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        scenario_outputs,
        uncertainty_df["scenario"].unique(),
        targets,
        is_logscale,
        STANDARD_X_LIMITS[0],
        426,
        title_font_size=STANDARD_TITLE_FONTSIZE,
        label_font_size=STANDARD_LABEL_FONTSIZE,
        file_name="multi_scenario"
    )


PLOT_FUNCS["Multi-output scenarios"] = plot_scenarios_multioutput
