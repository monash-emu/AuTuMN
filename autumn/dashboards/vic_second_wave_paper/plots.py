import os
import random
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yaml

from autumn.models.covid_19.parameters import Country, Population
from autumn.models.covid_19.preprocess.case_detection import get_testing_pop
from autumn.models.covid_19.preprocess.testing import find_cdr_function_from_test_data
from autumn.tools import plots
from autumn.tools.plots.calibration.plots import get_epi_params, calculate_r_hats
from autumn.tools.plots.plotter import StreamlitPlotter
from autumn.tools.plots.utils import get_plot_text_dict, REF_DATE, change_xaxis_to_date
from autumn.settings import Region
from autumn.dashboards.calibration_results.plots import (
    create_seroprev_csv,
    get_cdr_constants,
    get_uncertainty_db,
    get_uncertainty_df,
    write_mcmc_centiles,
)
from autumn.models.covid_19.constants import BASE_DATETIME
from autumn.tools.utils.utils import apply_moving_average
from autumn.tools.inputs import get_mobility_data
from autumn.tools.project import get_project


from autumn.tools.streamlit.utils import create_downloadable_csv, Dashboard

dash = Dashboard()

STANDARD_X_LIMITS = 153, 275
KEY_PARAMS = [
    "victorian_clusters.metro.mobility.microdistancing.behaviour_adjuster.parameters.effect",
    "victorian_clusters.metro.mobility.microdistancing.face_coverings_adjuster.parameters.effect",
]
STATEWIDE_OUTPUTS = ["notifications", "hospital_admissions", "icu_admissions", "infection_deaths"]
STANDARD_TITLE_FONTSIZE = 20
STANDARD_LABEL_FONTSIZE = 14
STANDARD_N_TICKS = 10
VIC_BURN_INS = 7500

# This has to be specified here, and we would generally want it to be the same as what you requested when asking for the
# full model runs to be triggered in BuildKite, but doesn't have to be.


def get_contact_rate_multipliers(mcmc_params):
    return [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "contact_rate_multiplier" in param
    ]


@dash.register("Multi-output uncertainty")
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
    chosen_outputs = STATEWIDE_OUTPUTS

    # Add vertical lines for the dates of specific policy interventions to the first panel
    multi_panel_vlines = [{}] * len(chosen_outputs)
    multi_panel_vlines[0] = {
        "postcodes": 184,  # 2nd July (11:59pm 1st)
        "stage 3": 191,  # 9th July (11:59pm 8th)
        "face coverings": 205,  # 23rd July (11:59pm 22nd)
        "stage 4": 215.75,  # 6pm 2nd August
    }

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
        multi_panel_vlines=multi_panel_vlines,
    )


def plot_regional_outputs(
    plotter,
    calib_dir_path,
    mcmc_tables,
    targets,
    regions,
    indicator,
    file_name,
    max_y_values=(),
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
        max_y_values=max_y_values,
        custom_titles=[i_region.replace("-", " ") for i_region in regions],
        custom_sup_title=indicator.replace("_", " "),
    )


@dash.register("Metro notifications")
def metro_notifications(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    max_y_value = 370.0
    plot_regional_outputs(
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_METRO,
        "notifications",
        "metro_notifications",
        max_y_values=(max_y_value,) * len(Region.VICTORIA_METRO),
    )


@dash.register("Regional notifications")
def regional_notifications(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    max_y_value = 40.0
    plot_regional_outputs(
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_RURAL,
        "notifications",
        "regional_notifications",
        max_y_values=(max_y_value,) * len(Region.VICTORIA_RURAL),
    )


@dash.register("Metro hospitalisations")
def metro_hospitalisations(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    max_y_value = 50.0
    plot_regional_outputs(
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_METRO,
        "hospital_admissions",
        "metro_hospital",
        max_y_values=(max_y_value,) * len(Region.VICTORIA_METRO),
    )


@dash.register("Regional hospitalisations")
def regional_hospitalisations(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    max_y_value = 5.0
    plot_regional_outputs(
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_RURAL,
        "hospital_admissions",
        "regional_hospital",
        max_y_values=(max_y_value,) * len(Region.VICTORIA_RURAL),
    )


@dash.register("Metro ICU admissions")
def metro_icu_admissions(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    max_y_value = 9.0
    plot_regional_outputs(
        plotter,
        calib_dir_path,
        mcmc_tables,
        targets,
        Region.VICTORIA_METRO,
        "icu_admissions",
        "metro_icu",
        max_y_values=(max_y_value,) * len(Region.VICTORIA_METRO),
    )


def get_vic_epi_params(mcmc_params):
    strings_to_ignore = [
        "dispersion_param",
        "contact_rate_multiplier",
        "target_output_ratio",
    ] + KEY_PARAMS
    params = get_epi_params(mcmc_params, strings_to_ignore=strings_to_ignore)
    return params


def get_vic_contact_params(mcmc_params):
    return [
        param
        for param in mcmc_params[0].loc[:, "name"].unique().tolist()
        if "contact_rate" in param
    ]


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
    ) = (VIC_BURN_INS, 16, 3, 8, 8, 300, False)
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


@dash.register("Epi posteriors")
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


@dash.register("Contact rate modifiers")
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


@dash.register("Key parameters")
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


def plot_param_matrix(
        plotter: StreamlitPlotter,
        mcmc_params: List[pd.DataFrame],
        parameters: List,
        label_param_string=False,
        show_ticks=False,
        file_name="",
        tight_layout=False,
        short_label=False,
):

    burn_in, label_font_size, label_chars, bins, style, dpi_request = (
        VIC_BURN_INS, 8, 2, 20, "Shade", 300
    )
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
        tight_layout=tight_layout,
        short_label=short_label,
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


@dash.register("All params matrix")
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


@dash.register("Epi params matrix")
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


@dash.register("Key params matrix")
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


@dash.register("Contact params matrix")
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
        get_vic_contact_params(mcmc_params),
        label_param_string=True,
        show_ticks=True,
        file_name="contact_param_matrix",
        tight_layout=True,
        short_label=True,
    )


@dash.register("Key params traces")
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
        8, 6, 300, False, VIC_BURN_INS,
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
        x_ticks_on=True,
    )


@dash.register("Epi param traces")
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
        x_ticks_on=True,
    )


@dash.register("Contact rate modifier traces")
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
        x_ticks_on=True,
    )


@dash.register("Seroprevalence by age and cluster")
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


@dash.register("CDR Curves")
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
    samples, label_rotation = 70, 90

    project = get_project("covid_19", "victoria", reload=True)
    params = project.param_set.dump_to_dict()['baseline']

    (
        iso3,
        testing_year,
        assumed_tests_parameter,
        smoothing_period,
        agegroup_params,
        time_params,
        times,
        agegroup_strata,
    ) = get_cdr_constants(params)

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


@dash.register("Contact tracing")
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

    scenario_outputs = [
        "prevalence", "prop_detected_traced", "cdr", "prop_contacts_with_detected_index", "prop_contacts_quarantined"
    ]

    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        scenario_outputs,
        [0],
        targets,
        is_logscale,
        STANDARD_X_LIMITS[0],
        426,
        title_font_size=STANDARD_TITLE_FONTSIZE,
        label_font_size=STANDARD_LABEL_FONTSIZE,
        file_name="contact_tracing",
        max_y_values=(2.0e-3, 1, 1, 1, 1),
    )


@dash.register("Multi-output worse scenarios")
def plot_worse_scenarios_multioutput(
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

    scenario_outputs = ["notifications", "infection_deaths", "hospital_admissions", "icu_occupancy"]

    # From Litton et al.
    icu_capacities = [{}] * len(scenario_outputs)
    icu_capacities[3] = {
        "base ICU beds": 499,
        "max physical ICU beds": 1092,
        "max surge ICU capacity": 1665,
    }

    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        scenario_outputs,
        [2, 3, 4],  # Scenarios to run in this plot
        targets,
        is_logscale,
        STANDARD_X_LIMITS[0],
        426,
        title_font_size=STANDARD_TITLE_FONTSIZE,
        label_font_size=STANDARD_LABEL_FONTSIZE,
        file_name="multi_scenario",
        multi_panel_hlines=icu_capacities,
        max_y_values=(3.2e4, 1.2e3, 6e3, 7e3),
    )


@dash.register("Multi-output better scenarios")
def plot_good_scenarios_multioutput(
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

    scenario_outputs = ["notifications", "infection_deaths", "hospital_admissions", "icu_occupancy"]

    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        scenario_outputs,
        [0, 5, 6],  # Scenarios to run in this plot
        targets,
        is_logscale,
        STANDARD_X_LIMITS[0],
        275,
        title_font_size=STANDARD_TITLE_FONTSIZE,
        label_font_size=STANDARD_LABEL_FONTSIZE,
        file_name="multi_scenario",
    )


@dash.register("Multi-output school scenario")
def plot_school_scenario_multioutput(
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

    scenario_outputs = ["notifications", "infection_deaths", "hospital_admissions", "icu_occupancy"]

    plots.uncertainty.plots.plot_multi_output_timeseries_with_uncertainty(
        plotter,
        uncertainty_df,
        scenario_outputs,
        [0, 1],  # Scenarios to run in this plot
        targets,
        is_logscale,
        STANDARD_X_LIMITS[0],
        275,
        title_font_size=STANDARD_TITLE_FONTSIZE,
        label_font_size=STANDARD_LABEL_FONTSIZE,
        file_name="multi_scenario",
    )


@dash.register("Mobility by cluster")
def plot_multicluster_mobility(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):
    project = get_project("covid_19", "victoria", reload=True)
    params = project.param_set.dump_to_dict()['baseline']

    all_cluster_mobility_values = {}
    fig, axes, max_dims, n_rows, n_cols, _ = plotter.get_figure()

    for i_region in Region.VICTORIA_METRO + Region.VICTORIA_RURAL:
        google_mobility_values, google_mobility_days = get_mobility_data(
            params["country"]["iso3"],
            i_region.replace("-", "_").upper(),
            BASE_DATETIME,
            params["mobility"]["google_mobility_locations"],
        )

        all_cluster_mobility_values[i_region] = google_mobility_values
    for i_region in Region.VICTORIA_METRO:
        axes.plot(
            google_mobility_days,
            apply_moving_average(all_cluster_mobility_values[i_region]["work"], 7),
            color="k",
            alpha=0.5,
        )
        axes.plot(
            google_mobility_days,
            apply_moving_average(all_cluster_mobility_values[i_region]["other_locations"], 7),
            color="g",
            alpha=0.5,
        )
    for i_region in Region.VICTORIA_RURAL:
        axes.plot(
            google_mobility_days,
            apply_moving_average(all_cluster_mobility_values[i_region]["work"], 7),
            color="b",
            alpha=0.5,
        )
        axes.plot(
            google_mobility_days,
            apply_moving_average(all_cluster_mobility_values[i_region]["other_locations"], 7),
            color="brown",
            alpha=0.5,
        )
    axes.set_xlim(left=STANDARD_X_LIMITS[0], right=STANDARD_X_LIMITS[1])
    axes.set_ylim(top=1.0)
    change_xaxis_to_date(axes, REF_DATE, rotation=0)

    plotter.save_figure(fig, filename=f"multi_cluster_mobility", title_text="Google mobility")


@dash.register("R_hat convergence statistics")
def display_parameters_r_hats(
    plotter: StreamlitPlotter,
    calib_dir_path: str,
    mcmc_tables: List[pd.DataFrame],
    mcmc_params: List[pd.DataFrame],
    targets: dict,
    app_name: str,
    region: str,
):

    r_hats = calculate_r_hats(mcmc_params, mcmc_tables, burn_in=VIC_BURN_INS)
    st.write("Convergence R_hat statistics for each parameter.\nWe want these values to be as close as possible to 1 (ideally < 1.1).")
    st.write(r_hats)

