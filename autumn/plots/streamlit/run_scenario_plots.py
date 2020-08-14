"""
Streamlit web UI for plotting model outputs
"""
import os
from importlib import import_module

import streamlit as st

from autumn.db.models import load_model_scenarios
from autumn.plots import plots
from autumn.plots.plotter import StreamlitPlotter
from autumn.tool_kit.model_register import RegionAppBase
from apps.covid_19.preprocess.mixing_matrix.adjust_location import (
    LocationMixingAdjustment,
    LOCATIONS,
    MICRODISTANCING_LOCATIONS,
)

from . import selectors, utils


def run_scenario_plots():
    app_name, app_dirpath = selectors.app(run_type="run")
    if not app_name:
        st.write("No applications have been run yet")
        return

    param_set_name, param_set_dirpath = selectors.param_set(app_dirpath)
    if not param_set_name:
        st.write("No parameter set folder found")
        return

    run_datestr, run_dirpath = selectors.model_run(param_set_dirpath)
    if not run_datestr:
        st.write("No model run folder found")
        return

    params = utils.load_params(run_dirpath)
    plot_config = utils.load_plot_config(app_name, param_set_name)

    # Get database from model data dir.
    db_path = os.path.join(run_dirpath, "outputs.db")
    scenarios = load_model_scenarios(db_path, params)

    # Create plotter which will write to streamlit UI.
    translations = plot_config["translations"]
    plotter = StreamlitPlotter(translations)

    # Import the app so we can re-build the model if we need to
    app_module = import_module(f"apps.{app_name}")
    region_app = app_module.get_region_app(param_set_name)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, region_app, scenarios, plot_config)


def plot_outputs_multi(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    chosen_scenarios = selectors.scenarios(scenarios)
    if chosen_scenarios:
        output_config = model_output_selector(chosen_scenarios, plot_config)
        is_logscale = st.sidebar.checkbox("Log scale")
        plots.plot_outputs_multi(plotter, chosen_scenarios, output_config, is_logscale)


def plot_compartment(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    chosen_scenarios = selectors.scenarios(scenarios)
    if chosen_scenarios:
        is_logscale = st.sidebar.checkbox("Log scale")
        if len(chosen_scenarios) == 1:
            # Plot many compartments for one scenario
            scenario = chosen_scenarios[0]
            compartment_options = scenario.model.compartment_names
            compartments = selectors.multi_compartment(scenario.model)
            plots.plot_multi_compartments_single_scenario(
                plotter, scenario, compartments, is_logscale
            )
        else:
            # Plot one compartment for many scenarios
            compartment_options = chosen_scenarios[0].model.compartment_names
            compartment = selectors.single_compartment(chosen_scenarios[0].model)
            if compartment:
                plots.plot_single_compartment_multi_scenario(
                    plotter, chosen_scenarios, compartment, is_logscale
                )
            else:
                st.write("Compartment does not exist")


def plot_compartment_aggregate(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    is_logscale = st.sidebar.checkbox("Log scale")
    names = selectors.multi_compartment(scenarios[0].model)
    plots.plot_agg_compartments_multi_scenario(plotter, scenarios, names, is_logscale)
    st.write(names)


def plot_dynamic_inputs(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    # Just use the base scenario for now
    chosen_scenarios = selectors.scenarios(scenarios, include_all=False)
    scenario = chosen_scenarios[0]
    # Assume a COVID model
    model = app.build_model(scenario.params["default"])
    tvs = model.time_variants
    tv_key = st.sidebar.selectbox("Select function", list(tvs.keys()))
    is_logscale = st.sidebar.checkbox("Log scale")
    tv_func = tvs[tv_key]
    plots.plot_time_varying_input(plotter, tv_key, tv_func, model.times, is_logscale)


def plot_location_mixing(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    chosen_scenarios = selectors.scenarios(scenarios, include_all=False)
    scenario = chosen_scenarios[0]

    # Assume a COVID model
    params = scenario.params["default"]
    mixing = params.get("mixing")
    if not mixing:
        st.write("This model does not have location based mixing")

    loc_key = st.sidebar.selectbox("Select location", LOCATIONS)
    is_logscale = st.sidebar.checkbox("Log scale")

    country_iso3 = params["iso3"]
    region = params["region"]
    microdistancing = params["microdistancing"]
    npi_effectiveness_params = params["npi_effectiveness"]
    google_mobility_locations = params["google_mobility_locations"]
    is_periodic_intervention = params.get("is_periodic_intervention")
    periodic_int_params = params.get("periodic_intervention")
    smooth_google_data = params.get("smooth_google_data")
    adjust = LocationMixingAdjustment(
        country_iso3,
        region,
        mixing,
        npi_effectiveness_params,
        google_mobility_locations,
        is_periodic_intervention,
        periodic_int_params,
        params["end_time"],
        microdistancing,
        smooth_google_data,
    )

    if adjust.microdistancing_function and loc_key in MICRODISTANCING_LOCATIONS:
        loc_func = lambda t: adjust.microdistancing_function(t) * adjust.loc_adj_funcs[loc_key](t)
    elif loc_key in adjust.loc_adj_funcs:
        loc_func = lambda t: adjust.loc_adj_funcs[loc_key](t)
    else:
        loc_func = lambda t: 1

    plots.plot_time_varying_input(plotter, loc_key, loc_func, scenario.model.times, is_logscale)


def plot_stacked_compartments_by_stratum(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    chosen_scenarios = selectors.scenarios(scenarios)
    compartment = selectors.single_compartment(chosen_scenarios[0].model).split("X")[0]
    stratify_by = "agegroup"
    plots.plot_stacked_compartments_by_stratum(plotter, chosen_scenarios, compartment, stratify_by)
    st.write(compartment)


def plot_stacked_derived_outputs_by_stratum(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    chosen_scenarios = selectors.scenarios(scenarios)
    output_config = model_output_selector(chosen_scenarios, plot_config)
    derived_output = output_config["name"].split("X")[0]
    stratify_by = "agegroup"
    plots.plot_stacked_compartments_by_stratum(
        plotter, chosen_scenarios, derived_output, stratify_by
    )
    st.write(derived_output)


def plot_multicountry_rainbow(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    root_path = os.path.join("data", "outputs", "run", "covid_19")

    for mode in ["by_age", "by_location"]:
        for config in [2, 3]:
            for objective in ["deaths", "yoll"]:
                full_path = os.path.join(
                    root_path, mode + "_config_" + str(config) + "_" + objective
                )
                country_scenarios = {}
                for country in countries:
                    country_dirpath = os.path.join(full_path, country)
                    dir_name = os.listdir(country_dirpath)[0]
                    run_dirpath = os.path.join(country_dirpath, dir_name)

                    params = utils.load_params(run_dirpath)

                    # Get database from model data dir.
                    db_path = os.path.join(run_dirpath, "outputs.db")
                    country_scenarios[country] = load_model_scenarios(db_path, params)

                print(
                    "Plotting multicountry rainbow for: "
                    + mode
                    + "_config_"
                    + str(config)
                    + "_"
                    + objective
                )

                plots.plot_multicountry_rainbow(country_scenarios, config, mode, objective)


def plot_multicounty_hospital(
    plotter: StreamlitPlotter, app: RegionAppBase, scenarios: list, plot_config: dict
):
    countries = ["belgium", "france", "italy", "spain", "sweden", "united-kingdom"]
    root_path = os.path.join("data", "outputs", "run", "covid_19")

    all_scenarios = {}
    for mode in ["by_age", "by_location"]:
        all_scenarios[mode] = {}
        for objective in ["deaths", "yoll"]:
            all_scenarios[mode][objective] = {}
            for config in [2, 3]:
                full_path = os.path.join(
                    root_path, mode + "_config_" + str(config) + "_" + objective
                )
                all_scenarios[mode][objective][config] = {}
                for country in countries:
                    country_dirpath = os.path.join(full_path, country)
                    dir_name = os.listdir(country_dirpath)[0]
                    run_dirpath = os.path.join(country_dirpath, dir_name)

                    params = utils.load_params(run_dirpath)

                    # Get database from model data dir.
                    db_path = os.path.join(run_dirpath, "outputs.db")
                    all_scenarios[mode][objective][config][country] = load_model_scenarios(
                        db_path, params
                    )

            print("Plotting multicountry hospital for: " + mode + "_" + objective)
            plots.plot_multicountry_hospital(all_scenarios, mode, objective)


PLOT_FUNCS = {
    "Compartment sizes": plot_compartment,
    "Compartments aggregate": plot_compartment_aggregate,
    "Scenario outputs": plot_outputs_multi,
    "Dynamic input functions": plot_dynamic_inputs,
    "Dynamic location mixing": plot_location_mixing,
    # "Stacked outputs by stratum": plot_stacked_compartments_by_stratum,
    # "Stacked derived by stratum": plot_stacked_derived_outputs_by_stratum,
    # "Multicountry rainbow": plot_multicountry_rainbow,
    # "Multicountry hospital": plot_multicounty_hospital,
}


def model_output_selector(scenarios, plot_config):
    """
    Allow user to select the output that they want to select.
    Returns an output config dictionary.
    """
    # Get a list of all the outputs requested by user
    outputs_to_plot = plot_config["outputs_to_plot"]

    # Get a list of all possible output names
    output_names = []
    base_scenario = scenarios[0]
    if base_scenario.generated_outputs:
        output_names += base_scenario.generated_outputs.keys()
    if base_scenario.model.derived_outputs:
        output_names += base_scenario.model.derived_outputs.keys()

    # Find the names of all the output types and get user to select one
    output_base_names = list(set([n.split("X")[0] for n in output_names]))
    output_base_name = st.sidebar.selectbox("Select output type", output_base_names)

    # Find the names of all the strata available to be plotted
    output_strata_names = [
        "X".join(n.split("X")[1:]) for n in output_names if n.startswith(output_base_name)
    ]
    has_strata_names = any(output_strata_names)
    if has_strata_names:
        # If there are strata names to select, get the user to select one.
        output_strata = st.sidebar.selectbox("Select output strata", output_strata_names)
        output_name = f"{output_base_name}X{output_strata}"
    else:
        # Otherwise just use the base name - no selection required.
        output_name = output_base_name

    # Construct an output config for the plotting code.
    try:
        output_config = next(o for o in outputs_to_plot if o["name"] == output_name)
    except StopIteration:
        output_config = {"name": output_name, "target_values": [], "target_times": []}

    return output_config
