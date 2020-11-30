"""
Streamlit web UI for plotting model outputs
"""
import os

import streamlit as st

from autumn import plots, db
from autumn.tool_kit.model_register import AppRegion
from autumn.plots.plotter import StreamlitPlotter

from dash import selectors

PLOT_FUNCS = {}


def plot_outputs_multi(plotter: StreamlitPlotter, app: AppRegion, scenarios: list):
    chosen_scenarios = selectors.scenarios(scenarios)
    if chosen_scenarios:
        output_config = model_output_selector(chosen_scenarios, app.targets)

        is_logscale = st.sidebar.checkbox("Log scale")
        is_custom_xrange = st.sidebar.checkbox("Use custom x axis range.")
        if is_custom_xrange:
            x_low, x_up = selectors.create_xrange_selector(0, 700)
        else:
            x_low = min(scenarios[0].model.times)
            x_up = max(scenarios[0].model.times)

        plots.model.plots.plot_outputs_multi(
            plotter, chosen_scenarios, output_config, is_logscale, x_low, x_up
        )


PLOT_FUNCS["Scenario outputs"] = plot_outputs_multi


def plot_compartment(plotter: StreamlitPlotter, app: AppRegion, scenarios: list):
    chosen_scenarios = selectors.scenarios(scenarios)
    if chosen_scenarios:
        is_logscale = st.sidebar.checkbox("Log scale")
        if len(chosen_scenarios) == 1:
            # Plot many compartments for one scenario
            scenario = chosen_scenarios[0]
            compartment_options = scenario.model.compartment_names
            compartments = selectors.multi_compartment(scenario.model)
            plots.model.plots.plot_multi_compartments_single_scenario(
                plotter, scenario, compartments, is_logscale
            )
        else:
            # Plot one compartment for many scenarios
            compartment_options = chosen_scenarios[0].model.compartment_names
            compartment = selectors.single_compartment(chosen_scenarios[0].model)
            if compartment:
                plots.model.plots.plot_single_compartment_multi_scenario(
                    plotter, chosen_scenarios, compartment, is_logscale
                )
            else:
                st.write("Compartment does not exist")


PLOT_FUNCS["Compartment sizes"] = plot_compartment


def plot_compartment_aggregate(plotter: StreamlitPlotter, app: AppRegion, scenarios: list):
    is_logscale = st.sidebar.checkbox("Log scale")
    names = selectors.multi_compartment(scenarios[0].model)
    plots.model.plots.plot_agg_compartments_multi_scenario(plotter, scenarios, names, is_logscale)
    st.write(names)


PLOT_FUNCS["Compartments aggregate"] = plot_compartment_aggregate


def plot_stacked_compartments_by_stratum(
    plotter: StreamlitPlotter, app: AppRegion, scenarios: list
):
    chosen_scenarios = selectors.scenarios(scenarios)
    compartment = selectors.single_compartment(chosen_scenarios[0].model).split("X")[0]
    stratify_by = "agegroup"
    plots.model.plots.plot_stacked_compartments_by_stratum(
        plotter, chosen_scenarios, compartment, stratify_by
    )
    st.write(compartment)


PLOT_FUNCS["Stacked outputs by stratum"] = plot_stacked_compartments_by_stratum


def plot_stacked_derived_outputs_by_stratum(
    plotter: StreamlitPlotter, app: AppRegion, scenarios: list
):
    chosen_scenarios = selectors.scenarios(scenarios)
    output_config = model_output_selector(chosen_scenarios, app.targets)
    derived_output = output_config["output_key"].split("X")[0]
    stratify_by = "agegroup"
    plots.model.plots.plot_stacked_compartments_by_stratum(
        plotter, chosen_scenarios, derived_output, stratify_by
    )
    st.write(derived_output)


PLOT_FUNCS["Stacked derived by stratum"] = plot_stacked_derived_outputs_by_stratum


def plot_multicountry_rainbow(plotter: StreamlitPlotter, app: AppRegion, scenarios: list):
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
                    country_scenarios[country] = db.load.load_model_scenarios(db_path, params)

                print(
                    "Plotting multicountry rainbow for: "
                    + mode
                    + "_config_"
                    + str(config)
                    + "_"
                    + objective
                )

                plots.model.plots.plot_multicountry_rainbow(
                    country_scenarios, config, mode, objective
                )


PLOT_FUNCS["Multicountry rainbow"] = plot_multicountry_rainbow


def plot_multicounty_hospital(plotter: StreamlitPlotter, app: AppRegion, scenarios: list):
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
                    all_scenarios[mode][objective][config][country] = db.load.load_model_scenarios(
                        db_path, params
                    )

            print("Plotting multicountry hospital for: " + mode + "_" + objective)
            plots.model.plots.plot_multicountry_hospital(all_scenarios, mode, objective)


PLOT_FUNCS["Multicountry hospital"] = plot_multicounty_hospital


def model_output_selector(scenarios, targets):
    """
    Allow user to select the output that they want to select.
    Returns an output config dictionary.
    """
    # Get a list of all the outputs requested by user
    outputs_to_plot = list(targets.values())

    # Get a list of all possible output names
    base_scenario = scenarios[0]
    output_names = base_scenario.model.derived_outputs.keys()
    output_names = [n for n in output_names if n not in ["chain", "run", "scenario"]]

    # Find the names of all the output types and get user to select one

    if any("for_cluster" in o for o in output_names):
        # Handle Victorian multi cluster model
        cluster_options = ["All"] + sorted(
            list(set([o.split("_for_cluster_")[-1] for o in output_names if "_for_cluster_" in o]))
        )
        cluster = st.sidebar.selectbox("Select cluster", cluster_options)
        if cluster == "All":
            output_base_names = sorted(list(o for o in output_names if "_for_cluster_" not in o))
            output_name = st.sidebar.selectbox("Select output type", output_base_names)
        else:
            output_base_names = sorted(
                list(o for o in output_names if f"_for_cluster_{cluster}" in o)
            )
            output_name = st.sidebar.selectbox("Select output type", output_base_names)
    else:
        output_base_names = sorted(list(set([n.split("X")[0] for n in output_names])))
        output_name = st.sidebar.selectbox("Select output type", output_base_names)

    # Construct an output config for the plotting code.
    try:
        output_config = next(o for o in outputs_to_plot if o["output_key"] == output_name)
    except StopIteration:
        output_config = {"output_key": output_name, "values": [], "times": []}

    return output_config
