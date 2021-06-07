"""
Streamlit web UI for plotting model outputs
"""
import os

import streamlit as st
from typing import List
from summer import CompartmentalModel

from autumn.tools import db, plots
from autumn.tools.plots.plotter import StreamlitPlotter
from autumn.tools.streamlit import selectors
from autumn.tools.streamlit.utils import Dashboard
from autumn.tools.project import Project

dash = Dashboard()


@dash.register("Scenario outputs")
def plot_outputs_multi(
    plotter: StreamlitPlotter, project: Project, models: List[CompartmentalModel]
):
    chosen_models = selectors.scenarios(models)
    if chosen_models:
        output_config = model_output_selector(chosen_models, project.plots)

        is_logscale = st.sidebar.checkbox("Log scale")
        is_custom_xrange = st.sidebar.checkbox("Use custom x axis range.")
        if is_custom_xrange:
            x_low, x_up = selectors.create_xrange_selector(0, 700)
        else:
            x_low = min(models[0].times)
            x_up = max(models[0].times)

        plots.model.plots.plot_outputs_multi(
            plotter, chosen_models, output_config, is_logscale, x_low, x_up
        )


@dash.register("Compartment sizes")
def plot_compartment(plotter: StreamlitPlotter, project: Project, models: List[CompartmentalModel]):
    chosen_models = selectors.scenarios(models)
    if chosen_models:
        is_logscale = st.sidebar.checkbox("Log scale")
        if len(chosen_models) == 1:
            # Plot many compartments for one scenario
            model = chosen_models[0]
            compartments = selectors.multi_compartment(model)
            plots.model.plots.plot_multi_compartments_single_scenario(
                plotter, model, compartments, is_logscale
            )
        else:
            # Plot one compartment for many scenarios
            compartment = selectors.single_compartment(chosen_models[0])
            if compartment:
                plots.model.plots.plot_single_compartment_multi_scenario(
                    plotter, chosen_models, compartment, is_logscale
                )
            else:
                st.write("Compartment does not exist")


@dash.register("Compartments aggregate")
def plot_compartment_aggregate(
    plotter: StreamlitPlotter, project: Project, models: List[CompartmentalModel]
):
    is_logscale = st.sidebar.checkbox("Log scale")
    names = selectors.multi_compartment(models[0])
    plots.model.plots.plot_agg_compartments_multi_scenario(plotter, models, names, is_logscale)
    st.write(names)


@dash.register("Stacked outputs by stratum")
def plot_stacked_compartments_by_stratum(
    plotter: StreamlitPlotter, project: Project, models: List[CompartmentalModel]
):
    chosen_models = selectors.scenarios(models)
    compartment = selectors.single_compartment(chosen_models[0]).split("X")[0]
    stratify_by = "agegroup"
    plots.model.plots.plot_stacked_compartments_by_stratum(
        plotter, chosen_models, compartment, stratify_by
    )
    st.write(compartment)


@dash.register("Stacked derived by stratum")
def plot_stacked_derived_outputs_by_stratum(
    plotter: StreamlitPlotter, project: Project, models: List[CompartmentalModel]
):
    chosen_models = selectors.scenarios(models)
    output_config = model_output_selector(chosen_models, project.plots)
    derived_output = output_config["output_key"].split("X")[0]
    stratify_by = "agegroup"
    plots.model.plots.plot_stacked_compartments_by_stratum(
        plotter, chosen_models, derived_output, stratify_by
    )
    st.write(derived_output)


@dash.register("Multicountry rainbow")
def plot_multicountry_rainbow(
    plotter: StreamlitPlotter, project: Project, models: List[CompartmentalModel]
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


@dash.register("Multicountry hospital")
def plot_multicounty_hospital(
    plotter: StreamlitPlotter, project: Project, models: List[CompartmentalModel]
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
                    all_scenarios[mode][objective][config][country] = db.load.load_model_scenarios(
                        db_path, params
                    )

            print("Plotting multicountry hospital for: " + mode + "_" + objective)
            plots.model.plots.plot_multicountry_hospital(all_scenarios, mode, objective)


def model_output_selector(models, targets):
    """
    Allow user to select the output that they want to select.
    Returns an output config dictionary.
    """
    # Get a list of all the outputs requested by user
    outputs_to_plot = list(targets.values())

    # Get a list of all possible output names
    base_scenario = models[0]
    output_names = base_scenario.derived_outputs.keys()

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
