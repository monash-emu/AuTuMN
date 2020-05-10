"""
Streamlit web UI for plotting model outputs
"""
import os

import streamlit as st

from autumn.db.models import load_model_scenarios
from autumn.plots import plots
from autumn.plots.plotter import StreamlitPlotter

from . import selectors, utils


def run_scenario_plots():
    app_dirname, app_dirpath = selectors.app()
    run_dirname, run_dirpath = selectors.model_run(app_dirpath)

    params = utils.load_params(run_dirpath)
    plot_config = utils.load_plot_config(app_dirname)
    post_processing_config = utils.load_post_processing_config(app_dirname)

    # Get database from model data dir.
    db_path = os.path.join(run_dirpath, "outputs.db")
    scenarios = load_model_scenarios(db_path, params, post_processing_config)

    # Create plotter which will write to streamlit UI.
    translations = plot_config["translations"]
    plotter = StreamlitPlotter(translations)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, scenarios, plot_config)


def plot_outputs_multi(plotter: StreamlitPlotter, scenarios: list, plot_config: dict):
    chosen_scenarios = selectors.scenario(scenarios)
    if chosen_scenarios:
        output_config = model_output_selector(chosen_scenarios, plot_config)
        is_logscale = st.sidebar.checkbox("Log scale")
        plots.plot_outputs_multi(plotter, chosen_scenarios, output_config, is_logscale)


def plot_compartment(plotter: StreamlitPlotter, scenarios: list, plot_config: dict):
    chosen_scenarios = selectors.scenario(scenarios)
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


def plot_compartment_aggregate(plotter: StreamlitPlotter, scenarios: list, plot_config: dict):
    is_logscale = st.sidebar.checkbox("Log scale")
    names = selectors.multi_compartment(scenarios[0].model)
    plots.plot_agg_compartments_multi_scenario(plotter, scenarios, names, is_logscale)
    st.write(names)


PLOT_FUNCS = {
    "Compartment sizes": plot_compartment,
    "Compartments aggregate": plot_compartment_aggregate,
    "Scenario outputs": plot_outputs_multi,
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
