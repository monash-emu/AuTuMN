"""
Entrypoint for Streamlit web UI
Run from a command line shell (with virtualenv active) using

    streamlit run apps/streamlit_plots.py

Website: https://www.streamlit.io/
Docs: https://docs.streamlit.io/
"""
import os
import sys
from datetime import datetime

# Fix import issue by adding autumn project directory to Python path.
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path += [PROJECT_PATH]

# Plus fix Streamlit hot reloading, which requires PYTHONPATH hacks
# https://github.com/streamlit/streamlit/issues/1176
MODULE_DIRNAMES = ["summer", "autumn"]
dirpaths = []
for module_dirname in MODULE_DIRNAMES:
    dirpaths += [
        os.path.join(PROJECT_PATH, d)
        for d, _, _ in os.walk(module_dirname)
        if not "__pycache__" in d
    ]

os.environ["PYTHONPATH"] = ":".join(dirpaths)

import yaml
import streamlit as st

from autumn import constants
from autumn.db import Database
from autumn.tb_model import load_model_scenarios
from autumn.tool_kit import Scenario
from autumn.outputs import scenario_plots
from autumn.outputs.plotter import Plotter, add_title_to_plot
from autumn.post_processing.processor import validate_post_process_config, post_process


class Apps:
    COVID = "covid"
    RMI = "marshall_islands"
    MONGOLIA = "mongolia"


APP_NAMES = [Apps.COVID, Apps.RMI, Apps.MONGOLIA]
APP_FOLDERS = {
    Apps.COVID: "covid_19",
    Apps.RMI: "marshall_islands",
    Apps.MONGOLIA: "mongolia",
}


def main():
    app_dirname = app_selector()
    app_data_dir_path = os.path.join(constants.DATA_PATH, app_dirname)

    model_run_dirname = app_model_run_selector(app_data_dir_path)
    model_run_path = os.path.join(app_data_dir_path, model_run_dirname)

    # Get params from model data dir.
    params_path = os.path.join(model_run_path, "params.yml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    plot_config_path = None
    for _app_name, _app_dir in APP_FOLDERS.items():
        if app_dirname.startswith(_app_name):
            app_code_path = os.path.join("apps", _app_dir)

    assert app_code_path, f"Could not find app code path from {model_run_path}"

    # Load plot config from project dir
    plot_config_path = os.path.join(app_code_path, "plots.yml")
    with open(plot_config_path, "r") as f:
        plot_config = yaml.safe_load(f)

    scenario_plots.validate_plot_config(plot_config)

    # Load post processing config from the project dir
    post_processing_path = os.path.join(app_code_path, "post-processing.yml")
    with open(post_processing_path, "r") as f:
        post_processing_config = yaml.safe_load(f)

    # Get database from model data dir.
    db_path = os.path.join(model_run_path, "outputs.db")
    scenarios = load_model_scenarios(db_path, params, post_processing_config)

    # Create plotter which will write to streamlit UI.
    translations = plot_config["translations"]
    plotter = StreamlitPlotter(translations)

    scenario_idx = scenario_idx_selector(scenarios)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, scenarios, scenario_idx, plot_config)


def plot_outputs_multi(plotter: Plotter, scenarios: list, scenario_idx: int, plot_config: dict):
    output_config = model_output_selector(scenarios, plot_config)
    scenario_plots.plot_outputs_multi(plotter, scenarios, output_config)


def plot_outputs_single(plotter: Plotter, scenarios: list, scenario_idx: int, plot_config: dict):
    output_config = model_output_selector(scenarios, plot_config)
    scenario = scenarios[scenario_idx]
    scenario_plots.plot_outputs_single(plotter, scenario, output_config)


PLOT_FUNCS = {
    "Multi scenario outputs": plot_outputs_multi,
    "Single scenario output": plot_outputs_single,
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


def scenario_idx_selector(scenarios):
    """
    Get user to select the scenario that they want.
    """
    options = ["Baseline"]
    for s in scenarios[1:]:
        options.append(f"Scenario {s.idx}")

    scenario_name = st.sidebar.selectbox("Select scenario", options)
    return options.index(scenario_name)


def app_selector():
    """
    Selector for users to choose which app they want to select
    from the model output data directory.
    """
    app_data_dirs = [
        f
        for f in os.listdir(constants.DATA_PATH)
        if os.path.isdir(os.path.join(f, constants.DATA_PATH))
        and any([f.startswith(n) for n in APP_NAMES])
    ]
    return st.sidebar.selectbox("Select app", app_data_dirs)


def app_model_run_selector(app_data_dir_path: str):
    """
    Allows a user to select what model run they want, given an app 
    """
    # Read model runs from filesystem
    model_run_dirs = [
        f for f in reversed(os.listdir(app_data_dir_path)) if not f.startswith("calibration")
    ]
    # Parse model run folder names
    model_runs = []
    for dirname in model_run_dirs:
        datestr = "-".join(dirname.split("-")[-7:])
        run_name = " ".join(dirname.split("-")[:-7]).title()
        run_datetime = datetime.strptime(datestr, "%d-%m-%Y--%H-%M-%S")
        model_runs.append([run_datetime, run_name, dirname])

    # Sort model runs by date
    model_runs = reversed(sorted(model_runs, key=lambda i: i[0]))

    # Create labels for the select box.
    labels = []
    model_run_dir_lookup = {}
    for run_datetime, run_name, dirname in model_runs:
        run_datestr = run_datetime.strftime("%-d %b at %-I:%M%p ")
        label = f'{run_datestr} "{run_name}"'
        model_run_dir_lookup[label] = dirname
        labels.append(label)

    label = st.sidebar.selectbox("Select app model run", labels)
    return model_run_dir_lookup[label]


class StreamlitPlotter(Plotter):
    """
    Plots stuff just like Plotter, but to Streamlit.
    """

    def __init__(self, translation_dict: dict):
        self.translation_dict = translation_dict

    def save_figure(self, fig, filename: str, subdir=None, title_text=None):
        """
        W
        """
        if title_text:
            pretty_title = self.get_plot_title(title_text).replace("X", " ")
            md = f"<p style='text-align: center;padding-left: 80px'>{pretty_title}</p>"
            st.markdown(md, unsafe_allow_html=True)

        st.pyplot(fig, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
