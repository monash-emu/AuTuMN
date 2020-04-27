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
from autumn.tb_model import LoadedModel
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
    st.sidebar.title("AuTuMN Plots")

    app_dirname = app_selector()
    app_data_dir_path = os.path.join(constants.DATA_PATH, app_dirname)

    model_run_dirname = app_model_run_selector(app_data_dir_path)
    model_run_path = os.path.join(app_data_dir_path, model_run_dirname)

    # Get database from model data dir.
    db_path = os.path.join(model_run_path, "outputs.db")
    out_db = Database(database_name=db_path)

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

    validate_post_process_config(post_processing_config)

    # Load scenarios from the dabase
    scenario_results = out_db.engine.execute("SELECT DISTINCT Scenario FROM outputs;").fetchall()
    scenario_names = sorted(([result[0] for result in scenario_results]))
    scenarios = []
    for scenario_name in scenario_names:
        # Load model outputs from database, build Scenario instance
        outputs = out_db.db_query("outputs", conditions=[f"Scenario='{scenario_name}'"])
        derived_outputs = out_db.db_query(
            "derived_outputs", conditions=[f"Scenario='{scenario_name}'"]
        )
        model = LoadedModel(outputs=outputs.to_dict(), derived_outputs=derived_outputs.to_dict())
        idx = int(scenario_name.split("_")[1])
        scenario = Scenario(model_builder=None, idx=idx, params=params)
        scenario.model = model
        scenario.generated_outputs = post_process(model, post_processing_config)
        scenarios.append(scenario)

    # Create plotter which will write to streamlit UI.
    translations = plot_config["translations"]
    plotter = StreamlitPlotter(translations)

    scenario_idx = scenario_idx_selector(scenarios)

    # Get user to select plot type / scenario
    plot_type = st.sidebar.selectbox("Select plot type", list(PLOT_FUNCS.keys()))
    plot_func = PLOT_FUNCS[plot_type]
    plot_func(plotter, scenarios, scenario_idx, plot_config)


def plot_outputs_multi(plotter: Plotter, scenarios: list, scenario_idx: int, plot_config: dict):
    output_config = _get_output_config(scenarios, plot_config)
    scenario_plots.plot_outputs_multi(plotter, scenarios, output_config)


def plot_outputs_single(plotter: Plotter, scenarios: list, scenario_idx: int, plot_config: dict):
    output_config = _get_output_config(scenarios, plot_config)
    scenario = scenarios[scenario_idx]
    scenario_plots.plot_outputs_single(plotter, scenario, output_config)


def _get_output_config(scenarios, plot_config):
    outputs_to_plot = plot_config["outputs_to_plot"]
    output_names = []
    base_scenario = scenarios[0]
    if base_scenario.generated_outputs:
        output_names += base_scenario.generated_outputs.keys()
    if base_scenario.model.derived_outputs:
        output_names += base_scenario.model.derived_outputs.keys()

    output_name = st.sidebar.selectbox("Select output", output_names)
    try:
        output_config = next(o for o in outputs_to_plot if o["name"] == output_name)
    except StopIteration:
        output_config = {"name": output_name, "target_values": [], "target_times": []}

    return output_config


PLOT_FUNCS = {
    "Multi scenario outputs": plot_outputs_multi,
    "Single scenario output": plot_outputs_single,
    # Request arbitrary output
}


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
    model_run_dirs = [
        f for f in reversed(os.listdir(app_data_dir_path)) if not f.startswith("calibration")
    ]
    model_run_dir_lookup = {}
    # TODO: Order datetimes properly
    for dirname in model_run_dirs:
        datestr = "-".join(dirname.split("-")[-7:])
        run_name = " ".join(dirname.split("-")[:-7]).title()
        run_datetime = datetime.strptime(datestr, "%d-%m-%Y--%H-%M-%S")
        run_datestr = run_datetime.strftime("%-d %b at %-I:%M%p ")
        label = f'{run_datestr} "{run_name}"'
        model_run_dir_lookup[label] = dirname

    labels = list(model_run_dir_lookup.keys())
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
            pretty_title = self.get_plot_title(title_text)
            add_title_to_plot(fig, 1, pretty_title)

        st.pyplot(fig, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
