"""
Utilities to plot data from existing databases.
"""
import os
import yaml

from autumn.db.models import load_model_scenarios

from .plots import validate_plot_config
from .scenario_plots import plot_scenarios
from .streamlit.utils import try_find_app_code_path

APP_DIRNAMES = ["covid_", "marshall_islands", "mongolia", "dummy"]


def plot_from_database(run_path: str):
    """
    Reads data from an existing model run and re-plots the outputs.
    """
    output_db_path = os.path.join(run_path, "outputs.db")
    assert os.path.exists(output_db_path), "Folder does not contain outputs.db"
    app_dirname = [x for x in run_path.split("/") if x][-2]
    params_path = os.path.join(run_path, "params.yml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    app_code_path = try_find_app_code_path(app_dirname)

    # Load plot config from project dir
    plot_config_path = os.path.join(app_code_path, "plots.yml")
    with open(plot_config_path, "r") as f:
        plot_config = yaml.safe_load(f)

    validate_plot_config(plot_config)

    # Load post processing config from the project dir
    post_processing_path = os.path.join(app_code_path, "post-processing.yml")
    with open(post_processing_path, "r") as f:
        post_processing_config = yaml.safe_load(f)

    # Get database from model data dir.
    db_path = os.path.join(run_path, "outputs.db")
    scenarios = load_model_scenarios(db_path, params, post_processing_config)

    plot_scenarios(scenarios, run_path, plot_config)
