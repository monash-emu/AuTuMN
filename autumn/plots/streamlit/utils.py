import os
import yaml

from autumn import constants
from autumn.plots.plots import validate_plot_config
from autumn.post_processing.processor import validate_post_process_config
from apps.covid_19.plots import load_plot_config as load_covid_plot_config


def load_params(run_dirpath: str) -> dict:
    """
    Loads the run params from the app's data dir
    """
    params_path = os.path.join(run_dirpath, "params.yml")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def load_post_processing_config(app_dirname: str) -> dict:
    """
    Loads the post-processing config from the app's code dir
    """
    app_code_path = try_find_app_code_path(app_dirname)
    pp_config_path = os.path.join(app_code_path, "post-processing.yml")
    with open(pp_config_path, "r") as f:
        config = yaml.safe_load(f)

    validate_post_process_config(config)
    return config


def load_plot_config(app_dirname: str) -> dict:
    """
    Loads the plot config from the app's code dir
    """
    if app_dirname.startswith("covid_"):
        region_name = app_dirname.replace("covid_", "")
        config = load_covid_plot_config(region_name)
    else:
        app_code_path = try_find_app_code_path(app_dirname)
        plot_config_path = os.path.join(app_code_path, "plots.yml")
        with open(plot_config_path, "r") as f:
            config = yaml.safe_load(f)

    validate_plot_config(config)
    return config


def try_find_app_code_path(app_dirname: str) -> str:
    """
    Try to use a model's app dirname in data/ to figure
    out the name of its code directory in apps/

    Returns the path to the app
    This is a little flakey.
    """
    app_folders = [
        d
        for d in os.listdir(constants.APPS_PATH)
        if os.path.isdir(os.path.join(constants.APPS_PATH, d))
    ]
    for app_name in app_folders:
        if app_name.split("_")[0] == app_dirname.split("_")[0]:
            return os.path.join("apps", app_name)

    raise ValueError(f"Could not find app for dirname {app_dirname}")
