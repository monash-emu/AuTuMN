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


def load_post_processing_config(app_name: str) -> dict:
    """
    Loads the post-processing config from the app's code dir
    """
    pp_config_path = os.path.join(constants.APPS_PATH, app_name, "post-processing.yml")
    with open(pp_config_path, "r") as f:
        config = yaml.safe_load(f)

    validate_post_process_config(config)
    return config


def load_plot_config(app_name: str, param_set_name: str) -> dict:
    """
    Loads the plot config from the app's code dir
    """
    if app_name in ["covid_19", "dr_tb_malancha"]:
        config = load_covid_plot_config(param_set_name)
    else:
        pp_config_path = os.path.join(constants.APPS_PATH, app_name, "plots.yml")
        with open(plot_config_path, "r") as f:
            config = yaml.safe_load(f)

    validate_plot_config(config)
    return config
