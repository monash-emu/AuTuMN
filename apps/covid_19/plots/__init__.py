"""
Used to load model parameters from file
"""
import os
import yaml
from os import path

from autumn.constants import APPS_PATH
from autumn.tool_kit.utils import merge_dicts


def load_plot_config(region_name: str):
    """
    Load  plot config requested COVID region.
    This is for loading only, please do not put any pre-processing in here.
    """
    plots_path = path.join(APPS_PATH, "covid_19", "plots")

    # Load base plot config
    base_yaml_path = path.join(plots_path, "base.yml")
    with open(base_yaml_path, "r") as f:
        base_plot_config = yaml.safe_load(f)

    # Load region plot config
    region_plot_path = path.join(plots_path, f"{region_name}.yml")
    with open(region_plot_path, "r") as f:
        region_plot_config = yaml.safe_load(f) or {}

    return merge_dicts(region_plot_config, base_plot_config)
