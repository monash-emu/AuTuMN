"""
Used to load model parameters from file
"""
import os
import yaml
import logging

logger = logging.getLogger(__file__)


def load_params(app_dir: str, application=None):
    """
    Load parameters for the application from the app directory.

    By convention, the parameters will either be 
        - in a single YAML file, named params.yml
        - in a folder called 'params'

    If there is a folder called 'params', then there should be:
        - a base YAML file called 'base.yml'
        - a set of application specific YAML files,
          each with the name of the application as the file name

    args:
        app_dir: the directory that contains params.yml or params folder
        application: the name of the application to be loaded,
                     only used if there are multiple parameter sets
    
    """
    logger.debug(f"Loading params from app dir {app_dir} for application {application}")
    if application:
        # Users wants to load base parameters + application specific params
        param_dir = os.path.join(app_dir, "params")
        base_yaml_path = os.path.join(param_dir, "base.yml")
        app_yaml_path = os.path.join(param_dir, f"{application}.yml")
        if not os.path.exists(param_dir):
            raise FileNotFoundError(f"Param dir not found at {param_dir}")
        if not os.path.exists(base_yaml_path):
            raise FileNotFoundError(f"Base param file not found at {base_yaml_path}")
        if not os.path.exists(app_yaml_path):
            raise FileNotFoundError(
                f"Application {application} param file not found at {app_yaml_path}"
            )

        with open(base_yaml_path, "r") as f:
            base_params = yaml.safe_load(f)
        with open(app_yaml_path, "r") as f:
            app_params = yaml.safe_load(f) or {}

        # Merge base and app params into one parameter set, with app params overwriting base params
        params = merge_dicts(app_params, base_params)

    else:
        # User wants to load params from a single file, no fancy merging
        param_yaml_path = os.path.join(app_dir, "params.yml")
        if not os.path.exists(param_yaml_path):
            raise FileNotFoundError(f"Param file not found at {param_yaml_path}")

        with open(param_yaml_path, "r") as f:
            params = yaml.safe_load(f)

    return params


def merge_dicts(src: dict, dest: dict):
    """
    Merge src dict into dest dict.
    """
    for key, value in src.items():
        if isinstance(value, dict):
            # get node or create one
            node = dest.setdefault(key, {})
            merge_dicts(value, node)
        else:
            dest[key] = value

    return dest
