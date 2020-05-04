"""
Used to load model parameters from file
"""
import os
import yaml
import logging
from autumn.constants import Compartment
from autumn.tool_kit.utils import find_relative_date_from_string_or_tuple

from .utils import merge_dicts

logger = logging.getLogger(__file__)


def get_mixing_lists_from_dict(working_dict):
    return [i_key for i_key in working_dict.keys()], [i_key for i_key in working_dict.values()]


def revise_mixing_data_for_dicts(parameters):
    list_of_possible_keys = ["home", "other_locations", "school", "work"]
    for age_index in range(16):
        list_of_possible_keys.append("age_" + str(age_index))
    for mixing_key in list_of_possible_keys:
        if mixing_key in parameters:
            (
                parameters[mixing_key + "_times"],
                parameters[mixing_key + "_values"],
            ) = get_mixing_lists_from_dict(parameters[mixing_key])
    return parameters


def revise_dates_if_ymd(mixing_params):
    """
    Find any mixing times parameters that were submitted as a three element list of year, month day - and revise to an
    integer representing the number of days from the reference time.
    """
    for key in (k for k in mixing_params if k.endswith("_times")):
        for i_time, time in enumerate(mixing_params[key]):
            if isinstance(time, (list, str)):
                mixing_params[key][i_time] = find_relative_date_from_string_or_tuple(time)


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

    # Revise any dates for mixing matrices submitted in YMD format
    for param in params:
        if type(params[param]) == dict and "mixing" in params[param]:
            params[param]["mixing"] = revise_mixing_data_for_dicts(params[param]["mixing"])
            revise_dates_if_ymd(params[param]["mixing"])

        if param == "scenarios":
            for scenario in params["scenarios"]:
                params[param][scenario]["mixing"] = revise_mixing_data_for_dicts(
                    params[param][scenario]["mixing"]
                )
                revise_dates_if_ymd(params[param][scenario]["mixing"])


    default = params["default"]
    # Adjust infection for relative all-cause mortality compared to South Korea, if process being applied
    if "ifr_multipliers" in default:
        default["infection_fatality_props"] = [
            i_prop * mult
            for i_prop, mult in zip(default["infection_fatality_props"], default["ifr_multipliers"])
        ]

    # Calculate presymptomatic period from exposed period and relative proportion of that period spent infectious
    if "prop_exposed_presympt" in default:
        default["compartment_periods"][Compartment.EXPOSED] = default["compartment_periods"][
            "incubation"
        ] * (1.0 - default["prop_exposed_presympt"])
        default["compartment_periods"][Compartment.PRESYMPTOMATIC] = (
            default["compartment_periods"]["incubation"] * default["prop_exposed_presympt"]
        )

    # Calculate early infectious period from total infectious period and proportion of that period spent isolated
    if "prop_infectious_early" in default:
        default["compartment_periods"][Compartment.EARLY_INFECTIOUS] = (
            default["compartment_periods"]["infectious"] * default["prop_infectious_early"]
        )
        default["compartment_periods"][Compartment.LATE_INFECTIOUS] = default[
            "compartment_periods"
        ]["infectious"] * (1.0 - default["prop_infectious_early"])

    return params
