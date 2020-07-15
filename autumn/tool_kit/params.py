"""
Used to load model parameters from file
"""
import os
import re
import yaml
from copy import deepcopy
from os import path

from autumn.constants import APPS_PATH
from autumn.tool_kit.utils import merge_dicts


def load_params(app_name: str, region_name: str):
    """
    Load  parameters for the requested app and region.
    This is for loading only, please do not put any pre-processing in here.

    The data structure returned by this function is a little wonky for
    backwards compatibility reasons.
    """

    param_path = path.join(APPS_PATH, app_name, "params")
    assert path.exists(param_path), f"App name {app_name} not found at {param_path}"
    param_dirnames = [
        d
        for d in os.listdir(param_path)
        if path.isdir(path.join(param_path, d)) and not d == "__pycache__"
    ]
    assert region_name in param_dirnames, f"Region name {region_name} is not in {param_dirnames}"
    app_param_dir = path.join(param_path, region_name)

    # Load base param config
    base_yaml_path = path.join(param_path, "base.yml")
    with open(base_yaml_path, "r") as f:
        base_params = yaml.safe_load(f)

    # Load app default param config
    default_param_path = path.join(app_param_dir, "default.yml")
    with open(default_param_path, "r") as f:
        app_default_params = yaml.safe_load(f)

    default_params = merge_dicts(app_default_params, base_params)

    # Load scenario specific params for the given app
    scenarios = {}
    scenaro_params_fnames = [
        fname
        for fname in os.listdir(app_param_dir)
        if fname.startswith("scenario-") and fname.endswith(".yml")
    ]
    for fname in scenaro_params_fnames:
        scenario_idx = int(fname.split("-")[-1].split(".")[0])
        yaml_path = path.join(app_param_dir, fname)
        with open(yaml_path, "r") as f:
            scenarios[scenario_idx] = yaml.safe_load(f)

    # By convention this is outside of the default params
    scenario_start_time = None
    if "scenario_start_time" in default_params:
        scenario_start_time = default_params["scenario_start_time"]
        del default_params["scenario_start_time"]

    # Return backwards-compatible data structure
    return {
        "default": default_params,
        "scenarios": scenarios,
        "scenario_start_time": scenario_start_time,
    }


def update_params(params: dict, updates: dict) -> dict:
    """
    Update parameter dict according to string based update requests in update dict.
    Update requests made as follows:

        - dict entries updated as "key": val
        - nested dict entries updated as "key1.key2": val
        - array entries updated as "arr(0)": 1

    Example

        params = {"foo": 1, "bar" {"baz": 2}, "bing": [1, 2]}
        updates = {"foo": 2, "bar.baz": 3, "bing[1]": 4}
        returns {"foo": 2, "bar" {"baz": 3}, "bing": [1, 4]}

    See tests for more details.
    """
    ps = deepcopy(params)
    for key, val in updates.items():
        ps = _update_params(ps, key, val)

    return ps


# Regex to match an array update request eg. "foo[1]"
ARRAY_REQUEST_REGEX = r"^\w+\(-?\d\)$"


def _update_params(params: dict, update_key: str, update_val) -> dict:
    ps = deepcopy(params)
    keys = update_key.split(".")
    current_key, nested_keys = keys[0], keys[1:]
    is_arr_update = re.match(ARRAY_REQUEST_REGEX, current_key)
    is_nested_update = bool(nested_keys)
    if is_arr_update and is_nested_update:
        # Array item replacement followed by nested dictionary replacement.
        key, idx_str = current_key.replace(")", "").split("(")
        idx = int(idx_str)
        child_key = ".".join(nested_keys)
        ps[key][idx] = _update_params(ps[key][idx], child_key, update_val)
    elif is_arr_update:
        # Array item replacement.
        key, idx_str = update_key.replace(")", "").split("(")
        idx = int(idx_str)
        ps[key][idx] = update_val
    elif is_nested_update:
        # Nested dictionary replacement.
        child_key = ".".join(nested_keys)
        key = current_key
        ps[key] = _update_params(ps[key], child_key, update_val)
    else:
        # Simple key lookup, just replace key with val.
        key = current_key
        ps[key] = update_val

    return ps
