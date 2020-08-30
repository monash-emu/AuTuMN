"""
Used to load model parameters from file
"""
import os
import re
import yaml
import json
from copy import deepcopy
from os import path

from autumn.constants import APPS_PATH, BASE_PATH
from autumn.tool_kit.utils import merge_dicts
from autumn.secrets import check_hash


def load_targets(app_name: str, region_name: str):
    """
    Load calibration targets from a YAML file
    """
    region_name = region_name.replace("-", "_")

    region_path = path.join(APPS_PATH, app_name, "regions")
    region_dirnames = [
        d
        for d in os.listdir(region_path)
        if path.isdir(path.join(region_path, d)) and not d == "__pycache__"
    ]
    assert region_name in region_dirnames, f"Region name {region_name} is not in {region_dirnames}"
    region_dir = path.join(region_path, region_name)

    # Load app default param config
    targets_path = path.join(region_dir, "targets.json")
    secret_targets_path = path.join(region_dir, "targets.secret.json")
    if os.path.exists(targets_path):
        with open(targets_path, "r") as f:
            targets = json.load(f)
    elif os.path.exists(secret_targets_path):
        check_hash(secret_targets_path)
        with open(secret_targets_path, "r") as f:
            targets = json.load(f)
    else:
        raise ValueError(
            f"No targets.json or targets.secrets.json found for {app_name} {region_name}"
        )

    return targets


def load_params(app_name: str, region_name: str):
    """
    Load  parameters for the requested app and region.
    This is for loading only, please do not put any pre-processing in here.

    The data structure returned by this function is a little wonky for
    backwards compatibility reasons.
    """
    region_name = region_name.replace("-", "_")
    param_filepaths = get_param_filepaths(app_name, region_name)
    params = {"scenarios": {}}
    is_name_correct = lambda n: re.match(r"^(default)|(scenario-\d+)$", n)
    for param_filepath in param_filepaths:
        name = param_filepath.replace("\\", "/").split("/")[-1].split(".")[0]
        if not is_name_correct(name):
            continue

        if name == "default":
            params["default"] = load_param_file(param_filepath)
        else:
            scenario_idx = int(name.split("-")[-1])
            params["scenarios"][scenario_idx] = load_param_file(param_filepath)

    assert "default" in params, "Region must have a 'default.yml' parameter file."
    return params


def load_param_file(path: str):
    data = read_yaml_file(path)
    parent_path = data["parent"]
    del data["parent"]
    if not parent_path:
        return data
    else:
        abs_parent_path = os.path.join(BASE_PATH, parent_path)
        parent_data = load_param_file(abs_parent_path)
        return merge_dicts(data, parent_data)


def read_yaml_file(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_param_filepaths(app_name: str, region_name: str):
    region_path = path.join(APPS_PATH, app_name, "regions")
    region_dirnames = [
        d
        for d in os.listdir(region_path)
        if path.isdir(path.join(region_path, d)) and not d == "__pycache__"
    ]
    assert region_name in region_dirnames, f"Region name {region_name} is not in {region_dirnames}"
    dirpath = path.join(region_path, region_name, "params")
    return [
        path.join(dirpath, fname)
        for fname in os.listdir(dirpath)
        if fname.endswith(".yml") or fname.endswith(".yaml")
    ]


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
