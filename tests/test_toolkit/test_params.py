import os
import yaml
from tempfile import TemporaryDirectory

from autumn.tool_kit.params import load_params


def test_load_params__with_no_apps():
    """
    Ensure params can be loaded when there is just params.yml
    """
    expected_params = {
        "scenario_start": 85,
        "default": {"n_compartment_repeats": 2, "stratify_by": ["agegroup", "clinical",]},
    }
    with TemporaryDirectory() as dir_path:
        params_path = os.path.join(dir_path, "params.yml")
        with open(params_path, "w") as f:
            yaml.dump(expected_params, f)

        actual_params = load_params(dir_path)
        assert expected_params == actual_params


def test_load_params__with_apps():
    """
    Ensure params can be loaded when there is just params.yml
    """
    base_params = {
        "scenario_start": 85,
        "default": {"n_compartment_repeats": 2, "stratify_by": ["agegroup", "clinical",]},
    }
    app_params = {
        "default": {"n_compartment_repeats": 3},
    }
    expected_params = {
        "scenario_start": 85,
        "default": {"n_compartment_repeats": 3, "stratify_by": ["agegroup", "clinical",]},
    }
    with TemporaryDirectory() as dir_path:
        param_dir = os.path.join(dir_path, "params")
        base_yaml_path = os.path.join(param_dir, "base.yml")
        app_yaml_path = os.path.join(param_dir, "app.yml")
        os.mkdir(param_dir)
        with open(base_yaml_path, "w") as f:
            yaml.dump(base_params, f)
        with open(app_yaml_path, "w") as f:
            yaml.dump(app_params, f)

        actual_params = load_params(dir_path, application="app")
        assert expected_params == actual_params
