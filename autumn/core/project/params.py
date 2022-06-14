from typing import Optional, Callable, Union, List
import operator
from functools import reduce
import re
import yaml
from copy import deepcopy

from autumn.core.utils.utils import merge_dicts


Validator = Callable[[dict], None]
PathOrDict = Union[dict, str]


class ParamFormat:
    # A dict with many nested dicts, arrays etc.
    STANDARD = "STANDARD"
    # A flat dict with target structure encoded in the keys.
    CALIBRATION = "CALIBRATION"


class Params:
    """
    A set of parameters that can be loaded by a model.
    """

    def __init__(
        self, data: PathOrDict, validator: Optional[Validator] = None, validate: bool = True
    ):
        self._params, self._fmts = [], []
        self._validator = validator
        self._update(data, ParamFormat.STANDARD, validate=validate)

    def to_dict(self) -> dict:
        """
        Returns params as a dict.
        """
        return self._build_dict()

    def update(self, new_params: PathOrDict, calibration_format=False, validate: bool = True):
        """
        Load some more parameters, overwriting existing where conflicts occur.
        Returns a copy of the current params
        """
        # Load in the data.
        fmt = ParamFormat.CALIBRATION if calibration_format else ParamFormat.STANDARD
        self_copy = self.copy()
        self_copy._update(new_params, fmt, validate)
        return self_copy

    def copy(self):
        self_copy = Params({})
        self_copy._params = deepcopy(self._params)
        self_copy._fmts = deepcopy(self._fmts)
        self_copy._validator = self._validator
        return self_copy

    def _update(self, new_params: PathOrDict, fmt: str, validate: bool = True):
        # Load the data
        params = self._load_path_or_dict(new_params)
        # Insert into list of params.
        self._params = [*self._params, params]
        self._fmts = [*self._fmts, fmt]
        # Check that everything is still valid.
        final_params = self._build_dict()
        if self._validator and validate:
            self._validator(final_params)

    def _build_dict(self) -> dict:
        final_params = {}
        for params, fmt in zip(self._params, self._fmts):
            if fmt == ParamFormat.CALIBRATION:
                final_params = update_params(final_params, params)
            elif fmt == ParamFormat.STANDARD:
                final_params = merge_dicts(params, final_params)
            else:
                raise ValueError(f"Unknown parameter format {fmt}")

        # Remove parent key, from old implementation.
        old_meta_keys = ["parent"]
        for k in old_meta_keys:
            if k in final_params:
                del final_params[k]

        # Add default key if not exists
        default_keys = ["description"]
        for k in default_keys:
            if k not in final_params:
                final_params[k] = None

        # Remove dispersion param keys
        keys = list(final_params.keys())
        for k in keys:
            if k.endswith("dispersion_param"):
                del final_params[k]

        return final_params

    def _load_path_or_dict(self, new_params: PathOrDict) -> dict:
        t = type(new_params)
        if t is str:
            # It's a path (hopefully), load it.
            return read_yaml_file(new_params)
        elif t is dict:
            # It's a dict so we don't need to do anything.
            return new_params
        else:
            raise ValueError(f"Loaded parameter data must be a string or dict, got {t}")

    def __reduce__(self):
        return (self.__class__, (self.to_dict(), ))

    def __repr__(self):
        return "Params" + repr(self.to_dict())

    def __pretty__(self, printer):
        return "Params\n" + printer.pformat(self.to_dict())

    def __getitem__(self, k):
        return self.to_dict()[k]


class ParameterSet:
    """
    A collection of model parameters, each representing a counterfactual scenario.
    Typically there is a single baseline parameter set, and zero or more scenarios.
    """

    baseline: Params
    scenarios: List[Params]

    def __init__(
        self,
        baseline: Params,
        scenarios: List[Params] = [],
    ):
        self.baseline = baseline
        self.scenarios = scenarios

    def update_baseline(self, new_params: PathOrDict, calibration_format=False):
        self.baseline.update(new_params, calibration_format)

    def update_scenarios(self, new_params: PathOrDict, calibration_format=False):
        for sc in self.scenarios:
            sc.update(new_params, calibration_format)

    def update_all(self, new_params: PathOrDict, calibration_format=False):
        self.update_baseline(new_params, calibration_format)
        self.update_scenarios(new_params, calibration_format)

    def dump_to_dict(self):
        return {
            "baseline": self.baseline.to_dict(),
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def __repr__(self):
        return "ParameterSet\n" + repr(self.dump_to_dict())

    def __pretty__(self, printer):
        return "ParameterSet\n" + printer.pformat(self.dump_to_dict())


def read_yaml_file(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


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
ARRAY_REQUEST_REGEX = r"^\w+\(-?\d+\)$"


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


def read_param_value_from_string(params: dict, update_key: str):

    keys = update_key.split(".")
    current_key, nested_keys = keys[0], keys[1:]
    is_arr_update = re.match(ARRAY_REQUEST_REGEX, current_key)
    assert (
        not is_arr_update
    ), "array items not supported by this function"  # FIXME only supports nested dictionaries for for the moment
    param_value = params[current_key]
        
    for nested_key in nested_keys:
        is_arr_key = re.match(ARRAY_REQUEST_REGEX, nested_key)
        if is_arr_key:
            key, idx_str = nested_key.replace(")", "").split("(")
            idx = int(idx_str)
            param_value = param_value[key][idx]
        else:
            param_value = param_value[nested_key]

    return param_value


def get_param_from_nest_string(
    parameters: dict, 
    param_request: str,
) -> Union[int, float, str]:
    """
    Get the value of a parameter from a parameters dictionary, using a single string
    defining the parameter name, with "." characters to separate the tiers of the
    keys in the nested parameter dictionary.
    
    Args:
        parameters: The full parameter set to look int
        param_request: The single request submitted by the user
    Return:
        The value of the parameter being requested
    """
    return reduce(operator.getitem, param_request.split("."), parameters.to_dict())
