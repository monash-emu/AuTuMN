from copy import deepcopy

import pytest
from summer.model import StratifiedModel
from summer.constants import Flow, IntegrationType

from apps import tuberculosis
from autumn.tool_kit.utils import merge_dicts
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


@pytest.mark.local_only
@pytest.mark.parametrize("region", tuberculosis.app.region_names)
def test_run_models_partial(region):
    """
    Smoke test: ensure we can build and run each default model with various stratification requests with nothing crashing.
    Does not include scenarios, plotting, etc.
    """
    region_app = tuberculosis.app.get_region(region)
    ps = deepcopy(region_app.params["default"])
    original_stratify_by = deepcopy(ps["stratify_by"])
    for stratify_by in powerset(original_stratify_by):
        ps = deepcopy(region_app.params["default"])
        ps["stratify_by"] = list(stratify_by)
        # Only run model for ~10 epochs.
        ps["end_time"] = ps["start_time"] + 10
        model = region_app.build_model(ps)
        model.run_model()


@pytest.mark.local_only
@pytest.mark.parametrize("region", tuberculosis.app.region_names)
def test_build_scenario_models(region):
    """
    Smoke test: ensure we can build the each model with nothing crashing.
    """
    region_app = tuberculosis.app.get_region(region)
    for idx, scenario_params in enumerate(region_app.params["scenarios"].values()):
        default_params = deepcopy(region_app.params["default"])
        params = merge_dicts(scenario_params, default_params)
        model = region_app.build_model(params)
        assert type(model) is StratifiedModel


@pytest.mark.run_models
@pytest.mark.github_only
@pytest.mark.parametrize("region", tuberculosis.app.region_names)
def test_run_models_full(region):
    """
    Smoke test: ensure our models run to completion for any stratification request without crashing.
    This takes ~30s per model.
    """
    for stratify_by in ([], ["organ"]):
        region_app = tuberculosis.app.get_region(region)
        ps = deepcopy(region_app.params["default"])
        ps["stratify_by"] = stratify_by
        model = region_app.build_model(ps)
        model.run_model()
