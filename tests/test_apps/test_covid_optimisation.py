from unittest import mock

import pytest

from autumn.constants import Region
from summer.model import StratifiedModel
from apps.covid_19.mixing_optimisation import mixing_opti as opti

AVAILABLE_MODES = [
    "by_age",
    "by_location",
]
AVAILABLE_CONFIGS = range(4)
DECISION_VARS = {
    "by_age": [1.0] * 16,
    "by_location": [1.0, 1.0, 1.0],
}


@pytest.mark.local_only
@pytest.mark.parametrize("region", Region.MIXING_OPTI_REGIONS)
@mock.patch("apps.covid_19.mixing_optimisation.mixing_opti.PHASE_2_START_TIME", 10)
def test_run_root_models_partial(region):
    """
    Smoke test: ensure we can build and run each root model with nothing crashing.
    """
    model = opti.run_root_model(region, {})
    assert type(model) is StratifiedModel
    assert model.outputs is not None


@pytest.mark.github_only
@pytest.mark.mixing_optimisation
@pytest.mark.parametrize("region", Region.MIXING_OPTI_REGIONS)
def test_run_root_models_full(region):
    """
    Smoke test: ensure we can build and run each root model with nothing crashing.
    """
    model = opti.run_root_model(region, {})
    assert type(model) is StratifiedModel
    assert model.outputs is not None


@pytest.mark.skip
def test_build_params_for_phases_2_and_3():
    for mode in AVAILABLE_MODES:
        for config in AVAILABLE_CONFIGS:
            scenario_params = opti.build_params_for_phases_2_and_3(
                DECISION_VARS[mode], config=config, mode=mode
            )
            assert "mixing" in scenario_params and "end_time" in scenario_params


@pytest.mark.skip
@pytest.mark.mixing_optimisation
@pytest.mark.github_only
def test_full_optimisation_iteration_for_uk():
    country = Region.UNITED_KINGDOM
    root_model = opti.run_root_model(country, {})
    for mode in AVAILABLE_MODES:
        for config in AVAILABLE_CONFIGS:
            h, d, yoll, p_immune, m = opti.objective_function(
                DECISION_VARS[mode], root_model, mode, country, config
            )
