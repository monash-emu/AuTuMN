from unittest import mock
from datetime import date

import pytest

from autumn.constants import Region
from summer.model import StratifiedModel
from apps.covid_19.mixing_optimisation import mixing_opti as opti


@pytest.mark.local_only
@pytest.mark.parametrize("region", Region.MIXING_OPTI_REGIONS)
@mock.patch("apps.covid_19.mixing_optimisation.mixing_opti.PHASE_2_START_TIME", 30)
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


AGE_GROUPS = [
    "0",
    "5",
    "10",
    "15",
    "20",
    "25",
    "30",
    "35",
    "40",
    "45",
    "50",
    "55",
    "60",
    "65",
    "70",
    "75",
]
AVAILABLE_MODES = [
    "by_age",
    "by_location",
]
AVAILABLE_CONFIGS = range(4)
DECISION_VARS = {
    "by_age": {ag: 1 for ag in AGE_GROUPS},
    "by_location": {
        "work": 1.0,
        "school": 1.0,
        "other_locations": 1.0,
    },
}


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


@pytest.mark.parametrize("config", AVAILABLE_CONFIGS)
@pytest.mark.parametrize("mode", AVAILABLE_MODES)
def test_build_params_for_phases_2_and_3__smoke_test(mode, config):
    opti.build_params_for_phases_2_and_3(DECISION_VARS[mode], config=config, mode=mode)


def test_build_params_for_phases_2_and_3__with_location_mode_and_microdistancing():
    scenario_params = opti.build_params_for_phases_2_and_3(
        decision_variables={
            "other_locations": 2,
            "school": 3,
            "work": 5,
        },
        config=2,
        mode="by_location",
    )
    loc_dates = [date(2020, 7, 31), date(2021, 1, 31), date(2021, 2, 1)]

    assert scenario_params == {
        "time": {
            "start": 213,
            "end": 669,
        },
        "mobility": {
            "age_mixing": {},
            "microdistancing": {"behaviour": {"parameters": {"sigma": 1.0}}},
            "mixing": {
                "other_locations": {
                    "times": loc_dates,
                    "values": [2, 2, 1.0],
                    "append": False,
                },
                "school": {
                    "times": loc_dates,
                    "values": [3, 3, 1.0],
                    "append": False,
                },
                "work": {
                    "times": loc_dates,
                    "values": [5, 5, 1.0],
                    "append": False,
                },
            },
        },
        "importation": {
            "props_by_age": None,
            "movement_prop": None,
            "quarantine_timeseries": {"times": [], "values": []},
            "case_timeseries": {
                "times": [397, 398, 399, 400],
                "values": [0, 5, 5, 0],
            },
        },
    }


def test_build_params_for_phases_2_and_3__with_age_mode():
    scenario_params = opti.build_params_for_phases_2_and_3(
        decision_variables={ag: i for i, ag in enumerate(AGE_GROUPS)}, config=0, mode="by_age"
    )
    age_dates = [date(2020, 7, 31), date(2020, 8, 1), date(2021, 1, 31), date(2021, 2, 1)]
    loc_dates = [date(2020, 7, 31), date(2021, 1, 31), date(2021, 2, 1)]

    assert scenario_params == {
        "time": {
            "start": 213,
            "end": 669,
        },
        "mobility": {
            "age_mixing": {
                "0": {"values": [1, 0, 0, 1], "times": age_dates},
                "5": {"values": [1, 1, 1, 1], "times": age_dates},
                "10": {"values": [1, 2, 2, 1], "times": age_dates},
                "15": {"values": [1, 3, 3, 1], "times": age_dates},
                "20": {"values": [1, 4, 4, 1], "times": age_dates},
                "25": {"values": [1, 5, 5, 1], "times": age_dates},
                "30": {"values": [1, 6, 6, 1], "times": age_dates},
                "35": {"values": [1, 7, 7, 1], "times": age_dates},
                "40": {"values": [1, 8, 8, 1], "times": age_dates},
                "45": {"values": [1, 9, 9, 1], "times": age_dates},
                "50": {"values": [1, 10, 10, 1], "times": age_dates},
                "55": {"values": [1, 11, 11, 1], "times": age_dates},
                "60": {"values": [1, 12, 12, 1], "times": age_dates},
                "65": {"values": [1, 13, 13, 1], "times": age_dates},
                "70": {"values": [1, 14, 14, 1], "times": age_dates},
                "75": {"values": [1, 15, 15, 1], "times": age_dates},
            },
            "mixing": {
                "other_locations": {
                    "times": loc_dates,
                    "values": [1.0, 1.0, 1.0],
                    "append": False,
                },
                "school": {
                    "times": loc_dates,
                    "values": [1.0, 1.0, 1.0],
                    "append": False,
                },
                "work": {
                    "times": loc_dates,
                    "values": [1.0, 1.0, 1.0],
                    "append": False,
                },
            },
        },
        "importation": {
            "props_by_age": None,
            "movement_prop": None,
            "quarantine_timeseries": {"times": [], "values": []},
            "case_timeseries": {
                "times": [397, 398, 399, 400],
                "values": [0, 5, 5, 0],
            },
        },
    }