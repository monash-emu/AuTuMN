from unittest import mock
from datetime import date

import pytest
import numpy as np

from autumn.constants import Region
from summer.model import StratifiedModel
from apps.covid_19.mixing_optimisation import mixing_opti as opti
from apps.covid_19.mixing_optimisation.constants import PHASE_2_START_TIME, OPTI_REGIONS
from apps.covid_19.mixing_optimisation import write_scenarios


@pytest.mark.local_only
@pytest.mark.parametrize("region", Region.MIXING_OPTI_REGIONS)
@mock.patch("apps.covid_19.mixing_optimisation.mixing_opti.PHASE_2_START_TIME", 100)
def test_run_root_models_partial(region):
    """
    Smoke test: ensure we can build and run each root model with nothing crashing.
    """
    model = opti.run_root_model(region)
    assert type(model) is StratifiedModel
    assert model.outputs is not None


@pytest.mark.github_only
@pytest.mark.mixing_optimisation
@pytest.mark.parametrize("region", Region.MIXING_OPTI_REGIONS)
def test_run_root_models_full(region):
    """
    Smoke test: ensure we can build and run each root model with nothing crashing.
    """
    model = opti.run_root_model(region)
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
AVAILABLE_DURATIONS = ["six_months", "twelve_months"]
DECISION_VARS = {
    "by_age": [1 for _ in range(len(AGE_GROUPS))],
    "by_location": [1, 1, 1],
}


@pytest.mark.mixing_optimisation
@pytest.mark.github_only
@pytest.mark.parametrize("duration", AVAILABLE_DURATIONS)
@pytest.mark.parametrize("mode", AVAILABLE_MODES)
def test_full_optimisation_iteration_for_uk(mode, duration):
    country = Region.UNITED_KINGDOM
    root_model = opti.run_root_model(country)
    h, d, yoll = opti.objective_function(DECISION_VARS[mode], root_model, mode, country, duration)
    assert h in (True, False)
    assert d >= 0
    assert yoll >= 0


@pytest.mark.parametrize("duration", AVAILABLE_DURATIONS)
@pytest.mark.parametrize("mode", AVAILABLE_MODES)
def test_build_params_for_phases_2_and_3__smoke_test(mode, duration):
    opti.build_params_for_phases_2_and_3(DECISION_VARS[mode], duration=duration, mode=mode)


@mock.patch("apps.covid_19.mixing_optimisation.mixing_opti.Scenario")
def test_objective_function_calculations(mock_scenario_cls):
    root_model = mock.Mock()
    sc_model = mock.Mock()
    mock_scenario_cls.return_value.model = sc_model
    phase_2_days = 183
    phase_3_days = 14 + 10
    num_timesteps = PHASE_2_START_TIME + phase_2_days + phase_3_days
    sc_model.times = np.array(range(num_timesteps))
    sc_model.derived_outputs = {
        # Expect 55 deaths as sum of vals.
        "infection_deaths": np.concatenate(
            [np.zeros(PHASE_2_START_TIME), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])],
        ),
        # Expect 108 yoll as sum of vals.
        "years_of_life_lost": np.concatenate(
            [np.zeros(PHASE_2_START_TIME), np.array([1, 3, 5, 7, 9, 11, 13, 17, 19, 23])],
        ),
        # Expect immunity because incidence decreasing
        "incidence": np.concatenate(
            [
                np.zeros(PHASE_2_START_TIME + phase_2_days + 14),
                np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]),
            ],
        ),
    }
    # Expect 10% immune.
    sc_model.compartment_names = ["a", "b_recovered", "c_recovered"]
    sc_model.outputs = np.zeros([num_timesteps, 3])
    sc_model.outputs[PHASE_2_START_TIME + phase_2_days, 0] = 90
    sc_model.outputs[PHASE_2_START_TIME + phase_2_days, 1] = 3
    sc_model.outputs[PHASE_2_START_TIME + phase_2_days, 2] = 7

    decision_variables = [1 for _ in range(len(AGE_GROUPS))]
    (herd_immunity, total_nb_deaths, years_of_life_lost,) = opti.objective_function(
        decision_variables,
        root_model,
        mode="by_age",
        country="france",
        duration="six_months",
    )
    assert herd_immunity
    assert total_nb_deaths == 55
    assert years_of_life_lost == 108


@pytest.mark.xfail
def test_build_params_for_phases_2_and_3__with_location_mode_and_microdistancing():
    scenario_params = opti.build_params_for_phases_2_and_3(
        decision_variables=[2, 3, 5],
        duration="six_months",
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


@pytest.mark.xfail
def test_build_params_for_phases_2_and_3__with_age_mode():
    scenario_params = opti.build_params_for_phases_2_and_3(
        decision_variables=[i for i in range(16)], duration="six_months", mode="by_age"
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


"""
Test write_scenarios module
"""


@pytest.mark.mixing_optimisation
def test_read_optimised_variables():
    test_file = "dummy_vars_for_test.csv"
    df = write_scenarios.read_opti_outputs(test_file)
    decision_vars = write_scenarios.read_decision_vars(
        df, "france", "by_age", "six_months", "deaths"
    )
    assert decision_vars == [0.99] * 16


@pytest.mark.mixing_optimisation
def test_build_all_scenarios():
    test_file = "dummy_vars_for_test.csv"
    all_sc_params = write_scenarios.build_all_scenario_dicts_from_outputs(test_file)

    assert set(list(all_sc_params.keys())) == set(OPTI_REGIONS)

    assert list(all_sc_params["france"].keys()) == [1, 9]

    assert list(all_sc_params["france"][1].keys()) == ["time", "mobility", "parent"]
