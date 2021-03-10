import os
import copy
from typing import List

from summer.model import StratifiedModel
from autumn.region import Region
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.params import update_params, merge_dicts
from datetime import date, timedelta

from apps import covid_19
from apps.covid_19.mixing_optimisation.constants import (
    PHASE_2_START_TIME,
    DURATION_PHASES_2_AND_3,
    PHASE_2_DURATION,
    MICRODISTANCING_OPTI_PARAMS,
    MIXING_FACTOR_BOUNDS,
)

REF_DATE = date(2019, 12, 31)
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

AGE_MODE = "by_age"
LOCATION_MODE = "by_location"
MODES = [AGE_MODE, LOCATION_MODE]

OBJECTIVE_DEATHS = "deaths"
OBJECTIVE_YOLL = "yoll"
OBJECTIVES = [OBJECTIVE_DEATHS, OBJECTIVE_YOLL]

DURATION_SIX_MONTHS = "six_months"
DURATION_TWELVE_MONTHS = "twelve_months"
DURATIONS = [DURATION_SIX_MONTHS, DURATION_TWELVE_MONTHS]

N_DECISION_VARS = {AGE_MODE: 16, LOCATION_MODE: 3}

MIXING_LOCS = ["other_locations", "school", "work"]
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

phase_2_end_times = {}
for _duration in DURATIONS:
    phase_2_end_times[_duration] = PHASE_2_START_TIME + PHASE_2_DURATION[_duration]


def objective_function(
    decision_variables: List[float],
    root_model: StratifiedModel,
    mode: str = AGE_MODE,
    country: str = Region.UNITED_KINGDOM,
    duration: str = DURATION_SIX_MONTHS,
    called_from_sensitivity_analysis: bool = False,
):
    """
    :param decision_variables: A list containing
        - mixing multipliers by age as a list if mode == "by_age"    OR
        - location multipliers as a list if mode == "by_location"
    :param root_model: integrated model supposed to model the past epidemic
    :param mode: either "by_age" or "by_location"
    :param country: the country name
    :param duration: string defining the duration of the optimised phase
    :param called_from_sensitivity_analysis: whether called from main optimisation or from sensitivity analysis
    """
    if not called_from_sensitivity_analysis:
        assert all(
            [MIXING_FACTOR_BOUNDS[0] <= d <= MIXING_FACTOR_BOUNDS[1] for d in decision_variables]
        )

    app_region = covid_19.app.get_region(country)
    build_model = app_region.build_model
    params = copy.deepcopy(app_region.params)

    # Create and run scenario 1
    sc_1_params_update = build_params_for_phases_2_and_3(
        decision_variables, params["default"]["elderly_mixing_reduction"], duration, mode
    )
    sc_1_params = merge_dicts(sc_1_params_update, params["default"])
    params["scenarios"][1] = sc_1_params
    scenario_1 = Scenario(build_model, idx=1, params=params)
    scenario_1.run(base_model=root_model)
    sc_1_model = scenario_1.model

    # Perform diagnostics.
    phase_2_start_idx = list(sc_1_model.times).index(PHASE_2_START_TIME)
    phase_2_end_idx = list(sc_1_model.times).index(phase_2_end_times[duration])

    # Determine number of deaths and years of life lost during Phase 2 and 3.
    total_nb_deaths = sum(sc_1_model.derived_outputs["infection_deaths"][phase_2_start_idx:])
    years_of_life_lost = sum(sc_1_model.derived_outputs["years_of_life_lost"][phase_2_start_idx:])

    # Has herd immunity been reached?
    herd_immunity = has_immunity_been_reached(sc_1_model, phase_2_end_idx)

    return herd_immunity, total_nb_deaths, years_of_life_lost


def has_immunity_been_reached(model: StratifiedModel, phase_2_end_index: int) -> bool:
    """
    Determine whether herd immunity has been reached after running a model
    :param model: a model run with Phase 2 and
    :return: a boolean
    """
    # validate herd immunity if incidence always decreases after 2 weeks in phase 3
    time_indices = range(phase_2_end_index, len(model.times))
    two_weeks_days = 14
    incidence_vals = [model.derived_outputs["incidence"][i] for i in time_indices[two_weeks_days:]]
    return max(incidence_vals) == incidence_vals[0]


def run_root_model(region: str):
    """
    Runs a model to simulate the past epidemic (up until 1/10/2020).
    Returns a the model which has been run.
    """
    app_region = covid_19.app.get_region(region)
    params = copy.deepcopy(app_region.params)

    # Set end time.
    params["default"]["time"]["end"] = PHASE_2_START_TIME

    base_scenario = Scenario(app_region.build_model, idx=0, params=params)
    base_scenario.run()
    return base_scenario.model


def build_params_for_phases_2_and_3(
    decision_variables: List[float],
    elderly_mixing_reduction,
    duration: str = DURATION_SIX_MONTHS,
    mode: str = AGE_MODE,
    final_mixing: float = 1.0,
):
    """
    Build the scenario parameters that includes phases 2 and 3
    """
    scenario_params = {}

    # Set start and end times.
    scenario_params["time"] = {
        "start": PHASE_2_START_TIME - 1,
        "end": PHASE_2_START_TIME + DURATION_PHASES_2_AND_3,
    }

    # Convert time integers to dates.
    phase_2_end_days = phase_2_end_times[duration]
    phase_2_first_day = REF_DATE + timedelta(days=PHASE_2_START_TIME)
    phase_1_end_date = phase_2_first_day + timedelta(days=-1)
    phase_2_end_date = REF_DATE + timedelta(days=phase_2_end_days)
    phase_3_first_day = phase_2_end_date + timedelta(days=1)

    # Apply social mixing adjustments
    scenario_params["mobility"] = {"mixing": {}, "age_mixing": {}}
    if mode == AGE_MODE:
        # Age based mixing optimisation.
        for age_idx in range(len(AGE_GROUPS)):
            age_group = AGE_GROUPS[age_idx]
            age_mixing = decision_variables[age_idx]
            final_mixing_value = final_mixing
            # need to reset elderly protection if it was previously in place during Phase 1
            if elderly_mixing_reduction is not None:
                if age_group in elderly_mixing_reduction["age_categories"]:
                    final_mixing_value = min(
                        1 - elderly_mixing_reduction["relative_reduction"], final_mixing_value
                    )
            scenario_params["mobility"]["age_mixing"][age_group] = {
                "times": [phase_1_end_date, phase_2_first_day, phase_2_end_date, phase_3_first_day],
                "values": [
                    1.0,
                    age_mixing,
                    age_mixing,
                    final_mixing_value,
                ],
            }

        # Set location-specific mixing back to pre-COVID rates on 1st of July.
        # ie. Use the pre-COVID contact matrices.
        scenario_params["mobility"]["mixing"] = {}
        for loc in MIXING_LOCS:
            scenario_params["mobility"]["mixing"][loc] = {
                "times": [phase_1_end_date, phase_2_end_date, phase_3_first_day],
                "values": [1.0, 1.0, 1.0],
                "append": False,
            }

    elif mode == LOCATION_MODE:
        # Set location-specific mixing to  use the opti decision variable.
        scenario_params["mobility"]["mixing"] = {}
        # Reformat decision vars.
        loc_mixing_map = {
            "other_locations": decision_variables[0],
            "school": decision_variables[1],
            "work": decision_variables[2],
        }
        for loc in MIXING_LOCS:
            # Use optimisation decision variables
            loc_mixing = loc_mixing_map[loc]
            scenario_params["mobility"]["mixing"][loc] = {
                "times": [phase_1_end_date, phase_2_end_date, phase_3_first_day],
                "values": [loc_mixing, loc_mixing, final_mixing],
                "append": False,
            }
    else:
        raise ValueError("The requested mode is not supported")

    # Add microdistancing params to the scenario.
    scenario_params["mobility"]["microdistancing"] = MICRODISTANCING_OPTI_PARAMS

    # Seed a new wave of infections with some importations.
    # # This tests whether herd immunity has actually been reached.

    # FIXME: Importation is no longer available so we need to find another way to test herd immunity
    # scenario_params["importation"] = {
    #     "props_by_age": None,
    #     "movement_prop": None,
    #     "quarantine_timeseries": {"times": [], "values": []},
    #     "case_timeseries": {
    #         "times": [phase_2_end_days + i for i in range(4)],
    #         "values": [0, 5, 5, 0],
    #     },
    # }

    return scenario_params
