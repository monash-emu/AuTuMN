import os
from datetime import timedelta

from autumn.projects.covid_19.mixing_optimisation.constants import (
    DURATION_PHASES_2_AND_3,
    MICRODISTANCING_OPTI_PARAMS,
    MIXING_FACTOR_BOUNDS,
    PHASE_2_DURATION,
    PHASE_2_START_TIME,
)
from autumn.models.covid_19.constants import COVID_BASE_DATETIME

APP_NAME = "covid_19"
ROOT_MODEL_PARAMS = {"time": {"end": PHASE_2_START_TIME}}

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


def make_scenario_func(
    root_params,
    duration: str = DURATION_SIX_MONTHS,
    mode: str = AGE_MODE,
    final_mixing: float = 1.0,
):  # a function that returns a scenario dictionary based on decision variables

    elderly_mixing_reduction = root_params["elderly_mixing_reduction"]

    # Convert time integers to dates.
    phase_2_end_days = phase_2_end_times[duration]
    phase_2_first_day = COVID_BASE_DATETIME + timedelta(days=PHASE_2_START_TIME)
    phase_1_end_date = phase_2_first_day + timedelta(days=-1)
    phase_2_end_date = COVID_BASE_DATETIME + timedelta(days=phase_2_end_days)
    phase_3_first_day = phase_2_end_date + timedelta(days=1)

    def scenario_func(decision_variables):
        """
        Build the scenario parameters that includes phases 2 and 3
        """
        scenario_params = {}

        # Set start and end times.
        scenario_params["time"] = {
            "start": PHASE_2_START_TIME - 1,
            "end": PHASE_2_START_TIME + DURATION_PHASES_2_AND_3,
        }

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
                    "times": [
                        phase_1_end_date,
                        phase_2_first_day,
                        phase_2_end_date,
                        phase_3_first_day,
                    ],
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

    return scenario_func


def has_immunity_been_reached(run_model, phase_2_end_index: int) -> bool:
    """
    Determine whether herd immunity has been reached after running a model
    :param model: a model run with Phase 2 and
    :return: a boolean
    """
    # validate herd immunity if incidence always decreases after 2 weeks in phase 3
    time_indices = range(phase_2_end_index, len(run_model.times))
    two_weeks_days = 14
    incidence_vals = [
        run_model.derived_outputs["incidence"][i] for i in time_indices[two_weeks_days:]
    ]
    return max(incidence_vals) == incidence_vals[0]


def make_objective_func(duration):
    def objective_func(run_model, decision_vars):
        # Perform diagnostics.
        phase_2_start_idx = list(run_model.times).index(PHASE_2_START_TIME)
        phase_2_end_idx = list(run_model.times).index(phase_2_end_times[duration])

        # Determine number of deaths and years of life lost during Phase 2 and 3.
        total_nb_deaths = sum(run_model.derived_outputs["infection_deaths"][phase_2_start_idx:])
        years_of_life_lost = sum(
            run_model.derived_outputs["years_of_life_lost"][phase_2_start_idx:]
        )

        # Has herd immunity been reached?
        herd_immunity = has_immunity_been_reached(run_model, phase_2_end_idx)

        return [herd_immunity, total_nb_deaths, years_of_life_lost]

    return objective_func
