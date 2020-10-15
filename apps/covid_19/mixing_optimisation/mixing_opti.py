import os
import copy

import yaml
import pandas as pd

from autumn.constants import Region
from autumn.model_runner import build_model_runner
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.params import update_params, merge_dicts
from datetime import date, timedelta

from apps import covid_19
from apps.covid_19.mixing_optimisation.constants import PHASE_2_START_TIME, DURATION_PHASES_2_AND_3
from apps.covid_19.model.preprocess.mixing_matrix.mixing_adjusters.age_adjuster import AGE_GROUPS


REF_DATE = date(2019, 12, 31)
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(FILE_DIR, "opti_params.yml")

with open(OPTI_PARAMS_PATH, "r") as yaml_file:
    opti_params = yaml.safe_load(yaml_file)


phase_2_end = [
    PHASE_2_START_TIME + config["phase_two_duration"] for config in opti_params["configurations"]
]


def run_root_model(region: str, calibrated_params: dict = {}):
    """
    Runs a model to simulate the past epidemic (up until 1/7/2020) using  a given calibrated parameter set.
    Returns a the model which has been run.
    """
    app_region = covid_19.app.get_region(region)
    params = copy.deepcopy(app_region.params)

    # Update params with optimisation default config.
    params["default"] = merge_dicts(opti_params["default"], params["default"])

    # Update params with calibrated parameters
    params["default"] = update_params(params["default"], calibrated_params)

    # Set start/end time.
    params["default"]["time"]["end"] = PHASE_2_START_TIME
    for scenario in params["scenarios"].values():
        scenario["time"]["start"] = PHASE_2_START_TIME - 1

    base_scenario = Scenario(app_region.build_model, idx=0, params=params)
    base_scenario.run()
    return base_scenario.model


AGE_MODE = "by_age"
LOCATION_MODE = "by_location"
MIXING_LOCS = ["other_locations", "school", "work"]


def build_params_for_phases_2_and_3(
    decision_variables: dict, config: int = 0, mode: str = AGE_MODE, final_mixing: float = 1.0
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
    phase_2_end_days = phase_2_end[config]
    phase_2_first_day = REF_DATE + timedelta(days=PHASE_2_START_TIME)
    phase_1_end_date = phase_2_first_day + timedelta(days=-1)
    phase_2_end_date = REF_DATE + timedelta(days=phase_2_end_days)
    phase_3_first_day = phase_2_end_date + timedelta(days=1)

    # Apply social mixing adjustments
    scenario_params["mobility"] = {"mixing": {}, "age_mixing": {}}
    if mode == AGE_MODE:
        # Age based mixing optimisation.
        for age_group in AGE_GROUPS:
            scenario_params["mobility"]["age_mixing"][age_group] = {
                "times": [phase_1_end_date, phase_2_first_day, phase_2_end_date, phase_3_first_day],
                "values": [
                    1.0,
                    decision_variables[age_group],
                    decision_variables[age_group],
                    final_mixing,
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
        for loc in MIXING_LOCS:
            # Use optimisation decision variables
            scenario_params["mobility"]["mixing"][loc] = {
                "times": [phase_1_end_date, phase_2_end_date, phase_3_first_day],
                "values": [decision_variables[loc], decision_variables[loc], final_mixing],
                "append": False,
            }
    else:
        raise ValueError("The requested mode is not supported")

    # Add optmisized microdistancing params to the scenario.
    microdistancing_opti_params = opti_params["configurations"][config].get("microdistancing")
    if microdistancing_opti_params:
        scenario_params["mobility"]["microdistancing"] = microdistancing_opti_params

    # Seed a new wave of infections with some importations.
    # This tests whether herd immunity has actually been reached.
    scenario_params["importation"] = {
        "props_by_age": None,
        "movement_prop": None,
        "quarantine_timeseries": {"times": [], "values": []},
        "case_timeseries": {
            "times": [phase_2_end_days + i for i in range(4)],
            "values": [0, 5, 5, 0],
        },
    }

    return scenario_params


def has_immunity_been_reached(_model, phase_2_end_index):
    """
    Determine whether herd immunity has been reached after running a model
    :param _model: a model run with Phase 2 and
    :return: a boolean
    """
    # validate herd immunity if incidence always decreases after 2 weeks in phase 3
    time_indices = range(phase_2_end_index, len(_model.derived_outputs["times"]))
    incidence_vals = [_model.derived_outputs["incidence"][i] for i in time_indices[14:]]
    return max(incidence_vals) == incidence_vals[0]


def objective_function(
    decision_variables,
    root_model,
    mode="by_age",
    country=Region.UNITED_KINGDOM,
    config=0,
    calibrated_params={},
):
    """
    :param decision_variables: dictionary containing
        - mixing multipliers by age as a list if mode == "by_age"    OR
        - location multipliers as a list if mode == "by_location"
    :param root_model: integrated model supposed to model the past epidemic
    :param mode: either "by_age" or "by_location"
    :param country: the country name
    :param config: the id of the configuration being considered
    :param calibrated_params: a dictionary containing a set of calibrated parameters
    """
    running_model = covid_19.app.get_region(country)
    build_model = running_model.build_model
    params = copy.deepcopy(running_model.params)

    # reformat decision vars if locations
    if mode == "by_location":
        new_decision_variables = {
            "other_locations": decision_variables[0],
            "school": decision_variables[1],
            "work": decision_variables[2],
        }
        decision_variables = new_decision_variables

    # Define scenario-1-specific params
    sc_1_params_update = build_params_for_phases_2_and_3(decision_variables, config, mode)

    # Rebuild the default parameters
    params["default"].update(opti_params["default"])
    params["default"] = update_params(params["default"], calibrated_params)
    for scenario in params["scenarios"].values():
        scenario["time"]["start"] = PHASE_2_START_TIME - 1

    # Create scenario 1
    sc_1_params = update_params(params["default"], sc_1_params_update)
    params["scenarios"][1] = sc_1_params
    scenario_1 = Scenario(build_model, idx=1, params=params)

    # Run scenario 1
    scenario_1.run(base_model=root_model)
    models = [root_model, scenario_1.model]

    # ____________________________       Perform diagnostics         ______________________
    # How many deaths and years of life lost during Phase 2 and 3
    start_phase2_index = models[1].derived_outputs["times"].index(PHASE_2_START_TIME)
    end_phase2_index = models[1].derived_outputs["times"].index(phase_2_end[config])
    total_nb_deaths = sum(models[1].derived_outputs["infection_deaths"][start_phase2_index:])
    years_of_life_lost = sum(models[1].derived_outputs["years_of_life_lost"][start_phase2_index:])

    # What proportion immune at end of Phase 2
    recovered_indices = [
        i
        for i in range(len(models[1].compartment_names))
        if "recovered" in models[1].compartment_names[i]
    ]
    nb_reco = sum([models[1].outputs[end_phase2_index, i] for i in recovered_indices])
    total_pop = sum(
        [models[1].outputs[end_phase2_index, i] for i in range(len(models[1].compartment_names))]
    )
    prop_immune = nb_reco / total_pop

    # Has herd immunity been reached?
    herd_immunity = has_immunity_been_reached(models[1], end_phase2_index)

    return herd_immunity, total_nb_deaths, years_of_life_lost, prop_immune, models
