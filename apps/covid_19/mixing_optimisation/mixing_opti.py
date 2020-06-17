import os
import copy

import yaml

from autumn.constants import Region
import autumn.post_processing as post_proc
from autumn.tool_kit.scenarios import Scenario

from apps.covid_19 import RegionApp


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(FILE_DIR, "opti_params.yml")

with open(OPTI_PARAMS_PATH, "r") as yaml_file:
    opti_params = yaml.safe_load(yaml_file)

available_countries = [Region.MALAYSIA]


def objective_function(decision_variables, mode="by_age", country=Region.MALAYSIA):
    """
    :param decision_variables: dictionary containing
        - mixing multipliers by age as a list if mode == "by_age"    OR
        - location multipliers as a dictionary if mode == "by_location"
    :param mode: either "by_age" or "by_location"
    :param country: the country name
    """
    running_model = RegionApp(country)
    build_model = running_model.build_model
    params = copy.deepcopy(running_model.params)

    # update params with optimisation config
    params["default"].update(opti_params["default"])

    # Define the two scenarios:
    #   baseline: using the decision variables
    #   scenario 1: after intervention to test immunity
    if mode == "by_age":
        mixing_update = {}
        for age_group in range(16):
            mixing_update["age_" + str(age_group) + "_times"] = [181, 183]
            mixing_update["age_" + str(age_group) + "_values"] = [
                1.0,
                decision_variables[age_group],
            ]
        params["default"]["mixing"].update(mixing_update)

    # set location-specific mixing back to pre-COVID rates on 1st of July or use the opti decision variable
    for loc in ["other_locations", "school", "work"]:
        latest_value = params["default"]["mixing"][loc + "_values"][-1]
        params["default"]["mixing"][loc + "_times"] += [181, 183]
        if mode == "by_age":  # just return mixing to pre-COVID
            new_mixing_adjustment = 1.0
        elif mode == "by_location":  # use optimisation decision variables
            new_mixing_adjustment = decision_variables[loc]
        else:
            raise ValueError("The requested mode is not supported")

        params["default"]["mixing"][loc + "_values"] += [
            latest_value,
            new_mixing_adjustment,
        ]

    # Add a scenario without any mixing multipliers
    end_time = params["default"]["end_time"]
    params["scenario_start_time"] = end_time - 1
    params["scenarios"][1] = {"end_time": end_time + 50, "mixing": {}}
    scenario_0 = Scenario(build_model, idx=0, params=params)
    scenario_1 = Scenario(build_model, idx=1, params=params)
    scenario_0.run()
    scenario_1.run(base_model=scenario_0.model)
    models = [scenario_0.model, scenario_1.model]

    # Has herd immunity been reached?
    herd_immunity = has_immunity_been_reached(models[1])

    # How many deaths after 1 July 2020
    first_july_index = models[0].derived_outputs["times"].index(183)
    total_nb_deaths = sum(models[0].derived_outputs["infection_deathsXall"][first_july_index:])
    recovered_indices = [
        i
        for i in range(len(models[0].compartment_names))
        if "recovered" in models[0].compartment_names[i]
    ]
    nb_reco = sum([models[0].outputs[-1, i] for i in recovered_indices])
    total_pop = sum(models[0].compartment_values)
    prop_immune = nb_reco / total_pop

    return herd_immunity, total_nb_deaths, prop_immune, models


def visualise_simulation(_models):

    pps = []
    for scenario_index in range(len(_models)):

        pps.append(
            post_proc.PostProcessing(
                _models[scenario_index],
                requested_outputs=["prevXinfectiousXamong", "prevXrecoveredXamong"],
                scenario_number=scenario_index,
                requested_times={},
            )
        )

    # FIXME: Matt broke this
    # old_outputs_plotter = Outputs(_models, pps, {}, plot_start_time=0)
    # old_outputs_plotter.plot_requested_outputs()


def has_immunity_been_reached(_model):
    """
    Determine whether herd immunity has been reached after running a model
    :param _model: a model run with no-intervention setting for testing herd-immunity
    :return: a boolean
    """
    return max(_model.derived_outputs["incidence"]) == _model.derived_outputs["incidence"][0]


if __name__ == "__main__":
    # looping through all countries and optimisation modes for testing purpose
    # optimisation will have to be performed separately for the different countries and modes.
    decision_vars = {
        "by_age": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
        "by_location": {"other_locations": 1.0, "school": 1.0, "work": 1.0},
    }

    for mode in ["by_age", "by_location"]:
        for country in available_countries:
            h, d, p_immune, m = objective_function(decision_vars[mode], mode, country)
            print(country)
            print("Immunity: " + str(h) + "\n" + "Deaths: " + str(round(d)))
