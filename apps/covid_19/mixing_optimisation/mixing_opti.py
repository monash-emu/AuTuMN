import os
import copy

import yaml
import numpy as np

import autumn.post_processing as post_proc
from autumn.tool_kit.scenarios import Scenario

from ..countries import Country, CountryModel


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(FILE_DIR, "opti_params.yml")

with open(OPTI_PARAMS_PATH, "r") as yaml_file:
    opti_params = yaml.safe_load(yaml_file)

aus = CountryModel(Country.AUSTRALIA)


def objective_function(decision_variables, mode="by_age"):
    """
    :param decision_variables: dictionary containing
        - mixing multipliers if mode == "by_age"    OR
        - location multipliers if mode == "by_location"
    :param mode: either "by_age" or "by_location"
    :return:
    """
    build_model = aus.build_model
    params = copy.deepcopy(aus.params)

    # Define the two scenarios:
    #   baseline: with intervention
    #   scenario 1: after intervention to test immunity
    if mode == "by_age":
        mixing_multipliers = decision_variables
        mixing_multipliers_matrix = build_mixing_multipliers_matrix(mixing_multipliers)
        params["default"].update({"mixing_matrix_multipliers": mixing_multipliers_matrix})

    elif mode == "by_location":
        mixing_update_dictionary = {}
        for loc in ["school", "work", "other_locations"]:
            mixing_update_dictionary[loc + "_times"] = [0]
            mixing_update_dictionary[loc + "_values"] = [decision_variables[loc]]

        params["default"].update({"mixing": mixing_update_dictionary})

    else:
        raise ValueError("The requested mode is not supported")

    # Add a scenario without any mixing multipliers
    end_time = params["default"]["end_time"]
    params["scenario_start_time"] = end_time - 1
    params["scenarios"][1] = {
        "end_time": end_time + 50,
        "mixing_matrix_multipliers": None,
        "mixing": None,
    }

    scenario_0 = Scenario(build_model, idx=0, params=params)
    scenario_1 = Scenario(build_model, idx=1, params=params)
    scenario_0.run()
    scenario_1.run(base_model=scenario_0.model)
    models = [scenario_0.model, scenario_1.model]

    # Has herd immunity been reached?
    herd_immunity = has_immunity_been_reached(models[1])

    # How many deaths
    total_nb_deaths = sum(models[0].derived_outputs["infection_deathsXall"])

    return herd_immunity, total_nb_deaths, models


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


def build_mixing_multipliers_matrix(mixing_multipliers):
    """
    Builds a full 16x16 matrix of multipliers based on the parameters found in mixing_multipliers
    :param mixing_multipliers: a dictionary with the parameters a, b, c ,d ,e ,f
    :return: a matrix of multipliers
    """
    mixing_multipliers_matrix = np.zeros((16, 16))
    mixing_multipliers_matrix[0:3, 0:3] = mixing_multipliers["a"] * np.ones((3, 3))
    mixing_multipliers_matrix[3:13, 3:13] = mixing_multipliers["b"] * np.ones((10, 10))
    mixing_multipliers_matrix[13:, 13:] = mixing_multipliers["c"] * np.ones((3, 3))
    mixing_multipliers_matrix[3:13, 0:3] = mixing_multipliers["d"] * np.ones((10, 3))
    mixing_multipliers_matrix[0:3, 3:13] = mixing_multipliers["d"] * np.ones((3, 10))
    mixing_multipliers_matrix[13:, 0:3] = mixing_multipliers["e"] * np.ones((3, 3))
    mixing_multipliers_matrix[0:3, 13:] = mixing_multipliers["e"] * np.ones((3, 3))
    mixing_multipliers_matrix[13:, 3:13] = mixing_multipliers["f"] * np.ones((3, 10))
    mixing_multipliers_matrix[3:13, 13:] = mixing_multipliers["f"] * np.ones((10, 3))
    return mixing_multipliers_matrix
