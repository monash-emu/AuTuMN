import os
import copy

import yaml
import numpy as np

import autumn.post_processing as post_proc
from autumn.tool_kit.scenarios import Scenario

from apps.covid_19.countries import Country, CountryModel


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(FILE_DIR, "opti_params.yml")

with open(OPTI_PARAMS_PATH, "r") as yaml_file:
    opti_params = yaml.safe_load(yaml_file)

avaialable_countries = ['malaysia', 'philippines', 'australia', 'liberia']


def objective_function(decision_variables, mode="by_age", country='malaysia'):
    """
    :param decision_variables: dictionary containing
        - mixing multipliers by age if mode == "by_age"    OR
        - location multipliers if mode == "by_location"
    :param mode: either "by_age" or "by_location"
    :return:
    """
    running_model = CountryModel(country)
    build_model = running_model.build_model
    params = copy.deepcopy(running_model.params)

    # Define the two scenarios:
    #   baseline: with intervention
    #   scenario 1: after intervention to test immunity
    if mode == "by_age":
        mixing_update = {}
        for age_group in range(15):
            mixing_update['age_' + str(age_group) + '_times'] = [10, 14]
            mixing_update['age_' + str(age_group) + '_values'] = [1., decision_variables[age_group]]

        params["default"]["mixing"].update(mixing_update)

    # elif mode == "by_location":
    #     mixing_update_dictionary = {}
    #     for loc in ["school", "work", "other_locations"]:
    #         mixing_update_dictionary[loc + "_times"] = [0]
    #         mixing_update_dictionary[loc + "_values"] = [decision_variables[loc]]
    #
    #     params["default"].update({"mixing": mixing_update_dictionary})
    #
    # else:
    #     raise ValueError("The requested mode is not supported")

    # Add a scenario without any mixing multipliers
    end_time = params["default"]["end_time"]
    params["scenario_start_time"] = end_time - 1
    params["scenarios"][1] = {
        "end_time": end_time + 50,
        # "mixing_matrix_multipliers": None,
        # "mixing": None,
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


if __name__ == '__main__':
    for country in avaialable_countries:
        mode = 'by_age'
        mixing_multipiers = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                             1., 1., 1., 1., 1., 1.]
        mixing_multipiers = [.1 * m for m in mixing_multipiers]
        h, d, m = objective_function(mixing_multipiers, mode, country)

        print(h)
        print(d)

