import os
import copy

import yaml
import pandas as pd

from autumn.model_runner import build_model_runner
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.params import update_params
from datetime import date, timedelta

from apps.covid_19 import RegionApp
from apps.covid_19.mixing_optimisation.constants import *


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(FILE_DIR, "opti_params.yml")

with open(OPTI_PARAMS_PATH, "r") as yaml_file:
    opti_params = yaml.safe_load(yaml_file)

available_countries = OPTI_REGIONS

phase_2_end = [PHASE_2_START_TIME + opti_params['configurations'][i]['phase_two_duration'] for
               i in opti_params['configurations'].keys()]


def run_root_model(country=Region.UNITED_KINGDOM, calibrated_params={}):
    """
    This function runs a model to simulate the past epidemic (up until 1/7/2020) using a given calibrated parameter set.
    Returns an integrated model for the past epidemic.
    """
    running_model = RegionApp(country)
    build_model = running_model.build_model

    params = copy.deepcopy(running_model.params)
    # update params with optimisation default config
    params["default"].update(opti_params["default"])
    # update params with calibrated parameters
    params["default"] = update_params(params['default'], calibrated_params)

    # prepare importation rates for herd immunity testing
    params["default"]["data"] = {
        'times_imported_cases': [0],
        'n_imported_cases': [0]
    }
    params["default"]["end_time"] = PHASE_2_START_TIME
    params["scenario_start_time"] = PHASE_2_START_TIME - 1


    scenario_0 = Scenario(build_model, idx=0, params=params)
    scenario_0.run()

    return scenario_0.model


def build_params_for_phases_2_and_3(decision_variables, config=0, mode='by_age'):
    # create parameters for scenario 1 which includes Phases 2 and 3
    ref_date = date(2019, 12, 31)
    phase_2_first_day = ref_date + timedelta(days=PHASE_2_START_TIME)
    phase_1_end_date = phase_2_first_day + timedelta(days=-1)
    phase_2_end_date = ref_date + timedelta(days=phase_2_end[config])
    phase_3_first_day = phase_2_end_date + timedelta(days=1)

    sc_1_params = {}
    if mode == "by_age":
        age_mixing_update = {}
        for age_group in range(16):
            age_mixing_update["age_" + str(age_group)] = {
                'times': [phase_1_end_date, phase_2_first_day, phase_2_end_date, phase_3_first_day],
                'values': [1.0, decision_variables[age_group], decision_variables[age_group], 1.0]
            }
        sc_1_params["mixing_age_adjust"] = age_mixing_update

    # set location-specific mixing back to pre-COVID rates on 1st of July or use the opti decision variable
    sc_1_params['mixing'] = {}
    for loc in ["other_locations", "school", "work"]:
        if mode == "by_age":  # just return mixing to pre-COVID
            new_mixing_adjustment = 1.0
        elif mode == "by_location":  # use optimisation decision variables
            new_mixing_adjustment = decision_variables[loc]
        else:
            raise ValueError("The requested mode is not supported")

        sc_1_params['mixing'][loc] = {
                'times': [phase_1_end_date, phase_2_end_date, phase_3_first_day],
                'values': [new_mixing_adjustment, new_mixing_adjustment, 1.],
                'append': False
        }

    sc_1_params['data'] = {
        'times_imported_cases': [phase_2_end[config], phase_2_end[config] + 1, phase_2_end[config] + 2,
                                 phase_2_end[config] + 3],
        'n_imported_cases': [0, 5, 5, 0]
    }
    sc_1_params['end_time'] = phase_2_end[config] + PHASE_3_DURATION

    if "microdistancing" in opti_params['configurations'][config]:
        if "microdistancing" in sc_1_params:
            sc_1_params['microdistancing'].update(opti_params['configurations'][config]['microdistancing'])
        else:
            sc_1_params['microdistancing'] = opti_params['configurations'][config]['microdistancing']
    return sc_1_params


def has_immunity_been_reached(_model, phase_2_end_index):
    """
    Determine whether herd immunity has been reached after running a model
    :param _model: a model run with Phase 2 and
    :return: a boolean
    """
    # validate herd immunity if exposed prevalence always decreases after 2 weeks in phase 3
    early_compartments_indices = [i for i, c in enumerate(_model.compartment_names) if 'exposedX' in c]
    time_indices = range(phase_2_end_index, len(_model.derived_outputs["times"]))
    early_infections = [sum([_model.outputs[t, i] for i in early_compartments_indices]) for t in time_indices[14:]]
    return max(early_infections) == early_infections[0]


def objective_function(decision_variables, root_model, mode="by_age", country=Region.UNITED_KINGDOM, config=0,
                       calibrated_params={}):
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
    running_model = RegionApp(country)
    build_model = running_model.build_model
    params = copy.deepcopy(running_model.params)

    # reformat decision vars if locations
    if mode == "by_location":
        new_decision_variables = {
            "other_locations": decision_variables[0],
            "school": decision_variables[1],
            "work": decision_variables[2]
        }
        decision_variables = new_decision_variables

    # Define scenario-1-specific params
    sc_1_params_update = build_params_for_phases_2_and_3(decision_variables, config, mode)

    # Rebuild the default parameters
    params["default"].update(opti_params["default"])
    params["default"] = update_params(params['default'], calibrated_params)
    params['scenario_start_time'] = PHASE_2_START_TIME - 1

    # Create scenario 1
    sc_1_params = update_params(params['default'], sc_1_params_update)
    params["scenarios"][1] = sc_1_params
    scenario_1 = Scenario(build_model, idx=1, params=params)

    # Run scenario 1
    scenario_1.run(base_model=root_model)
    models = [root_model, scenario_1.model]

    #____________________________       Perform diagnostics         ______________________
    # How many deaths during Phase 2
    start_phase2_index = models[1].derived_outputs["times"].index(PHASE_2_START_TIME)
    end_phase2_index = models[1].derived_outputs["times"].index(phase_2_end[config])
    total_nb_deaths = sum(models[1].derived_outputs["infection_deathsXall"][start_phase2_index: end_phase2_index + 1])
    print("Max deaths: " + str(max(models[1].derived_outputs["infection_deathsXall"])))


    # What proportion immune at end of Phase 2
    recovered_indices = [
        i
        for i in range(len(models[1].compartment_names))
        if "recovered" in models[1].compartment_names[i]
    ]
    nb_reco = sum([models[1].outputs[end_phase2_index, i] for i in recovered_indices])
    total_pop = sum([models[1].outputs[end_phase2_index, i] for i in range(len(models[1].compartment_names))])
    prop_immune = nb_reco / total_pop

    # Has herd immunity been reached?
    herd_immunity = has_immunity_been_reached(models[1], end_phase2_index)   # FIXME change this

    return herd_immunity, total_nb_deaths, prop_immune, models


def read_list_of_param_sets_from_csv(country, config):
    """
    Read a csv file containing the MCMC outputs and return a list of calibrated parameter sets. Each parameter set is
    described as a dictionary.
    :param country: string
    :param config: integer used to refer to different sensitivity analyses
    :return: a list of dictionaries
    """
    path_to_csv = os.path.join('calibrated_param_sets', country + '_config_' + str(config) + ".csv")
    table = pd.read_csv(path_to_csv)

    col_names_to_skip = ["idx", "loglikelihood", "best_deaths", "all_vars_to_1_deaths",
                         "best_p_immune", "all_vars_to_1_p_immune",
                         "notifications_dispersion_param", "infection_deathsXall_dispersion_param"]
    for i in range(16):
        col_names_to_skip.append("best_x" + str(i))

    list_of_param_sets = []

    for index, row in table.iterrows():
        par_dict = {}
        for col_name in [c for c in table.columns if c not in col_names_to_skip]:
            par_dict[col_name] = row[col_name]
        list_of_param_sets.append(par_dict)

    return list_of_param_sets


def run_all_phases(decision_variables, country=Region.UNITED_KINGDOM, config=0, calibrated_params={}, mode="by_age"):
    running_model = RegionApp(country)
    build_model = running_model.build_model

    params = copy.deepcopy(running_model.params)
    # update params with optimisation default config
    params["default"].update(opti_params["default"])
    # update params with calibrated parameters
    params["default"] = update_params(params['default'], calibrated_params)

    # prepare importation rates for herd immunity testing
    params["default"]["data"] = {
        'times_imported_cases': [0],
        'n_imported_cases': [0]
    }

    params["scenarios"][1] = build_params_for_phases_2_and_3(decision_variables, config, mode)

    run_models = build_model_runner(
        model_name="covid_19",
        param_set_name=country,
        build_model=build_model,
        params=params
    )

    run_models()


if __name__ == "__main__":
    # looping through all countries and optimisation modes for testing purpose
    # optimisation will have to be performed separately for the different countries and modes.

    decision_vars = {
        "by_age": [1.] * 16,
        "by_location": [1., 1., 1.]
    }

    # to produce graph with 3 phases
    # run_all_phases(decision_vars["by_age"], "belgium", 2, {}, "by_age")
    # exit()

    for _mode in ["by_age", "by_location"]:  # , "by_location"]:
        for _country in ['belgium', 'united-kingdom']:  # available_countries:
            print("*********** " + _country + " ***********")
            for _config in [2, 3]:  # opti_params["configurations"]:
                # param_set_list = read_list_of_param_sets_from_csv(country, config)
                param_set_list = [{}]
                for param_set in param_set_list:
                    # Run this line of code every time we use a new param_set and before performing optimisation
                    # This is an initialisation step
                    _root_model = run_root_model(_country, param_set)

                    # The following line is the one to be run again and again during optimisation
                    h, d, p_immune, m = objective_function(decision_vars[_mode], _root_model, _mode, _country, _config,
                                                           param_set)
                    print("Immunity: " + str(h) + "\n" + "Deaths: " + str(round(d)) + "\n" + "Prop immune: " +
                          str(round(p_immune, 3)))

                    # run_all_phases(decision_vars[_mode], _country, _config, param_set, _mode)
                    # break
