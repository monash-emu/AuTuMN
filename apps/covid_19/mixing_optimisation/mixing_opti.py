import os
import copy

import yaml
import pandas as pd

from autumn.constants import Region
import autumn.post_processing as post_proc
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.params import update_params
from datetime import date, timedelta

from apps.covid_19 import RegionApp


FILE_DIR = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(FILE_DIR, "opti_params.yml")

with open(OPTI_PARAMS_PATH, "r") as yaml_file:
    opti_params = yaml.safe_load(yaml_file)

available_countries = [Region.UNITED_KINGDOM]
phase_2_end = [366, 274, 366 + 181]  # depending on the config (0: 6 months, 1: 3 months, 2: 1 year)


def build_params_for_phases_2_and_3(decision_variables, config):
    # create parameters for scenario 1 which includes Phases 2 and 3
    ref_date = date(2019, 12, 31)
    phase_2_end_date = ref_date + timedelta(days=phase_2_end[config])
    phase_3_first_day = phase_2_end_date + timedelta(days=1)
    sc_1_params = {}
    if mode == "by_age":
        age_mixing_update = {}
        for age_group in range(16):
            age_mixing_update["age_" + str(age_group)] = {
                'times': [date(2020, 6, 30), date(2020, 7, 1), phase_2_end_date, phase_3_first_day],
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
                'times': [date(2020, 6, 30), phase_2_end_date, phase_3_first_day],
                'values': [new_mixing_adjustment, new_mixing_adjustment, 1.],
                'append': False
        }

    sc_1_params['data'] = {
        'times_imported_cases': [phase_2_end[config], phase_2_end[config] + 1, phase_2_end[config] + 2,
                                 phase_2_end[config] + 3],
        'n_imported_cases': [0, 80, 80, 0]
    }
    sc_1_params['end_time'] = phase_2_end[config] + 365

    return sc_1_params


def objective_function(decision_variables, root_model, mode="by_age", country=Region.UNITED_KINGDOM, config=0,
                       calibrated_params={}):
    """
    :param decision_variables: dictionary containing
        - mixing multipliers by age as a list if mode == "by_age"    OR
        - location multipliers as a dictionary if mode == "by_location"
    :param mode: either "by_age" or "by_location"
    :param country: the country name
    :param config: the id of the configuration being considered
    :param calibrated_params: a dictionary containing a set of calibrated parameters
    """
    running_model = RegionApp(country)
    build_model = running_model.build_model
    params = copy.deepcopy(running_model.params)

    # update params with optimisation default config
    params["default"].update(opti_params["default"])

    # update params with calibrated parameters
    params["default"] = update_params(params['default'], calibrated_params)

    # update params with specific config (Sensitivity analyses)
    params["default"].update(opti_params["configurations"][config])

    # Define scenario 1
    sc_1_params = build_params_for_phases_2_and_3(decision_variables, config)
    params["scenarios"][1] = sc_1_params
    scenario_1 = Scenario(build_model, idx=1, params=params)

    # Run scenario 1
    scenario_1.run(base_model=root_model)
    models = [root_model, scenario_1.model]

    #____________________________       Perform diagnostics         ______________________
    # How many deaths during Phase 2
    first_july_index = models[1].derived_outputs["times"].index(183)
    end_phase2_index = models[1].derived_outputs["times"].index(phase_2_end[config])
    total_nb_deaths = sum(models[1].derived_outputs["infection_deathsXall"][first_july_index: end_phase2_index + 1])

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

    scenario_0 = Scenario(build_model, idx=0, params=params)
    scenario_0.run()

    return scenario_0.model


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


def has_immunity_been_reached(_model, phase_2_end_index):
    """
    Determine whether herd immunity has been reached after running a model
    :param _model: a model run with Phase 2 and
    :return: a boolean
    """
    # validate herd immunity if disease incidence always decreases after 1 week in phase 3
    future_incidence_list = _model.derived_outputs["incidence"][phase_2_end_index + 7:]
    print(future_incidence_list)
    return max(future_incidence_list) == future_incidence_list[0]


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


def run_all_phases(decision_variables, country=Region.UNITED_KINGDOM, calibrated_params={}, mode="by_age"):
    pass



if __name__ == "__main__":
    # looping through all countries and optimisation modes for testing purpose
    # optimisation will have to be performed separately for the different countries and modes.
    decision_vars = {
        "by_age": [.5] * 16, # [0.30210397, 0.455783819,	0.250627758,	0.903096598,	0.075936739,	0.24088156,	0.002722042,	0.129826402,	0.131136458,	0.119729594,	0.000211481,	0.003760947,	0.103899082,	0.137976494,	0.057792135,	0.072422987]
        "by_location": {"other_locations": 1.0, "school": 1.0, "work": 1.0},
    }

    for mode in ["by_age"]:  # , "by_location"]:
        for country in available_countries:
            for config in [0]:  # opti_params["configurations"]:
                param_set_list = read_list_of_param_sets_from_csv(country, config)
                for param_set in param_set_list:
                    # Run this line of code every time we use a new param_set and before performing optimisation
                    # This is an initialisation step
                    root_model = run_root_model(country, param_set)

                    # The following line is the one to be run again and again during optimisation
                    h, d, p_immune, m = objective_function(decision_vars[mode], root_model, mode, country, config,
                                                           param_set)
                    print("Immunity: " + str(h) + "\n" + "Deaths: " + str(round(d)) + "\n" + "Prop immune: " +
                          str(round(p_immune, 3)))
