import os
import copy

import yaml
import pandas as pd
import numpy as np

from autumn.model_runner import build_model_runner
from autumn.tool_kit.scenarios import Scenario
from autumn.tool_kit.params import update_params
from datetime import date, timedelta

from apps.covid_19 import RegionApp
from apps.covid_19.mixing_optimisation.constants import *
from apps.covid_19.preprocess.mixing_matrix.dynamic import build_dynamic
from autumn.inputs.demography.queries import get_iso3_from_country_name

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


def build_params_for_phases_2_and_3(decision_variables, config=0, mode='by_age', final_mixing=1.):
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
                'values': [1.0, decision_variables[age_group], decision_variables[age_group], final_mixing]
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
                'values': [new_mixing_adjustment, new_mixing_adjustment, final_mixing],
                'append': False
        }

    sc_1_params['data'] = {
        'times_imported_cases': [phase_2_end[config], phase_2_end[config] + 1, phase_2_end[config] + 2,
                                 phase_2_end[config] + 3],
        'n_imported_cases': [0,0, 0, 0]  # FIXME TURN ON IMPORTAION BEFORE RUNNIN ANY OPTIMISATION. use [0, 5, 5, 0]
    }

    # assert sum(sc_1_params['data']['n_imported_cases']) > 0., "Romain needs to turn on importation before running any opti!"

    sc_1_params['end_time'] = PHASE_2_START_TIME + DURATION_PHASES_2_AND_3

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
    # validate herd immunity if incidence always decreases after 2 weeks in phase 3
    time_indices = range(phase_2_end_index, len(_model.derived_outputs["times"]))
    incidence_vals = [_model.derived_outputs['incidence'][i] for i in time_indices[14:]]
    return max(incidence_vals) == incidence_vals[0]


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
    # How many deaths and years of life lost during Phase 2 and 3
    start_phase2_index = models[1].derived_outputs["times"].index(PHASE_2_START_TIME)
    end_phase2_index = models[1].derived_outputs["times"].index(phase_2_end[config])
    total_nb_deaths = sum(models[1].derived_outputs["infection_deathsXall"][start_phase2_index:])
    years_of_life_lost = sum(models[1].derived_outputs["years_of_life_lost"][start_phase2_index:])

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
    herd_immunity = has_immunity_been_reached(models[1], end_phase2_index)

    return herd_immunity, total_nb_deaths, years_of_life_lost, prop_immune, models


def read_list_of_param_sets_from_csv(country):
    """
    Read a csv file containing the MCMC outputs and return a list of calibrated parameter sets. Each parameter set is
    described as a dictionary.
    :param country: string
    :param config: integer used to refer to different sensitivity analyses
    :return: a list of dictionaries
    """
    path_to_csv = os.path.join('calibrated_param_sets', country + "_calibrated_params.csv")
    table = pd.read_csv(path_to_csv)

    col_names_to_skip = ["idx", "loglikelihood", "best_deaths", "all_vars_to_1_deaths",
                         "best_p_immune", "all_vars_to_1_p_immune",
                         "best_yoll", "all_vars_to_1_yoll",
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


def make_model_builder(decision_variables, country=Region.UNITED_KINGDOM, config=0, calibrated_params={}, mode="by_age"):
    running_model = RegionApp(country)
    build_model = running_model.build_model

    if mode == "by_location":
        new_decision_variables = {
            "other_locations": decision_variables[0],
            "school": decision_variables[1],
            "work": decision_variables[2]
        }
        decision_variables = new_decision_variables

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

    return build_model_runner(
        model_name="covid_19",
        param_set_name=country,
        build_model=build_model,
        params=params
    )


def run_all_phases(decision_variables, country=Region.UNITED_KINGDOM, config=0, calibrated_params={}, mode="by_age"):
    run_models = make_model_builder(decision_variables, country, config, calibrated_params, mode)
    run_models()
    return


def read_csv_output_file(output_dir, country, config=2, mode="by_age", objective="deaths", from_streamlit=False):
    path_to_input_csv = os.path.join('calibrated_param_sets', country + "_calibrated_params.csv")
    if from_streamlit:
        path_to_input_csv = os.path.join('apps', 'covid_19', 'mixing_optimisation', path_to_input_csv)
    input_table = pd.read_csv(path_to_input_csv)

    col_names = [c for c in input_table.columns if c not in ["loglikelihood", 'idx'] and "dispersion_param" not in c]

    if mode == "by_location":
        removed_columns = ["best_x" + str(i) for i in range(3, 16)]
        col_names = [c for c in col_names if c not in removed_columns]

    output_file_name = os.path.join(output_dir, "results_" + country + "_" + mode + "_" + str(config) + "_" + objective + ".csv")
    out_table = pd.read_csv(output_file_name, sep=" ", header=None)
    out_table.columns = col_names
    out_table["loglikelihood"] = input_table["loglikelihood"]

    return out_table


def get_mle_params_and_vars(output_dir, country, config=2, mode="by_age", objective="deaths", from_streamlit=False):

    out_table = read_csv_output_file(output_dir, country, config, mode, objective, from_streamlit)
    n_vars = {"by_age": 16, "by_location": 3}

    mle_rows = out_table[out_table["loglikelihood"] == out_table.loc[len(out_table) - 1, "loglikelihood"]]
    mle_rows = mle_rows.sort_values(by="best_" + objective)

    decision_vars = [float(mle_rows.loc[mle_rows.index[0], "best_x" + str(i)]) for i in range(n_vars[mode])]

    params = {}
    for c in out_table.columns:
        if c in ["idx", "loglikelihood"]:
            continue
        elif "best_" in c:
            break
        params[c] = float(mle_rows.loc[mle_rows.index[0], c])

    # print("Guillaume's outputs for best_death: " + str(float(mle_rows.loc[mle_rows.index[0], 'best_deaths'])))

    return params, decision_vars


def reformat_date_to_integer(_date):
    ref_date = date(2019, 12, 31)

    delta = (_date - ref_date).days
    return delta


def get_params_for_phases_2_and_3_from_opti_outptuts(output_dir, country, config, mode, objective, final_mixing=1.):
    _, decision_vars = get_mle_params_and_vars(output_dir, country, config, mode, objective)
    if mode == "by_location":
        new_decision_variables = {
            "other_locations": decision_vars[0],
            "school": decision_vars[1],
            "work": decision_vars[2]
        }
        decision_vars = new_decision_variables

    sc_params = build_params_for_phases_2_and_3(decision_vars, config, mode, final_mixing)
    # clean up times
    for par in ["mixing_age_adjust", "mixing"]:
        if par in sc_params:
            for mixing_key in sc_params[par].keys():
                sc_params[par][mixing_key]['times'] = [
                    reformat_date_to_integer(d) for d in sc_params[par][mixing_key]['times']
                ]
    return sc_params


def drop_yml_scenario_file(output_dir, country, config=2, mode="by_age", objective="deaths", final_mixing=1.):
    sc_params = get_params_for_phases_2_and_3_from_opti_outptuts(output_dir, country, config, mode, objective, final_mixing)

    scenario_mapping = {
        1: mode + "_2_deaths",
        2: mode + "_2_yoll",
        3: mode + "_3_deaths",
        4: mode + "_3_yoll",
    }

    for key, val in scenario_mapping.items():
        if val == mode + "_" + str(config) + "_" + objective:
            sc_index = key
    param_file_path = "../params/" + country + "/scenario-" + str(sc_index) + ".yml"

    with open(param_file_path, "w") as f:
        yaml.dump(sc_params, f)


def write_all_yml_files_from_outputs(output_dir, mode="by_age"):
    for country in OPTI_REGIONS:
        for config in [2, 3]:
            for objective in ["deaths", "yoll"]:
                drop_yml_scenario_file(output_dir, country, config, mode, objective)
        if mode == "by_age":
        # make extra scenario for unmitigated
            sc_params = build_params_for_phases_2_and_3([1.] * 16, 2, 'by_age')
            # clean up times
            for par in ["mixing_age_adjust", "mixing"]:
                if par in sc_params:
                    for mixing_key in sc_params[par].keys():
                        sc_params[par][mixing_key]['times'] = [
                            reformat_date_to_integer(d) for d in sc_params[par][mixing_key]['times']
                        ]
            param_file_path = "../params/" + country + "/scenario-5.yml"

            with open(param_file_path, "w") as f:
                yaml.dump(sc_params, f)


def write_all_yml_files_for_immunity_scenarios(output_dir, final_mixing=1.):

    durations = [183., 2*365.25]
    rel_prop_sympts = [1., .5]

    mode = "by_age"
    config = 2
    objective = "deaths"

    for country in OPTI_REGIONS:
        sc_params = get_params_for_phases_2_and_3_from_opti_outptuts(output_dir, country, config, mode, objective, final_mixing)
        sc_params['end_time'] = 366 + 365
        param_file_path = "../params/" + country + "/scenario-1.yml"
        with open(param_file_path, "w") as f:
            yaml.dump(sc_params, f)

        sc_params['full_immunity'] = False
        sc_index = 1
        for duration in durations:
            sc_params['immunity_duration'] = duration
            for rel_prop_sympt in rel_prop_sympts:
                sc_index += 1
                sc_params['rel_prop_symptomatic_experienced'] = rel_prop_sympt

                param_file_path = "../params/" + country + "/scenario-" + str(sc_index) + ".yml"
                with open(param_file_path, "w") as f:
                    yaml.dump(sc_params, f)


def evaluate_extra_deaths(decision_vars, extra_contribution, i, root_model, mode, country,
                                            config, mle_params, best_objective, objective, direction="up"):
    tested_decision_vars = copy.deepcopy(decision_vars)
    if direction == "up":
        tested_decision_vars[i] += extra_contribution
    else:
        tested_decision_vars[i] -= extra_contribution
    h, this_d, this_yoll, p_immune, m = objective_function(tested_decision_vars, root_model, mode, country,
                                                  config, mle_params)
    this_objective = {
        'deaths': this_d,
        'yoll': this_yoll
    }

    if not h:
        delta_deaths_per_million = 1.e6
    else:
        population = sum(m[0].compartment_values)
        delta_deaths_per_million = (this_objective[objective] - best_objective) / population * 1.e6

    return delta_deaths_per_million


def run_sensitivity_perturbations(output_dir, country, config=2, mode="by_age", objective="deaths",
                                  target_objective_per_million=20, tol=.02, direction="up"):
    # target_deaths is a number of deaths per million people
    mle_params, decision_vars = get_mle_params_and_vars(output_dir, country, config, mode, objective)
    root_model = run_root_model(country, mle_params)

    h, best_d, best_yoll, p_immune, m = objective_function(decision_vars, root_model, mode, country, config, mle_params)
    best_objective = {
        'deaths': best_d,
        'yoll': best_yoll,
    }

    delta_contributions = []
    for i in range(len(decision_vars)):
        print("Age group " + str(i))
        extra_contribution_lower = 0.
        if direction == "up":
            extra_contribution_upper = 1. - decision_vars[i]
        else:
            extra_contribution_upper = decision_vars[i]

        if extra_contribution_upper < tol:
            best_solution = extra_contribution_upper if direction == "up" else decision_vars[i]
        else:
            # find an upper bound (lower if direction is down):
            delta_deaths_per_million = evaluate_extra_deaths(decision_vars, extra_contribution_upper, i, root_model, mode, country,
                                                             config, mle_params, best_objective[objective], objective,
                                                             direction)
            if delta_deaths_per_million < target_objective_per_million:
                best_solution = extra_contribution_upper
            else:
                loop_count = 0
                while (extra_contribution_upper - extra_contribution_lower) > tol:
                    evaluation_point = (extra_contribution_lower + extra_contribution_upper) / 2.
                    delta_deaths_per_million = evaluate_extra_deaths(decision_vars, evaluation_point, i, root_model, mode, country,
                                                                     config, mle_params, best_objective[objective],
                                                                     objective, direction)
                    if delta_deaths_per_million > target_objective_per_million:
                        extra_contribution_upper = evaluation_point
                    else:
                        extra_contribution_lower = evaluation_point
                    loop_count += 1
                    if loop_count >= 20:
                        print("FLAG INFINITE LOOP")
                        break

                if (extra_contribution_upper - target_objective_per_million) < (target_objective_per_million - extra_contribution_lower):
                    best_solution = extra_contribution_upper
                else:
                    best_solution = extra_contribution_lower

        delta_contributions.append(best_solution)

        print(best_solution)
    output_file_path = os.path.join("optimisation_outputs", "sensitivity", country + "_" + mode + "_" + str(config) + "_" + objective + "_" + direction + ".yml")

    with open(output_file_path, "w") as f:
        yaml.dump(delta_contributions, f)


def run_sensitivity_minimum_mixing(output_dir):
    mode = 'by_age'
    results = {}
    for country in OPTI_REGIONS:
        results[country] = {}
        for config in [2, 3]:
            results[country][config] = {}
            for objective in ['deaths', 'yoll']:
                results[country][config][objective] = {}
                mle_params, decision_vars = get_mle_params_and_vars(output_dir, country, config, mode, objective)
                if config == 2 and objective == 'deaths':  # to run the root_model only once
                    print("running root model for " + country)
                    root_model = run_root_model(country, mle_params)
                for min_mixing in [0., .10, .20, .30, .40, .50]:
                    modified_vars = copy.deepcopy(decision_vars)
                    modified_vars = [max([v, min_mixing]) for v in modified_vars]
                    print("evaluate objective for " + country + " " + str(config) + " " + objective)
                    h, d, yoll, p_immune, _ = objective_function(modified_vars, root_model, mode, country,
                                                                 config, mle_params)
                    res_dict = {'h': bool(h), 'd': float(d), 'yoll': float(yoll), 'p_immune': float(p_immune)}
                    results[country][config][objective][min_mixing] = res_dict

    param_file_path = "optimisation_outputs/sensitivity_min_mixing/results.yml"
    with open(param_file_path, "w") as f:
        yaml.dump(results, f)
    return results


def read_sensitivity_min_mix_res():
    res_path = "optimisation_outputs/sensitivity_min_mixing/results.yml"

    with open(res_path, "r") as yaml_file:
        results = yaml.safe_load(yaml_file)

    return results


def explore_optimal_plan(decision_vars, _root_model, mode, _country, config, param_set, delta=.1):
    n_vars = {'by_age': 16, "by_location": 3}
    h, d, yoll, p_immune, m = objective_function(decision_vars, _root_model, mode, _country, config,
                                                    param_set)
    print("Baseline Deaths: " + str(round(d)))
    print("Baseline herd immunity: " + str(h))
    print("#############################")

    for i in range(n_vars[mode]):
        test_vars = copy.deepcopy(decision_vars)
        test_vars[i] += delta
        test_vars[i] = min([test_vars[i], 1.])
        test_vars[i] = max([test_vars[i], 0.])

        h, d, yoll, p_immune, m = objective_function(test_vars, _root_model, mode, _country, config,
                                                    param_set)
        print("Deaths " + str(i) + ": " + str(int(round(d))) + " " + str(h) + "\t" + str(decision_vars[i]) + " -->" + str(test_vars[i]))


def get_mixing_matrices(output_dir, country, config=2, mode="by_age", objective="deaths", from_streamlit=False):

    iso_3 = get_iso3_from_country_name(country.title()) if country != "united-kingdom" else "GBR"
    params, decision_vars = get_mle_params_and_vars(output_dir, country, config, mode, objective, from_streamlit)

    if mode == "by_location":
        new_decision_variables = {
            "other_locations": decision_vars[0],
            "school": decision_vars[1],
            "work": decision_vars[2]
        }
        decision_vars = new_decision_variables

    sc_1_params = build_params_for_phases_2_and_3(decision_vars, config, mode)
    if mode == 'by_location':
        sc_1_params['mixing_age_adjust'] = {}


    mixing_func = build_dynamic(iso_3, country, mixing=sc_1_params['mixing'], mixing_age_adjust=sc_1_params['mixing_age_adjust'],
                                npi_effectiveness_params={},
                                google_mobility_locations={'work': ['workplaces'], 'other_locations': ['retail_and_recreation', 'grocery_and_pharmacy', 'transit_stations']},
                                is_periodic_intervention=False, periodic_int_params={}, periodic_end_time=0.,
                                microdistancing_params={}, smooth_google_data=True)

    original_prem = mixing_func(10000.)
    optimised = mixing_func(PHASE_2_START_TIME + 10.)

    return original_prem, optimised


if __name__ == "__main__":
    # looping through all countries and optimisation modes for testing purpose
    # optimisation will have to be performed separately for the different countries and modes.
    # res = read_sensitivity_min_mix_res()
    # exit()

    output_dir = "optimisation_outputs/6Aug2020/"
    # run_sensitivity_minimum_mixing(output_dir)
    # exit()
    # write_all_yml_files_from_outputs(output_dir, mode="by_age")
    write_all_yml_files_for_immunity_scenarios(output_dir, final_mixing=.8)
    for _country in available_countries:
        print("Running for " + _country + " ...")
        mode = 'by_age'
        for config in [2]:
            for objective in ["deaths"]:
                print("config_" + str(config) + " objective_" + objective)
                param_set, decision_vars = get_mle_params_and_vars(output_dir, _country, config=config, mode=mode,
                                                              objective=objective)
                # _root_model = run_root_model(_country, param_set)
                #
                # h, d, yoll, p_immune, m = objective_function(decision_vars, _root_model, mode, _country, config,
                #                             param_set)
                # print("Immunity: " + str(h) + "\n" + "Deaths: " + str(round(d)) + "\n" + "Years of life lost: " +
                # str(round(yoll)) + "\n" + "Prop immune: " + str(round(p_immune, 3))
                # )

                run_all_phases(decision_vars, _country, config, param_set, mode)
                print("... done.")
    exit()

    # decision_vars = {
    #     "by_age": [0.99403736,0.966716181,0.99528575,0.996704989,0.999250901,0.99909351,0.996430804,0.99494714,0.999902635,0.999955508,0.988036486,0.970353795,0.03743012,0.170611743,0.004352714,0.243200946],
    #     "by_location": [1., 1., 1.]
    # }

    # to produce graph with 3 phases
    # run_all_phases(decision_vars["by_age"], "sweden", 2, {}, "by_age")
    # exit()

    for _mode in ["by_age", "by_location"]:
        for _country in available_countries:
            print("*********** " + _country + " ***********")
            for _config in [2, 3]:
                param_set_list = read_list_of_param_sets_from_csv(_country)
                # param_set_list = [param_set_list[-1]]
                for param_set in param_set_list:
                    # Run this line of code every time we use a new param_set and before performing optimisation
                    # This is an initialisation step
                    _root_model = run_root_model(_country, param_set)

                    # The following line is the one to be run again and again during optimisation
                    h, d, yoll, p_immune, m = objective_function(decision_vars[_mode], _root_model, _mode, _country, _config,
                                                           param_set)
                    print("Immunity: " + str(h) + "\n" + "Deaths: " + str(round(d)) + "\n" + "Years of life lost: " +
                          str(round(yoll)) + "\n" + "Prop immune: " + str(round(p_immune, 3))
                          )

                    # run_all_phases(decision_vars[_mode], _country, _config, param_set, _mode)
                    # break
