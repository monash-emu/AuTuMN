from applications.run_single_application import *
from autumn.db import Database, get_iso3_from_country_name
import numpy as np
import copy

INPUT_DB_PATH = os.path.join(constants.DATA_PATH, 'inputs.db')
input_database = Database(database_name=INPUT_DB_PATH)

# read parameter values
main_params_path = os.path.join(constants.BASE_PATH, 'applications', 'covid_19', 'params.yml')
with open(main_params_path, 'r') as yaml_file:
        main_params = yaml.safe_load(yaml_file)
main_params['default'] = add_agegroup_breaks(main_params['default'])

this_file_dir = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(this_file_dir, 'opti_params.yml')
with open(OPTI_PARAMS_PATH, 'r') as yaml_file:
        opti_params = yaml.safe_load(yaml_file)

main_params['default'].update(opti_params['default'])


# select the country
COUNTRY = 'Australia'
ISO3 = get_iso3_from_country_name(input_database, COUNTRY)
main_params['default']['country'] = COUNTRY
main_params['default']['iso3'] = ISO3


def has_immunity_been_reached(_model):
    """
    Determine whether herd immunity has been reached after running a model
    :param _model: a model run with no-intervention setting for testing herd-immunity
    :return: a boolean
    """
    return max(_model.derived_outputs['incidence']) == _model.derived_outputs['incidence'][0]


def build_mixing_multipliers_matrix(mixing_multipliers):
    """
    Builds a full 16x16 matrix of multipliers based on the parameters found in mixing_multipliers
    :param mixing_multipliers: a dictionary with the parameters a, b, c ,d ,e ,f
    :return: a matrix of multipliers
    """
    mixing_multipliers_matrix = np.zeros((16, 16))
    mixing_multipliers_matrix[0:3, 0:3] = mixing_multipliers['a'] * np.ones((3, 3))
    mixing_multipliers_matrix[3:13, 3:13] = mixing_multipliers['b'] * np.ones((10, 10))
    mixing_multipliers_matrix[13:, 13:] = mixing_multipliers['c'] * np.ones((3, 3))
    mixing_multipliers_matrix[3:13, 0:3] = mixing_multipliers['d'] * np.ones((10, 3))
    mixing_multipliers_matrix[0:3, 3:13] = mixing_multipliers['d'] * np.ones((3, 10))
    mixing_multipliers_matrix[13:, 0:3] = mixing_multipliers['e'] * np.ones((3, 3))
    mixing_multipliers_matrix[0:3, 13:] = mixing_multipliers['e'] * np.ones((3, 3))
    mixing_multipliers_matrix[13:, 3:13] = mixing_multipliers['f'] * np.ones((3, 10))
    mixing_multipliers_matrix[3:13, 13:] = mixing_multipliers['f'] * np.ones((10, 3))
    return mixing_multipliers_matrix


def objective_function(mixing_multipliers):
    # build the model
    model_function = build_covid_model
    mixing_progression = {}

    mixing_multipliers_matrix = build_mixing_multipliers_matrix(mixing_multipliers)

    # save params without mixing multipliers for scenario 1
    main_params_sc1 = copy.copy(main_params['default'])
    main_params_sc1['end_time'] = 300

    # Prepare scenario data
    main_params['default'].update({'mixing_matrix_multipliers': mixing_multipliers_matrix})
    scenario_params = {0: main_params['default'],
                       1: main_params_sc1}

    # run the model
    models = run_multi_scenario(
        scenario_params,
        main_params['default']['end_time'] - 1,
        model_function,
        mixing_progression,
        run_kwargs=SOLVER_KWARGS
    )

    # Has herd immunity been reached?
    herd_immunity = has_immunity_been_reached(models[1])

    # How many deaths
    total_nb_deaths = sum(models[0].derived_outputs['infection_deathsXall'])

    return herd_immunity, total_nb_deaths, models


def visualise_simulation(_models):

    pps = []
    for scenario_index in range(len(_models)):

        pps.append(
                    post_proc.PostProcessing(
                        _models[scenario_index],
                        requested_outputs=['prevXinfectiousXamong', 'prevXrecoveredXamong'],
                        scenario_number=scenario_index,
                        requested_times={}
                    )
                )

    old_outputs_plotter = Outputs(_models, pps, {}, plot_start_time=0)
    old_outputs_plotter.plot_requested_outputs()


# parameters to optimise (all bounded in [0., 2.])
mixing_mult = {'a': 1., 'b': 1.5, 'c': 0., 'd': .5, 'e': .7, 'f': 1.3}

h_immu, obj, models = objective_function(mixing_multipliers=mixing_mult)

# visualise_simulation(models)

