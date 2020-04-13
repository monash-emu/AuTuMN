from applications.run_single_application import *
from autumn.db import Database, get_iso3_from_country_name

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

# select the country
COUNTRY = 'Australia'
ISO3 = get_iso3_from_country_name(input_database, COUNTRY)
main_params['default']['country'] = COUNTRY
main_params['default']['iso3'] = ISO3

# check the format of the mixing_matrix_multipliers
if any([len(opti_params['mixing_matrix_multipliers']) != len(opti_params['mixing_matrix_multipliers'][i])
        for i in opti_params['mixing_matrix_multipliers']]):
    raise ValueError("mixing multipliers are ill-defined")


def objective_function(mixing_multipliers):
    # build the model
    model_function = build_covid_model
    mixing_progression = build_covid_matrices(main_params['default']['country'], main_params['mixing'])

    # Prepare scenario data
    scenario_params = {0: {}}

    # run the model
    model = run_multi_scenario(
        scenario_params,
        main_params['scenario_start'],
        model_function,
        mixing_progression,
        run_kwargs=SOLVER_KWARGS
    )[0]

    # Has herd immunity been reached?
    herd_immunity = True

    # How many deaths
    total_nb_deaths = sum(model.derived_outputs['infection_deathsXall'])

    return herd_immunity, total_nb_deaths


# h_i, obj = objective_function(mixing_multipliers=opti_params['mixing_matrix_multipliers'])
