from applications.run_single_application import *

main_params_path = os.path.join(constants.BASE_PATH, 'applications', 'covid_19', 'params.yml')
with open(main_params_path, 'r') as yaml_file:
        main_params = yaml.safe_load(yaml_file)

this_file_dir = os.path.dirname(os.path.abspath(__file__))
OPTI_PARAMS_PATH = os.path.join(this_file_dir, 'opti_params.yml')
with open(OPTI_PARAMS_PATH, 'r') as yaml_file:
        opti_params = yaml.safe_load(yaml_file)

# INPUT_DB_PATH = os.path.join(constants.DATA_PATH, 'inputs.db')

if any([len(opti_params['mixing_matrix_multipliers']) != len(opti_params['mixing_matrix_multipliers'][i])
        for i in opti_params['mixing_matrix_multipliers']]):
    raise ValueError("mixing multipliers are ill-defined")


def objective_function(mixing_multipliers):
    # build the model



    # run the model



    # Has herd immunity been reached?
    herd_immunity = True

    # How many deaths
    total_nb_deaths = 1000

    return herd_immunity, total_nb_deaths
