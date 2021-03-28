from autumn.optimisation.opti import Opti
from apps.covid_19.vaccine_optimisation.vaccine_opti import (
    APP_NAME,
    ROOT_MODEL_PARAMS,
    make_objective_func,
    make_scenario_func,
)

COUNTRY = "malaysia"


def run_sample_code():
    # Initialisation of the optimisation object. This needs to be run once before optimising.
    opti_object = Opti(APP_NAME, COUNTRY, root_model_params=ROOT_MODEL_PARAMS)
    root_params = opti_object.run_root_model()  # run the baseline
    opti_object.scenario_func = make_scenario_func(root_params)
    opti_object.objective_func = make_objective_func()

    # Evaluation of the objective function
    decision_vars = [.1, .1, .1]
    [objective] = opti_object.evaluate_objective(decision_vars)
    print(objective)

    decision_vars = [.2, .2, .2]
    [objective] = opti_object.evaluate_objective(decision_vars)
    print(objective)
