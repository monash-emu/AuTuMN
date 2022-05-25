from autumn.runners.optimisation.opti import Opti

from autumn.projects.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from autumn.projects.covid_19.mixing_optimisation.mixing_opti import (
    APP_NAME,
    ROOT_MODEL_PARAMS,
    AGE_MODE,
    DURATIONS,
    LOCATION_MODE,
    MODES,
    OBJECTIVES,
    make_objective_func,
    make_scenario_func,
)

PRINT_OUTPUT = True


def run_sample_code():
    """
    Sample code for Guillaume. Please do not delete.
    """
    for country in OPTI_REGIONS:
        # need to run this line of code once to initialise the simulator for a given country
        opti_object = Opti(APP_NAME, country, root_model_params=ROOT_MODEL_PARAMS)
        root_params = opti_object.run_root_model()

        for mode in MODES:
            for duration in DURATIONS:

                opti_object.objective_func = make_objective_func(duration)
                opti_object.scenario_func = make_scenario_func(
                    root_params, duration, mode, final_mixing=1.0
                )

                for objective in OBJECTIVES:

                    # use some example decision variables (size 16 or 3 depending on the mode)
                    if mode == AGE_MODE:
                        decision_vars = [1.0] * 16
                    elif mode == LOCATION_MODE:
                        decision_vars = [1.0] * 3

                    # The decision variables must now have a lower bound of 0.1  and upper bound of 1
                    # run the objective function
                    [h, total_nb_deaths, years_of_life_lost] = opti_object.evaluate_objective(
                        decision_vars
                    )

                    if PRINT_OUTPUT:
                        print(f"{country} / {mode} / {duration} / {objective}")
                        print(h)
                        print(total_nb_deaths)
                        print(years_of_life_lost)

                    exit()

                    # ----------  objective to minimise ----------
                    #   - total_nb_deaths if objective == 'deaths'
                    #   - years_of_life_lost if objective == 'yoll'

                    # ----------       constraint      ----------
                    #   - h must be True
