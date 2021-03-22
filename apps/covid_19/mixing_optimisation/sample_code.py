from apps.covid_19.mixing_optimisation.constants import OPTI_REGIONS
from apps.covid_19.mixing_optimisation.mixing_opti import (
    AGE_MODE,
    DURATIONS,
    LOCATION_MODE,
    MODES,
    OBJECTIVES,
    objective_function,
    run_root_model,
)

"""
Sample code for Guillaume. Please do not delete.
"""
for country in OPTI_REGIONS:
    # need to run this line of code once to initialise the simulator for a given country
    root_model = run_root_model(country)
    for mode in MODES:
        for duration in DURATIONS:
            for objective in OBJECTIVES:

                # use some example decision variables (size 16 or 3 depending on the mode)
                if mode == AGE_MODE:
                    decision_vars = [1.0] * 16
                elif mode == LOCATION_MODE:
                    decision_vars = [1.0] * 3

                # The decision variables must now have a lower bound of 0.1  and upper bound of 1

                # run the objective function
                h, total_nb_deaths, years_of_life_lost = objective_function(
                    decision_vars, root_model, mode, country, duration
                )

                exit()
                # ----------  objective to minimise ----------
                #   - total_nb_deaths if objective == 'deaths'
                #   - years_of_life_lost if objective == 'yoll'

                # ----------       constraint      ----------
                #   - h must be True
