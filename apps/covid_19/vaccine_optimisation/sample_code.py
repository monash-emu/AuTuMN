from apps.covid_19.vaccine_optimisation.vaccine_opti import (
    get_decision_vars_names,
    initialise_opti_object
)

import numpy as np

COUNTRY = "malaysia"


def run_sample_code():
    # Initialisation of the optimisation object. This needs to be run once before optimising.
    opti_object = initialise_opti_object(COUNTRY)

    # Create decision variables for random allocations and random relaxation
    decision_vars = []
    for phase_number in range(2):
        sample = list(np.random.uniform(low=0., high=1., size=(8,)))
        sum = np.sum(sample)
        decision_vars += [s/sum for s in sample]
    decision_vars.append(np.random.uniform(low=0., high=1.))

    # Evaluate objective function
    [total_deaths, max_hospital, relaxation] = opti_object.evaluate_objective(decision_vars)

    # Print decision vars and outputs
    print(get_decision_vars_names())
    print(f"Decision variables: {decision_vars}")
    print(f"N deaths: {total_deaths} / Max hospital: {max_hospital} / Relaxation: {relaxation}")


# This can be run using:
# python -m apps runsamplevaccopti
