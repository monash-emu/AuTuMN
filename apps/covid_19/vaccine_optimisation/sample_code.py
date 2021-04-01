from apps.covid_19.vaccine_optimisation.vaccine_opti import (
    get_decision_vars_names,
    initialise_opti_object
)

import numpy as np
import yaml

COUNTRY = "malaysia"  # should use "malaysia" or "philippines"


def run_sample_code():
    # Initialisation of the optimisation object. This needs to be run once before optimising.
    opti_object = initialise_opti_object(COUNTRY)

    # Create decision variables for random allocations and random relaxation
    decision_vars = []
    for phase_number in range(2):
        sample = list(np.random.uniform(low=0., high=1., size=(8,)))
        _sum = sum(sample)
        decision_vars += [s/_sum for s in sample]
    decision_vars.append(np.random.uniform(low=0., high=1.))

    # Evaluate objective function
    [total_deaths, max_hospital, relaxation] = opti_object.evaluate_objective(decision_vars)

    # Print decision vars and outputs
    print(get_decision_vars_names())
    print(f"Decision variables: {decision_vars}")
    print(f"N deaths: {total_deaths} / Max hospital: {max_hospital} / Relaxation: {relaxation}")


def dump_decision_vars_sample(n_samples):
    decision_vars_sample = []
    for i in range(n_samples):
        decision_vars = []
        for phase_number in range(2):
            sample = list(np.random.uniform(low=0., high=1., size=(8,)))
            _sum = sum(sample)
            decision_vars += [s/_sum for s in sample]
        decision_vars.append(float(np.random.uniform(low=0., high=1.)))

        decision_vars = [float(v) for v in decision_vars]

        decision_vars_sample.append(decision_vars)

    file_path = "comparison_test/vars_sample.yml"
    with open(file_path, "w") as f:
        yaml.dump(decision_vars_sample, f)


def evaluate_sample_decision_vars(user="Guillaume"):

    file_path = "comparison_test/vars_sample.yml"
    with open(file_path) as file:
        vars_samples = yaml.load(file)

    opti_object = initialise_opti_object(COUNTRY)
    dumped_dict = {
        'deaths': [],
        'hosp': []
    }
    for decision_vars in vars_samples:
        [total_deaths, max_hospital, _] = opti_object.evaluate_objective(decision_vars)
        dumped_dict['deaths'].append(float(total_deaths))
        dumped_dict['hosp'].append(float(max_hospital))

    file_path = f"comparison_test/obj_values_{user}.yml"
    with open(file_path, "w") as f:
        yaml.dump(dumped_dict, f)


def compare_outputs():
    outputs = {}

    for name in ["Romain", "Guillaume"]:
        file_path = f"comparison_test/obj_values_{name}.yml"
        with open(file_path) as file:
            outputs[name] = yaml.load(file)

    for obj in ["deaths", "hosp"]:
        perc_diff = [int(100 * (outputs["Guillaume"][obj][i] - outputs["Romain"][obj][i]) / outputs["Romain"][obj][i]) for i in range(len(outputs["Romain"][obj]))]
        average_perc_diff = sum(perc_diff) / len(perc_diff)

        print(f"Comparison for {obj}:")
        print("Percentage difference (ref. Romain):")
        print(perc_diff)
        print(f"Average perc diff: {average_perc_diff}%")

        for name in ["Romain", "Guillaume"]:
            x = outputs[name][obj]
            ordered_output = sorted(x)
            ranks = [ordered_output.index(v) for v in x]
            print(f"Ranks {name}:")
            print(ranks)

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        print()


# evaluate_sample_decision_vars("Guillaume")
# compare_outputs()


# This can be run using:
# python -m apps runsamplevaccopti



