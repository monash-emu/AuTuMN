import copy

def find_first_list_element_above(list, value):
    """
    Simple method to return the index of the first element of a list that is greater than a specified value.

    Args:
        list: List of floats
        value: The value that the element must be greater than
    """
    return next(x[0] for x in enumerate(list) if x[1] > value)


def initialise_scenario_run(baseline_model, update_params, model_builder):
    """
    function to run a scenario. Running time starts at start_time.the initial conditions will be loaded form the
    run baseline_model
    :return: the run scenario model
    """

    # find last integrated time and its index before start_time in baseline_model
    first_index_over = min([x[0] for x in enumerate(baseline_model.times) if x[1] > update_params['start_time']])
    index_of_interest = max([0, first_index_over - 1])
    integration_start_time = baseline_model.times[index_of_interest]
    init_compartments = baseline_model.outputs[index_of_interest, :]

    update_params['start_time'] = integration_start_time

    sc_model = model_builder(update_params)
    sc_model.compartment_values = init_compartments

    return sc_model


def run_multi_scenario(scenario_params, scenario_start_time, model_builder):
    """
    Run a baseline model and scenarios
    :param scenario_params: a dictionary keyed with scenario numbers (0 for baseline). values are dictionaries
    containing parameter updates
    :return: a list of model objects
    """
    param_updates_for_baseline = scenario_params[0] if 0 in scenario_params.keys() else {}
    baseline_model = model_builder(param_updates_for_baseline)
    print("____________________  Now running Baseline Scenario ")
    baseline_model.run_model()

    models = [baseline_model]

    for scenario_index in scenario_params.keys():
        print("____________________  Now running Scenario " + str(scenario_index))
        if scenario_index == 0:
            continue
        scenario_params[scenario_index]['start_time'] = scenario_start_time
        scenario_model = initialise_scenario_run(baseline_model, scenario_params[scenario_index], model_builder)
        scenario_model.run_model()
        models.append(copy.deepcopy(scenario_model))

    return models

