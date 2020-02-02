"""
Utilities for running multiple model scenarios
"""
import copy


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
        if scenario_index == 0:
            continue
        print("____________________  Now running Scenario " + str(scenario_index))
        scenario_params[scenario_index]["start_time"] = scenario_start_time
        scenario_model = initialise_scenario_run(
            baseline_model, scenario_params[scenario_index], model_builder
        )
        scenario_model.run_model()
        models.append(copy.deepcopy(scenario_model))

    return models


def initialise_scenario_run(baseline_model, update_params, model_builder):
    """
    function to run a scenario. Running time starts at start_time.the initial conditions will be loaded form the
    run baseline_model
    :return: the run scenario model
    """

    if isinstance(baseline_model, dict):
        baseline_model_times = baseline_model["times"]
        baseline_model_outputs = baseline_model["outputs"]
    else:
        baseline_model_times = baseline_model.times
        baseline_model_outputs = baseline_model.outputs

    # find last integrated time and its index before start_time in baseline_model
    first_index_over = min(
        [x[0] for x in enumerate(baseline_model_times) if x[1] > update_params["start_time"]]
    )
    index_of_interest = max([0, first_index_over - 1])
    integration_start_time = baseline_model_times[index_of_interest]
    init_compartments = baseline_model_outputs[index_of_interest, :]

    update_params["start_time"] = integration_start_time

    sc_model = model_builder(update_params)
    sc_model.compartment_values = init_compartments

    return sc_model
