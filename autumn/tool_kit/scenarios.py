"""
Utilities for running multiple model scenarios
"""

def run_multi_scenario(param_lookup, scenario_start_time, model_builder):
    """
    Run a baseline model and scenarios
    FIXME: [Matt] Wouldn't it make more sense if param_lookup was just a list?

    :param param_lookup: 
        A dictionary keyed with scenario numbers (0 for baseline). 
        Values are dictionaries containing parameter updates.
    :return: a list of model objects
    """
    # Run baseline model as scenario '0'
    baseline_params =  param_lookup[0] if 0 in param_lookup else {}
    baseline_model = model_builder(baseline_params)
    print("____________________  Running baseline scenario ")
    baseline_model.run_model()
    models = [baseline_model]

    for scenario_idx, scenario_params in param_lookup.items():
        # Ignore scenario '0' because we've already run it.
        if scenario_idx == 0:
            continue

        print(f"____________________  Running scenario #{scenario_idx}")
        scenario_params["start_time"] = scenario_start_time
        scenario_model = initialise_scenario_run(
            baseline_model, scenario_params, model_builder
        )
        scenario_model.run_model()
        models.append(scenario_model)

    return models


def initialise_scenario_run(baseline_model, scenario_params, model_builder):
    """
    Build a model to run a scenario. Running time starts at start_time, which must
    be specified in `scenario_params`. The initial conditions will be loaded from the
    run baseline_model.

    :return: the run scenario model
    """
    baseline_times = baseline_model.times
    baseline_outputs = baseline_model.outputs

    # Find the timestep from which we will start the scenario.
    scenario_start_index = get_scenario_start_index(baseline_times, scenario_params["start_time"])
    scenario_start_time = baseline_times[scenario_start_index]
    scenario_init_compartments = baseline_outputs[scenario_start_index, :]

    # Create the new scenario model using the scenario-specific params,
    # ensuring the initial conditions are the same for the given start time.
    scenario_params["start_time"] = scenario_start_time
    model = model_builder(scenario_params)
    model.compartment_values = scenario_init_compartments
    return model


def get_scenario_start_index(baseline_times, scenario_start_time):
    """
    Returns the index of the closest timestep that is at, or before the scenario start time.
    """
    indexs_after_start_index = [
        idx for idx, time in enumerate(baseline_times) if time > scenario_start_time
    ]
    if not indexs_after_start_index:
        msg = f"Scenario start time {scenario_start_time} is set after the baseline time range."
        raise ValueError(msg)

    index_after_start_index = min(indexs_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index
