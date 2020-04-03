"""
Utilities for running multiple model scenarios
"""
from autumn.tool_kit.timer import Timer
import numpy
from autumn.demography.social_mixing import change_mixing_matrix_for_scenario


def run_multi_scenario(param_lookup, scenario_start_time, model_builder, default_params, run_kwargs={}):
    """
    Run a baseline model and scenarios

    :param param_lookup: 
        A dictionary keyed with scenario numbers (0 for baseline)
        Values are dictionaries containing parameter updates
    :return: a list of model objects
    """
    # Run baseline model as scenario '0'
    baseline_params = param_lookup[0] if 0 in param_lookup else {}
    baseline_model = model_builder(baseline_params)
    with Timer("Running baseline scenario"):
        baseline_model.run_model(**run_kwargs)
    models = [baseline_model]

    # Run scenario models
    for scenario_idx, scenario_params in param_lookup.items():

        # Ignore the baseline because it has already been run
        if scenario_idx == 0:
            continue

        with Timer(f'Running scenario #{scenario_idx}'):
            scenario_params['start_time'] = scenario_start_time
            scenario_model = initialise_scenario_run(baseline_model, scenario_params, model_builder)
            scenario_model = change_mixing_matrix_for_scenario(scenario_model, scenario_params, default_params)
            scenario_model.run_model(**run_kwargs)
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

    # Find the time step from which we will start the scenario
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
    Returns the index of the closest time step that is at, or before the scenario start time.
    """
    indices_after_start_index = [
        idx for idx, time in enumerate(baseline_times) if time > scenario_start_time
    ]
    if not indices_after_start_index:
        raise ValueError(
            f"Scenario start time {scenario_start_time} is set after the baseline time range."
        )

    index_after_start_index = min(indices_after_start_index)
    start_index = max([0, index_after_start_index - 1])
    return start_index


def get_model_times_from_inputs(start_time, end_time, time_step):
    """
    Find the time steps for model integration from the submitted requests, ensuring the time points are evenly spaced.
    """
    n_times = int(round(
        (end_time - start_time) / time_step
    )) + 1
    return numpy.linspace(
        start_time, end_time, n_times
    ).tolist()

