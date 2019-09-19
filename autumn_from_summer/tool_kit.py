import copy
import numpy


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

    if isinstance(baseline_model, dict):
        baseline_model_times = baseline_model['times']
        baseline_model_outputs = baseline_model['outputs']
    else:
        baseline_model_times = baseline_model.times
        baseline_model_outputs = baseline_model.outputs

    # find last integrated time and its index before start_time in baseline_model
    first_index_over = min([x[0] for x in enumerate(baseline_model_times) if x[1] > update_params['start_time']])
    index_of_interest = max([0, first_index_over - 1])
    integration_start_time = baseline_model_times[index_of_interest]
    init_compartments = baseline_model_outputs[index_of_interest, :]

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


def get_cost_from_coverage(coverage, saturation, unit_cost, popsize, inflection_cost=0., alpha=1.):
    """
    Estimate the raw cost associated with a given coverage for a specific intervention.

    Args:
        coverage: The intervention's coverage as a proportion
        inflection_cost: Cost at which inflection occurs on the curve (also the point of maximal efficiency)
        saturation: Maximal possible coverage, i.e. upper asymptote of the logistic curve
        unit_cost: Unit cost of the intervention
        popsize: Size of the population targeted by the intervention
        alpha: Steepness parameter determining the curve's shape
    Returns:
        The raw cost from the logistic function
    """

    # if unit cost or pop_size or coverage is null, return 0
    if popsize == 0. or unit_cost == 0. or coverage == 0.: return 0.
    #assert 0. <= coverage <= saturation, 'Coverage must satisfy 0 <= coverage <= saturation'
    if coverage == saturation: coverage = saturation - 1e-6

    # logistic curve function code
    a = saturation / (1. - 2. ** alpha)
    b = 2. ** (alpha + 1.) / (alpha * (saturation - a) * unit_cost * popsize)
    return inflection_cost - 1. / b * numpy.log(((saturation - a) / (coverage - a)) ** (1. / alpha) - 1.)


def get_coverage_from_cost(spending, saturation, unit_cost, popsize, inflection_cost=0., alpha=1.):
    """
    Estimate the coverage associated with a spending in a program.

    Args:
        spending: The amount of money allocated to a program (absolute value, not a proportion of all funding)
        inflection_cost: Cost at which inflection occurs on the curve (also the point of maximal efficiency)
        saturation: Maximal possible coverage, i.e. upper asymptote of the logistic curve
        unit_cost: Unit cost of the intervention
        popsize: Size of the population targeted by the intervention
        alpha: Steepness parameter determining the curve's shape
    Returns:
        The proportional coverage of the intervention given the spending
   """

    # if cost is smaller thar c_inflection_cost, then the starting cost necessary to get coverage has not been reached
    if popsize == 0. or unit_cost == 0. or spending == 0. or spending <= inflection_cost: return 0.

    a = saturation / (1. - 2. ** alpha)
    b = 2. ** (alpha + 1.) / (alpha * (saturation - a) * unit_cost * popsize)
    return a + (saturation - a) / ((1. + numpy.exp((-b) * (spending - inflection_cost))) ** alpha)

