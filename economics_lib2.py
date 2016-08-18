import numpy
import matplotlib.pyplot as plt
import math
from autumn.spreadsheet import read_input_data_xls
from autumn.tool_kit import indices
import autumn.model
import autumn.data_processing


# Functions related to the estimation of the costs associated to a programmatic configuration. Is run after full integration
def coverage_over_time(model, param_key):
    """
    Define a function which returns the coverage over time associated with an intervention
    Args:
        model: model object, after integration
        param_key: the key of the parameter associated with the intervention

    Returns:
        a function which takes a time for argument an will return a coverage
    """
    coverage_function = model.scaleup_fns[param_key]
    return coverage_function


def get_cost_from_coverage(coverage, c_inflection_cost, saturation, unit_cost, pop_size, alpha=1.0):
    """
    Estimate the global uninflated cost associated with a given coverage
    Args:
        coverage: the coverage (as a proportion, then lives in 0-1)
        c_inflection_cost: cost at which inflection occurs on the curve. It's also the configuration leading to the
                            best efficiency.
        saturation: maximal acceptable coverage, ie upper asymptote
        unit_cost: unit cost of the intervention
        pop_size: size of the population targeted by the intervention
        alpha: steepness parameter

    Returns:
        uninflated cost

    """
    if pop_size * unit_cost == 0:  # if unit cost or pop_size is null, return 0
        return 0
    assert 0 <= coverage < saturation, 'coverage must verify 0 <= coverage < saturation'

    a = saturation / (1.0 - 2 ** alpha)
    b = ((2.0 ** (alpha + 1.0)) / (alpha * (saturation - a) * unit_cost * pop_size))
    cost_uninflated = c_inflection_cost - 1.0 / b * math.log(
        (((saturation - a) / (coverage - a)) ** (1.0 / alpha)) - 1.0)
    return cost_uninflated


def inflate_cost(cost_uninflated, current_cpi, cpi_time_variant):
    """
    Calculate the inflated cost associated with cost_uninflated and considering the current cpi and the cpi correponding
    to the date considered (cpi_time_variant)

    Returns:
        the inflated cost
    """
    return cost_uninflated * current_cpi / cpi_time_variant


def discount_cost(cost_uninflated, discount_rate, t_into_future):
    """
    Calculate the discounted cost associated with cost_uninflated at time (t + t_into_future)
    Args:
        cost_uninflated: cost without accounting for discounting
        discount_rate: discount rate (/year)
        t_into_future: number of years into future at which we want to calculate the discounted cost

    Returns:
        the discounted cost
    """
    assert t_into_future >= 0, 't_into_future must be >= 0'
    return (cost_uninflated / ((1 + discount_rate) ** t_into_future))


def economics_diagnostic(model, period_end,
                         interventions=['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf']):
    """
    Run the economics diagnostics associated with a model run. Integration is supposed to have been run at this point
    Args:
        model: the model object
        period_end: date of the end of the period considered for total cost calculations
        interventions: list of interventions considered for costing

    Returns:
        the costs associated to each intervention and at each time step. These are yearly costs.
    """
    start_time = model.inputs.model_constants['recent_time']  # start time for cost calculations
    start_index = indices(model.times, lambda x: x >= start_time)[0]

    end_time_integration = model.inputs.model_constants['scenario_end_time']
    assert period_end <= end_time_integration, 'period_end must be <= end_time_integration'
    end_index = indices(model.times, lambda x: x >= period_end)[0]

    # prepare the references to fetch the data and model outputs
    param_key_base = 'econ_program_prop_'
    c_inflection_cost_base = 'econ_program_inflectioncost_'
    unitcost_base = 'econ_program_unitcost_'
    popsize_label_base = 'popsize_'

    discount_rate = 0.03  # not ideal... perhaps best to get it from a spreadsheet
    cpi_function = model.scaleup_fns['econ_cpi']
    year_current = model.inputs.model_constants['current_time']
    current_cpi = cpi_function(year_current)

    # prepare the storage. 'costs' will store all the costs and will be returned
    costs = {'cost_times': []}

    count_intervention = 0  # to count the interventions
    for intervention in interventions:  # for each intervention
        count_intervention += 1
        costs[intervention] = {'uninflated_cost': [], 'inflated_cost': [], 'discounted_cost': []}

        param_key = param_key_base + intervention  # name of the corresponding parameter
        coverage_function = coverage_over_time(model, param_key)  # create a function to calculate coverage over time

        c_inflection_cost_label = c_inflection_cost_base + intervention  # key of the scale up function for inflection cost
        c_inflection_cost_function = model.scaleup_fns[c_inflection_cost_label]

        unitcost_label = unitcost_base + intervention  # key of the scale up function for unit cost
        unit_cost_function = model.scaleup_fns[unitcost_label]

        popsize_label = popsize_label_base + intervention
        pop_size_index = model.var_labels.index(
            popsize_label)  # column index in model.var_array that corresponds to the intervention

        saturation = 1.001  # provisional

        for i in range(start_index,
                       end_index + 1):  # for each step time. We may want to change this bit. No need for all time steps
            t = model.times[i]
            if count_intervention == 1:
                costs['cost_times'].append(t)  # storage of the time
            # calculate the time variants that feed into the logistic function
            coverage = coverage_function(t)
            c_inflection_cost = c_inflection_cost_function(t)
            unit_cost = unit_cost_function(t)
            pop_size = model.var_array[i, pop_size_index]

            # calculate uninflated cost
            cost = get_cost_from_coverage(coverage, c_inflection_cost, saturation, unit_cost, pop_size)
            costs[intervention]['uninflated_cost'].append(cost)  # storage

            # calculate inflated cost
            cpi_time_variant = cpi_function(t)
            inflated_cost = inflate_cost(cost, current_cpi, cpi_time_variant)
            costs[intervention]['inflated_cost'].append(inflated_cost)  # storage

            # calculate discounted cost
            t_into_future = max(0, (t - year_current))
            discounted_cost = discount_cost(cost, discount_rate, t_into_future)
            costs[intervention]['discounted_cost'].append(discounted_cost)  # storage

    return costs

def total_cost_intervention(costs, intervention, time_from=None, time_to=None):

    if time_from is None:
        indice_time_from = 0
    else:
        assert time_from >= costs['cost_times'][0], 'time_from is too early'
        indice_time_from = indices(costs['cost_times'], lambda x: x >= time_from)[0]

    if time_to is None:
        indice_time_to = len(costs['cost_times']) - 1
    else:
        assert time_to <= costs['cost_times'][-1], 'time_to is too late'
        indice_time_to = indices(costs['cost_times'], lambda x: x <= time_to)[-1]

    time_step = costs['cost_times'][1] - costs['cost_times'][0]

    total_intervention_uninflated = 0
    total_intervention_inflated = 0
    total_intervention_discounted = 0
    for i in range(indice_time_from, indice_time_to):
        total_intervention_uninflated += time_step * costs[intervention]['uninflated_cost'][i]
        total_intervention_inflated += time_step * costs[intervention]['inflated_cost'][i]
        total_intervention_discounted += time_step * costs[intervention]['discounted_cost'][i]
    total_cost_int = {'total_intervention_uninflated': total_intervention_uninflated,
                      'total_intervention_inflated': total_intervention_inflated,
                      'total_intervention_discounted': total_intervention_discounted}
    return total_cost_int

# Functions related to the more complicated direction, where there is a feed back loop
def get_coverage_from_cost(cost, c_inflection_cost, saturation, unit_cost, pop_size, alpha=1.0):
    """
    Estimate the coverage associated with a spending in a programme
    Args:
       cost: the amount of money allocated to a programme (absolute number, not a proportion of global funding)
       c_inflection_cost: cost at which inflection occurs on the curve. It's also the configuration leading to the
                           best efficiency.
       saturation: maximal acceptable coverage, ie upper asymptote
       unit_cost: unit cost of the intervention
       pop_size: size of the population targeted by the intervention
       alpha: steepness parameter

    Returns:
       coverage (as a proportion, then lives in 0-1)
   """
    assert cost >= 0, 'cost must be positive or null'
    if cost <= c_inflection_cost:  # if cost is smaller thar c_inflection_cost, then the starting cost necessary to get coverage has not been reached
        return 0

    a = saturation / (1.0 - 2 ** alpha)
    b = ((2.0 ** (alpha + 1.0)) / (alpha * (saturation - a) * unit_cost * pop_size))
    coverage_estimated = a + (saturation - a) / ((1 + math.exp((-b) * (cost - c_inflection_cost))) ** alpha)
    return coverage_estimated

if __name__ == "__main__":
    country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
    print(country)

    inputs = autumn.data_processing.Inputs(True)
    inputs.read_and_load_data()

    scenario = None

    model = autumn.model.ConsolidatedModel(scenario, inputs)
    model.integrate()
    costs = economics_diagnostic(model, period_end=2020.)

    costs_bcg = total_cost_intervention(costs, 'vaccination', time_from=2012, time_to=2014)
    print(costs_bcg)