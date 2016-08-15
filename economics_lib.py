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

def get_cost_from_coverage(coverage, c_reflection_cost, saturation, unit_cost, pop_size, alpha = 1.0):
    """
    Estimate the global uninflated cost associated with a given coverage
    Args:
        coverage: the coverage (as a proportion, then lives in 0-1)
        c_reflection_cost: cost at which inflection occurs on the curve. It's also the configuration leading to the
                            best efficiency.
        saturation: maximal acceptable coverage, ie upper asymptote
        unit_cost: unit cost of the intervention
        pop_size: size of the population targeted by the intervention
        alpha: steepness parameter

    Returns:
        uninflated cost

    """
    a = saturation / (1.0 - 2**alpha)
    b = ((2.0**(alpha + 1.0)) / (alpha * (saturation - a) * unit_cost * pop_size))
    cost_uninflated = c_reflection_cost - 1.0/b * math.log((((saturation - a) / (coverage - a))**(1.0 / alpha)) - 1.0)
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
    return cost_uninflated / ((1 + discount_rate)**t_into_future)

def economics_diagnostic(model, period_end, interventions = ['bcg'] ):
    """
    Run the economics diagnostics associated with a model run. Integration is supposed to have been run at this point
    Args:
        model: the model object
        period_end: date of the end of the period considered for total cost calculations
        interventions: list of interventions considered for costing

    Returns:
        the spending
    """
    start_time = model.inputs.model_constants['recent_time'] # start time for cost calculations
    start_index = indices(model.times, lambda x: x >= start_time)[0]

    end_time_integration = model.inputs.model_constants['scenario_end_time']
    assert period_end <= end_time_integration, 'period_end must be <= end_time_integration'
    end_index = indices(model.times, lambda x: x >= period_end)[0]

    param_keys = {'bcg': 'program_prop_vaccination', 'ipt': 'econ_program_prop_ipt'}  # to be completed
    c_reflection_costs = {'bcg': 'econ_program_reflectioncost_vaccination'}  # to be completed
    unitcosts = {'bcg': 'econ_program_unitcost_vaccination'}  # to be completed

    discount_rate = 0.03
    cpi_function = model.scaleup_fns['econ_cpi']
    year_current = model.inputs.model_constants['current_time']
    current_cpi = cpi_function(year_current)

    for intervention in interventions: # for each intervention
        print '#######################################'
        print intervention
        param_key = param_keys[intervention]
        coverage_function = coverage_over_time(model, param_key)
        c_reflection_cost_function = model.scaleup_fns[c_reflection_costs[intervention]]
        unit_cost_function = model.scaleup_fns[unitcosts[intervention]]

        for i in range(start_index, end_index + 1): # for each step time
            t = model.times[i]
            print t
            coverage = coverage_function(t)
            c_reflection_cost = c_reflection_cost_function(t)
            saturation = 1.001  # provisional
            unit_cost = unit_cost_function(t)
            pop_size = 15000. # To be linked with James' work
            cost = get_cost_from_coverage(coverage, c_reflection_cost, saturation, unit_cost, pop_size)
            print cost

            cpi_time_variant = cpi_function(t)
            inflated_cost = inflate_cost(cost, current_cpi, cpi_time_variant)
            print inflated_cost

            t_into_future = max(0, (t - year_current))
            discounted_cost = discount_cost(cost, discount_rate, t_into_future)
            print discounted_cost


# Functions related to the more complicated direction, where there is a feed back loop
def get_coverage_from_cost(cost, c_reflection_cost, saturation, unit_cost, pop_size, alpha = 1.0):
    """
    Estimate the coverage associated with a spending in a programme
    Args:
       cost: the amount of money allocated to a programme (absolute number, not a proportion of global funding)
       c_reflection_cost: cost at which inflection occurs on the curve. It's also the configuration leading to the
                           best efficiency.
       saturation: maximal acceptable coverage, ie upper asymptote
       unit_cost: unit cost of the intervention
       pop_size: size of the population targeted by the intervention
       alpha: steepness parameter

    Returns:
       coverage (as a proportion, then lives in 0-1)
   """
    a = saturation / (1.0 - 2 ** alpha)
    b = ((2.0 ** (alpha + 1.0)) / (alpha * (saturation - a) * unit_cost * pop_size))
    coverage_estimated = a + (saturation - a) /((1 + math.exp((-b) * (cost - c_reflection_cost)))**alpha)
    return coverage_estimated

"""
Needed for integration in the epi model:

    For both directions:
        Read economics data (ok)

    For the easy direction: Get the cost associated with a programmatic configuration
        - Run the model  (ok)

        - Run economic diagnostic:
            - For each intervention
                - get coverage (ok)

                - for each time_step
                    - get pop_size (James)
                    - calculate and store the associated cost (calculate ok)

                - Calculate and store total cost of the intervention over the period of time

            -  Calculate and store the global total cost over the period of time


"""

country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
print(country)

inputs = autumn.data_processing.Inputs(True)
inputs.read_and_load_data()

scenario = None

model = autumn.model.ConsolidatedModel(scenario, inputs)
model.integrate()

economics_diagnostic(model, period_end = 2020., interventions=['bcg'])