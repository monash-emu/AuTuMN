
import pylab
import numpy
from matplotlib import patches
import matplotlib.pyplot as plt
import plotting
import math

"""

Module for estimating cost of a program

"""
params_default = {
    "saturation": 1,
    "coverage": 0.33,
    "funding": 2.5e6,
    "scale_up_factor": 0.75,
    "unitcost": 20,
    "popsize": 1e5,
    "method": 1,
    "outcome_zerocov": 20,
    "outcome_fullcov": 5}

def get_coverage_from_outcome_program_as_param(outcome):
    coverage = numpy.array(outcome)
    return coverage


start_coverage = 0.0001
end_coverage = params_default["saturation"]
delta_coverage = 0.001
method = 1

def make_coverage_steps(start_coverage, end_coverage, delta_coverage):
    steps = []
    step = start_coverage
    while step <= end_coverage:
        steps.append(step)
        step += delta_coverage
    return steps
coverage_values = make_coverage_steps(start_coverage, end_coverage, delta_coverage)


def get_cost_from_coverage(coverage_range, saturation, coverage, funding, scale_up_factor, unitcost, popsize):
    if method in (1, 2):
        if method == 1: # For new programs which requires significant start-up cost. Unit cost unknown
            cost = funding / (((saturation - coverage_range) / (coverage_range * (saturation / coverage)))**((1 - scale_up_factor) / 2))
            return cost
        elif method == 2: # For well-established programs of which start-up cost should be ignored. Unit cost known
            cost = - unitcost * popsize * math.log(((2 * saturation) / (coverage_range + saturation)) - 1)
            return cost

def cost_scaleup_fns(model,
                     functions,
                     start_time_str = 'start_time',
                     end_time_str = '',
                     parameter_type='',
                     country=u'',
                     figure_number = 1):

    if  start_time_str == 'recent_time':
        start_time = model.data['attributes'][start_time_str]
    else:
        start_time = model.data['country_constants'][start_time_str]

    end_time = model.data['attributes'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, end_time - start_time + 1)

    for figure_number, function in enumerate(functions):
        if function == str('program_prop_vaccination'):
            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)

            for coverage_range in coverage_values:
                cost = get_cost_from_coverage(coverage_range,
                                              params_default['saturation'],
                                              coverage,
                                              params_default['funding'],
                                              params_default['scale_up_factor'],
                                              params_default['unitcost'],
                                              params_default['popsize'])


            plt.figure(111)
            plt.plot(x_vals, coverage)
            title = str(country) + ' ' + \
                        plotting.replace_underscore_with_space(parameter_type) + \
                        ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title(title)
            plt.show()

            plt.figure(222)
            plt.plot(x_vals, cost)
            title = str(country) + ' ' + \
                        plotting.replace_underscore_with_space(parameter_type) + \
                        ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title(title)
            plt.show()