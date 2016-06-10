import matplotlib.pyplot as plt
from numpy import exp
from numpy import linspace
import numpy as np
import scipy.optimize
import autumn.logistic_fitting as logistic_fitting
from autumn.spreadsheet import read_and_process_data

saturation = 0.8
coverage = 0.33
funding = 2.5e6
scale_up_factor = 0.75
unitcost = 20
popsize = 1e5
method = 1
outcome_zerocov = 20
outcome_fullcov = 5


start_coverage = 0.001
end_coverage = 0.78
delta_coverage = 0.01

def make_coverage_steps(start_coverage, end_coverage, delta_coverage):
    steps = []
    step = start_coverage
    while step <= end_coverage:
        steps.append(step)
        step += delta_coverage
    return steps

coverage_values = make_coverage_steps(start_coverage, end_coverage, delta_coverage)
#print(coverage_values)




cost_values = []
outcome_values = []


def coverage_cost_fx (coverage_range, saturation, coverage, funding, scale_up_factor, unitcost, popsize):
        if method in (1, 2):
            if method == 1: # For new programs which requires significant start-up cost. Unit cost unknown
                x = funding / (((saturation - coverage_range) / (coverage_range * (saturation / coverage)))**((1 - scale_up_factor) / 2))
                return x

        elif method == 2: # For well-established programs of which start-up cost should be ignored. Unit cost known
                x = - unitcost * popsize * math.log(((2 * saturation) / (coverage_range + saturation)) - 1)
                return x

def new_coverage_cost_fx (coverage_range):
        return coverage_cost_fx (coverage_range, saturation, coverage, funding, scale_up_factor, unitcost, popsize)  # closure

for coverage_range in coverage_values:
        cost_values.append (new_coverage_cost_fx(coverage_range))
        print(cost_values)

plt.plot(coverage_values, cost_values, 'r')
plt.xlim([0, saturation])
plt.ylim([0, 5e6])
plt.show()