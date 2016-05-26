"""
Cost-coverage and coverage-outcome curves
"""

import matplotlib.pyplot as plt
from numpy import exp
import numpy as np


#Logistic function for cost-coverage curve
def cost_coverage_fx(cost, saturation, coverage, funding, scale_up_factor, unitcost, popsize):

    if method in (1, 2):
        if method == 1: # For new programs which requires significant start-up cost. Unit cost unknown
            y =  saturation / (1 + (saturation / coverage) * (funding / cost)**(2 / (1 - scale_up_factor)))
            return y
        elif method == 2: # For well-established programs of which start-up cost should be ignored. Unit cost known
            y =  (2 * saturation / (1 + exp((-2 * cost) / (unitcost * popsize)))) - saturation
            return y

# Logistic function for coverage-outcome curve
def coverage_outcome_fx(cov, outcome_zerocov, outcome_fullcov):
    y = (outcome_fullcov - outcome_zerocov) * np.array (cov) + outcome_zerocov
    return y

def make_cost_steps(start_cost, end_cost, delta_cost):
    steps = []
    step = start_cost
    while step <= end_cost:
        steps.append(step)
        step += delta_cost
    return steps

start_cost = 0.01
end_cost = 1e7
delta_cost = 100

cost_values = make_cost_steps (start_cost, end_cost, delta_cost)

# Parameters for cost-coverage function
saturation = 0.8
coverage = 0.33
funding = 2.5e6
scale_up_factor = 0.5
unitcost = 20
popsize = 1e5

method = 1

if method == 1:
    print ('Method 1, new prpgrams, unit cost not available')
elif method ==2:
    print ('Method 2, established-programs, unit cost available')

# Parameters for coverage-outcome function
outcome_zerocov = 20
outcome_fullcov = 5

def new_cost_coverage_fx (cost):
    return cost_coverage_fx (cost, saturation, coverage, funding, scale_up_factor, unitcost, popsize)  # closure
    
coverage_values = []
outcome_values = []

for cost in cost_values:
    coverage_values.append (new_cost_coverage_fx(cost))

for cov in coverage_values:
    outcome_values.append (coverage_outcome_fx(cov, outcome_zerocov, outcome_fullcov))


fig = plt.figure(1)
fig.suptitle('Cost-coverage-outcome curve')
plt.subplot (121)
#fig.suptitle('Cost-coverage curve')
plt.plot(cost_values, coverage_values, 'r', linewidth = 3)
plt.xlim([start_cost, end_cost])
plt.ylim ([0, 1])
plt.xlabel('$ Cost')
plt.ylabel('% Coverage')
#plt.show()
#fig.savefig('cost_coverage.jpg')
#fig = plt.figure(2)
plt.subplot (122)
#fig.suptitle('Coverage-outcome curve')
plt.plot(coverage_values, outcome_values, 'b', linewidth = 3)
plt.xlim([0, 1])
plt.ylim ([outcome_zerocov, outcome_fullcov])
plt.xlabel('% Coverage')
plt.ylabel('Outcome')
plt.show()
#fig.savefig('coverage_outcome.jpg')
fig.savefig('Cost_coverage_outcome.jpg')