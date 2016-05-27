"""
Cost-coverage and coverage-outcome curves
"""

"""
TO DO LIST
1. By year
2. Like to transmission parameters
3. Read data from spreadsheet
"""

import matplotlib.pyplot as plt
from numpy import exp
import numpy as np
import os
from autumn.spreadsheet import read_and_process_data

"""
# Decide on country
country = u'Fiji'
keys_of_sheets_to_read = [
    'bcg', 'birth_rate', 'life_expectancy', 'attributes', 'parameters', 'miscellaneous', 'programs', 'tb',
    'notifications', 'outcomes']
data = read_and_process_data(True, keys_of_sheets_to_read, country)

programs = []

for key, value in data['programs'].items():
    print(key, value)
    #print(u'program_prop_treatment_success_mdr', value)
    #programs = key[1]
    #print(programs)
"""

#Define the programs using a list of strings
program_name = [
    "vaccination",
    "detect",
    "algorithm_sensitivity",
    "lowquality",
    "firstline_dst",
    "secondline_dst",
    "treatment_success",
    "treatment_death",
    "treatment_success_mdr",
    "treatment_death_mdr",
    "treatment_success_xdr",
    "treatment_death_xdr"
]

#Year = 1986
prog_cov = {
    "program_prop_vaccination":
        0.,
    "program_prop_detect":
        0.,
    "program_prop_algorithm_sensitivity":
        0.,
    "program_prop_lowquality":
        0.,
    "program_prop_firstline_dst":
        0.,
    "program_prop_secondline_dst":
        0.,
    "program_prop_treatment_success":
        50.,
    "program_prop_treatment_death":
        12.,
    "program_prop_treatment_success_mdr":
        30.,
    "program_prop_treatment_death_mdr":
        50.,
    "program_prop_treatment_success_xdr":
        15.,
    "program_prop_treatment_death_xdr":
        62.,
}

prog_cost = {
    "program_cost_vaccination":
        0.,
    "program_cost_detect":
        0.,
    "program_cost_algorithm_sensitivity":
        0.,
    "program_cost_lowquality":
        0.,
    "program_cost_firstline_dst":
        0.,
    "program_cost_secondline_dst":
        0.,
    "program_cost_treatment_success":
        5e6,
    "program_cost_treatment_death":
        0.,
    "program_cost_treatment_success_mdr":
        256491.,
    "program_cost_treatment_death_mdr":
        0.,
    "program_cost_treatment_success_xdr":
        0.,
    "program_cost_treatment_death_xdr":
        0.,
}

print(len(prog_cost))


#Logistic function for cost-coverage curve
def cost_coverage_fx (cost, saturation, coverage, funding, scale_up_factor, unitcost, popsize):

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


#### DEFAULT Parameters for cost-coverage function
params_default = {
    "saturation": 0.8,
    "coverage": 0.33,
    "funding": 2.5e6,
    "scale_up_factor": 0.5,
    "unitcost": 20,
    "popsize": 1e5,
    "method": 1,
    "outcome_zerocov": 20,
    "outcome_fullcov": 5
}

#### Define which program
prog = 9

if prog == 7:
    print("Program: treatment success, Year 1986")
    method = 2
    saturation = params_default["saturation"]
    coverage = prog_cov["program_prop_treatment_success"]
    funding = prog_cost["program_cost_treatment_success"]
    scale_up_factor = params_default["scale_up_factor"]
    unitcost = params_default["unitcost"]
    popsize = params_default["popsize"]
    outcome_zerocov = params_default["outcome_zerocov"]
    outcome_fullcov = params_default["outcome_fullcov"]
elif prog == 9:
    print("Program: treatment success MDR, Year 1986")
    method = 2
    saturation = params_default["saturation"] - 0.2
    coverage = prog_cov["program_prop_treatment_success_mdr"] / 100.
    funding = prog_cost["program_cost_treatment_success_mdr"]
    scale_up_factor = params_default["scale_up_factor"]
    unitcost = params_default["unitcost"]
    popsize = params_default["popsize"]
    outcome_zerocov = params_default["outcome_zerocov"]
    outcome_fullcov = params_default["outcome_fullcov"]
else:
    method = params_default["method"]
    saturation = params_default["saturation"]
    coverage = params_default["coverage"]
    funding = params_default["funding"]
    scale_up_factor = params_default["scale_up_factor"]
    unitcost = params_default["unitcost"]
    popsize = params_default["popsize"]
    outcome_zerocov = params_default["outcome_zerocov"]
    outcome_fullcov = params_default["outcome_fullcov"]

if method == 1:
    print ('Method 1, new prpgrams, unit cost not available')
elif method ==2:
    print ('Method 2, established-programs, unit cost available')

# Parameters for coverage-outcome function

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