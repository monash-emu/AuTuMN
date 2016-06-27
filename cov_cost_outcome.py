"""
COVERAGE-cost and coverage-outcome curves
"""

"""
TO DO LIST
1. Like to transmission parameters
"""

import matplotlib.pyplot as plt
from numpy import exp
from numpy import linspace
import numpy as np
import math
import scipy.optimize
import autumn.logistic_fitting2 as logistic_fitting2
from autumn.spreadsheet import read_and_process_data, read_input_data_xls


####### RUNNING CONDITIONS ############
#country = u'Fiji'
country = read_input_data_xls(True, ['attributes'])['attributes'][u'country']
print(country)

prog = 9   #refer to program_name below
multi_data = False

divider = 100.

year = 1986

start_year = 1920
end_year = 2015
delta_year = 1

########################################

def make_year_steps(start_year, end_year, delta_year):
    steps = []
    step = start_year
    while step <= end_year:
        steps.append(step)
        step += delta_year
    return steps
year_values = make_year_steps (start_year, end_year, delta_year)

#########################################

if multi_data == True:
    print("Multiple cost-coverage data points available")
else:
    print("Only one cost-coverage data point available")

#########################################

# Read data

#keys_of_sheets_to_read = [
#    'bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters', 'miscellaneous', 'time_variants', 'tb',
#    'notifications', 'outcomes']
#data = read_and_process_data(True, keys_of_sheets_to_read, country)

data = read_and_process_data(True,
                             ['bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters',
                              'country_constants', 'time_variants', 'tb', 'notifications', 'outcomes',
                              'country_economics', 'default_economics'],
                             country)

initials = data['time_variants'][u'program_prop_treatment_success_mdr'][year]


########################################################

#Define the programs using a list of strings
program_name = [
    "vaccination",                   #prog 1
    "detect",                        #prog 2
    "algorithm_sensitivity",         #prog 3
    "lowquality",                    #prog 4
    "firstline_dst",                 #prog 5
    "secondline_dst",                #prog 6
    "treatment_success",             #prog 7
    "treatment_death",               #prog 8
    "treatment_success_mdr",         #prog 9
    "treatment_death_mdr",           #prog 10
    "treatment_success_xdr",         #prog 11
    "treatment_death_xdr"            #prog 12
]

#print("PROGRAM" "" + program_name[8])
#print("YEAR" "" + str(year))
#####################################################
#Single data point

prog_cov = {
    "program_prop_vaccination":
        data['time_variants'][u'program_prop_vaccination'][year] / divider,

    "program_prop_detect":
        data['time_variants'][u'program_prop_detect'][year] / divider,

    "program_prop_algorithm_sensitivity":
        data['time_variants'][u'program_prop_algorithm_sensitivity'][year] / divider,

    "program_prop_lowquality":
        data['time_variants'][u'program_prop_lowquality'][year] / divider,

    "program_prop_firstline_dst":
        data['time_variants'][u'program_prop_firstline_dst'][year] / divider,

    "program_prop_secondline_dst":
        data['time_variants'][u'program_prop_secondline_dst'][year] / divider,

    "program_prop_treatment_success":
        data['time_variants'][u'program_prop_treatment_success'][year] / divider,

    "program_prop_treatment_death":
        data['time_variants'][u'program_prop_treatment_death'][year] / divider,

    "program_prop_treatment_success_mdr":
        data['time_variants'][u'program_prop_treatment_success_mdr'][year] / divider,

    "program_prop_treatment_death_mdr":
        data['time_variants'][u'program_prop_treatment_death_mdr'][year] / divider,

    "program_prop_treatment_success_xdr":
        data['time_variants'][u'program_prop_treatment_success_xdr'][year] / divider,

    "program_prop_treatment_death_xdr":
        data['time_variants'][u'program_prop_treatment_death_xdr'][year] / divider,
}

#print(prog_cov["program_prop_treatment_success_mdr"])

prog_cost = {
    "program_cost_vaccination":
        data['time_variants'][u'program_cost_vaccination'][year],

    "program_cost_detect":
        data['time_variants'][u'program_cost_detect'][year],

    "program_cost_algorithm_sensitivity":
        data['time_variants'][u'program_cost_algorithm_sensitivity'][year],

    "program_cost_lowquality":
        data['time_variants'][u'program_cost_lowquality'][year],

    "program_cost_firstline_dst":
        data['time_variants'][u'program_cost_firstline_dst'][year],

    "program_cost_secondline_dst":
        data['time_variants'][u'program_cost_secondline_dst'][year],

    "program_cost_treatment_success":
        data['time_variants'][u'program_cost_treatment_success'][year],

    "program_cost_treatment_death":
        data['time_variants'][u'program_cost_treatment_death'][year],

    "program_cost_treatment_success_mdr":
        data['time_variants'][u'program_cost_treatment_success_mdr'][year],

    "program_cost_treatment_death_mdr":
        data['time_variants'][u'program_cost_treatment_death_mdr'][year],

    "program_cost_treatment_success_xdr":
        data['time_variants'][u'program_cost_treatment_success_xdr'][year],

    "program_cost_treatment_death_xdr":
        data['time_variants'][u'program_cost_treatment_death_xdr'][year],
}

#Multiple data points

costcost = np.array([6e5, 4.2e5, 3.5e5, 3e5, 2e5], dtype='float')
covcov = np.array  ([0.8, 0.7,   0.6,   0.3, 0.08], dtype='float')

###########################################################
#Default parameters for cost-coverage function
params_default = {
    "saturation": 0.8,
    "coverage": 0.33,
    "funding": 2.5e6,
    "scale_up_factor": 0.75,
    "unitcost": 20,
    "popsize": 1e5,
    "method": 1,
    "outcome_zerocov": 20,
    "outcome_fullcov": 5
}

if prog == 7:
    print("PROGRAM" "" + program_name[prog - 1], "YEAR" "" + str(year))
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
    print("PROGRAM" " " + program_name[prog - 1],
          "YEAR" " " + str(year))
    method = 2
    saturation = params_default["saturation"]
    coverage = prog_cov["program_prop_treatment_success_mdr"]
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
    print ("METHOD" " " + str(method), "new programs", "unit cost not available")
elif method ==2:
    print ("METHOD" " " + str(method), "established-programs", "unit cost available")


# Values for cost axis
start_coverage = 0.0001
end_coverage = saturation
delta_coverage = 0.001

def make_coverage_steps(start_coverage, end_coverage, delta_coverage):
    steps = []
    step = start_coverage
    while step <= end_coverage:
        steps.append(step)
        step += delta_coverage
    return steps

coverage_values = make_coverage_steps(start_coverage, end_coverage, delta_coverage)


cost_values = []
outcome_values = []


###################################################
#Logistic functions

if multi_data == True:
    print(costcost)
    print(covcov)
    p_guess = (np.median(costcost), np.min(covcov), 0.8, 20000)
    p, cov, infodict, mesg, ier = scipy.optimize.leastsq(logistic_fitting2.residuals, p_guess, args=(covcov, costcost), full_output=1)
    cost_values = logistic_fitting2.logistic(p, coverage_values)
    #plt.plot(costcost, covcov, 'bo', cost_values, coverage_values, 'r-', linewidth = 3)
    #plt.xlim([start_cost, end_cost])
    #plt.ylim ([0, 1])
    #plt.xlabel('$ Cost')
    #plt.ylabel('% Coverage')
    #plt.show()

else:
#Logistic function for cost-coverage curve
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

# Logistic function for coverage-outcome curve

"""
def outcome_coverage_fx(cost, outcome_zerocov, outcome_fullcov):

    x = (np.array (outcome) - outcome_zerocov) / (outcome_fullcov - outcome_zerocov)
    #y = (outcome_fullcov - outcome_zerocov) * np.array (cov) + outcome_zerocov
    return x

for cost in cost_values:
    coverage_values.append (outcome_coverage_fx(cost, outcome_zerocov, outcome_fullcov))

"""

if multi_data == True:
    fig = plt.figure(1)
    fig.suptitle('Coverage-cost-outcome curve - Multiple data points')
    plt.subplot (121)
    plt.plot(covcov, costcost, 'bo', coverage_values, cost_values, 'r-', linewidth = 3)
    plt.xlim([0, 1])
    plt.ylim ([0, 1e6])
    plt.xlabel('% Coverage')
    plt.ylabel('$ Cost')
    plt.grid(True)
    """
    plt.subplot (122)
    plt.plot(outcome_values, coverage_values, 'b', linewidth = 3)
    plt.xlim ([outcome_zerocov, outcome_fullcov])
    plt.ylim([0, 1])
    plt.xlabel('Outcome')
    plt.ylabel('% coverage')
    plt.grid(True)
    """
    plt.show()
    fig.savefig('Coverage_cost_outcome_multiple.jpg')

else:
    fig = plt.figure(1)
    fig.suptitle('Coverage-cost-outcome curve - Single data point')
    plt.subplot (121)
    #fig.suptitle('Cost-coverage curve')
    plt.plot(coverage_values, cost_values, 'r', linewidth = 3)
    plt.plot(prog_cov["program_prop_treatment_success_mdr"], prog_cost["program_cost_treatment_success_mdr"], 'o')
    plt.xlim([start_coverage, end_coverage])
    plt.ylim ([0, 1e6])
    plt.grid(True)
    plt.xlabel('% Coverage')
    plt.ylabel('$ Cost')
    #plt.show()
    # #fig.savefig('cost_coverage.jpg')
    # #fig = plt.figure(2)
    """
    plt.subplot (122)
    #fig.suptitle('Coverage-outcome curve')
    plt.plot(outcome_values, coverage_values, 'b', linewidth = 3)
    plt.xlim ([outcome_zerocov, outcome_fullcov])
    plt.ylim([0, 1])
    plt.xlabel('Outcome')
    plt.ylabel('% Coverage')
    plt.grid(True)
    """
    plt.show()
    #fig.savefig('coverage_outcome.jpg')
    fig.savefig('Coverage_cost_outcome_single.jpg')
