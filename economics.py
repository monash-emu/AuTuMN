
import pylab
import numpy
from matplotlib import patches
import matplotlib.pyplot as plt
import autumn.plotting as plotting
import math
from autumn.spreadsheet import read_and_process_data, read_input_data_xls
import itertools
"""

Module for estimating cost of a program

"""

############### INITIAL CONDITIONS##################

params_default = {
    "saturation": 1,
    "coverage": 0.33,
    "funding": 1.5e6,
    "scale_up_factor": 0.75,
    "unitcost": 90,
    "popsize": 881e3,
    "method": 2,
    "outcome_zerocov": 20,
    "outcome_fullcov": 5}

start_coverage = 0.0001
end_coverage = params_default["saturation"]
delta_coverage = 0.001
method = 1
year_index = 1985 # To plot/use cost function of a particular year. 1995 is just an exmaple
year_ref = 2015 # Reference year for inflation calculation

if method == 1:
    print ("METHOD" " " + str(method), "new programs", "unit cost not available")
elif method ==2:
    print ("METHOD" " " + str(method), "established-programs", "unit cost available")

###################################################

########### GET INFLATION RATE DATA ###############

inflation = {1970: 0.041, 1971: 0.091, 1972: 0.22, 1973: 0.111, 1974: 0.145, 1975: 0.131, 1976: 0.114, 1977: 0.07,
            1978: 0.061, 1979: 0.078, 1980: 0.145, 1981: 0.112, 1982: 0.07, 1983: 0.067, 1984: 0.053, 1985: 0.044,
            1986: 0.018, 1987: 0.057, 1988: 0.118, 1989: 0.062, 1990: 0.082, 1991: 0.065, 1992: 0.049, 1993: 0.052,
            1994: 0.008, 1995: 0.022, 1996: 0.031, 1997: 0.034, 1998: 0.057, 1999: 0.02, 2000: 0.011, 2001: 0.043,
            2002: 0.008, 2003: 0.042, 2004: 0.028, 2005: 0.024, 2006: 0.025, 2007: 0.048, 2008: 0.077, 2009: 0.032,
            2010: 0.037, 2011: 0.073, 2012: 0.034, 2013: 0.029, 2014: 0.005, 2015: 0.014}

#print(inflation[2010])

cpi = {1981: 28.8, 1982: 30.8, 1983: 32.9, 1984: 34.6, 1985: 36.1, 1986: 36.8, 1987: 38.9, 1988: 43.4, 1989: 46.1,
       1990: 49.9, 1991: 53.1, 1992: 55.7, 1993: 58.6, 1994: 59.1, 1995: 60.4, 1996: 62.2, 1997: 64.3, 1998: 68,
       1999: 69.3, 2000: 70.1, 2001: 73.1, 2002: 73.6, 2003: 76.7, 2004: 78.9, 2005: 80.8, 2006: 82.8, 2007: 86.7,
       2008: 93.4, 2009: 96.5, 2010: 100, 2011: 107.3, 2012: 110.9, 2013: 114.2, 2014: 114.8, 2015: 116.4}

'''
country = read_input_data_xls(True, ['attributes'])['attributes'][u'country']
print(country)
data = read_and_process_data(True,
                             ['bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters',
                              'country_constants', 'time_variants', 'tb', 'notifications', 'outcomes'],
                             country)
inflation = data['time_variants'][u'inflation']
cpi = data['time_variants'][u'cpi']
'''

#inft = map(inflation, years)
#print(inft)


#######CREATE EMPTY LIST TO STORE RESULTS LATER #####

cost = []
coverage_values = []

#####################################################


######## MAKE COVERAGE RANGE #######################

def make_coverage_steps(start_coverage, end_coverage, delta_coverage):
    steps = []
    step = start_coverage
    while step <= end_coverage:
        steps.append(step)
        step += delta_coverage
    return steps
coverage_values = make_coverage_steps(start_coverage, end_coverage, delta_coverage)

def make_year_steps(start_time, end_time, delta_time):
    steps = []
    step = start_time
    while step <= end_time:
        steps.append(step)
        step += delta_time
    return steps
year_steps = make_year_steps(0, 121, 1)
#print(year_steps[120])

#######################################################


##### FX TO GET COVERAGE FROM OUTCOME #################
# Because Vaccination is both a TB program and model param, COVERAGE = OUTCOME

def get_coverage_from_outcome_program_as_param(outcome):
    coverage = numpy.array(outcome)
    return coverage

######################################################

##### FX TO GET COST FROM COVERAGE ##################

def get_cost_from_coverage(coverage_range, saturation, coverage, funding, scale_up_factor, unitcost, popsize):
    if method in (1, 2):
        if method == 1: # For new programs which requires significant start-up cost. Unit cost unknown
            cost_uninflated = funding / (((saturation - coverage_range) / (coverage_range * (saturation / coverage)))**((1 - scale_up_factor) / 2))
            return cost_uninflated
        elif method == 2: # For well-established programs of which start-up cost should be ignored. Unit cost known
            cost_uninflated = - unitcost * popsize * math.log(((2 * saturation) / (coverage_range + saturation)) - 1)
            return cost_uninflated


def cost_scaleup_fns(model,
                     functions,
                     start_time_str = 'start_time',
                     end_time_str = '',
                     parameter_type='',
                     country=u''):

    if start_time_str == 'recent_time':
        start_time = model.data['attributes'][start_time_str]
    else:
        start_time = model.data['country_constants'][start_time_str]

    end_time = model.data['attributes'][end_time_str]
    x_vals = numpy.linspace(start_time, end_time, end_time - start_time + 1)  # years

    year_pos = year_index - start_time
    #print(year_index, year_pos)


    for i, function in enumerate(functions):

        cpi_scaleup = map(model.scaleup_fns['cpi'], x_vals)
        inflation_scaleup = map(model.scaleup_fns['inflation'], x_vals)
        #print(cpi_scaleup)

        if function == str('program_prop_vaccination'):
            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            print(scaleup_param_vals[int(year_pos)], x_vals[int(year_pos)])
            #print(len(x_vals))
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            for coverage_range in coverage_values:
                cost_uninflated = get_cost_from_coverage(coverage_range,
                                              params_default['saturation'],
                                              coverage,
                                              params_default['funding'],
                                              params_default['scale_up_factor'],
                                              params_default['unitcost'],
                                              params_default['popsize'])
                cost_inflated = cost_uninflated * cpi[2015] / cpi_scaleup



            plt.figure(111)
            plt.plot(x_vals, coverage)
            title = str(country) + ' ' + \
                        plotting.replace_underscore_with_space(parameter_type) + \
                        ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title(title)
            plt.ylim([0, 1.1])
            plt.xlabel("Years")
            plt.ylabel('program_prop_vaccination')
            plt.show()

            #print(len(x_vals))
            #print(len(cost))


            plt.figure(222)
            plt.plot(x_vals, cost_uninflated, x_vals, cost_inflated)
            title = str(country) + ' ' + \
                        plotting.replace_underscore_with_space(parameter_type) + \
                        ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title(title)
            plt.xlabel('Years')
            plt.ylabel('$ Cost of program_vaccination')
            plt.show()


            fig, ax1 = plt.subplots()
            ax1.plot(x_vals, cost_uninflated, 'b-', label = 'Un-inflated cost')
            ax1.plot(x_vals, cost_inflated, 'b--', label = 'Inflated cost')
            ax1.set_xlabel('Years')
            # Make the y-axis label and tick labels match the line color.
            ax1.set_ylabel('Yearly total BCG cost (USD)', color='b')
            for tl in ax1.get_yticklabels():
                tl.set_color('b')

            ax2 = ax1.twinx()
            ax2.plot(x_vals, cpi_scaleup, 'r.', label = 'Consumer price index')
            ax2.set_ylabel('Consumer price index', color='r')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')
            plt.grid(True)
            legend = ax1.legend(loc='upper center', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')

            plt.show()



            '''
            plt.figure(333)
            plt.plot(cost, coverage_values)
            title = str(country) + ' ' + \
                        plotting.replace_underscore_with_space(parameter_type) + \
                        ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title(title)
            plt.xlabel('Cost')
            plt.ylabel('Coverage')
            plt.show()
            '''




