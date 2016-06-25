

import numpy
import matplotlib.pyplot as plt
import autumn.plotting as plotting
import math
from autumn.spreadsheet import read_and_process_data, read_input_data_xls

"""

Module for estimating cost of a program

"""

"""
TO DO LIST

2. Funding value for each year. This will be challenging to get data for

4. Check with James: popsize is susceptible_fully?
5. Use plotting.py to plot all results. At the moment, economics.py has its own plotting function
6. Move the initial conditions to a Excel spreadsheet.
7. Think about discounting. At the moment, it is not need as the model only runs until 2015.
If we consider future cost projections, then discounting is needed. Formula for discounting:
Present value = cost / (1+ discount rate)^t, where t is number of years into the future
http://www.crest.uts.edu.au/pdfs/FactSheet_Discounting.pdf
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
year_index = 2000 # To plot/use cost function of a particular year. 1995 is just an example
year_ref = 2015 # Reference year for inflation calculation

if method == 1:
    print ("METHOD" " " + str(method), "new programs", "unit cost not available")
elif method ==2:
    print ("METHOD" " " + str(method), "established-programs", "unit cost available")

###################################################

########### GET INFLATION RATE DATA ###############

inflation = {1920: 0.123, 1930: 0.123, 1940: 0.123, 1950: 0.123, 1955: 0.123, 1960: 0.123, 1965: 0.123,
            1970: 0.041, 1971: 0.091, 1972: 0.22, 1973: 0.111, 1974: 0.145, 1975: 0.131, 1976: 0.114, 1977: 0.07,
            1978: 0.061, 1979: 0.078, 1980: 0.145, 1981: 0.112, 1982: 0.07, 1983: 0.067, 1984: 0.053, 1985: 0.044,
            1986: 0.018, 1987: 0.057, 1988: 0.118, 1989: 0.062, 1990: 0.082, 1991: 0.065, 1992: 0.049, 1993: 0.052,
            1994: 0.008, 1995: 0.022, 1996: 0.031, 1997: 0.034, 1998: 0.057, 1999: 0.02, 2000: 0.011, 2001: 0.043,
            2002: 0.008, 2003: 0.042, 2004: 0.028, 2005: 0.024, 2006: 0.025, 2007: 0.048, 2008: 0.077, 2009: 0.032,
            2010: 0.037, 2011: 0.073, 2012: 0.034, 2013: 0.029, 2014: 0.005, 2015: 0.014}
#Inflation: 1970 onwards are actual data. 1920 - 1970 calculated as average of 1970 - 1975

econ_cpi = {1970: 9.04, 1971: 9.41, 1972: 11.48, 1973: 12.75, 1974: 14.6, 1975: 16.5, 1976: 18.38, 1977: 19.67, 1978: 20.87,
            1979: 22.5, 1980: 25.9,
            1981: 28.8, 1982: 30.8, 1983: 32.9, 1984: 34.6, 1985: 36.1, 1986: 36.8, 1987: 38.9, 1988: 43.4, 1989: 46.1,
            1990: 49.9, 1991: 53.1, 1992: 55.7, 1993: 58.6, 1994: 59.1, 1995: 60.4, 1996: 62.2, 1997: 64.3, 1998: 68,
            1999: 69.3, 2000: 70.1, 2001: 73.1, 2002: 73.6, 2003: 76.7, 2004: 78.9, 2005: 80.8, 2006: 82.8, 2007: 86.7,
            2008: 93.4, 2009: 96.5, 2010: 100, 2011: 107.3, 2012: 110.9, 2013: 114.2, 2014: 114.8, 2015: 116.4}
#CPI: 1981 onwards are actual data. 1970 - 1980 calculated from inflation rate (inflation = (CPI new - CPI old)/CPI old.
#1920 - 1970 also calculated fron inflation rate but less reliable as inflation data are not actual data

country = read_input_data_xls(True, ['attributes'])['attributes'][u'country']
print(country)
data = read_and_process_data(True,
                             ['bcg', 'rate_birth', 'life_expectancy', 'attributes', 'parameters',
                              'country_constants', 'time_variants', 'tb', 'notifications', 'outcomes'],
                             country)
inflation_excel = data['time_variants'][u'inflation']
econ_cpi_excel = data['time_variants']['econ_cpi']
time_step = data['attributes']['time_step']


#######CREATE EMPTY LIST TO STORE RESULTS LATER #####

cost_uninflated = []
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
            # In this function, funding and coverage are time-variant. Coverage_range varies from a pre-define range every year
            cost_uninflated = funding / (((saturation - coverage_range) / (coverage_range * (saturation / coverage)))**((1 - scale_up_factor) / 2))
            return cost_uninflated

        elif method == 2: # For well-established programs of which start-up cost should be ignored. Unit cost known
            # In this function, popsize and unit cost are time-variant. Unitcost can be constant if only has 1 data
            cost_uninflated = - unitcost * popsize * math.log(((2 * saturation) / (coverage_range + saturation)) - 1)
            return cost_uninflated


def cost_scaleup_fns(model,
                     functions,
                     start_time_str = 'start_time',
                     end_time_str = '',
                     parameter_type='',
                     country = u''):

    if start_time_str == 'recent_time':
        start_time = model.data['attributes'][start_time_str]
    else:
        start_time = model.data['country_constants'][start_time_str]

    end_time = model.data['attributes'][end_time_str]
    #x_vals = numpy.linspace(start_time, end_time, end_time - start_time + 1)  # years
    x_vals = numpy.linspace(start_time, end_time, len(model.times))  # years

    #year_pos = year_index - start_time
    year_pos = ((year_index - start_time) / ((end_time - start_time) / len(model.times)))
    year_pos = int(year_pos)
    #print(year_index, model.times[year_pos])

    '''
    def make_time_steps(start_time, end_time, time_step):
        steps = []
        step = start_time
        while step <= end_time:
            steps.append(step)
            step += time_step
        return steps
    x_vals = make_time_steps(start_time, end_time, time_step)
    #For it to work, time_step = (start_time - end_time) / len(model.times))
    print(len(x_vals))
    print(model.times)
    '''

    for i, function in enumerate(functions):
        econ_cpi_scaleup = map(model.scaleup_fns['cpi'], x_vals)
        inflation_scaleup = map(model.scaleup_fns['inflation'], x_vals)

        if function == str('program_prop_vaccination'):
            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['program_cost_vaccination'], x_vals)
            #print(scaleup_param_vals[int(year_pos)], x_vals[int(year_pos)])
            #print(len(x_vals))
            popsize = model.compartment_soln['susceptible_fully']
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            for coverage_range in coverage_values:
                cost_uninflated = get_cost_from_coverage(coverage_range,
                                              params_default['saturation'],
                                              coverage, #Add [year_pos] to get cost-coverage curve at that year
                                              funding_scaleup, #Add [year_pos] to get cost-coverage curve at that year
                                              params_default['scale_up_factor'],
                                              params_default['unitcost'],
                                              popsize)

                print(cost_uninflated[year_pos])
                cost_inflated = cost_uninflated * econ_cpi_excel[year_ref] / econ_cpi_scaleup


################### PLOTTING ##############################################




            data_to_plot = {}
            data_to_plot = model.scaleup_data[function]

            plt.figure('Coverage (program_prop_vaccination)')
            lineup = plt.plot(x_vals, coverage, 'b', linewidth = 2, label = 'scaleup_program_pop_vaccination')
            plt.scatter(data_to_plot.keys(), data_to_plot.values(), label = 'data_progrom_pop_vaccination')
            #title = str(country) + ' ' + \
            #            plotting.replace_underscore_with_space(parameter_type) + \
            #            ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title('Coverage (program_prop_vaccination)')
            plt.xlim([start_time, end_time])
            plt.ylim([0, 1.1])
            plt.xlabel("Years")
            plt.ylabel('program_prop_vaccination')
            plt.legend(loc = 'upper left')
            plt.grid(True)
            plt.show()


            plt.figure('Cost of BCG program (USD)')
            plt.plot(x_vals, cost_uninflated, 'b', linewidth = 3, label = 'Uninflated')
            plt.plot(x_vals, cost_inflated, 'b--', linewidth = 3, label = 'Inflated')
            #title = str(country) + ' ' + \
            #            plotting.replace_underscore_with_space(parameter_type) + \
            #            ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title('Cost of BCG program (USD')
            plt.xlabel('Years')
            plt.ylabel('$ Cost of program_vaccination (USD')
            plt.xlim([start_time, end_time])
            plt.legend(loc = 'upper right')
            plt.title('Cost of BCG program (USD)')
            plt.grid(True)
            plt.show()


            plt.figure('BCG spending')
            plt.plot(x_vals, funding_scaleup, 'b', linewidth = 3, label = 'BCG spending')
            a = {}
            a = model.scaleup_data['program_cost_vaccination']
            plt.scatter(a.keys(), a.values())
            plt.title('BCG spending')
            plt.legend(loc = 'upper left')
            plt.xlim([start_time, end_time])
            plt.grid(True)
            plt.show()

            plt.figure('Population size')
            plt.plot(x_vals, popsize, 'r', linewidth = 3, label = 'Population size (susceptible_fully')
            plt.title('Population size (susceptible_fully')
            plt.legend(loc = 'upper left')
            plt.xlim([start_time, end_time])
            plt.grid(True)
            plt.show()


            fig, ax1 = plt.subplots()
            ax1.plot(x_vals, cost_uninflated, 'b-', linewidth = 3, label = 'Un-inflated cost')
            ax1.plot(x_vals, cost_inflated, 'b--', linewidth = 3, label = 'Inflated cost')
            ax1.set_xlabel('Years')
            # Make the y-axis label and tick labels match the line color.
            ax1.set_ylabel('Yearly total BCG cost (USD)', color='b')
            for tl in ax1.get_yticklabels():
                tl.set_color('b')
            ax2 = ax1.twinx()
            ax2.plot(x_vals, econ_cpi_scaleup, 'r-', linewidth = 3, label = 'Consumer price index - actual data')
            ax2.plot(econ_cpi.keys(), econ_cpi.values(), 'ro', label ='Consumer price index - fitted')
            ax2.set_ylabel('Consumer price index', color='r')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')
            plt.grid(True)
            legend = ax1.legend(loc = 'upper left', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            plt.show()



            '''
            plt.figure(333)
            plt.plot(cost_uninflated, coverage_values)
            title = str(country) + ' ' + \
                        plotting.replace_underscore_with_space(parameter_type) + \
                        ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title(title)
            plt.xlabel('Cost')
            plt.ylabel('Coverage')
            plt.title('Cost coverage curve for year ' + str(year_index))
            plt.show()
            '''








