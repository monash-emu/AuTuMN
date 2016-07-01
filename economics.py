
import numpy
import matplotlib.pyplot as plt
import math
from autumn.spreadsheet import read_and_process_data, read_input_data_xls
import autumn.plotting as plotting
import autumn.model as model

"""

Module for estimating cost of a program

"""

"""
TO DO LIST

2. Funding value for each year. This will be challenging to get data for
5. Use plotting.py to plot all results. At the moment, economics.py has its own plotting function
6. Move the initial conditions to a Excel spreadsheet.

"""

############ READ SOME DATA FROM SPREADSHEET ########

country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
print(country)
data = read_and_process_data(country,
                             keys_of_sheets_to_read=['bcg', 'rate_birth', 'life_expectancy', 'control_panel',
                              'default_parameters',
                              'tb', 'notifications', 'outcomes',
                              'country_constants', 'default_constants',
                              'country_economics', 'default_economics',
                              'country_programs', 'default_programs'],
                             from_test=True)
inflation = data['country_economics']['econ_inflation']
cpi = data['country_economics']['econ_cpi']
time_step = data['model_constants']['time_step']

###################################################


############### INITIAL CONDITIONS#################

params_default = {
    "saturation": 1.001,
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
plot_costcurve = True
method = 2
discount_rate = 0.03
year_index = 2014 # To plot/use cost function of a particular year. 1995 is just an example
year_current = data['model_constants']['current_time'] # Reference year for inflation calculation (2015)
print("Current year " + str(year_current))


if method == 1:
    print ("METHOD" " " + str(method), "new programs", "unit cost not available")
elif method ==2:
    print ("METHOD" " " + str(method), "established-programs", "unit cost available")

###################################################

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


#######CREATE EMPTY LIST TO STORE RESULTS LATER #####

#cost_uninflated = []
#cost_uninflated_toplotcostcurve = []
#cost_inflated = []
#popsize = []
#x_vals_2015onwards_array =[]
#cost_discounted_array = []

#####################################################


##### FX TO GET COVERAGE FROM OUTCOME #################
# Because Vaccination is both a TB program and model param, COVERAGE = OUTCOME

def get_coverage_from_outcome_program_as_param(outcome):
    coverage = numpy.array(outcome)
    return coverage

######################################################


##### FX TO GET COST FROM COVERAGE ##################

def get_cost_from_coverage(saturation, coverage, funding_mid, scale_up_factor, unitcost, popsize, coverage_mid):
    if method in (1, 2):
        if method == 1: # For new programs which requires significant start-up cost. Unit cost unknown
            # In this function, funding and coverage are time-variant. Coverage_range varies from a pre-define range every year
            cost_uninflated = funding_mid / (((saturation - coverage) / (coverage * (saturation / coverage_mid - 1)))**((1 - scale_up_factor) / 2))
            return cost_uninflated

        elif method == 2: # For well-established programs of which start-up cost should be ignored. Unit cost known
            # In this function, popsize and unit cost are time-variant. Unitcost can be constant if only has 1 data
            cost_uninflated = - unitcost * popsize * math.log(((2 * saturation) / (coverage + saturation)) - 1)
            return cost_uninflated

def cost_scaleup_fns(model,
                     functions,
                     start_time_str = 'start_time',
                     end_time_str = '',
                     parameter_type='',
                     country = u''):

    if start_time_str == 'recent_time':
        start_time = model.data['model_constants'][start_time_str]
    else:
        start_time = model.data['country_constants'][start_time_str]

    end_time = model.data['model_constants'][end_time_str]
    print('Start time ' + str(start_time) + ' End time ' + str(end_time))
    #x_vals = numpy.linspace(start_time, end_time, end_time - start_time + 1)  # years
    x_vals = numpy.linspace(start_time, end_time, len(model.times))  # years

    year_pos = ((year_index - start_time) / ((end_time - start_time) / len(model.times)))
    year_pos = int(year_pos)
    print('Index year ' + str(x_vals[year_pos]))
    #print(year_index, model.times[year_pos])


    for i, function in enumerate(functions):
        cpi_scaleup = map(model.scaleup_fns['econ_cpi'], x_vals)
        inflation_scaleup = map(model.scaleup_fns['econ_inflation'], x_vals)

        if function == str('program_prop_vaccination'):

            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_vaccination'], x_vals)
            #print(scaleup_param_vals[int(year_pos)], x_vals[int(year_pos)])
            unitcost = map(model.scaleup_fns['econ_program_unitcost_vaccination'], x_vals)
            #popsize = model.compartment_soln['susceptible_fully']
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            coverage_mid = coverage
            for i in numpy.arange(0, len(x_vals), 1):
                all_flows = model.var_array[int(i)]
                for a, b in enumerate(model.var_labels):
                    if b == 'births_vac':
                        pop = all_flows[a]
                        popsize.append(pop)
                        cost_uninflated.append(get_cost_from_coverage(params_default['saturation'],
                                                        coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                        funding_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                        params_default['scale_up_factor'],
                                                        unitcost[int(i)],
                                                        popsize[int(i)],
                                                        coverage_mid[int(i)]))
                        cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])

                        current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))
                        if i >= current_year_pos and i <= len(x_vals):
                            x_vals_2015onwards = x_vals[i]
                            x_vals_2015onwards_array.append(x_vals_2015onwards)
                            cost_todiscount = cost_uninflated[int(i)]
                            if x_vals[i] <= 2015:
                                    years_into_future = 0
                            elif x_vals[i] > 2015 and x_vals[i] <= 2016:
                                    years_into_future = 1
                            elif x_vals[i] > 2016 and x_vals[i] <= 2017:
                                    years_into_future = 2
                            elif x_vals[i] > 2017 and x_vals[i] <= 2018:
                                    years_into_future = 3
                            elif x_vals[i] > 2018 and x_vals[i] <= 2019:
                                    years_into_future = 4
                            elif x_vals[i] > 2019 and x_vals[i] <= 2020:
                                    years_into_future = 5
                            elif x_vals[i] > 2020 and x_vals[i] <= 2021:
                                    years_into_future = 6
                            elif x_vals[i] > 2021 and x_vals[i] <= 2022:
                                    years_into_future = 7
                            elif x_vals[i] > 2022 and x_vals[i] <= 2023:
                                    years_into_future = 8
                            elif x_vals[i] > 2023 and x_vals[i] <= 2024:
                                    years_into_future = 9
                            elif x_vals[i] > 2024 and x_vals[i] <= 2025:
                                    years_into_future = 10
                            elif x_vals[i] > 2025 and x_vals[i] <= 2026:
                                    years_into_future = 11
                            elif x_vals[i] > 2026 and x_vals[i] <= 2027:
                                    years_into_future = 12
                            elif x_vals[i] > 2027 and x_vals[i] <= 2028:
                                    years_into_future = 13
                            elif x_vals[i] > 2028 and x_vals[i] <= 2029:
                                    years_into_future = 14
                            elif x_vals[i] > 2029 and x_vals[i] <= 2030:
                                    years_into_future = 15
                            elif x_vals[i] > 2030 and x_vals[i] <= 2031:
                                    years_into_future = 16
                            elif x_vals[i] > 2031 and x_vals[i] <= 2032:
                                    years_into_future = 17
                            elif x_vals[i] > 2032 and x_vals[i] <= 2033:
                                    years_into_future = 18
                            elif x_vals[i] > 2033 and x_vals[i] <= 2034:
                                    years_into_future = 19
                            else:
                                    years_into_future = 20
                            cost_discounted = cost_todiscount / ((1 + discount_rate)**years_into_future)
                            cost_discounted_array.append(cost_discounted)

                            #print(len(cost_discounted_array))



########## PLOT COST COVERAGE CURVE #######################################

            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(params_default['saturation'],
                                                coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                funding_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year
                                                params_default['scale_up_factor'],
                                                unitcost[year_pos],
                                                popsize[year_pos],
                                                coverage_mid[year_pos]))

###########################################################################


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
            plt.plot(x_vals, cost_inflated, 'g', linewidth = 3, label = 'Inflated')
            plt.plot(x_vals_2015onwards_array, cost_discounted_array, 'r', linewidth =3, label = 'Discounted')
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
            a = model.scaleup_data['econ_program_totalcost_vaccination']
            plt.scatter(a.keys(), a.values())
            plt.title('BCG spending')
            plt.legend(loc = 'upper left')
            plt.xlim([start_time, end_time])
            plt.grid(True)
            plt.show()

            plt.figure('Population size')
            plt.plot(x_vals, popsize, 'r', linewidth = 3, label = 'Population size (birth_vac)')
            plt.title('Population size (biths_vac)')
            plt.legend(loc = 'upper left')
            plt.xlim([start_time, end_time])
            plt.xlabel('Year')
            plt.ylabel('Number of people')
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
            ax2.plot(x_vals, cpi_scaleup, 'r-', linewidth = 3, label = 'Consumer price index')
            ax2.set_ylabel('Consumer price index', color='r')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')
            plt.grid(True)
            legend = ax1.legend(loc = 'upper left', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            plt.show()


            plt.figure('Cost-coverage curve')
            plt.plot(cost_uninflated_toplotcostcurve, coverage_values, 'r', linewidth = 3)
            plt.title('Cost coverage curve for year ' + str(year_index))
            plt.xlabel('$ Cost')
            plt.ylabel('Coverage')
            plt.ylim([0, params_default['saturation']])
            plt.show()

########## IPT #####################################################################

        elif function == str('program_prop_ipt'):

            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_ipt'], x_vals)
            #print(scaleup_param_vals[int(year_pos)], x_vals[int(year_pos)])
            unitcost = map(model.scaleup_fns['econ_program_unitcost_ipt'], x_vals)
            #popsize = model.compartment_soln['susceptible_fully']
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            coverage_mid = coverage
            for i in numpy.arange(0, len(x_vals), 1):
                all_flows = model.var_array[int(i)]
                for a, b in enumerate(model.var_labels):
                    if b == 'ipt_commencements':
                        pop = all_flows[a]
                        popsize.append(pop)
                        cost_uninflated.append(get_cost_from_coverage(params_default['saturation'],
                                                        coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                        funding_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                        params_default['scale_up_factor'],
                                                        unitcost[int(i)],
                                                        popsize[int(i)],
                                                        coverage_mid[int(i)]))

                        cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])

                        current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))
                        if i >= current_year_pos and i <= len(x_vals):
                            x_vals_2015onwards = x_vals[i]
                            x_vals_2015onwards_array.append(x_vals_2015onwards)
                            cost_todiscount = cost_uninflated[int(i)]
                            if x_vals[i] <= 2015:
                                    years_into_future = 0
                            elif x_vals[i] > 2015 and x_vals[i] <= 2016:
                                    years_into_future = 1
                            elif x_vals[i] > 2016 and x_vals[i] <= 2017:
                                    years_into_future = 2
                            elif x_vals[i] > 2017 and x_vals[i] <= 2018:
                                    years_into_future = 3
                            elif x_vals[i] > 2018 and x_vals[i] <= 2019:
                                    years_into_future = 4
                            elif x_vals[i] > 2019 and x_vals[i] <= 2020:
                                    years_into_future = 5
                            elif x_vals[i] > 2020 and x_vals[i] <= 2021:
                                    years_into_future = 6
                            elif x_vals[i] > 2021 and x_vals[i] <= 2022:
                                    years_into_future = 7
                            elif x_vals[i] > 2022 and x_vals[i] <= 2023:
                                    years_into_future = 8
                            elif x_vals[i] > 2023 and x_vals[i] <= 2024:
                                    years_into_future = 9
                            elif x_vals[i] > 2024 and x_vals[i] <= 2025:
                                    years_into_future = 10
                            elif x_vals[i] > 2025 and x_vals[i] <= 2026:
                                    years_into_future = 11
                            elif x_vals[i] > 2026 and x_vals[i] <= 2027:
                                    years_into_future = 12
                            elif x_vals[i] > 2027 and x_vals[i] <= 2028:
                                    years_into_future = 13
                            elif x_vals[i] > 2028 and x_vals[i] <= 2029:
                                    years_into_future = 14
                            elif x_vals[i] > 2029 and x_vals[i] <= 2030:
                                    years_into_future = 15
                            elif x_vals[i] > 2030 and x_vals[i] <= 2031:
                                    years_into_future = 16
                            elif x_vals[i] > 2031 and x_vals[i] <= 2032:
                                    years_into_future = 17
                            elif x_vals[i] > 2032 and x_vals[i] <= 2033:
                                    years_into_future = 18
                            elif x_vals[i] > 2033 and x_vals[i] <= 2034:
                                    years_into_future = 19
                            else:
                                    years_into_future = 20
                            cost_discounted = cost_todiscount / ((1 + discount_rate)**years_into_future)
                            cost_discounted_array.append(cost_discounted)

                            #print(len(cost_discounted_array))



########## PLOT COST COVERAGE CURVE #######################################

            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(params_default['saturation'],
                                                coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                funding_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year
                                                params_default['scale_up_factor'],
                                                unitcost[year_pos],
                                                popsize[year_pos],
                                                coverage_mid[year_pos]))

###########################################################################


################### PLOTTING ##############################################

            data_to_plot = {}
            data_to_plot = model.scaleup_data[function]

            plt.figure('Coverage (program_prop_ipt)')
            lineup = plt.plot(x_vals, coverage, 'b', linewidth = 2, label = 'scaleup_program_pop_ipt')
            plt.scatter(data_to_plot.keys(), data_to_plot.values(), label = 'data_progrom_pop_ipt')
            #title = str(country) + ' ' + \
            #            plotting.replace_underscore_with_space(parameter_type) + \
            #            ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title('Coverage (program_prop_ipt)')
            plt.xlim([start_time, end_time])
            plt.ylim([0, 1.1])
            plt.xlabel("Years")
            plt.ylabel('program_prop_ipt')
            plt.legend(loc = 'upper left')
            plt.grid(True)
            plt.show()


            plt.figure('Cost of IPT program (USD)')
            plt.plot(x_vals, cost_uninflated, 'b', linewidth = 3, label = 'Uninflated')
            plt.plot(x_vals, cost_inflated, 'g', linewidth = 3, label = 'Inflated')
            plt.plot(x_vals_2015onwards_array, cost_discounted_array, 'r', linewidth =3, label = 'Discounted')
            #title = str(country) + ' ' + \
            #            plotting.replace_underscore_with_space(parameter_type) + \
            #            ' parameter' + ' from ' + plotting.replace_underscore_with_space(start_time_str)
            plt.title('Cost of IPT program (USD')
            plt.xlabel('Years')
            plt.ylabel('$ Cost of program_IPT (USD')
            plt.xlim([start_time, end_time])
            plt.legend(loc = 'upper right')
            plt.title('Cost of IPT program (USD)')
            plt.grid(True)
            plt.show()


            plt.figure('IPT spending')
            plt.plot(x_vals, funding_scaleup, 'b', linewidth = 3, label = 'IPT spending')
            a = {}
            a = model.scaleup_data['econ_program_totalcost_ipt']
            plt.scatter(a.keys(), a.values())
            plt.title('IPT spending')
            plt.legend(loc = 'upper left')
            plt.xlim([start_time, end_time])
            plt.grid(True)
            plt.show()

            plt.figure('Population size')
            plt.plot(x_vals, popsize, 'r', linewidth = 3, label = 'Population size (ipt_commencements)')
            plt.title('Population size (ipt_commencements)')
            plt.legend(loc = 'upper left')
            plt.xlim([start_time, end_time])
            plt.xlabel('Year')
            plt.ylabel('Number of people')
            plt.grid(True)
            plt.show()


            fig, ax1 = plt.subplots()
            ax1.plot(x_vals, cost_uninflated, 'b-', linewidth = 3, label = 'Un-inflated cost')
            ax1.plot(x_vals, cost_inflated, 'b--', linewidth = 3, label = 'Inflated cost')
            ax1.set_xlabel('Years')
            # Make the y-axis label and tick labels match the line color.
            ax1.set_ylabel('Yearly total IPT cost (USD)', color='b')
            for tl in ax1.get_yticklabels():
                tl.set_color('b')
            ax2 = ax1.twinx()
            ax2.plot(x_vals, cpi_scaleup, 'r-', linewidth = 3, label = 'Consumer price index')
            ax2.set_ylabel('Consumer price index', color='r')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')
            plt.grid(True)
            legend = ax1.legend(loc = 'upper left', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            plt.show()


            plt.figure('Cost-coverage curve')
            plt.plot(cost_uninflated_toplotcostcurve, coverage_values, 'r', linewidth = 3)
            plt.title('Cost coverage curve for year ' + str(year_index))
            plt.xlabel('$ Cost')
            plt.ylabel('Coverage')
            plt.ylim([0, params_default['saturation']])
            plt.show()

############################################################################################

