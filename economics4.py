
import numpy
import matplotlib.pyplot as plt
import math
from autumn.spreadsheet import read_input_data_xls
import autumn.data_processing
import scipy.optimize
import os
import autumn.model
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy

'''
TO DO:
0-15 years population for IPT
Position of "program_rate_detect" and "program_rate_miss" in Xpert
'''
############ READ SOME DATA FROM SPREADSHEET ########

country = read_input_data_xls(True, ['control_panel'])['control_panel']['country']
#print(country)

inputs = autumn.data_processing.Inputs(True)
inputs.read_and_load_data()

inflation = inputs.original_data['country_economics']['econ_inflation']
cpi = inputs.original_data['country_economics']['econ_cpi']
time_step = inputs.model_constants['time_step']

###################################################


############### INITIAL CONDITIONS#################

saturation = 1.001
alpha = 1.

start_coverage = 0.0001
end_coverage = saturation
delta_coverage = 0.001
plot_costcurve = True

discount_rate = 0.03
year_index = 2015 # To plot/use cost function of a particular year. 1995 is just an example
year_current = inputs.model_constants['current_time'] # Reference year for inflation calculation (2015)
#print("Current year " + str(year_current))

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


##### FX TO GET COVERAGE FROM OUTCOME #################
# COVERAGE = OUTCOME

def get_coverage_from_outcome_program_as_param(outcome):
    coverage = numpy.array(outcome)
    return coverage

######################################################


##### FX TO GET COST FROM COVERAGE ##################

def get_cost_from_coverage(c_reflection_cost, b_growth_rate, saturation, a, coverage, alpha):
    cost_uninflated = c_reflection_cost - 1/b_growth_rate * math.log((((saturation - a) / (coverage - a))**(1 / alpha)) - 1)
    return cost_uninflated

def get_coverage_from_cost(a, saturation, b_growth_rate, cost, c_relfection_cost, alpha):
    coverage_estimated = a + (saturation - a) /((1 + math.exp((-b_growth_rate) * (cost - c_relfection_cost)))**alpha)
    return coverage_estimated

def cost_scaleup_fns(model,
                     functions,
                     start_time_str = 'start_time',
                     end_time_str = '',
                     parameter_type='',
                     country = u''):

    if start_time_str == 'recent_time':
        start_time = model.inputs.model_constants[start_time_str]
    else:
        start_time = model.inputs.original_data['country_constants'][start_time_str]

    end_time = model.inputs.model_constants[end_time_str]
    print('Start time ' + str(start_time) + ' End time ' + str(end_time))
    #x_vals = numpy.linspace(start_time, end_time, end_time - start_time + 1)  # years
    x_vals = numpy.linspace(start_time, end_time, len(model.times))  # years
    print(len(x_vals))

    year_pos = ((year_index - start_time) / ((end_time - start_time) / len(model.times)))
    year_pos = int(year_pos)
    #print('Index year ' + str(x_vals[year_pos]))
    #print('year position ' + str(year_pos))
    year_pos_end = ((end_time - start_time) / ((end_time - start_time) / len(model.times)))
    #print(year_pos_end)

    #indexes = numpy.linspace (year_pos, year_pos_end, 12001)

    #print(len(indexes))

    #print(len(replacement))
    #print(year_index, model.times[year_pos])

    for i, function in enumerate(functions):
        cpi_scaleup = map(model.scaleup_fns['econ_cpi'], x_vals)
        inflation_scaleup = map(model.scaleup_fns['econ_inflation'], x_vals)


#######################################################################################
###  BCG VACCINATION
#######################################################################################

        if function == str('program_prop_vaccination'): #using data from data_fiji
        #if function == str('econ_program_prop_vaccination'):
            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []
            popsize_vac =[]
            popsize_unvac = []
            b_growth_rate = []
            coverage_estimated =[]

            indexes = numpy.arange(12001, 14002, 1)
            replacements = numpy.linspace(15000, 16000, 2001)

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_vaccination'], x_vals)
            reflection_cost_scaleup = map(model.scaleup_fns['econ_program_reflectioncost_vaccination'], x_vals)
            #print(scaleup_param_vals[int(year_pos)], x_vals[int(year_pos)])

            unitcost = map(model.scaleup_fns['econ_program_unitcost_vaccination'], x_vals)
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            for i in numpy.arange(0, len(x_vals), 1):
                all_flows = model.var_array[int(i)]
                for a, b in enumerate(model.var_labels):
                    if b == 'births_vac':
                        #pop = all_flows[a] #actually vaccinated
                        pop_vac = all_flows[a]
                        pop_unvac = (pop_vac * (1 - coverage[int(i)])) / coverage[int(i)]
                        popsize_vac.append(pop_vac)
                        popsize_unvac.append(pop_unvac)
                        pop = pop_vac + pop_unvac
                        popsize.append(pop)
                        a = saturation / (1 - 2**alpha)
                        b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                        b_growth_rate.append(b)
                        cost_uninflated.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
                        cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])
                        current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))
                        cost_predefined = cost_uninflated # create another copy of cost_uninfltated, then replace with user-defined funding values

                        # Discounting
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

                        coverage_estimated.append(get_coverage_from_cost(a,
                                                                    saturation,
                                                                    b_growth_rate[int(i)],
                                                                    cost_uninflated[int(i)],
                                                                    reflection_cost_scaleup[int(i)],
                                                                    alpha))
            #Replace costs values of certain years with new funding values, and then estimate corresponding new coverage levels
            for index_element, replacement_element in zip (indexes, replacements):
                cost_predefined [index_element] = replacement_element
            b_growth_rate_2 = []
            coverage_new_bcg = []
            for i in numpy.arange (0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate_2.append(b)
                coverage_new_bcg.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate_2[int(i)],
                                                            cost_predefined[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))
                coverage_new_bcg_nan = numpy.nan_to_num(coverage_new_bcg)

            # Write new coverage levels into workbook so that they can be fed and read back into epi module to estimate new TB burden
            book = xlwt.Workbook(encoding="utf-8")
            sheet1 = book.add_sheet("new_coverage")
            sheet1.write(0, 0, "year")
            sheet1.write(0, 1, "new_bcg_cov")
            i=0
            ii=0
            for m in range (len(x_vals)):
                ii = ii+1
                yr = (m*(end_time - start_time) + len(x_vals)*start_time)/len(x_vals)
                sheet1.write(ii, 0, yr)
            for n in coverage_new_bcg_nan:
                i = i+1
                sheet1.write(i, 1, n)
            book.save('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls') # Romain will need to change the path

            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    a = saturation / (1 - 2**alpha)
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(reflection_cost_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[year_pos],
                                                                          saturation,
                                                                          a,
                                                                          coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
            #Plotting results
            data_to_plot = model.scaleup_data[function]
            fig = plt.figure('BCG')
            ax1 = fig.add_subplot(221)
            ax1.plot(cost_uninflated_toplotcostcurve, coverage_values, 'b-', linewidth = 3)
            ax1.set_title('BCG cost-coverage curve ' + str(year_index))
            ax1.set_xlabel('Cost (USD)')
            ax1.set_ylabel('Coverage (%)')
            ax1.set_ylim([0, 1.1])
            ax1.grid(True)

            ax2 = fig.add_subplot(222)
            ax2.plot(x_vals, cost_uninflated, 'r-', linewidth = 3, label = 'Cost uninflated')
            ax2.plot(x_vals, cost_inflated, 'r--', linewidth = 3, label = 'Cost inflated')
            ax2.plot(x_vals_2015onwards_array, cost_discounted_array, 'b--', linewidth = 3, label = 'Cost discounted')
            ax2.set_title('BCG yearly total cost (USD)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Cost (USD)')
            ax2.set_xlim([1950, end_time])
            ax2.legend(loc = 'upper right')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(x_vals, popsize, 'r-', linewidth = 3, label = 'Popsize (BCG births + nonBCG births)')
            ax3.plot(x_vals, popsize_vac, 'b-', linewidth =3, label = 'BCG births')
            ax3.plot(x_vals, popsize_unvac, 'b--', linewidth =3, label = 'nonBCG births')
            ax3.set_title('BCG population size')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('People')
            ax3.set_xlim([1950, end_time])
            ax3.legend(loc = 'upper right')
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(x_vals, coverage, 'r-', linewidth =3)
            ax4.scatter(data_to_plot.keys(), data_to_plot.values())
            ax4.plot(x_vals, coverage_estimated, 'b-', linewidth = 3)
            ax4.set_title('BCG coverage')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Coverage (%)')
            ax4.set_ylim([0, 1.1])
            ax4.grid(True)
            plt.show()

            plt.figure('new BCG coverage level')
            plt.plot(x_vals, coverage_new_bcg, 'r-', linewidth = 3)
            plt.xlabel('years')
            plt.ylabel('coverage')
            plt.title('new BCG coverage level')
            plt.show()

#######################################################################################
###  IPT
#######################################################################################

        #elif function == str('program_prop_vaccination'): #using data from data_fiji
        elif function == str('econ_program_prop_ipt'):
            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []
            popsize_vac =[]
            popsize_unvac = []
            b_growth_rate = []
            coverage_estimated = []
            cost_predefined = []

            indexes = numpy.arange(12001, 17002, 1)
            replacements = numpy.linspace(20000, 30000, 5001)

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_ipt'], x_vals)
            reflection_cost_scaleup = map(model.scaleup_fns['econ_program_reflectioncost_ipt'], x_vals)
            unitcost = map(model.scaleup_fns['econ_program_unitcost_ipt'], x_vals)
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            coverage_mid = coverage
            for i in numpy.arange(0, len(x_vals), 1):
                all_flows = model.var_array[int(i)]
                for a, b in enumerate(model.var_labels):
                    if b == 'ipt_commencements':
                        #pop = all_flows[a] #actually vaccinated
                        pop = all_flows[a]
                        popsize.append(pop)
                        a = saturation / (1 - 2**alpha)
                        b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                        b_growth_rate.append(b)
                        cost_uninflated.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
                        cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])
                        current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))
                        cost_predefined.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))


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

                        coverage_estimated.append(get_coverage_from_cost(a,
                                                                         saturation,
                                                                         b_growth_rate[int(i)],
                                                                         cost_uninflated[int(i)],
                                                                         reflection_cost_scaleup[int(i)],
                                                                         alpha))

            for index_element, replacement_element in zip (indexes, replacements):
                cost_predefined [index_element] = replacement_element
            b_growth_rate_2 = []
            coverage_new_ipt = []
            for i in numpy.arange (0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate_2.append(b)
                coverage_new_ipt.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate_2[int(i)],
                                                            cost_predefined[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))
                coverage_new_ipt_nan = numpy.nan_to_num(coverage_new_ipt)

            rb = open_workbook('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')
            book = copy(rb)
            sheet1 = book.get_sheet(0)
            sheet1.write(0, 2, 'new_ipt_cov')
            i=0
            for n in coverage_new_ipt_nan:
                i = i+1
                sheet1.write(i, 2, n)
            book.save('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')

            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    a = saturation / (1 - 2**alpha)
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(reflection_cost_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[year_pos],
                                                                          saturation,
                                                                          a,
                                                                          coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
            '''
            data_to_plot = model.scaleup_data[function]
            fig = plt.figure('IPT')
            ax1 = fig.add_subplot(221)
            ax1.plot(cost_uninflated_toplotcostcurve, coverage_values, 'b-', linewidth = 3)
            ax1.set_title('IPT cost-coverage curve ' + str(year_index))
            ax1.set_xlabel('Cost (USD)')
            ax1.set_ylabel('Coverage (%)')
            ax1.set_ylim([0, 1.1])
            ax1.grid(True)

            ax2 = fig.add_subplot(222)
            ax2.plot(x_vals, cost_uninflated, 'r-', linewidth = 3, label = 'Cost uninflated')
            ax2.plot(x_vals, cost_inflated, 'r--', linewidth = 3, label = 'Cost inflated')
            ax2.plot(x_vals_2015onwards_array, cost_discounted_array, 'b--', linewidth = 3, label = 'Cost discounted')
            ax2.set_title('IPT yearly total cost (USD)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Cost (USD)')
            ax2.set_xlim([2009, end_time])
            ax2.legend(loc = 'upper right')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(x_vals, popsize, 'r-', linewidth = 3)
            ax3.set_title('IPT population size')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('People')
            ax3.set_xlim([2009, end_time])
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(x_vals, coverage, 'r-', linewidth =3)
            ax4.plot(x_vals, coverage_estimated, 'b-', linewidth = 3)
            ax4.scatter(data_to_plot.keys(), data_to_plot.values())
            ax4.set_title('IPT coverage')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Coverage (%)')
            ax4.set_ylim([0, 1.1])
            ax4.grid(True)
            plt.show()
            '''

#######################################################################################
###  Xpert
#######################################################################################

        #elif function == str('program_prop_vaccination'): #using data from data_fiji
        elif function == str('econ_program_prop_xpert'):
            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []
            b_growth_rate = []
            coverage_estimated = []
            cost_predefined = []

            indexes = numpy.arange(12001, 17002, 1)
            replacements = numpy.linspace(200000, 300000, 5001)

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_xpert'], x_vals)
            reflection_cost_scaleup = map(model.scaleup_fns['econ_program_reflectioncost_xpert'], x_vals)
            unitcost = map(model.scaleup_fns['econ_program_unitcost_xpert'], x_vals)
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            coverage_mid = coverage
            active_all = model.compartment_soln['active_extrapul_age0to5'] + \
                        model.compartment_soln['active_extrapul_age15up'] + \
                        model.compartment_soln['active_extrapul_age5to15'] + \
                        model.compartment_soln['active_smearneg_age0to5'] + \
                        model.compartment_soln['active_smearneg_age15up'] + \
                        model.compartment_soln['active_smearneg_age5to15'] + \
                        model.compartment_soln['active_smearpos_age0to5'] + \
                        model.compartment_soln['active_smearpos_age15up'] + \
                        model.compartment_soln['active_smearpos_age5to15']

            for i in numpy.arange(0, len(x_vals), 1):
                all_flows = model.var_array[int(i)]
                program_rate_detect = all_flows[52]
                program_rate_missed = all_flows[12]
                presenting_for_care_rate = program_rate_detect + program_rate_missed
                presentation_per_time =  presenting_for_care_rate * active_all[int(i)]
                pop = presentation_per_time
                popsize.append(pop)
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate.append(b)
                cost_uninflated.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
                cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])
                current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))
                cost_predefined.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))

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

                coverage_estimated.append(get_coverage_from_cost(a,
                                                                 saturation,
                                                                 b_growth_rate[int(i)],
                                                                 cost_uninflated[int(i)],
                                                                 reflection_cost_scaleup[int(i)],
                                                                 alpha))

            for index_element, replacement_element in zip (indexes, replacements):
                cost_predefined [index_element] = replacement_element
            b_growth_rate_2 = []
            coverage_new_xpert = []
            for i in numpy.arange (0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate_2.append(b)
                coverage_new_xpert.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate_2[int(i)],
                                                            cost_predefined[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))
                coverage_new_xpert_nan = numpy.nan_to_num(coverage_new_xpert)

            rb = open_workbook('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')
            book = copy(rb)
            sheet1 = book.get_sheet(0)
            sheet1.write(0, 3, 'new_xpert_cov')
            i=0
            for n in coverage_new_xpert_nan:
                i = i+1
                sheet1.write(i, 3, n)
            book.save('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')

            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    a = saturation / (1 - 2**alpha)
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(reflection_cost_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[year_pos],
                                                                          saturation,
                                                                          a,
                                                                          coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
            '''
            data_to_plot = model.scaleup_data[function]
            fig = plt.figure('XPERT')
            ax1 = fig.add_subplot(221)
            ax1.plot(cost_uninflated_toplotcostcurve, coverage_values, 'b-', linewidth = 3)
            ax1.set_title('XPERT cost-coverage curve ' + str(year_index))
            ax1.set_xlabel('Cost (USD)')
            ax1.set_ylabel('Coverage (%)')
            ax1.set_ylim([0, 1.1])
            ax1.grid(True)

            ax2 = fig.add_subplot(222)
            ax2.plot(x_vals, cost_uninflated, 'r-', linewidth = 3, label = 'Cost uninflated')
            ax2.plot(x_vals, cost_inflated, 'r--', linewidth = 3, label = 'Cost inflated')
            ax2.plot(x_vals_2015onwards_array, cost_discounted_array, 'b--', linewidth = 3, label = 'Cost discounted')
            ax2.set_title('XPERT yearly total cost (USD)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Cost (USD)')
            ax2.set_xlim([2012, end_time])
            ax2.legend(loc = 'upper right')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(x_vals, popsize, 'r-', linewidth = 3)
            ax3.set_title('XPERT population size')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('People')
            ax3.set_xlim([start_time, end_time])
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(x_vals, coverage, 'r-', linewidth =3)
            ax4.plot(x_vals, coverage_estimated, 'b-', linewidth = 3)
            ax4.scatter(data_to_plot.keys(), data_to_plot.values())
            ax4.set_title('XPERT coverage')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Coverage (%)')
            ax4.set_ylim([0, 1.1])
            ax4.grid(True)
            plt.show()
            '''

#######################################################################################
### TREATMENT SUPPORT
#######################################################################################

        elif function == str('econ_program_prop_treatment_support'):
            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []
            b_growth_rate = []
            coverage_estimated = []
            cost_predefined = []

            indexes = numpy.arange(12001, 17002, 1)
            replacements = numpy.linspace(50000, 80000, 5001)

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_treatment_support'], x_vals)
            reflection_cost_scaleup = map(model.scaleup_fns['econ_program_reflectioncost_treatment_support'], x_vals)
            unitcost = map(model.scaleup_fns['econ_program_unitcost_treatment_support'], x_vals)
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            coverage_mid = coverage
            treatment_all = model.compartment_soln['treatment_infect_extrapul_age0to5'] + \
                            model.compartment_soln['treatment_infect_extrapul_age15up'] + \
                            model.compartment_soln['treatment_infect_extrapul_age5to15'] + \
                            model.compartment_soln['treatment_infect_smearneg_age0to5'] + \
                            model.compartment_soln['treatment_infect_smearneg_age15up'] + \
                            model.compartment_soln['treatment_infect_smearneg_age5to15'] + \
                            model.compartment_soln['treatment_infect_smearpos_age0to5'] + \
                            model.compartment_soln['treatment_infect_smearpos_age15up'] + \
                            model.compartment_soln['treatment_infect_smearpos_age5to15'] + \
                            model.compartment_soln['treatment_noninfect_extrapul_age0to5'] + \
                            model.compartment_soln['treatment_noninfect_extrapul_age15up'] + \
                            model.compartment_soln['treatment_noninfect_extrapul_age5to15'] + \
                            model.compartment_soln['treatment_noninfect_smearneg_age0to5'] + \
                            model.compartment_soln['treatment_noninfect_smearneg_age15up'] + \
                            model.compartment_soln['treatment_noninfect_smearneg_age5to15'] + \
                            model.compartment_soln['treatment_noninfect_smearpos_age0to5'] + \
                            model.compartment_soln['treatment_noninfect_smearpos_age15up'] + \
                            model.compartment_soln['treatment_noninfect_smearpos_age5to15']
            popsize = treatment_all
            for i in numpy.arange(0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate.append(b)
                cost_uninflated.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
                cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])
                current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))

                cost_predefined.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
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

                coverage_estimated.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate[int(i)],
                                                            cost_uninflated[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))

            for index_element, replacement_element in zip (indexes, replacements):
                cost_predefined [index_element] = replacement_element
            b_growth_rate_2 = []

            coverage_new_treatment_support = []
            for i in numpy.arange (0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate_2.append(b)
                coverage_new_treatment_support.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate_2[int(i)],
                                                            cost_predefined[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))
                coverage_new_treatment_support_nan = numpy.nan_to_num(coverage_new_treatment_support)
            rb = open_workbook('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')
            book = copy(rb)
            sheet1 = book.get_sheet(0)
            sheet1.write(0, 4, 'new_treatment_support_cov')
            i=0
            for n in coverage_new_treatment_support_nan:
                i = i+1
                sheet1.write(i, 4, n)
            book.save('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')


            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    a = saturation / (1 - 2**alpha)
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(reflection_cost_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[year_pos],
                                                                          saturation,
                                                                          a,
                                                                          coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))


            '''
            data_to_plot = model.scaleup_data[function]
            fig = plt.figure('TREATMENT SUPPORT')
            ax1 = fig.add_subplot(221)
            ax1.plot(cost_uninflated_toplotcostcurve, coverage_values, 'b-', linewidth = 3)
            ax1.set_title('TREATMENT SUPPORT cost-coverage curve ' + str(year_index))
            ax1.set_xlabel('Cost (USD)')
            ax1.set_ylabel('Coverage (%)')
            ax1.set_ylim([0, 1.1])
            ax1.grid(True)

            ax2 = fig.add_subplot(222)
            ax2.plot(x_vals, cost_uninflated, 'r-', linewidth = 3, label = 'Cost uninflated')
            ax2.plot(x_vals, cost_inflated, 'r--', linewidth = 3, label = 'Cost inflated')
            ax2.plot(x_vals_2015onwards_array, cost_discounted_array, 'b--', linewidth = 3, label = 'Cost discounted')
            ax2.set_title('TREATMENT SUPPORT yearly total cost (USD)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Cost (USD)')
            ax2.set_xlim([2014, end_time])
            ax2.legend(loc = 'upper right')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(x_vals, popsize, 'r-', linewidth = 3)
            ax3.set_title('TREATMENT SUPPORT population size')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('People')
            ax3.set_xlim([start_time, end_time])
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(x_vals, coverage, 'r-', linewidth =3)
            ax4.plot(x_vals, coverage_estimated, 'b-', linewidth = 3)
            ax4.scatter(data_to_plot.keys(), data_to_plot.values())
            ax4.set_title('TREATMENT SUPPORT coverage')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Coverage (%)')
            ax4.set_ylim([0, 1.1])
            ax4.grid(True)
            plt.show()
            '''

############################
#SMEAR BASED ACF
###########################

        elif function == str('econ_program_prop_smearacf'):
            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []
            b_growth_rate = []
            coverage_estimated = []
            cost_predefined = []

            indexes = numpy.arange(12001, 17002, 1)
            replacements = numpy.linspace(500000, 800000, 5001)

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_smearacf'], x_vals)
            reflection_cost_scaleup = map(model.scaleup_fns['econ_program_reflectioncost_smearacf'], x_vals)
            unitcost = map(model.scaleup_fns['econ_program_unitcost_smearacf'], x_vals)
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            coverage_mid = coverage

            popsize =   model.compartment_soln['active_smearpos_age0to5'] + \
                        model.compartment_soln['active_smearpos_age15up'] + \
                        model.compartment_soln['active_smearpos_age5to15']

            for i in numpy.arange(0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate.append(b)
                cost_uninflated.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
                cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])
                current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))

                cost_predefined.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
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

                coverage_estimated.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate[int(i)],
                                                            cost_uninflated[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))

            for index_element, replacement_element in zip (indexes, replacements):
                cost_predefined [index_element] = replacement_element
            b_growth_rate_2 = []

            coverage_new_smearacf = []
            for i in numpy.arange (0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate_2.append(b)
                coverage_new_smearacf.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate_2[int(i)],
                                                            cost_predefined[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))
                coverage_new_smearacf_nan = numpy.nan_to_num(coverage_new_smearacf)
            rb = open_workbook('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')
            book = copy(rb)
            sheet1 = book.get_sheet(0)
            sheet1.write(0, 5, 'new_smearacf_cov')
            i=0
            for n in coverage_new_smearacf_nan:
                i = i+1
                sheet1.write(i, 5, n)
            book.save('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')

            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    a = saturation / (1 - 2**alpha)
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(reflection_cost_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[year_pos],
                                                                          saturation,
                                                                          a,
                                                                          coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))

            '''
            data_to_plot = model.scaleup_data[function]
            fig = plt.figure('SMEAR ACF')
            ax1 = fig.add_subplot(221)
            ax1.plot(cost_uninflated_toplotcostcurve, coverage_values, 'b-', linewidth = 3)
            ax1.set_title('SMEAR ACF cost-coverage curve ' + str(year_index))
            ax1.set_xlabel('Cost (USD)')
            ax1.set_ylabel('Coverage (%)')
            ax1.set_ylim([0, 1.1])
            ax1.grid(True)

            ax2 = fig.add_subplot(222)
            ax2.plot(x_vals, cost_uninflated, 'r-', linewidth = 3, label = 'Cost uninflated')
            ax2.plot(x_vals, cost_inflated, 'r--', linewidth = 3, label = 'Cost inflated')
            ax2.plot(x_vals_2015onwards_array, cost_discounted_array, 'b--', linewidth = 3, label = 'Cost discounted')
            ax2.set_title('SMEAR ACF yearly total cost (USD)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Cost (USD)')
            ax2.set_xlim([2014, end_time])
            ax2.legend(loc = 'upper right')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(x_vals, popsize, 'r-', linewidth = 3)
            ax3.set_title('SMEAR ACF population size')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('People')
            ax3.set_xlim([start_time, end_time])
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(x_vals, coverage, 'r-', linewidth =3)
            ax4.plot(x_vals, coverage_estimated, 'b-', linewidth = 3)
            ax4.scatter(data_to_plot.keys(), data_to_plot.values())
            ax4.set_title('SMEAR ACF coverage')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Coverage (%)')
            ax4.set_ylim([0, 1.1])
            ax4.grid(True)
            plt.show()
            '''

############################
#XPERT ACF
###########################

        elif function == str('econ_program_prop_xpertacf'):
            cost_uninflated = []
            cost_uninflated_toplotcostcurve = []
            cost_inflated = []
            popsize = []
            x_vals_2015onwards_array =[]
            cost_discounted_array = []
            b_growth_rate = []
            coverage_estimated = []
            cost_predefined = []

            indexes = numpy.arange(12001, 17002, 1)
            replacements = numpy.linspace(500000, 800000, 5001)

            scaleup_param_vals = map(model.scaleup_fns[function], x_vals)
            funding_scaleup = map(model.scaleup_fns['econ_program_totalcost_xpertacf'], x_vals)
            reflection_cost_scaleup = map(model.scaleup_fns['econ_program_reflectioncost_xpertacf'], x_vals)
            unitcost = map(model.scaleup_fns['econ_program_unitcost_xpertacf'], x_vals)
            coverage = get_coverage_from_outcome_program_as_param(scaleup_param_vals)
            coverage_mid = coverage

            popsize =   model.compartment_soln['active_smearpos_age0to5'] + \
                        model.compartment_soln['active_smearpos_age15up'] + \
                        model.compartment_soln['active_smearpos_age5to15'] + \
                        model.compartment_soln['active_smearneg_age0to5'] + \
                        model.compartment_soln['active_smearneg_age15up'] + \
                        model.compartment_soln['active_smearneg_age5to15']

            for i in numpy.arange(0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate.append(b)
                cost_uninflated.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
                cost_inflated.append(cost_uninflated[int(i)] * cpi[int(year_current)] / cpi_scaleup[int(i)])
                current_year_pos = ((year_current - start_time) / ((end_time - start_time) / len(model.times)))

                cost_predefined.append(get_cost_from_coverage(reflection_cost_scaleup[int(i)], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[int(i)],
                                                                          saturation,
                                                                          a,
                                                                          coverage[int(i)], #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))

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

                coverage_estimated.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate[int(i)],
                                                            cost_uninflated[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))

            for index_element, replacement_element in zip (indexes, replacements):
                cost_predefined [index_element] = replacement_element
            b_growth_rate_2 = []

            coverage_new_xpertacf = []
            for i in numpy.arange (0, len(x_vals), 1):
                a = saturation / (1 - 2**alpha)
                b = ((2**(alpha + 1)) / (alpha * (saturation - a) * unitcost[int(i)] * popsize[int(i)]))
                b_growth_rate_2.append(b)
                coverage_new_xpertacf.append(get_coverage_from_cost(a,
                                                            saturation,
                                                            b_growth_rate_2[int(i)],
                                                            cost_predefined[int(i)],
                                                            reflection_cost_scaleup[int(i)],
                                                            alpha))
                coverage_new_xpertacf_nan = numpy.nan_to_num(coverage_new_xpertacf)

            rb = open_workbook('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')
            book = copy(rb)
            sheet1 = book.get_sheet(0)
            sheet1.write(0, 6, 'new_xpertacf_cov')
            i=0
            for n in coverage_new_xpertacf_nan:
                i = i+1
                sheet1.write(i, 6, n)
            book.save('C:/Users/ntdoan/Github/AuTuMN/autumn/xls/new_bcg_cov_fiji.xls')

            if plot_costcurve is True:
                for coverage_range in coverage_values:
                    a = saturation / (1 - 2**alpha)
                    cost_uninflated_toplotcostcurve.append(get_cost_from_coverage(reflection_cost_scaleup[year_pos], #Add [year_pos] to get cost-coverage curve at that year,
                                                                          b_growth_rate[year_pos],
                                                                          saturation,
                                                                          a,
                                                                          coverage_range, #Add [year_pos] to get cost-coverage curve at that year
                                                                          alpha))
            '''
            data_to_plot = model.scaleup_data[function]
            fig = plt.figure('XPERT ACF')
            ax1 = fig.add_subplot(221)
            ax1.plot(cost_uninflated_toplotcostcurve, coverage_values, 'b-', linewidth = 3)
            ax1.set_title('XPERT ACF cost-coverage curve ' + str(year_index))
            ax1.set_xlabel('Cost (USD)')
            ax1.set_ylabel('Coverage (%)')
            ax1.set_ylim([0, 1.1])
            ax1.grid(True)

            ax2 = fig.add_subplot(222)
            ax2.plot(x_vals, cost_uninflated, 'r-', linewidth = 3, label = 'Cost uninflated')
            ax2.plot(x_vals, cost_inflated, 'r--', linewidth = 3, label = 'Cost inflated')
            ax2.plot(x_vals_2015onwards_array, cost_discounted_array, 'b--', linewidth = 3, label = 'Cost discounted')
            ax2.set_title('XPERT yearly total cost (USD)')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Cost (USD)')
            ax2.set_xlim([2014, end_time])
            ax2.legend(loc = 'upper right')
            ax2.grid(True)

            ax3 = fig.add_subplot(223)
            ax3.plot(x_vals, popsize, 'r-', linewidth = 3)
            ax3.set_title('XPERT ACF population size')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('People')
            ax3.set_xlim([start_time, end_time])
            ax3.grid(True)

            ax4 = fig.add_subplot(224)
            ax4.plot(x_vals, coverage, 'r-', linewidth =3)
            ax4.plot(x_vals, coverage_estimated, 'b-', linewidth = 3)
            ax4.scatter(data_to_plot.keys(), data_to_plot.values())
            ax4.set_title('XPERT ACF coverage')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Coverage (%)')
            ax4.set_ylim([0, 1.1])
            ax4.grid(True)
            plt.show()
            '''
    #plt.plot(x_vals, coverage_new_treatment_support, 'r-', linewidth = 3)
    #plt.show()



    '''
    plt.plot(x_vals, coverage_new_bcg, 'r-', linewidth = 3, label = "New BCG coverage")
    plt.plot(x_vals, coverage_new_ipt, 'b-', linewidth = 3, label = 'New IPT coverage')
    plt.plot(x_vals, coverage_new_xpert, 'k-', linewidth = 3, label = 'New XPERT coverage')
    plt.plot(x_vals, coverage_new_treatment_support, 'y-', linewidth = 3, label = 'New TREATMENT SUPPORT coverage')
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.show()
    '''

########## RUNNING CODE #############

scenario = None
n_organs = inputs.model_constants['n_organs'][0]
n_strains = inputs.model_constants['n_strains'][0]
is_quality = inputs.model_constants['is_lowquality'][0]
is_amplification = inputs.model_constants['is_amplification'][0]
is_misassignment = inputs.model_constants['is_misassignment'][0]
is_amplification = inputs.model_constants['is_amplification'][0]
is_misassignment = inputs.model_constants['is_misassignment'][0]

model = autumn.model.ConsolidatedModel(
    scenario,  # Scenario to run
    inputs)

print(str(n_organs) + " organ(s),   " +
      str(n_strains) + " strain(s),   " +
      "Low quality? " + str(is_quality) + ",   " +
      "Amplification? " + str(is_amplification) + ",   " +
      "Misassignment? " + str(is_misassignment) + ".")

model.integrate()


# Classify scale-up functions for plotting
classified_scaleups = {'program_prop': [],
                       'program_other': [],
                       'birth': [],
                       'non_program': []}
for fn in model.scaleup_fns:
    if 'program_prop' in fn:
        classified_scaleups['program_prop'] += [fn]
    elif 'program' in fn:
        classified_scaleups['program_other'] += [fn]
    elif 'demo_rate_birth' in fn:
        classified_scaleups['birth'] += [fn]
    else:
        classified_scaleups['non_program'] += [fn]

# Plot them from the start of the model and from "recent_time"
for i, classification in enumerate(classified_scaleups):
   cost_scaleup_fns(model,
                    classified_scaleups[classification],
                    'start_time',
                    'scenario_end_time',
                    classification,
                    model.inputs.country)

