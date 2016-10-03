from numpy import isfinite
import tool_kit
import model
import os
import data_processing
import numpy
import datetime
from scipy.stats import norm, beta
from Tkinter import *
from scipy.optimize import minimize
from random import uniform
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import outputs


def is_positive_definite(v):

    return isfinite(v) and v > 0.


def generate_candidates(n_candidates, param_ranges_unc):

    """
    Function for generating candidate parameters

    """

    # Dictionary for storing candidates
    param_candidates = {}
    for param_dict in param_ranges_unc:

        # Find bounds of parameter
        bound_low, bound_high = param_dict['bounds'][0], param_dict['bounds'][1]

        # Draw from distribution
        if param_dict['distribution'] == 'beta':
            x = numpy.random.beta(2., 2., n_candidates)
            x = bound_low + x * (bound_high - bound_low)
        elif param_dict['distribution'] == 'uniform':
            x = numpy.random.uniform(bound_low, bound_high, n_candidates)

            # Return values
        param_candidates[param_dict['key']] = x
    return param_candidates


class ModelRunner:

    def __init__(self, gui_inputs, runtime_outputs, figure_frame):

        self.gui_inputs = gui_inputs
        self.runtime_outputs = runtime_outputs
        self.figure_frame = figure_frame
        self.inputs = data_processing.Inputs(gui_inputs, runtime_outputs, from_test=True)
        self.inputs.read_and_load_data()
        self.model_dict = {}
        self.is_last_run_success = False
        self.interventions_to_cost = ['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf',
                                      'ipt_age0to5', 'ipt_age5to15', 'decentralisation']
        self.loglikelihoods = []
        self.outputs_unc = [{'key': 'incidence',
                             'posterior_width': None,
                             'width_multiplier': 2.  # for incidence for ex. Width of Normal posterior relative to CI width in data
                             }]
        self.results = {}
        self.all_parameters_tried = {}
        self.whether_accepted_list = []
        self.accepted_indices = []
        self.results['scenarios'] = {}
        self.solns_for_extraction = ['compartment_soln', 'fraction_soln']
        self.arrays_for_extraction = ['flow_array', 'fraction_array', 'soln_array', 'var_array']
        self.optimization = False
        self.total_funding = 1e7

    def master_runner(self):

        for scenario in self.gui_inputs['scenarios_to_run']:

            # Name and initialise model
            scenario_name = tool_kit.find_scenario_string_from_number(scenario)
            self.model_dict[scenario_name] = model.ConsolidatedModel(scenario, self.inputs, self.gui_inputs)

            # Sort out times for scenario runs
            if scenario is None:
                self.model_dict[scenario_name].start_time = self.inputs.model_constants['start_time']
            else:
                scenario_start_time_index = \
                    self.model_dict['baseline'].find_time_index(self.inputs.model_constants['recent_time'])
                self.model_dict[scenario_name].start_time = \
                    self.model_dict['baseline'].times[scenario_start_time_index]
                self.model_dict[scenario_name].loaded_compartments = \
                    self.model_dict['baseline'].load_state(scenario_start_time_index)

            # Describe model
            self.add_comment_to_gui_window('Running ' + scenario_name + ' conditions for ' + self.gui_inputs['country']
                                           + ' using point estimates for parameters.')

            # Integrate and add result to outputs object
            self.model_dict[scenario_name].integrate()

            # Store
            self.store_scenario_results(scenario_name)

        if self.gui_inputs['output_uncertainty']:

            # Describe process
            self.add_comment_to_gui_window('Uncertainty analysis commenced')

            # Prepare directory for eventual pickling
            out_dir = 'pickles'
            if not os.path.isdir(out_dir):
                os.makedirs(out_dir)
            results_file = os.path.join(out_dir, 'results_uncertainty.pkl')
            indices_file = os.path.join(out_dir, 'indices_uncertainty.pkl')

            # Don't run uncertainty but load a saved simulation
            if self.gui_inputs['pickle_uncertainty'] == 'Load':
                self.add_comment_to_gui_window('Uncertainty results loaded from previous simulation')
                self.results['uncertainty'] = tool_kit.pickle_load(results_file)
                self.accepted_indices = tool_kit.pickle_load(indices_file)

            # Run uncertainty
            else:
                self.run_uncertainty()

            # Write uncertainty if requested
            if self.gui_inputs['pickle_uncertainty'] == 'Save':
                tool_kit.pickle_save(self.results['uncertainty'], results_file)
                tool_kit.pickle_save(self.accepted_indices, indices_file)
                self.add_comment_to_gui_window('Uncertainty results saved to disc')

        if self.optimization:
            self.run_optimization()

    def store_scenario_results(self, scenario):

        """
        This method is designed to store all the results that will be needed for later analysis in separate
        attributes to the individual models, to avoid them being over-written during the uncertainty process.
        Args:
            scenario: The name of the model run.

        """

        self.results['scenarios'][scenario] = {}

        self.results['scenarios'][scenario]['compartment_soln'] \
            = self.model_dict[scenario].compartment_soln
        self.results['scenarios'][scenario]['costs'] \
            = self.model_dict[scenario].costs
        self.results['scenarios'][scenario]['flow_array'] \
            = self.model_dict[scenario].flow_array
        self.results['scenarios'][scenario]['fraction_array'] \
            = self.model_dict[scenario].fraction_array
        self.results['scenarios'][scenario]['fraction_soln'] \
            = self.model_dict[scenario].fraction_soln
        self.results['scenarios'][scenario]['soln_array'] \
            = self.model_dict[scenario].soln_array
        self.results['scenarios'][scenario]['var_array'] \
            = self.model_dict[scenario].var_array

    def run_uncertainty(self):

        """
        Main method to run all the uncertainty processes.

        """

        # If not doing an adaptive search, only need to start with a single parameter set
        if self.gui_inputs['adaptive_uncertainty']:
            n_candidates = 1
        else:
            n_candidates = self.gui_inputs['uncertainty_runs'] * 10

        # Define an initial set of parameter candidates only
        param_candidates = generate_candidates(n_candidates, self.inputs.param_ranges_unc)
        normal_char = self.get_normal_char()

        # Prepare for uncertainty loop
        for param_dict in self.inputs.param_ranges_unc:
            self.all_parameters_tried[param_dict['key']] = []
        n_accepted = 0
        i_candidates = 0
        run = 0
        prev_log_likelihood = -1e10
        params = []
        self.results['uncertainty'] = {}
        self.prepare_uncertainty_dictionaries('baseline')

        # Until a sufficient number of parameters are accepted
        while n_accepted < self.gui_inputs['uncertainty_runs'] + self.gui_inputs['burn_in_runs']:

            # Set timer
            start_timer_run = datetime.datetime.now()

            # Update parameters
            new_params = []
            if self.gui_inputs['adaptive_uncertainty']:
                if i_candidates == 0:
                    new_params = []
                    for param_dict in self.inputs.param_ranges_unc:
                        new_params.append(param_candidates[param_dict['key']][run])
                        params.append(param_candidates[param_dict['key']][run])
                else:
                    new_params = self.update_params(params)
            else:
                for param_dict in self.inputs.param_ranges_unc:
                    new_params.append(param_candidates[param_dict['key']][run])

            # Run the integration
            # (includes checking parameters, setting parameters and recording success/failure of run)
            self.run_with_params(new_params)

            # Now storing regardless of acceptance
            if self.is_last_run_success:
                self.store_uncertainty_results('baseline')

                # Calculate prior
                prior_log_likelihood = 0.
                for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                    param_val = new_params[p]
                    self.all_parameters_tried[param_dict['key']].append(new_params[p])

                    # Calculate the density of param_val
                    bound_low, bound_high = param_dict['bounds'][0], param_dict['bounds'][1]

                    # Normalise value and find log of PDF from beta distribution
                    if param_dict['distribution'] == 'beta':
                        prior_log_likelihood \
                            += beta.logpdf((param_val - bound_low) / (bound_high - bound_low),
                                           2., 2.)

                    # Find log of PDF from uniform distribution
                    elif param_dict['distribution'] == 'uniform':
                        prior_log_likelihood \
                            += numpy.log(1. / (bound_high - bound_low))

                # Calculate posterior
                posterior_log_likelihood = 0.
                for output_dict in self.outputs_unc:
                    working_output_dictionary = normal_char[output_dict['key']]
                    for year in working_output_dictionary.keys():
                        year_index \
                            = tool_kit.find_first_list_element_at_least_value(self.model_dict['baseline'].times,
                                                                              year)
                        model_result_for_output \
                            = self.model_dict['baseline'].get_var_soln(output_dict['key'])[year_index]
                        mu, sd = working_output_dictionary[year][0], working_output_dictionary[year][1]
                        posterior_log_likelihood += norm.logpdf(model_result_for_output, mu, sd)

                # Sum for overall likelihood of run
                log_likelihood = prior_log_likelihood + posterior_log_likelihood

                # Determine acceptance
                if log_likelihood >= prev_log_likelihood:
                    accepted = 1
                else:
                    accepted = numpy.random.binomial(n=1, p=numpy.exp(log_likelihood - prev_log_likelihood))

                # Record information for accepted runs
                if accepted != 1:
                    self.whether_accepted_list.append(False)
                elif accepted == 1:
                    self.whether_accepted_list.append(True)
                    self.accepted_indices += [run]
                    n_accepted += 1

                    # Update likelihood and parameter set for next run
                    prev_log_likelihood = log_likelihood
                    params = new_params

                    # Store outputs once burn-in complete
                    if n_accepted > self.gui_inputs['burn_in_runs']:

                        # Model storage
                        params_dict = {}
                        for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                            params_dict[param_dict['key']] = new_params[p]
                        self.loglikelihoods.append(log_likelihood)

                        # Run scenarios other than baseline and store uncertainty
                        for scenario in self.gui_inputs['scenarios_to_run']:
                            scenario_name = tool_kit.find_scenario_string_from_number(scenario)
                            if scenario is not None:
                                scenario_start_time_index = \
                                    self.model_dict['baseline'].find_time_index(
                                        self.inputs.model_constants['recent_time'])
                                self.model_dict[scenario_name].start_time = \
                                    self.model_dict['baseline'].times[scenario_start_time_index]
                                self.model_dict[scenario_name].loaded_compartments = \
                                    self.model_dict['baseline'].load_state(scenario_start_time_index)
                                self.model_dict[scenario_name].integrate()

                                self.prepare_uncertainty_dictionaries(scenario_name)
                                self.store_uncertainty_results(scenario_name)

                i_candidates += 1
                run += 1

            self.plot_progressive_parameters(run)

            # Generate more candidates if required
            if not self.gui_inputs['adaptive_uncertainty'] and run >= len(param_candidates.keys()):
                param_candidates = generate_candidates(n_candidates, self.inputs.param_ranges_unc)
                run = 0
            self.add_comment_to_gui_window(str(n_accepted) + ' accepted / ' + str(i_candidates) +
                                           ' candidates. Running time: '
                                           + str(datetime.datetime.now() - start_timer_run))

    def set_model_with_params(self, param_dict):

        """
        Populates baseline model with params from uncertainty calculations.

        Args:
            param_dict: Dictionary of the parameters to be set within the model (keys parameter name strings and values
                parameter values).

        """

        n_set = 0
        for key in param_dict:
            if key in self.model_dict['baseline'].params:
                n_set += 1
                self.model_dict['baseline'].set_parameter(key, param_dict[key])
            else:
                raise ValueError("%s not in model params" % key)

    def convert_param_list_to_dict(self, params):

        """
        Extract parameters from list into dictionary that can be used for setting in the model
        through the set_model_with_params method.

        Args:
            params: The parameter names for extraction.

        Returns:
            param_dict: The dictionary returned in appropriate format.

        """

        param_dict = {}

        for names, vals in zip(self.inputs.param_ranges_unc, params):
            param_dict[names['key']] = vals

        return param_dict

    def get_normal_char(self):

        """
        Define the characteristics (mean and standard deviation) of the normal distribution for model outputs
        (incidence, mortality).

        Returns:
            normal_char: Dictionary with keys outputs and values dictionaries. Sub-dictionaries have keys years
                and values lists, with first element of list means and second standard deviations.

        """

        # Dictionary storing the characteristics of the normal distributions
        normal_char = {}
        for output_dict in self.inputs.outputs_unc:
            normal_char[output_dict['key']] = {}

            # Mortality
            if output_dict['key'] == 'mortality':
                sd = output_dict['posterior_width'] / (2.0 * 1.96)
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    mu = self.inputs.data_to_fit[output_dict['key']][year]
                    normal_char[output_dict['key']][year] = [mu, sd]

            # Incidence
            elif output_dict['key'] == 'incidence':
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    low = self.inputs.data_to_fit['incidence_low'][year]
                    high = self.inputs.data_to_fit['incidence_high'][year]
                    sd = output_dict['width_multiplier'] * (high - low) / (2.0 * 1.96)
                    mu = (high + low) / 2.
                    normal_char[output_dict['key']][year] = [mu, sd]

        return normal_char

    def update_params(self, old_params):

        """
        Update all the parameter values being used in the uncertainty analysis.

        Args:
            old_params:

        Returns:
            new_params: The new parameters to be used in the next model run.

        """

        new_params = []

        # Iterate through the parameters being used
        for p, param_dict in enumerate(self.inputs.param_ranges_unc):
            bounds = param_dict['bounds']
            sd = self.gui_inputs['search_width'] * (bounds[1] - bounds[0]) / (2.0 * 1.96)
            random = -100.

            # Search for new parameters
            while random < bounds[0] or random > bounds[1]:
                random = norm.rvs(loc=old_params[p], scale=sd, size=1)

            # Add them to the dictionary
            new_params.append(random[0])

        return new_params

    def run_with_params(self, params):

        """
        Integrate the model with the proposed parameter set.

        Args:
            params: The parameters to be set in the model.

        """

        # Check whether parameter values are acceptable
        for p, param in enumerate(params):

            # Whether the parameter value is valid
            if not is_positive_definite(param):
                print 'Warning: parameter%d=%f is invalid for model' % (p, param)
                self.is_last_run_success = False
                return
            bounds = self.inputs.param_ranges_unc[p]['bounds']

            # Whether the parameter value is within acceptable ranges
            if (param < bounds[0]) or (param > bounds[1]):
                # print "Warning: parameter%d=%f is outside of the allowed bounds" % (p, param)
                self.is_last_run_success = False
                return

        param_dict = self.convert_param_list_to_dict(params)

        self.set_model_with_params(param_dict)
        self.is_last_run_success = True
        try:
            self.model_dict['baseline'].integrate()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False

    def prepare_uncertainty_dictionaries(self, scenario):

        """
        Prepare dictionaries for uncertainty results to be populated. There are three different formats to deal with,
        one for "solns", which are dictionary format, one for "arrays", which are numpy ndarray formats and
        "costs", which are dictionaries of lists.

        Args:
            scenario: The scenario currently being run.

        """

        # First dictionary tier declaration
        self.results['uncertainty'][scenario] = {}

        # For dictionary of lists formats
        for attribute in self.solns_for_extraction:
            self.results['uncertainty'][scenario][attribute] = {}

        # For array formats
        for attribute in self.arrays_for_extraction:
            self.results['uncertainty'][scenario][attribute] = []

        # Costs (which have a different structure)
        # self.results['uncertainty'][scenario]['costs'] = {}
        # for program in self.model_dict['baseline'].costs:
        #     if program == 'cost_times':
        #         self.results['uncertainty'][scenario]['costs'][program] = []
        #     else:
        #         self.results['uncertainty'][scenario]['costs'][program] = {}

    def store_uncertainty_results(self, scenario):

        """
        Store the uncertainty results in the dictionaries created in prepare_uncertainty_dictionaries.

        Args:
            scenario: The scenario that has been run.

        """

        # For dictionary of lists formats
        for attribute in self.solns_for_extraction:
            for compartment in self.model_dict[scenario].compartment_soln:
                if compartment in self.results['uncertainty'][scenario][attribute]:
                    self.results['uncertainty'][scenario][attribute][compartment] \
                        = numpy.vstack([self.results['uncertainty'][scenario][attribute][compartment],
                                        getattr(self.model_dict[scenario], attribute)[compartment]])
                else:
                    self.results['uncertainty'][scenario][attribute][compartment] \
                        = getattr(self.model_dict[scenario], attribute)[compartment]

        # For array formats
        for attribute in self.arrays_for_extraction:
            if self.results['uncertainty'][scenario][attribute] == []:
                self.results['uncertainty'][scenario][attribute] \
                    = getattr(self.model_dict[scenario], attribute)
            else:
                self.results['uncertainty'][scenario][attribute] \
                    = numpy.dstack([self.results['uncertainty'][scenario][attribute],
                                    getattr(self.model_dict[scenario], attribute)])

        # # For costs (which have a different structure)
        # for program in self.model_dict[scenario].costs:
        #     if program == 'cost_times':
        #         if self.results['uncertainty'][scenario]['costs'][program] == []:
        #             self.results['uncertainty'][scenario]['costs'][program] \
        #                 = self.model_dict[scenario].costs[program]
        #         else:
        #             self.results['uncertainty'][scenario]['costs'][program] \
        #                 = numpy.vstack([self.results['uncertainty'][scenario]['costs'][program],
        #                                 self.model_dict[scenario].costs[program]])
        #     else:
        #         for cost in self.model_dict[scenario].costs['vaccination']:
        #             if cost in self.results['uncertainty'][scenario]['costs'][program]:
        #                 self.results['uncertainty'][scenario]['costs'][program][cost] \
        #                     = numpy.vstack([self.results['uncertainty'][scenario]['costs'][program][cost],
        #                                     self.model_dict[scenario].costs[program][cost]])
        #             else:
        #                 self.results['uncertainty'][scenario]['costs'][program][cost] \
        #                     = self.model_dict[scenario].costs[program][cost]

    def run_optimization(self):
        print 'Start optimization'

        # Initialise a new model that will be run from 'recent_time' for optimisation
        opti_model_init = model.ConsolidatedModel(None, self.inputs, self.gui_inputs)
        start_time_index = \
            self.model_dict['baseline'].find_time_index(self.inputs.model_constants['recent_time'])
        opti_model_init.start_time = \
            self.model_dict['baseline'].times[start_time_index]
        opti_model_init.loaded_compartments = \
            self.model_dict['baseline'].load_state(start_time_index)

        opti_model_init.eco_drives_epi = True

        nb_int = len(self.model_dict['baseline'].interventions_to_cost)

        # function to minimize: incidence in 2035
        def func(x):
            """
            Args:
                x: defines the resource allocation (as absolute funding over the total period (2015 - 2035))

            Returns:
                predicted incidence for 2035
            """
            i = 0
            for int in self.model_dict['baseline'].interventions_to_cost:
                opti_model_init.available_funding[int] = x[i]*self.total_funding
                i += 1
            opti_model_init.distribute_funding_across_years()
            opti_model_init.integrate()
            print opti_model_init.get_var_soln('incidence')[-1]
            return opti_model_init.get_var_soln('incidence')[-1]

        use_packages = False
        if use_packages:
            # Some initial funding
            x_0 = []
            for i in range(nb_int):
                x_0.append(1./nb_int)

            #######  define constraints   #######
            # Equality constraint:  Sum(x)=Total funding
            cons =[{'type':'ineq',
                   # 'fun' : lambda x: sum(x) - self.total_funding,    # if x is absolute funding
                    'fun': lambda x: 1-sum(x),    # if x is proportion
                    'jac' : lambda x: -numpy.ones(nb_int-1)}]

            # # Each funding must be positive or null
            for i in range(nb_int-1):
                jac_array = numpy.zeros(nb_int-1)
                jac_array[i] = 1.
                cons.append({'type': 'ineq',
                             'fun': lambda x: x[i],
                             'jac': lambda x: jac_array})

            bnds = []
            for int in range(nb_int-1):
                bnds.append((0, None))
            # Ready to run optimization
            res = minimize(func, x_0, constraints=cons, method='SLSQP', options={'disp': True})
        else:
            n_random = 5
            best_x = None
            best_objective = 1e9
            for i in range(n_random):
                x = numpy.zeros(nb_int)
                sum_generated = 0
                for j in range(nb_int-1):
                    x[j] = uniform(0., 1.-sum_generated)
                    sum_generated += x[j]
                x[nb_int-1] = 1. - sum_generated
                objective = func(x)
                print x
                print objective
                if objective < best_objective:
                    best_x = x
                    best_objective = objective

            print 'The best allocation is:'
            print best_x

    def add_comment_to_gui_window(self, comment):

        self.runtime_outputs.insert(END, comment + '\n')
        self.runtime_outputs.see(END)

    def plot_progressive_parameters(self, run):

        # Initialise plotting
        figure = plt.Figure()
        parameter_plots = FigureCanvasTkAgg(figure, master=self.figure_frame)
        subplot_grid = outputs.find_subplot_numbers(len(self.all_parameters_tried))

        # Cycle through parameters with one subplot for each parameter
        for p, param in enumerate(self.all_parameters_tried):
            ax = figure.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)
            ax.plot(range(1, run + 1), self.all_parameters_tried[param])
            ax.set_xlim((0., self.gui_inputs['uncertainty_runs']))

            # Find the y-limits from the parameter bounds and the parameter values tried
            for param_number in range(len(self.inputs.param_ranges_unc)):
                if self.inputs.param_ranges_unc[param_number]['key'] == param:
                    bounds = self.inputs.param_ranges_unc[param_number]['bounds']
            min_ylimit = min(self.all_parameters_tried[param] + [bounds[0]])
            max_ylimit = max(self.all_parameters_tried[param] + [bounds[1]])
            ax.set_ylim((min_ylimit, max_ylimit))

            # Finalise
            parameter_plots.show()
            parameter_plots.draw()
            parameter_plots.get_tk_widget().grid(row=1, column=1)



