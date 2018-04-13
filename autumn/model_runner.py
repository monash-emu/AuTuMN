
# external imports
import os
import numpy
import datetime
from scipy.stats import norm, beta, gamma
from scipy.optimize import minimize
from pyDOE import lhs
import itertools
import copy

# AuTuMN imports
import tool_kit as t_k
import model
import inputs
import economics


''' static functions relevant to model runner only '''


def find_uncertainty_output_weights(output_series, approach, relative_weights=(1., 2.)):
    """
    Creates a set of "weights" to determine the proportion of the log-likelihood to be contributed by the years
    considered in the automatic calibration / uncertainty process.

    Args:
        output_series: List of the points (usually years) that the weights are to be applied to
        approach: Choice of method
        relative_weights: Relative size of the starting and ending weights if method is 1
    """

    # return a single weight of 1.0 in case len(output_series)==1
    if len(output_series) == 1:
        return [1.0]

    # linearly scaling weights summing to one
    if approach == 1:
        weights = numpy.linspace(relative_weights[0], relative_weights[1], num=len(output_series))
        return [i / sum(weights) for i in weights]

    # equally distributed weights summing to one
    elif approach == 2:
        return [1. / float(len(output_series))] * len(output_series)

    # all weights equal to one
    elif approach == 3:
        return [1.] * len(output_series)

    # 0.5 weight for the most recent point. Reminder 0.5 allocated evenly between the other points.
    elif approach == 4:
        old_times_weight = 0.5/(len(output_series) - 1)
        weights = [old_times_weight for i in range(len(output_series) - 1)]
        weights.append(0.5)
        return weights


def find_log_probability_density(distribution, param_val, bounds, additional_params=None):
    """
    Find the log probability density for the parameter value being considered. Uniform is the default distribution if no
    distribution is specified.

    Args:
        distribution: String specifying the general type of distribution (uniform is default)
        param_val: The parameter value
        bounds: Two element list for the upper and lower limits of the distribution
        additional_params: Any additional parameters to the distribution if not completely specified with bounds
    Returns:
        prior_log_likelihood: Prior log likelihood associated with the individual parameter fed in to this function
    """

    # finding the parameter value's distance through the distribution width
    normalised_param_value = (param_val - bounds[0]) / (bounds[1] - bounds[0])

    # find the log probability density
    if distribution == 'beta_2_2':
        prior_log_likelihood = beta.logpdf(normalised_param_value, 2., 2.)
    elif distribution == 'beta_mean_stdev':
        alpha_value = ((1. - additional_params[0]) / additional_params[1] ** 2. - 1. / additional_params[0]) \
                      * additional_params[0] ** 2.
        beta_value = alpha_value * (1. / additional_params[0] - 1.)
        prior_log_likelihood = beta.logpdf(normalised_param_value, alpha_value, beta_value)
    elif distribution == 'beta_params':
        prior_log_likelihood = beta.logpdf(normalised_param_value, additional_params[0], additional_params[1])
    elif distribution == 'gamma_mean_stdev':
        prior_log_likelihood = gamma.logpdf(param_val, (additional_params[0] / additional_params[1]) ** 2.,
                                            scale=additional_params[1] ** 2. / additional_params[0])
    elif distribution == 'gamma_params':
        prior_log_likelihood = gamma.logpdf(param_val, additional_params[0])
    else:
        prior_log_likelihood = numpy.log(1. / (bounds[1] - bounds[0]))
    return prior_log_likelihood


def find_rate_from_aggregate(output_dict, output_string, numerator_string, denominator_string, strata,
                             percentage=False):
    """
    Find any outputs that do not depend upon the strain being evaluated.

    Args:
        output_dict: Output data structure to be updated
        strata: Population strata being evaluated over
        output_string: String for the key to go into the dictionary
        numerator_string: String to index the numerator from the output_dict being evaluated
        denominator_string: String to index the denominator from the output_dict being evaluated
        percentage: Whether to return result as a percentage (default being proportion)
    Returns:
        Updated version of epi_outputs
    """

    # annual risk of infection
    for stratum in strata:
        output_dict[output_string + stratum] = t_k.elementwise_list_division(
            output_dict[numerator_string + stratum], output_dict[denominator_string + stratum], percentage=percentage)
    return output_dict


def solve_by_dichotomy(f, objective, a, b, tolerance):
    """
    Apply the dichotomy method to solve the equation f(x)=objective, x being the unknown variable.
    a and b are initial x values satisfying the following inequation: (f(a) - objective) * (f(b) - objective) < 0.
    The iterative method stops once f(x) is found sufficiently close to the objective, that is when |objective-f(x)|
    <= tolerance. Returns the value of x.
    """

    # check that f(a) - objective) * (f(b) - objective) < 0
    y_a, y_b = f(a), f(b)
    assert (y_a - objective) * (y_b - objective) <= 0.

    # check the order
    assert a < b

    # will be used as a multiplier in various tests that depend on the function monotony (incr. or decr.)
    monotony = -1. if y_a >= objective else 1.

    while min(abs(y_a - objective), abs(y_b - objective)) > tolerance:
        dist_a, dist_b = abs(y_a - objective), abs(y_b - objective)
        # the x update is based on the Thales theorem
        # - that is, if f is linear, the exact solution is obtained in one step
        x = a + (dist_a * b - a * dist_a) / (dist_a + dist_b)
        y = f(x)
        if (y - objective) * monotony >= 0:
            b, y_b = x, y
        else:
            a, y_a = x, y

    if abs(y_a - objective) <= abs(y_b - objective):
        return a
    else:
        return b


''' base class for running models for any disease '''


class ModelRunner:
    def __init__(self, gui_inputs, gui_console_fn=None):
        """
        Instantiation method for model runner - currently including some attributes that should be set externally, e.g.
        in the GUI(s).

        Args:
            gui_inputs: Inputs from the off-line Tkinter GUI
            gui_console_fn: JavaScript GUI inputs
        """

        self.gui_inputs = gui_inputs
        self.inputs = inputs.Inputs(gui_inputs, gui_console_fn=gui_console_fn)
        self.inputs.process_inputs()
        (self.scenarios, self.standard_rate_outputs, self.divide_population, self.epi_outputs_to_analyse, \
          self.interventions_to_cost) \
            = [[] for _ in range(5)]
        (self.models, self.from_labels, self.to_labels, self.multipliers, self.uncertainty_percentiles) \
            = [{} for _ in range(5)]
        (self.is_last_run_success, self.is_adjust_population) \
            = [False for _ in range(2)]
        (self.n_centiles_for_shading, self.plot_count) \
            = [0 for _ in range(2)]
        for attribute in ['scenarios', 'interventions_to_cost', 'is_adjust_population', 'n_centiles_for_shading']:
            setattr(self, attribute, getattr(self.inputs, attribute))
        self.outputs = {'epi_uncertainty': {}}
        self.percentiles = [50., 2.5, 97.5] + list(numpy.linspace(0., 100., self.n_centiles_for_shading * 2 + 1))
        self.additional_cost_types = ['inflated', 'discounted', 'discounted_inflated']
        self.cost_types = self.additional_cost_types + ['raw']
        self.non_disease_compartment_strings = ['susceptible']
        self.emit_delay, self.gui_console_fn = 0.1, gui_console_fn
        if self.gui_console_fn:
            self.gui_console_fn('init')

        # optimisation attributes - note that this is currently dead
        # self.optimisation = False  # leave True even if loading optimisation results
        # self.opti_outputs_dir = 'saved_optimisation_analyses'
        # self.indicator_to_minimise = 'incidence'  # currently must be 'incidence' or 'mortality'
        # self.annual_envelope = [112.5e6]  # size of funding envelope in scenarios to be run
        # self.save_opti = True
        # self.load_optimisation = False  # optimisation will not be run if true
        # self.total_funding = None  # funding for entire period
        # self.f_tol = {'incidence': 0.5,
        #               'mortality': 0.05}  # stopping condition for optimisation algorithm (differs by indicator)
        # self.year_end_opti = 2035.  # model is run until that date during optimisation
        # self.acceptable_combinations = []  # list of intervention combinations that can be considered with funding
        # self.opti_results = {}  # store all the results that we need for optimisation
        # self.optimised_combinations = []
        # self.optimal_allocation = {}
        # self.interventions_considered_for_opti \
        #     = ['engage_lowquality', 'xpert', 'cxrxpertacf_prison', 'cxrxpertacf_urbanpoor', 'ipt_age0to5',
        #        'intensive_screening']  # interventions that must appear in optimal plan
        # self.interventions_forced_for_opti = ['engage_lowquality', 'ipt_age0to5', 'intensive_screening']

    ''' master methods to run other methods '''

    def master_runner(self):
        """
        Calls methods to run model with each of the three fundamental approaches.
        """

        # prepare file for saving
        out_dir = 'saved_uncertainty_analyses'
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        storage_file_name = os.path.join(out_dir, 'store.pkl')

        # load a saved simulation
        if self.gui_inputs['pickle_uncertainty'] == 'Load':
            self.add_comment_to_gui_window('Loading results from previous simulation')
            self.outputs = t_k.pickle_load(storage_file_name)

        # or run the manual scenarios as requested by user
        else:
            if self.inputs.run_mode == 'rapid_calibration':
                self.run_rapid_univariate_calibration()
            self.run_manual_calibration()
            if self.inputs.run_mode == 'epi_uncertainty':
                self.run_epi_uncertainty()
            if self.inputs.run_mode == 'int_uncertainty':
                self.run_intervention_uncertainty()

        # save uncertainty if requested
        if self.gui_inputs['pickle_uncertainty'] == 'Save':
            self.add_comment_to_gui_window('Uncertainty results saved to disc')
            t_k.pickle_save(self.outputs, storage_file_name)

        # master optimisation method
        # if self.optimisation and not self.load_optimisation:
        #     self.run_optimisation()
        #
        # prepare file for saving, save and load as requested
        # if not os.path.isdir(self.opti_outputs_dir):
        #     os.makedirs(self.opti_outputs_dir)
        # self.load_opti_results()
        # self.save_opti_results()

        # notify user that model running has finished
        self.add_comment_to_gui_window('Model running complete')

    def run_manual_calibration(self):
        """
        Runs the scenarios once each, starting from baseline with parameter values as specified in spreadsheets.
        """

        # run for each scenario, including baseline which is always included by default
        self.outputs['manual'] = {'epi': {}, 'cost': {}}
        for scenario in self.scenarios:

            # name, initialise and describe model, with appropriate times for scenario runs if required
            self.models[scenario] = model.ConsolidatedModel(scenario, self.inputs, self.gui_inputs, self.gui_console_fn)
            if scenario > 0:
                self.prepare_new_model_from_baseline(scenario)
            self.add_comment_to_gui_window(
                'Running %s conditions for %s using point estimates for parameters.'
                % ('manual_' + t_k.find_scenario_string_from_number(scenario), self.gui_inputs['country']))

            # integrate
            self.models[scenario].integrate()

            # interpret
            self.outputs['manual']['epi'][scenario] \
                = self.find_epi_outputs(scenario, strata_to_analyse=[self.models[scenario].agegroups,
                                                                     self.models[scenario].riskgroups])
            self.outputs['manual']['epi'][scenario].update(
                self.find_population_fractions(scenario=scenario,
                                               strata_to_analyse=[self.models[scenario].agegroups,
                                                                  self.models[scenario].riskgroups]))
            if self.interventions_to_cost[scenario]:
                self.outputs['manual']['cost'][scenario] = self.find_cost_outputs(scenario)

    def prepare_new_model_from_baseline(self, scenario):
        """
        Method to set the start time of a model and load the compartment values from the baseline run. From memory, I
        think that it was essential to set both start_time and next_time_point to the new time to ensure that the new
        model run took up where the last one left off.

        Args:
            scenario: Scenario number
        """

        start_time_index = self.models[0].find_time_index(self.inputs.model_constants['before_intervention_time'])
        self.models[scenario].start_time, self.models[scenario].next_time_point \
            = [self.models[0].times[start_time_index]] * 2
        self.models[scenario].loaded_compartments = self.models[0].load_state(start_time_index)

    ''' output interpretation methods '''

    def get_rate_for_output(self, scenario, flow_type, flow):
        """
        Find the numeric output for a flow rate, regardless of whether the intercompartmental flow is fixed or variable.

        Args:
            scenario: Integer for scenario
            flow_type: String for the type of flow
            flow: The tuple representing the individual flow
        Returns:
            The flow rate as a float value
        """

        mapper = self.models[scenario].flow_type_index[flow_type]
        return flow[mapper['rate']] if 'fixed_' in flow_type \
            else self.models[scenario].get_var_soln(flow[mapper['rate']])

    def find_epi_outputs(self, scenario, epi_outputs_to_analyse=None, strata_to_analyse=([])):
        """
        Method to extract all requested epidemiological outputs from the models. Intended ultimately to be flexible
        enough for use for analysis of scenarios, uncertainty and optimisation.

        Args:
            scenario: The integer value representing the scenario of the model to be analysed
            epi_outputs_to_analyse: List of strings for the outputs of interest to be worked through
            strata_to_analyse: List of any stratifications that outputs are required over, i.e. list of lists
        """

        # preliminaries
        outputs_to_analyse = epi_outputs_to_analyse if epi_outputs_to_analyse else self.epi_outputs_to_analyse
        epi_outputs, strata = {'times': self.models[scenario].times}, ['']
        for stratification_type in strata_to_analyse:
            strata += stratification_type

        # all outputs should cycle over each stratum, or at least be able to
        for stratum in strata:

            # population
            epi_outputs.update(self.find_population_totals(scenario, stratum, len(self.models[scenario].times)))

            # to allow calculation by strain and the total output
            for strain in [''] + self.models[scenario].strains:

                # standard rate outputs
                for output in self.standard_rate_outputs:
                    if output in outputs_to_analyse:
                        epi_outputs = self.find_standard_rate_output(epi_outputs, scenario, output, strain, stratum)

                # mortality and prevalence calculations
                if 'mortality' in outputs_to_analyse:
                    epi_outputs = self.find_mortality_output(epi_outputs, scenario, strain, stratum)
                if 'prevalence' in outputs_to_analyse:
                    epi_outputs = self.find_prevalence_output(epi_outputs, scenario, strain, stratum)

            # aggregate outputs ignoring strain status
            if 'infections' in outputs_to_analyse:
                epi_outputs = find_rate_from_aggregate(epi_outputs, 'annual_risk_infection', 'infections', 'population',
                                                       strata, percentage=True)

            # proportional outputs by strain
            if 'incidence' in outputs_to_analyse:
                epi_outputs = self.find_outputs_proportional_by_strain(epi_outputs, scenario, stratum)

        return epi_outputs

    def find_population_totals(self, scenario, stratum, output_length):
        """
        Find the total population sizes for each population stratum.

        Args:
            scenario: Integer for scenario value
            stratum: Population stratum being evaluated
            output_length: Length of all the outputs (i.e. number of time points evaluated at)
        Returns:
            Updated version of epi_outputs
        """

        new_outputs = {'population' + stratum: [0.] * output_length}
        for label in self.models[scenario].labels:
            if stratum in label:
                new_outputs['population' + stratum] = t_k.elementwise_list_addition(
                    self.models[scenario].get_compartment_soln(label), new_outputs['population' + stratum])
        return new_outputs

    def find_standard_rate_output(self, epi_outputs, scenario, output, strain, stratum):
        """
        Standard method for looping through epidemiological outputs that are defined by their from and to compartments
        only (as specified in self.from_labels and self.to_labels).

        Args:
            epi_outputs: Output data structure to be updated
            scenario: Integer for scenario value
            output: String representing the output of interest
            strain: Strain being evaluated
            stratum: Population stratum being evaluated
        Returns:
            Updated version of epi_outputs
        """

        blank_output_list = [0.] * len(epi_outputs['times'])
        master_mapper = self.models[scenario].flow_type_index
        strain_stratum = strain + stratum
        denominator \
            = t_k.prepare_denominator(epi_outputs['population' + stratum]) if output in self.divide_population else 1.
        multiplier = self.multipliers[output] if output in self.multipliers else 1.
        epi_outputs[output + strain_stratum] = blank_output_list
        for flow_type in self.models[scenario].flows_by_type:
            mapper = master_mapper[flow_type]
            for flow in self.models[scenario].flows_by_type[flow_type]:
                if t_k.are_strings_in_subdict(mapper, flow, self.from_labels[output] + [strain, stratum], 'from') \
                        and t_k.are_strings_in_subdict(mapper, flow, self.to_labels[output], 'to'):
                    increment = self.models[scenario].get_compartment_soln(flow[mapper['from']]) \
                                * self.get_rate_for_output(scenario, flow_type, flow) / denominator * multiplier
                    epi_outputs[output + strain_stratum] \
                        = t_k.elementwise_list_addition(increment, epi_outputs[output + strain_stratum])
        return epi_outputs

    def find_mortality_output(self, epi_outputs, scenario, strain, stratum):
        """
        Currently only coded for TB, which is quite disease-specific.

        Args:
            epi_outputs: Output data structure to be updated
            scenario: Integer for scenario value
            strain: Strain being evaluated
            stratum: Population stratum being evaluated
        Returns:
            Updated version of epi_outputs
        """

        return epi_outputs

    def find_outputs_proportional_by_strain(self, epi_outputs, scenario, stratum):
        """
        Find percentage incidence by strain.

        Args:
            epi_outputs: Output data structure to be updated
            scenario: Integer for scenario value
            stratum: Population stratum being evaluated
        Returns:
            Updated version of epi_outputs
        """

        for strain in self.models[scenario].strains:
            if strain:
                epi_outputs['perc_incidence' + strain + stratum] \
                    = t_k.elementwise_list_division(epi_outputs['incidence' + strain + stratum],
                                                    t_k.prepare_denominator(epi_outputs['incidence' + stratum]),
                                                    percentage=True)
        return epi_outputs

    def find_prevalence_output(self, epi_outputs, scenario, strain, stratum):
        """
        General method to calculate prevalence of disease based on a list of all the compartment strings that
        represent states without disease (i.e. self.non_disease_compartment_strings).

        Args:
            epi_outputs: Output data structure to be updated
            scenario: Integer for scenario value
            strain: Strain being evaluated
            stratum: Population stratum being evaluated
        Returns:
            Updated version of epi_outputs
        """

        blank_output_list = [0.] * len(epi_outputs['times'])
        strain_stratum = strain + stratum
        denominator = t_k.prepare_denominator(epi_outputs['population' + stratum])
        epi_outputs['prevalence' + strain_stratum] = blank_output_list
        for label in self.models[scenario].labels:
            if not any(s in label for s in self.non_disease_compartment_strings) \
                    and all(s in label for s in [strain, stratum]):
                prevalence_increment = self.models[scenario].get_compartment_soln(label) / denominator * 1e5
                epi_outputs['prevalence' + strain_stratum] = t_k.elementwise_list_addition(
                    prevalence_increment, epi_outputs['prevalence' + strain_stratum])
        return epi_outputs

    def find_population_fractions(self, scenario, strata_to_analyse=()):
        """
        Find the proportion of the population in various strata. The stratifications must apply to the entire
        population, so this method should not be used for strains, health systems, etc.

        Args:
            scenario: The numeral for the scenario to be analysed
            strata_to_analyse: List/tuple with each element being a list of model stratifications
        """

        fractions = {}
        for stratification in strata_to_analyse:
            if len(stratification) > 1:
                for stratum in stratification:
                    fractions['fraction' + stratum] \
                        = t_k.elementwise_list_division(self.outputs['manual']['epi'][scenario]['population' + stratum],
                                                        self.outputs['manual']['epi'][scenario]['population'])

        return fractions

    def find_cost_outputs(self, scenario):
        """
        Master method to call methods to find and update costs below. Note that cost_outputs has to be passed through
        the sub-methods below and then returned, so that something can be returned to both manual and uncertainty
        analysis methods for the appropriate handling.

        Args:
            scenario: Numeral for the scenario being costed
        """

        cost_outputs = self.find_raw_cost_outputs(scenario)
        cost_outputs.update({'raw_cost_all_programs': self.find_cost_all_programs(scenario, cost_outputs)})
        cost_outputs.update(self.find_adjusted_costs(scenario, cost_outputs))
        return cost_outputs

    def find_raw_cost_outputs(self, scenario):
        """
        Find cost dictionaries to add to cost_outputs attribute.

        Args:
            scenario: Numeral for the scenario being costed
        """

        cost_outputs = {'times': self.models[scenario].cost_times}
        for i, intervention in enumerate(self.interventions_to_cost[scenario]):
            cost_outputs['raw_cost_' + intervention] = self.models[scenario].costs[:, i]
        return cost_outputs

    def find_cost_all_programs(self, scenario, cost_outputs):
        """
        Sum costs across all programs and populate to cost_outputs dictionary for each scenario.

        Args:
            scenario: Numeral for the scenario being costed
            cost_outputs: The cost output structure being passed through these various cost methods
        """

        # arbitrary program (the first) to get the list length for elementwise addition
        cost_all_programs = [0.] * len(cost_outputs['raw_cost_' + self.interventions_to_cost[scenario][0]])

        # increment for each intervention
        for intervention in self.interventions_to_cost[scenario]:
            cost_all_programs = t_k.elementwise_list_addition(cost_outputs['raw_cost_' + intervention],
                                                              cost_all_programs)
        return cost_all_programs

    def find_adjusted_costs(self, scenario, cost_outputs):
        """
        Find costs adjusted for inflation, discounting and both.

        Args:
            scenario: Scenario being costed
            cost_outputs: The cost output structure being passed through these various cost methods
        """

        # get some preliminary parameters
        year_current = self.inputs.model_constants['recent_time']
        current_cpi = self.inputs.scaleup_fns[scenario]['econ_cpi'](year_current)
        discount_rate = self.inputs.model_constants['econ_discount_rate']

        # loop over interventions for costing and cost types
        for intervention in self.interventions_to_cost[scenario] + ['all_programs']:
            for cost_type in self.additional_cost_types:
                cost_outputs[cost_type + '_cost_' + intervention] = []
                for t, time in enumerate(self.models[scenario].cost_times):
                    cost_outputs[cost_type + '_cost_' + intervention].append(
                        economics.get_adjusted_cost(cost_outputs['raw_cost_' + intervention][t], cost_type,
                                                    current_cpi, self.inputs.scaleup_fns[0]['econ_cpi'](time),
                                                    discount_rate, max(0., (time - year_current))))
        return cost_outputs

    ''' epidemiological uncertainty methods '''

    def run_epi_uncertainty(self):
        """
        Main method to run all the uncertainty processes using a Metropolis-Hastings algorithm with normal proposal
        distribution.
        """
        self.add_comment_to_gui_window('Uncertainty analysis commenced')

        # prepare basic storage and local variables for uncertainty loop
        n_accepted, prev_log_likelihood, starting_params, run, accepted, accepted_params = 0, -5e2, [], 0, 0, None
        for key_for_dicts in ['epi', 'cost', 'all_parameters', 'accepted_parameters', 'rejected_parameters',
                              'all_compartment_values']:
            self.outputs['epi_uncertainty'][key_for_dicts] = {}
        for key_for_lists in ['loglikelihoods', 'whether_accepted', 'accepted_indices', 'rejected_indices']:
            self.outputs['epi_uncertainty'][key_for_lists] = []
        for param in self.inputs.param_ranges_unc:
            self.outputs['epi_uncertainty']['all_parameters'][param['key']] = []
            self.outputs['epi_uncertainty']['accepted_parameters'][param['key']] = {}
            self.outputs['epi_uncertainty']['rejected_parameters'][param['key']] = {n_accepted: []}
            starting_params.append(self.inputs.model_constants[param['key']])
        for compartment_type in self.inputs.compartment_types:
            if compartment_type in self.inputs.model_constants:
                self.outputs['epi_uncertainty']['all_compartment_values'][compartment_type] = []

        # find values of mu and sd for the likelihood calculation. Process uncertainty weights in the same loop.
        years_to_compare = range(2010, 2017)
        mu_values, sd_values, mean_sd_value, weights = {}, {}, {}, {}
        for output_dict in self.inputs.outputs_unc:
            mu_values[output_dict['key']] = {}
            sd_values[output_dict['key']] = {}
            # the GTB values for the output of interest
            working_output_dictionary = self.get_fitting_data()[output_dict['key']]
            available_years = []
            for y, year in enumerate(years_to_compare):
                if year in working_output_dictionary.keys():
                    available_years.append(year)
                    mu_values[output_dict['key']][year], sd_values[output_dict['key']][year] = \
                        working_output_dictionary[year]
            mean_sd_value[output_dict['key']] = numpy.mean(sd_values[output_dict['key']].values())
            weights[output_dict['key']] = find_uncertainty_output_weights(available_years, 4)
            self.add_comment_to_gui_window('"Weights for ": ' + output_dict['key'] + '\n' + str(weights))

        # start main uncertainty loop
        while n_accepted < self.gui_inputs['uncertainty_runs']:
            # instantiate model objects
            for scenario in self.scenarios:
                params_copy = copy.deepcopy(self.models[scenario].params)  # we need to reuse the updated params
                self.models[scenario] = model.ConsolidatedModel(scenario, self.inputs, self.gui_inputs)
                self.models[scenario].params = params_copy

            # set timer
            start_timer_run = datetime.datetime.now()
            proposed_params = self.update_params(accepted_params) if accepted_params else starting_params

            # run baseline scenario (includes parameter checking, parameter setting and recording success/failure)
            self.run_with_params(proposed_params, scenario=0)

            # store outputs regardless of acceptance, provided run was completed successfully
            if self.is_last_run_success:

                # get outputs for calibration and store results
                self.store_uncertainty(0, 'epi_uncertainty')
                last_run_output_index = None if self.outputs['epi_uncertainty']['epi'][0]['mortality'].ndim == 1 else -1
                outputs_for_comparison \
                    = [self.outputs['epi_uncertainty']['epi'][0]['incidence'][
                           last_run_output_index, t_k.find_first_list_element_at_least(
                               self.outputs['manual']['epi'][0]['times'], float(year))] for year in years_to_compare]

                # calculate likelihood
                prior_log_likelihood, posterior_log_likelihood = 0., 0.

                # calculate prior
                for p, param in enumerate(self.inputs.param_ranges_unc):
                    param_val = proposed_params[p]
                    self.outputs['epi_uncertainty']['all_parameters'][param['key']].append(proposed_params[p])
                    if 'additional_params' not in param:
                        param['additional_params'] = None
                    prior_log_likelihood += find_log_probability_density(
                        param['distribution'], param_val, param['bounds'], additional_params=param['additional_params'])

                # calculate likelihood
                for output_dict in self.inputs.outputs_unc:
                    index_for_available_years = 0
                    for y, year in enumerate(years_to_compare):
                        if year in working_output_dictionary.keys():
                            model_result_for_output = outputs_for_comparison[y]
                            mu = mu_values[output_dict['key']][year]
                            sd = mean_sd_value[output_dict['key']] if self.average_sd_for_likelihood else \
                                sd_values[output_dict['key']][year]
                            posterior_log_likelihood += norm.logpdf(model_result_for_output, mu, sd) * \
                                                        weights[output_dict['key']][index_for_available_years]
                            index_for_available_years += 1

                # determine acceptance
                log_likelihood = prior_log_likelihood + posterior_log_likelihood
                accepted = numpy.random.binomial(n=1, p=min(1., numpy.exp(log_likelihood - prev_log_likelihood)))

                # describe progression of likelihood analysis
                self.add_comment_to_gui_window(
                    'Previous log likelihood:\n%4.3f\nLog likelihood this run:\n%4.3f\nAcceptance probability:\n%4.3f'
                    % (prev_log_likelihood, log_likelihood, min(1., numpy.exp(log_likelihood - prev_log_likelihood)))
                    + '\nWhether accepted:\n%s\n________________\n' % str(bool(accepted)))
                self.outputs['epi_uncertainty']['loglikelihoods'].append(log_likelihood)

                # record starting population
                if self.gui_inputs['write_uncertainty_outcome_params']:
                    for compartment_type in self.outputs['epi_uncertainty']['all_compartment_values'].keys():
                        self.outputs['epi_uncertainty']['all_compartment_values'][compartment_type].append(
                            self.models[0].params[compartment_type])

                # record uncertainty calculations for all runs
                if accepted:
                    self.outputs['epi_uncertainty']['whether_accepted'].append(True)
                    self.outputs['epi_uncertainty']['accepted_indices'].append(run)
                    n_accepted += 1
                    for p, param in enumerate(self.inputs.param_ranges_unc):
                        self.outputs['epi_uncertainty']['accepted_parameters'][param['key']][n_accepted] \
                            = proposed_params[p]
                        self.outputs['epi_uncertainty']['rejected_parameters'][param['key']][n_accepted] = []

                    # update likelihood and parameter set for next run
                    prev_log_likelihood, accepted_params = log_likelihood, proposed_params

                    # run scenarios - only if accepted and not for baseline
                    for scenario in self.scenarios:
                        if scenario:
                            self.prepare_new_model_from_baseline(scenario)
                            self.run_with_params(accepted_params, scenario=scenario)
                            self.store_uncertainty(scenario, 'epi_uncertainty')

                    self.make_disease_specific_adjustments(last_run_output_index, years_to_compare)

                    # make algorithmic adjustments
                    if self.is_adjust_population:
                        self.adjust_start_population(last_run_output_index)

                else:
                    self.outputs['epi_uncertainty']['whether_accepted'].append(False)
                    self.outputs['epi_uncertainty']['rejected_indices'].append(run)
                    for p, param in enumerate(self.inputs.param_ranges_unc):
                        self.outputs['epi_uncertainty']['rejected_parameters'][param['key']][n_accepted].append(
                            proposed_params[p])

                # plot parameter progression and report on progress
                if self.gui_inputs['uncertainty_runs'] <= 10:
                    self.plot_progressive_parameters()
                self.add_comment_to_gui_window(
                    str(n_accepted) + ' accepted / ' + str(run) + ' candidates. Running time: '
                    + str(datetime.datetime.now() - start_timer_run))

                self.record_disease_specific_adjustments()

                run += 1

    def get_fitting_data(self):
        """
        Get data to fit the model outputs to, which will always be disease-specific.
        """

        pass

    def make_disease_specific_adjustments(self, last_run_output_index, years_to_compare):
        """
        Make any disease specific adjustments to improve automatic calibration.
        """

        pass

    def record_disease_specific_adjustments(self):
        """
        Record the values of the adjustments made in disease-specific automatic calibration.
        """

        pass

    def run_with_params(self, params, scenario=0):
        """
        Integrate the model with the proposed parameter set.

        Args:
            params: The parameters to be set in the model
            scenario: Numeral for the scenario to be run
        """

        # check whether parameter values are acceptable
        for p, param in enumerate(params):

            # whether the parameter value is valid
            if not numpy.isfinite(param) or param < 0.:
                self.add_comment_to_gui_window('Warning: parameter%d=%f is invalid for model' % (p, param))
                self.is_last_run_success = False
                return

            # whether the parameter value is within acceptable ranges
            bounds = self.inputs.param_ranges_unc[p]['bounds']
            if param < bounds[0] or param > bounds[1]:
                self.add_comment_to_gui_window('Warning: parameter%d=%f is outside of the allowed bounds' % (p, param))
                self.is_last_run_success = False
                return

        param_dict = {names['key']: vals for names, vals in zip(self.inputs.param_ranges_unc, params)}

        # set parameters and run
        self.set_model_with_params(param_dict, scenario)
        self.is_last_run_success = True

        try:
            self.models[scenario].integrate()
        except:
            self.add_comment_to_gui_window('Warning: parameters=%s failed with model' % params)
            self.is_last_run_success = False

    def set_model_with_params(self, param_dict, scenario=0):
        """
        Populates baseline model with params from uncertainty calculations, including adjusting starting time.
        Also adjusts starting population to better match target population at current time using target_population input
        from country sheet. (Not currently in default sheet.)

        Args:
            param_dict: Dictionary of the parameters to be set within the model (keys parameter name strings and values
                parameter values)
            scenario: Numeral for scenario to run
        """

        for key in param_dict:

            # start time usually set in instantiation, which has already been done here, so needs to be set separately
            if key == 'start_time':
                self.models[scenario].start_time = param_dict[key]

            # set parameters
            elif key in self.models[scenario].params:
                self.models[scenario].set_parameter(key, param_dict[key])
            else:
                raise ValueError('%s not in model_object params' % key)

    def store_uncertainty(self, scenario, uncertainty_type='epi_uncertainty'):
        """
        Add model results from one uncertainty run to the appropriate outputs dictionary, vertically stacking
        results on to the previous matrix.

        Args:
            scenario: The scenario being run
            uncertainty_type: String for the approach to uncertainty, either epi_uncertainty or intervention_uncertainty
        Updates:
            self.outputs: Which contains all the outputs in a tiered dictionary format
        """

        # get outputs to add to outputs attribute
        new_outputs = {'epi': self.find_epi_outputs(scenario)}
        # new_outputs['cost'] = self.find_cost_outputs(scenario) if self.models[scenario].interventions_to_cost else {}
        if self.models[scenario].interventions_to_cost:
            new_outputs.update({'cost': self.find_cost_outputs(scenario)})

        # incorporate new data
        for output_type in ['epi', 'cost']:

            # create third tier if outputs haven't been recorded yet for this scenario
            if scenario not in self.outputs[uncertainty_type][output_type]:
                self.outputs[uncertainty_type][output_type] = {scenario: {}}

            # for each epi or cost output available
            for output in new_outputs[output_type]:

                # if output hasn't been added yet, start array off from list output
                if output not in self.outputs[uncertainty_type][output_type][scenario]:
                    self.outputs[uncertainty_type][output_type][scenario].update(
                        {output: numpy.array(new_outputs[output_type][output])})

                # otherwise append new data
                else:

                    # adjust list size if necessary or just use output directly
                    if output_type == 'epi' and scenario == 0:
                        shape_index = 0 if self.outputs[uncertainty_type]['epi'][scenario][output].ndim == 1 else 1

                        # extend new output data with zeros if too short
                        if len(new_outputs['epi'][output]) \
                                < self.outputs[uncertainty_type]['epi'][scenario][output].shape[shape_index]:
                            new_outputs['epi'][output] \
                                = t_k.join_zero_array_to_left(
                                    self.outputs[uncertainty_type]['epi'][scenario][output].shape[shape_index]
                                    - len(new_outputs['epi'][output]), new_outputs['epi'][output])

                        # extend existing output data array if too long
                        elif len(new_outputs['epi'][output]) \
                                > self.outputs[uncertainty_type]['epi'][scenario][output].shape[shape_index]:
                            self.outputs[uncertainty_type]['epi'][scenario][output] \
                                = t_k.join_zero_array_to_left(
                                    len(new_outputs['epi'][output])
                                    - self.outputs[uncertainty_type]['epi'][scenario][output].shape[shape_index],
                                    self.outputs[uncertainty_type]['epi'][scenario][output])

                    # stack onto previous output if seen before
                    self.outputs[uncertainty_type][output_type][scenario][output] \
                        = numpy.vstack((self.outputs[uncertainty_type][output_type][scenario][output],
                                        new_outputs[output_type][output]))

    def update_params(self, old_params):
        """
        Update all the parameter values being used in the uncertainty analysis.

        Args:
            old_params:
        Returns:
            new_params: The new parameters to be used in the next model run.
        """

        new_params = []

        # Manually define the width of the interval containing 95% of the proposal Gaussian density
        overwritten_abs_search_width = {'tb_n_contact': 2., 'start_time': 5.}

        # iterate through the parameters being used
        for p, param_dict in enumerate(self.inputs.param_ranges_unc):
            bounds, random = param_dict['bounds'], -100.
            if param_dict['key'] in overwritten_abs_search_width.keys():
                abs_search_width = overwritten_abs_search_width[param_dict['key']]
            else:
                abs_search_width = self.gui_inputs['search_width'] * (bounds[1] - bounds[0])
            sd = abs_search_width / (2. * 1.96)

            # search for new parameters and add to dictionary
            while random < bounds[0] or random > bounds[1]:
                random = norm.rvs(loc=old_params[p], scale=sd, size=1)
            new_params.append(random[0])

        return new_params

    def adjust_start_population(self, last_run_output_index):
        """
        Algorithmically adjust the starting population to better match the modern population targeted.

        Args:
            last_run_output_index: Integer to index last output
        """

        if 'target_population' in self.inputs.model_constants:
            population_adjustment \
                = self.inputs.model_constants['target_population'] / float(self.outputs['epi_uncertainty']['epi'][0][
                    'population'][last_run_output_index, t_k.find_first_list_element_above(
                        self.outputs['manual']['epi'][0]['times'], self.inputs.model_constants['current_time'])])
            for compartment in self.inputs.compartment_types:
                if compartment in self.models[0].params:
                    self.models[0].set_parameter(compartment,
                                                 self.models[0].params[compartment] * population_adjustment)

    def plot_progressive_parameters(self):
        """
        Produce real-time parameter plot, according to which GUI is in use.
        """

        if self.gui_console_fn:
            self.gui_console_fn('graph', {
                'all_parameters_tried': self.outputs['epi_uncertainty']['all_parameters'],
                'whether_accepted_list': self.outputs['epi_uncertainty']['whether_accepted'],
                'rejection_dict': self.outputs['epi_uncertainty']['rejected_parameters'],
                'accepted_indices': self.outputs['epi_uncertainty']['accepted_indices'],
                'acceptance_dict': self.outputs['epi_uncertainty']['accepted_parameters'],
                'names': {param: t_k.find_title_from_dictionary(param)
                          for p, param in enumerate(self.outputs['epi_uncertainty']['all_parameters'])},
                'param_ranges_unc': self.inputs.param_ranges_unc})

    ''' other run type methods '''

    def run_intervention_uncertainty(self):
        """
        Master method for running intervention uncertainty. That is, starting from the calibrated baseline simulated,
        project forward scenarios based on varying parameters for the effectiveness of the intervention under
        consideration.
        """

        self.outputs['int_uncertainty'] = {'epi': {}, 'cost': {}}

        # extract relevant intervention parameters from the intervention uncertainty dictionary
        working_param_dict = {}
        for param in self.inputs.intervention_param_dict[self.inputs.uncertainty_intervention]:
            for int_param in range(len(self.inputs.int_ranges_unc)):
                if self.inputs.int_ranges_unc[int_param]['key'] \
                        in self.inputs.intervention_param_dict[self.inputs.uncertainty_intervention]:
                    working_param_dict[param] = self.inputs.int_ranges_unc[int_param]

        # generate samples using latin hypercube design
        sample_values = lhs(len(working_param_dict), samples=self.inputs.n_samples)
        self.outputs['int_uncertainty']['parameter_values'] = {}
        for p, param in enumerate(working_param_dict):
            self.outputs['int_uncertainty']['parameter_values'][param] = []
            for sample in range(self.inputs.n_samples):
                self.outputs['int_uncertainty']['parameter_values'][param].append(
                    working_param_dict[param]['bounds'][0]
                    + (working_param_dict[param]['bounds'][1] - working_param_dict[param]['bounds'][0])
                    * sample_values[sample][p])

        # loop through parameter values
        for sample in range(self.inputs.n_samples):

            # prepare for integration of scenario
            self.models[15] = model.ConsolidatedModel(15, self.inputs, self.gui_inputs, self.gui_console_fn)
            self.prepare_new_model_from_baseline(15)
            self.models[15].relevant_interventions.append(self.inputs.uncertainty_intervention)
            for param in self.outputs['int_uncertainty']['parameter_values']:
                self.models[15].set_parameter(param, self.outputs['int_uncertainty']['parameter_values'][param][sample])

            # integrate and save
            self.models[15].integrate()
            self.store_uncertainty(15, uncertainty_type='int_uncertainty')

    def run_rapid_univariate_calibration(self):
        """
        Perform a least-square minimisation on the distance between the model outputs and the datapoints to
        calibrate a single parameter.
        Return the value of the calibrated parameter.
        """
        self.add_comment_to_gui_window('Rapid calibration commenced')
        params_to_calibrate = 'tb_n_contact'   # hard-coded
        param_dict = {}
        targeted_indicator = 'incidence'  # hard-coded
        single_point_calibration = True

        years_to_compare = range(2010, 2017)
        working_output_dictionary = self.get_fitting_data()[targeted_indicator]
        available_years = []
        target_values = {}

        for y, year in enumerate(years_to_compare):
            if year in working_output_dictionary.keys():
                available_years.append(year)
                target_values[year] = working_output_dictionary[year][0]

        if single_point_calibration:
            available_years = [available_years[-1]]
            weights = [1.0]
        else:
            weights = find_uncertainty_output_weights(available_years, 4)

        def objective_function(param_val):
            # run the model
            self.models[0] = model.ConsolidatedModel(0, self.inputs, self.gui_inputs)
            self.outputs['manual'] = {'epi': {}, 'cost': {}}
            # set parameters and run
            param_dict[params_to_calibrate] = param_val
            self.set_model_with_params(param_dict, 0)
            self.models[0].integrate()
            self.outputs['manual']['epi'][0] \
                = self.find_epi_outputs(0, strata_to_analyse=[self.models[0].agegroups,
                                                                     self.models[0].riskgroups])

            outputs_for_comparison \
                = [self.outputs['manual']['epi'][0]['incidence'][t_k.find_first_list_element_at_least(
                           self.outputs['manual']['epi'][0]['times'], float(year))] for year in years_to_compare]

            sum_of_squares = 0.
            abs_diff = 0.
            index_for_available_years = 0

            for y, year in enumerate(years_to_compare):
                if year in working_output_dictionary.keys():
                    if not single_point_calibration or year == available_years[-1]:
                        model_result_for_output = outputs_for_comparison[y]
                        data = target_values[year]
                        sum_of_squares += weights[index_for_available_years] * (data-model_result_for_output)**2
                        abs_diff += model_result_for_output - data
                        index_for_available_years += 1

            if single_point_calibration:
                return abs_diff
            else:
                return sum_of_squares

        if single_point_calibration:
            param_low, param_high = 10., 20.   # starting points
            f_low = objective_function(param_low)
            f_high = objective_function(param_high)
            if f_low*f_high > 0:
                exit('the interval [param_low - param_high] does not contain the solution')

            param_tol = 0.1
            while (param_high - param_low) / 2. > param_tol:
                midpoint = (param_low + param_high) / 2.
                obj = objective_function(midpoint)
                print "param value: " + str(midpoint)
                print "distance to target: " + str(obj)
                if obj == 0:
                    return midpoint
                elif obj*f_low < 0:
                    param_high = midpoint
                else:
                    param_low = midpoint
            best_param_value = midpoint
        else:
            x_0 = self.inputs.model_constants[params_to_calibrate]
            optimisation_result = minimize(fun=objective_function, x0=x_0, method='Nelder-Mead',options={'fatol':5.})
            best_param_value = optimisation_result.x

        print "The best value found for " + params_to_calibrate + " is " + str(best_param_value)

    ''' optimisation methods '''

    def run_optimisation(self):
        """
        Master optimisation method for the different levels of funding defined in self.annual_envelope
        """

        # initialise the optimisation output data structures
        standard_optimisation_attributes = ['best_allocation', 'incidence', 'mortality']
        self.opti_results['indicator_to_minimise'] = self.indicator_to_minimise
        self.opti_results['annual_envelope'] = self.annual_envelope
        for attribute in standard_optimisation_attributes:
            self.opti_results[attribute] = []

        # run optimisation for each envelope
        for envelope in self.annual_envelope:
            self.add_comment_to_gui_window('Start optimisation for annual total envelope of: ' + str(envelope))
            self.total_funding = envelope * (self.inputs.model_constants['scenario_end_time']
                                             - self.inputs.model_constants['scenario_start_time'])
            self.optimise_single_envelope()
            full_results = self.get_full_results_opti()
            for attribute in standard_optimisation_attributes:
                self.opti_results[attribute].append(full_results[attribute])

    def get_acceptable_combinations(self):
        """
        Determines the acceptable combinations of interventions according to the related starting costs and given a
        total amount of funding populates the acceptable_combinations attribute of model_runner.
        """

        # find all possible combinations of the considered interventions
        all_possible_combinations \
            = list(itertools.chain.from_iterable(
                itertools.combinations(range(len(self.interventions_considered_for_opti)), n) for n in
                range(len(self.interventions_considered_for_opti) + 1)[1:]))

        # determine whether each combination is fund-able given start-up costs
        fundable_combinations = []
        for combination in all_possible_combinations:
            total_startup_costs = 0.
            for intervention in combination:
                if self.inputs.intervention_startdates[0][self.interventions_considered_for_opti[intervention]] is 0:
                    total_startup_costs \
                        += self.inputs.model_constants['econ_startupcost_' +
                                                       self.interventions_considered_for_opti[intervention]]
            if total_startup_costs <= self.total_funding:
                fundable_combinations.append(combination)

        # determine whether a forced intervention is missing from each fund-able intervention
        combinations_missing_a_forced_intervention = []
        for c, combination in enumerate(fundable_combinations):
            for intervention in self.interventions_forced_for_opti:
                if self.interventions_considered_for_opti.index(intervention) not in combination \
                        and combination not in combinations_missing_a_forced_intervention:
                    combinations_missing_a_forced_intervention.append(combination)

        # populate final list of acceptable combinations
        acceptable_combinations = []
        for combination in fundable_combinations:
            if combination not in combinations_missing_a_forced_intervention:
                acceptable_combinations.append(combination)
        self.acceptable_combinations = acceptable_combinations
        self.add_comment_to_gui_window('Number of combinations to consider: ' + str(len(self.acceptable_combinations)))

    def optimise_single_envelope(self):
        """
        Method to fully run optimisation for a single funding envelope.
        """

        start_timer_opti = datetime.datetime.now()
        self.optimised_combinations = []

        # initialise a new model that will be run from recent_time and set basic attributes for optimisation
        self.models['optimisation'] = model.ConsolidatedModel(0, self.inputs, self.gui_inputs, self.gui_console_fn)
        self.prepare_new_model_from_baseline(0)
        self.models['optimisation'].eco_drives_epi = True
        self.models['optimisation'].inputs.model_constants['scenario_end_time'] = self.year_end_opti
        self.models['optimisation'].interventions_considered_for_opti = self.interventions_considered_for_opti

        # find the combinations of interventions to be optimised
        self.get_acceptable_combinations()

        # for each acceptable combination of interventions
        for c, combination in enumerate(self.acceptable_combinations):

            # prepare storage
            dict_optimised_combi = {'interventions': [], 'distribution': [], 'objective': None}

            for i in range(len(combination)):
                intervention = self.interventions_considered_for_opti[combination[i]]
                dict_optimised_combi['interventions'].append(intervention)

            if self.gui_console_fn:
                self.gui_console_fn('console', {
                    'message': 'Optimisation of the distribution across: \n' +
                                str(dict_optimised_combi['interventions'])
                })

            # function to minimise: incidence in 2035
            def minimisation_function(x):

                """
                Args:
                    x: defines the resource allocation (as absolute funding over the total period (2015 - 2035))
                Returns:
                    x has same length as combination
                    predicted incidence for 2035
                """

                # initialise funding at zero for each intervention
                for inter in self.interventions_considered_for_opti:
                    self.models['optimisation'].available_funding[inter] = 0.

                # input values from x
                for x_i in range(len(x)):
                    inter = self.interventions_considered_for_opti[combination[x_i]]
                    self.models['optimisation'].available_funding[inter] = x[x_i] * self.total_funding
                self.models['optimisation'].distribute_funding_across_years()
                self.models['optimisation'].integrate()
                output_list = self.find_epi_outputs(0)
                return output_list[self.indicator_to_minimise][-1]

            # if only one intervention, the distribution is obvious
            if len(combination) == 1:
                dict_optimised_combi['distribution'] = [1.]
                dict_optimised_combi['objective'] = minimisation_function([1.])

            # otherwise
            else:

                # initial guess
                starting_distribution = []
                for i in range(len(combination)):
                    starting_distribution.append(1. / len(combination))

                # equality constraint is that the sum of the proportions has to be equal to one
                sum_to_one_constraint = [{'type': 'ineq',
                                          'fun': lambda x: 1. - sum(x),
                                          'jac': lambda x: -numpy.ones(len(x))}]
                cost_bounds = []
                for i in range(len(combination)):
                    minimal_allocation = 0.

                    # if start-up costs apply
                    if self.inputs.intervention_startdates[0][
                            self.models['manual_baseline'].interventions_to_cost[combination[i]]] is None:
                        minimal_allocation \
                            = self.models['manual_baseline'].inputs.model_constants[
                                  'econ_startupcost_'
                                  + self.models['manual_baseline'].interventions_to_cost[combination[i]]] \
                            / self.total_funding
                    cost_bounds.append((minimal_allocation, 1.))

                # ready to run optimisation
                optimisation_result \
                    = minimize(minimisation_function, starting_distribution, jac=None, bounds=cost_bounds,
                               constraints=sum_to_one_constraint, method='SLSQP',
                               options={'disp': False, 'ftol': self.f_tol[self.indicator_to_minimise]})
                dict_optimised_combi['distribution'] = optimisation_result.x
                dict_optimised_combi['objective'] = optimisation_result.fun

            self.optimised_combinations.append(dict_optimised_combi)
            self.add_comment_to_gui_window('Combination ' + str(c + 1) + '/' + str(len(self.acceptable_combinations))
                                           + ' completed.')

        # update optimal allocation
        self.optimal_allocation = {}
        best_dict = {}
        best_obj = 1e10
        for i, dict_opti in enumerate(self.optimised_combinations):
            if dict_opti['objective'] < best_obj:
                best_dict = dict_opti
                best_obj = dict_opti['objective']

        for intervention in self.interventions_considered_for_opti:
            self.optimal_allocation[intervention] = 0.
        for i, intervention in enumerate(best_dict['interventions']):
            self.optimal_allocation[intervention] = best_dict['distribution'][i]

        self.add_comment_to_gui_window('End optimisation after ' + str(datetime.datetime.now() - start_timer_opti))

    def get_full_results_opti(self):
        """
        We need to run the best allocation scenario until 2035 to obtain the final incidence and mortality.
        """

        # prepare new model to run full scenario duration
        self.models['optimisation'] = model.ConsolidatedModel(0, self.inputs, self.gui_inputs, self.gui_console_fn)
        self.prepare_new_model_from_baseline(0)
        self.models['optimisation'].eco_drives_epi = True
        self.models['optimisation'].interventions_considered_for_opti = self.interventions_considered_for_opti

        # initialise funding at zero for each intervention
        for intervention in self.interventions_considered_for_opti:
            self.models['optimisation'].available_funding[intervention] = 0.

        # distribute funding and integrate
        for intervention, prop in self.optimal_allocation.iteritems():
            self.models['optimisation'].available_funding[intervention] = prop * self.total_funding
        self.models['optimisation'].distribute_funding_across_years()
        self.models['optimisation'].integrate()

        # find epi results
        output_list = self.find_epi_outputs(0)
        del self.models['optimisation']
        return {'best_allocation': self.optimal_allocation, 'incidence': output_list['incidence'][-1],
                'mortality': output_list['mortality'][-1]}

    def load_opti_results(self):
        """
        Load optimisation results if attribute to self is True.
        """

        if self.load_optimisation:
            storage_file_name = os.path.join(self.opti_outputs_dir, 'opti_outputs.pkl')
            self.opti_results = t_k.pickle_load(storage_file_name)
            self.add_comment_to_gui_window('Optimisation results loaded')

    def save_opti_results(self):
        """
        Save optimisation results, which is expected to be the usual behaviour for the model runner.
        """

        # save only if optimisation has been run and save requested
        if self.save_opti and self.optimisation:
            filename = os.path.join(self.opti_outputs_dir, 'opti_outputs.pkl')
            t_k.pickle_save(self.opti_results, filename)
            self.add_comment_to_gui_window('Optimisation results saved')

    ''' GUI-related methods '''

    def add_comment_to_gui_window(self, comment):

        if self.gui_console_fn:
            self.gui_console_fn('console', {'message': comment})


''' disease-specific model runners '''


class TbRunner(ModelRunner):
    def __init__(self, gui_inputs, gui_console_fn=None):

        ModelRunner.__init__(self, gui_inputs, gui_console_fn)

        # outputs
        self.epi_outputs_to_analyse = ['incidence', 'prevalence', 'mortality', 'true_mortality', 'notifications']
        self.add_strain_specific_outputs_to_analyse()

        self.average_sd_for_likelihood = True  # whether to use a common sd for all data points iun the likelihood calculation
        self.standard_rate_outputs = ['incidence', 'notifications', 'infections']
        self.from_labels \
            = {'incidence': ['latent'],
               'notifications': ['active'],
               'infections': []}
        self.to_labels \
            = {'incidence': ['active'],
               'notifications': ['detect'],
               'infections': ['latent_early']}
        self.divide_population = ['incidence']
        self.multipliers \
            = {'incidence': 1e5,
               'mortality': 1e5}
        self.non_disease_compartment_strings \
            = ['susceptible', 'latent']

        # uncertainty adjustments
        self.outputs['epi_uncertainty'] \
            = {'adjustments': {'program_prop_death_reporting': [], 'mdr_introduce_time': []}}
        self.relative_difference_to_adjust_mortality = 1.1
        self.relative_difference_to_adjust_mdr = 1.1
        self.amount_to_adjust_mortality = .02
        self.amount_to_adjust_mdr_year = 1.
        self.prop_death_reporting = self.inputs.model_constants['program_prop_death_reporting']
        self.adjust_mortality = True
        adjust_mdr = False
        self.adjust_mdr = False if len(self.inputs.strains) < 2 else adjust_mdr
        self.mdr_introduce_time = self.inputs.model_constants['mdr_introduce_time']

    def add_strain_specific_outputs_to_analyse(self):
        """
        If several strains are simulated, we also want the strain-specific outputs to be analysed
        """
        if len(self.inputs.strains) > 1:
            new_epi_outputs_to_analyse = []
            for output in self.epi_outputs_to_analyse:
                new_epi_outputs_to_analyse.append(output)
                for strain in self.inputs.strains:
                    new_epi_outputs_to_analyse.append(output + strain)
            self.epi_outputs_to_analyse = new_epi_outputs_to_analyse
    ''' output interpretation methods '''

    def find_mortality_output(self, epi_outputs, scenario, strain, stratum):
        """
        TB-specific method to calculate mortality rates, accounting for under-reporting of community deaths.

        Args:
            epi_outputs: Output data structure to be updated
            scenario: Integer for scenario value
            strain: Strain being evaluated
            stratum: Population stratum being evaluated
        Returns:
            Updated version of epi_outputs
        """

        blank_output_list = [0.] * len(epi_outputs['times'])
        master_mapper = self.models[scenario].flow_type_index
        strain_stratum = strain + stratum
        denominator = t_k.prepare_denominator(epi_outputs['population' + stratum])
        multiplier = self.multipliers['mortality'] if 'mortality' in self.multipliers else 1.
        epi_outputs['mortality' + strain_stratum], epi_outputs['true_mortality' + strain_stratum] \
            = [blank_output_list] * 2
        for flow_type in self.models[scenario].flows_by_type:
            mapper = master_mapper[flow_type]
            for flow in self.models[scenario].flows_by_type[flow_type]:
                if 'death' in flow_type and strain in flow[mapper['from']] and stratum in flow[mapper['from']]:
                    for mortality_type in ['true_mortality', 'mortality']:
                        prop_death_reporting = self.prop_death_reporting \
                            if mortality_type == 'mortality' and 'fixed' in flow_type else 1.
                        mortality_increment \
                            = self.models[scenario].get_compartment_soln(flow[mapper['from']]) \
                            * self.get_rate_for_output(scenario, flow_type, flow) / denominator * multiplier
                        epi_outputs[mortality_type + strain_stratum] \
                            = t_k.elementwise_list_addition(mortality_increment * prop_death_reporting,
                                                            epi_outputs[mortality_type + strain_stratum])
        return epi_outputs

    ''' epidemiological uncertainty-related methods '''

    def get_fitting_data(self):
        """
        Define the characteristics (mean and standard deviation) of the normal distribution for model outputs
        - currently incidence and mortality only.

        Returns:
            norm_dist_params: Dictionary with keys outputs and values dictionaries. Sub-dictionaries have keys years
                and values lists, with first element of list means and second standard deviations.
        """

        # dictionary storing the characteristics of the normal distributions
        norm_dist_params = {}
        for output_dict in self.inputs.outputs_unc:
            norm_dist_params[output_dict['key']] = {}

            # incidence
            if output_dict['key'] == 'incidence':
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    low, high = self.inputs.data_to_fit['incidence_low'][year], \
                                self.inputs.data_to_fit['incidence_high'][year]
                    norm_dist_params[output_dict['key']][year] \
                        = [(high + low) / 2., output_dict['width_multiplier'] * (high - low) / (2. * 1.96)]

            # mortality
            elif output_dict['key'] == 'mortality':
                sd = output_dict['posterior_width'] / (2. * 1.96)
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    mu = self.inputs.data_to_fit[output_dict['key']][year]
                    norm_dist_params[output_dict['key']][year] = [mu, sd]

        return norm_dist_params

    def make_disease_specific_adjustments(self, last_run_output_index, years_to_compare):
        """
        Make any TB-specific adjustments required for automatic calibration here. Arguments are just passed through to
        the specific methods below.
        """

        if self.adjust_mortality:
            self.adjust_mortality_reporting(last_run_output_index, years_to_compare)
        if self.adjust_mdr:
            self.adjust_mdr_introduction(last_run_output_index)

    def adjust_mortality_reporting(self, last_run_output_index, years_to_compare):
        """
        Adjust the proportion of mortality captured by reporting systems to better match reported data.

        Args:
            last_run_output_index: Integer to index the run
            years_to_compare: Years to compare the reported mortality to the modelled mortality over
        """

        ratios = []
        for year in years_to_compare:
            if year in self.inputs.original_data['gtb']['e_mort_exc_tbhiv_100k']:
                ratios.append(self.outputs['epi_uncertainty']['epi'][0]['mortality'][
                                  last_run_output_index, t_k.find_first_list_element_above(
                                      self.outputs['manual']['epi'][0]['times'], float(year))]
                              / self.inputs.original_data['gtb']['e_mort_exc_tbhiv_100k'][year])
        average_ratio = numpy.mean(ratios)
        if average_ratio < 1. / self.relative_difference_to_adjust_mortality:
            self.prop_death_reporting += self.amount_to_adjust_mortality
        elif average_ratio > self.relative_difference_to_adjust_mortality:
            self.prop_death_reporting -= self.amount_to_adjust_mortality

    def adjust_mdr_introduction(self, last_run_output_index):
        """
        Adjust timing of MDR-TB algorithmically to better match modern proportionate burden observed. May not work if
        introducing earlier doesn't consistently result in a greater burden of disease - as was the case for Bulgaria.
        """

        ratio_mdr_prevalence \
            = float(self.outputs['epi_uncertainty']['epi'][0]['perc_incidence_mdr'][
                        last_run_output_index, t_k.find_first_list_element_at_least(
                            self.outputs['manual']['epi'][0]['times'], self.inputs.model_constants['current_time'])]) \
            / self.inputs.model_constants['tb_perc_mdr_prevalence']
        if ratio_mdr_prevalence < 1. / self.relative_difference_to_adjust_mdr:
            self.mdr_introduce_time -= self.amount_to_adjust_mdr_year
        elif ratio_mdr_prevalence > self.relative_difference_to_adjust_mdr:
            self.mdr_introduce_time += self.amount_to_adjust_mdr_year
        for scenario in self.scenarios:
            self.models[scenario].set_parameter('mdr_introduce_time', self.mdr_introduce_time)

    def record_disease_specific_adjustments(self):
        # record death reporting proportion and mdr introduction time, which may or may not have been adjusted
        if self.adjust_mortality:
            self.outputs['epi_uncertainty']['adjustments']['program_prop_death_reporting'].append(
                self.prop_death_reporting)
        if self.adjust_mdr:
            self.outputs['epi_uncertainty']['adjustments']['mdr_introduce_time'].append(self.mdr_introduce_time)
