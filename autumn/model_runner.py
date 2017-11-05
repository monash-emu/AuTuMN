
import tool_kit
import model
import os
import data_processing
import numpy
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, gamma
from Tkinter import *
from scipy.optimize import minimize
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import outputs
import autumn.economics
import itertools
from pyDOE import lhs


def generate_candidates(n_candidates, param_ranges_unc):
    """
    Function for generating candidate parameters.
    """

    # dictionary for storing candidates
    param_candidates = {}
    for param_dict in param_ranges_unc:

        # find bounds of parameter
        bound_low, bound_high = param_dict['bounds'][0], param_dict['bounds'][1]

        # draw from distribution
        if param_dict['distribution'] == 'beta':
            x = bound_low + numpy.random.beta(2., 2., n_candidates) * (bound_high - bound_low)
        elif param_dict['distribution'] == 'uniform':
            x = numpy.random.uniform(bound_low, bound_high, n_candidates)
        else:
            x = .5 * (param_dict['bounds'][0] + param_dict['bounds'][1])
            print 'Unsupported distribution specified to parameter. Defaulting to the midpoint of the range.'

        # return values
        param_candidates[param_dict['key']] = x
    return param_candidates


def elementwise_list_addition(increment, list_to_increment):
    """
    Simple method to element-wise increment a list by the values in another list of the same length.
    """

    assert len(increment) == len(list_to_increment), 'Attempted to add two lists of different lengths'
    return [sum(x) for x in zip(list_to_increment, increment)]


def elementwise_list_division(numerator, denominator):
    """
    Simple method to element-wise divide a list by the values in another list of the same length.
    """

    assert len(numerator) == len(denominator), 'Attempted to divide two lists of different lengths'
    return [n / d for n, d in zip(numerator, denominator)]


def elementwise_list_percentage(numerator, denominator):
    """
    Simple method to element-wise divide a list by the values in another list of the same length to produce a
    percentage.
    """

    assert len(numerator) == len(denominator), 'Attempted to divide two lists of different lengths'
    return [n / d * 1e2 for n, d in zip(numerator, denominator)]


def find_integer_dict_from_float_dict(float_dict):
    # Method may be redundant with better code

    times = float_dict.keys()
    times.sort()
    start, finish = (numpy.floor(times[0]), numpy.floor(times[-1]))
    float_years = numpy.linspace(start, finish, finish - start + 1.)
    integer_dict = {}
    for year in float_years:
        key = [t for t in times if t >= year][0]
        integer_dict[int(key)] = float_dict[key]
    return integer_dict


def extract_integer_dicts(models_to_analyse={}, dict_to_extract_from={}):
    # method may be redundant with better code

    integer_dict = {}
    for scenario in models_to_analyse:
        integer_dict[scenario] = {}
        for output in dict_to_extract_from[scenario]:
            integer_dict[scenario][output] = find_integer_dict_from_float_dict(dict_to_extract_from[scenario][output])
    return integer_dict


def get_output_dicts_from_lists(models_to_analyse={}, output_dict_of_lists={}):
    """
    Convert output lists to dictionaries. Also may ultimately be unnecessary.
    """

    output_dictionary = {}
    for scenario in models_to_analyse:
        output_dictionary[scenario] = {}
        for output in output_dict_of_lists[scenario]:
            if output != 'times':
                output_dictionary[scenario][output] \
                    = dict(zip(output_dict_of_lists[scenario]['times'], output_dict_of_lists[scenario][output]))
    return output_dictionary


def find_uncertainty_output_weights(list, method, relative_weights=[1., 2.]):
    """
    Creates a set of "weights" to determine the proportion of the log-likelihood to be contributed by the years
    considered in the calibration.

    Args:
        list: A list of the years that the weights are to be applied to.
        method: Choice of method.
        relative_weights: Relative size of the starting and ending weights if method is 1.
    """

    # linearly scaling weights summing to one
    if method == 1:
        if len(list) == 1:
            return [1.]
        else:
            weights = numpy.linspace(relative_weights[0], relative_weights[1], num=len(list))
            return [i / sum(weights) for i in weights]

    # equally distributed weights summing to one
    elif method == 2:
        return [1. / float(len(list))] * len(list)

    # all weights equal to one
    elif method == 3:
        return [1.] * len(list)


def is_parameter_value_valid(parameter):
    """
    Determine whether a number is finite and positive and so valid for the model as a parameter.
    """

    return numpy.isfinite(parameter) and parameter > 0.


def find_log_probability_density(distribution, param_val, bounds, additional_params=None):
    """
    Find the log probability density for the parameter value being considered.

    Args:
        distribution: String specifying the general type of distribution
        param_val: The parameter value
        bounds: Two element list for the upper and lower limits of the distribution
        prior_log_likelihood: Preceding log likelihood for incrementing
        additional_params: Any additional parameters to the distribution if not completely specified with bounds
    Returns:
        prior_log_likelihood: Prior log likelihood associated with the individual parameter fed in to this function
    """

    # save some code repetition by finding the parameter value's distance through the distribution width
    normalised_param_value = (param_val - bounds[0]) / (bounds[1] - bounds[0])

    # find the log probability density
    if distribution == 'uniform':
        prior_log_likelihood = numpy.log(1. / (bounds[1] - bounds[0]))
    elif distribution == 'beta_2_2':
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
    return prior_log_likelihood


class ModelRunner:
    def __init__(self, gui_inputs, runtime_outputs, figure_frame, js_gui=None):
        """
        Instantiation method for model runner - currently including some attributes that should be set externally, e.g.
        in the GUI(s).

        Args:
            gui_inputs: Inputs from the off-line Tkinter GUI
            runtime_outputs: Offline GUI window for commenting
            figure_frame: Uncertainty parameter plotting window of Tkinter GUI
            js_gui: JavaScript GUI inputs
        """

        # conversion of inputs to attributes
        self.gui_inputs = gui_inputs
        self.scenarios = self.gui_inputs['scenarios_to_run']
        self.runtime_outputs = runtime_outputs
        self.figure_frame = figure_frame
        self.inputs = data_processing.Inputs(gui_inputs, runtime_outputs, js_gui=js_gui)
        self.inputs.read_and_load_data()

        # preparing for basic runs
        self.models = {}
        self.interventions_to_cost = self.inputs.interventions_to_cost

        # uncertainty-related attributes
        self.is_last_run_success = False
        self.loglikelihoods = []
        self.outputs_unc = [{'key': 'incidence',
                             'posterior_width': None,
                             'width_multiplier': 2.  # width of normal posterior relative to range of allowed values
                             }]
        self.all_parameters_tried = {}  # all refers to applying to every model run (rather than accepted only)
        self.all_compartment_values_tried = {}
        self.all_other_adjustments_made = {}
        self.whether_accepted_list = []
        self.accepted_indices = []
        self.rejected_indices = []
        self.solns_for_extraction = ['compartment_soln', 'fraction_soln']
        self.arrays_for_extraction = ['flow_array', 'fraction_array', 'soln_array', 'var_array', 'costs']
        self.acceptance_dict = {}
        self.rejection_dict = {}
        self.uncertainty_percentiles = {}
        self.n_centiles_for_shading = 100
        self.percentiles = [2.5, 50., 97.5] + list(numpy.linspace(0., 100., self.n_centiles_for_shading * 2 + 1))
        self.random_start = False  # whether to start from a random point, as opposed to the manually calibrated value
        self.intervention_uncertainty = self.inputs.intervention_uncertainty
        self.relative_difference_to_adjust_mortality = 1.1
        self.amount_to_adjust_mortality = .02

        # optimisation attributes
        self.optimisation = False  # leave True even if loading optimisation results
        self.opti_outputs_dir = 'saved_optimisation_analyses'
        self.indicator_to_minimise = 'incidence'  # currently must be 'incidence' or 'mortality'
        self.annual_envelope = [112.5e6]  # size of funding envelope in scenarios to be run
        self.save_opti = True
        self.load_optimisation = False  # optimisation will not be run if true
        self.total_funding = None  # funding for entire period
        self.f_tol = {'incidence': 0.5,
                      'mortality': 0.05}  # stopping condition for optimisation algorithm (differs by indicator)
        self.year_end_opti = 2035.  # model is run until that date during optimisation
        self.acceptable_combinations = []  # list of intervention combinations that can be considered with funding
        self.opti_results = {}  # store all the results that we need for optimisation
        self.optimised_combinations = []
        self.optimal_allocation = {}
        self.interventions_considered_for_opti \
            = ['engage_lowquality', 'xpert', 'cxrxpertacf_prison', 'cxrxpertacf_urbanpoor', 'ipt_age0to5',
               'intensive_screening']  # interventions that must appear in optimal plan
        self.interventions_forced_for_opti = ['engage_lowquality', 'ipt_age0to5', 'intensive_screening']

        # output-related attributes
        self.epi_outputs_to_analyse = ['incidence', 'prevalence', 'mortality', 'true_mortality', 'notifications']
        self.epi_outputs = {}
        self.epi_outputs_uncertainty = {}
        self.epi_outputs_uncertainty_centiles = None
        self.cost_outputs = {}
        self.cost_outputs_dict = {}
        self.cost_outputs_integer_dict = {}
        self.cost_outputs_uncertainty = {}
        self.cost_outputs_uncertainty_centiles = None
        self.additional_cost_types = ['inflated', 'discounted', 'discounted_inflated']
        self.cost_types = self.additional_cost_types + ['raw']

        # saving-related
        self.attributes_to_save \
            = ['epi_outputs', 'epi_outputs_uncertainty', 'cost_outputs', 'cost_outputs_dict',
               'cost_outputs_integer_dict', 'cost_outputs_uncertainty', 'accepted_indices', 'rejected_indices',
               'all_parameters_tried', 'whether_accepted_list', 'acceptance_dict', 'rejection_dict', 'loglikelihoods',
               'all_other_adjustments_made']

        # GUI-related
        self.emit_delay = 0.1
        self.plot_count = 0
        self.js_gui = js_gui
        if self.js_gui: self.js_gui('init')

    ''' master methods to run other methods '''

    def master_runner(self):
        """
        Calls methods to run model with each of the three fundamental approaches.
        """

        # prepare file for saving
        out_dir = 'saved_uncertainty_analyses'
        if not os.path.isdir(out_dir): os.makedirs(out_dir)
        storage_file_name = os.path.join(out_dir, 'store.pkl')

        # load a saved simulation
        if self.gui_inputs['pickle_uncertainty'] == 'Load':
            self.add_comment_to_gui_window('Loading results from previous simulation')
            loaded_data = tool_kit.pickle_load(storage_file_name)
            self.add_comment_to_gui_window('Loading finished')
            for attribute in loaded_data: setattr(self, attribute, loaded_data[attribute])

        # or run the manual scenarios as requested by user
        else:
            self.run_manual_calibration()
            if self.gui_inputs['output_uncertainty']: self.run_epi_uncertainty()
            if self.intervention_uncertainty: self.run_intervention_uncertainty()

        # save uncertainty if requested
        if self.gui_inputs['pickle_uncertainty'] == 'Save':
            data_to_save = {}
            for attribute in self.attributes_to_save: data_to_save[attribute] = getattr(self, attribute)
            tool_kit.pickle_save(data_to_save, storage_file_name)
            self.add_comment_to_gui_window('Uncertainty results saved to disc')

        # master optimisation method
        if self.optimisation and not self.load_optimisation: self.run_optimisation()

        # prepare file for saving, save and load as requested
        if not os.path.isdir(self.opti_outputs_dir): os.makedirs(self.opti_outputs_dir)
        self.load_opti_results()
        self.save_opti_results()

        # notify user that model running has finished
        self.add_comment_to_gui_window('Model running complete')

    def run_manual_calibration(self):
        """
        Runs the scenarios a single time, starting from baseline with parameter values as specified in spreadsheets.
        """

        for scenario in self.scenarios:

            # name and initialise model
            scenario_name = 'manual_' + tool_kit.find_scenario_string_from_number(scenario)
            self.models[scenario_name] = model.ConsolidatedModel(scenario, self.inputs, self.gui_inputs)

            # sort out times for scenario runs
            if scenario > 0: self.prepare_new_model_from_baseline('manual', scenario_name)

            # describe model to user and integrate
            self.add_comment_to_gui_window('Running %s conditions for %s using point estimates for parameters.'
                                           % (scenario_name, self.gui_inputs['country']))
            self.models[scenario_name].integrate()

            # model interpretation for each scenario
            self.epi_outputs[scenario_name] \
                = self.find_epi_outputs(scenario_name, outputs_to_analyse=self.epi_outputs_to_analyse,
                                        stratifications=[self.models[scenario_name].agegroups,
                                                         self.models[scenario_name].riskgroups])
            if len(self.models[scenario_name].interventions_to_cost) > 0: self.find_cost_outputs(scenario_name)
            self.find_population_fractions(
                scenario_name=scenario_name, stratifications=[self.models[scenario_name].agegroups,
                                                              self.models[scenario_name].riskgroups])

    def prepare_new_model_from_baseline(self, run_type, scenario_name):
        """
        Method to set the start time of a model and load the compartment values from the baseline run.

        Args:
            run_type: The type of run for the model object to be set
            scenario_name: Either the scenario name or optimisation if during optimisation run
        """

        scenario_start_time_index = \
            self.models[run_type + '_baseline'].find_time_index(
                self.inputs.model_constants['before_intervention_time'])
        start_time = self.models[run_type + '_baseline'].times[scenario_start_time_index]
        self.models[scenario_name].start_time = start_time
        self.models[scenario_name].next_time_point = start_time
        self.models[scenario_name].loaded_compartments \
            = self.models[run_type + '_baseline'].load_state(scenario_start_time_index)

    ''' output interpretation methods '''

    def find_epi_outputs(self, scenario, outputs_to_analyse, stratifications=[]):
        """
        Method to extract all requested epidemiological outputs from the models. Intended ultimately to be flexible\
        enough for use for analysis of scenarios, uncertainty and optimisation.

        Args:
            scenario: The number value representing the scenario of the model to be analysed
            outputs_to_analyse: List of strings for the outputs of interest to be worked through
            stratifications: Whether it is necessary to provide outputs by any model compartmental stratifications
        """

        ''' compulsory elements to calculate '''

        if 'population' not in outputs_to_analyse: outputs_to_analyse.append('population')
        epi_outputs = {'times': self.models[scenario].times}

        ''' unstratified outputs '''

        # initialise lists to zeros to allow incrementation
        for output in outputs_to_analyse:
            epi_outputs[output] = [0.] * len(epi_outputs['times'])
            for strain in self.models[scenario].strains:
                epi_outputs[output + strain] = [0.] * len(epi_outputs['times'])

        # population
        for compartment in self.models[scenario].compartments:
            epi_outputs['population'] \
                = elementwise_list_addition(self.models[scenario].get_compartment_soln(compartment),
                                            epi_outputs['population'])

        # replace zeroes with small numbers for division
        total_denominator = tool_kit.prepare_denominator(epi_outputs['population'])

        # to allow calculation by strain and the total output
        strains = self.models[scenario].strains + ['']

        # incidence
        if 'incidence' in outputs_to_analyse:
            for strain in strains:
                for from_label, to_label, rate in self.models[scenario].var_transfer_rate_flows:  # variable flows
                    if 'latent' in from_label and 'active' in to_label and strain in to_label:
                        incidence_increment = self.models[scenario].get_compartment_soln(from_label) \
                                              * self.models[scenario].get_var_soln(rate) / total_denominator * 1e5
                        epi_outputs['incidence' + strain] \
                            = elementwise_list_addition(incidence_increment, epi_outputs['incidence' + strain])
                for from_label, to_label, rate in self.models[scenario].fixed_transfer_rate_flows:  # fixed flows
                    if 'latent' in from_label and 'active' in to_label and strain in to_label:
                        incidence_increment = self.models[scenario].get_compartment_soln(from_label) \
                                              * rate / total_denominator * 1e5
                        epi_outputs['incidence' + strain] \
                            = elementwise_list_addition(incidence_increment, epi_outputs['incidence' + strain])

            # find percentage incidence by strain
            if len(self.models[scenario].strains) > 1:
                for strain in self.models[scenario].strains:
                    epi_outputs['perc_incidence' + strain] \
                        = elementwise_list_percentage(epi_outputs['incidence' + strain],
                                                      tool_kit.prepare_denominator(epi_outputs['incidence']))

        # notifications
        if 'notifications' in outputs_to_analyse:
            for strain in strains:
                for from_label, to_label, rate in self.models[scenario].var_transfer_rate_flows:
                    if 'active' in from_label and 'detect' in to_label and strain in to_label:
                        notifications_increment \
                            = self.models[scenario].get_compartment_soln(from_label) \
                              * self.models[scenario].get_var_soln(rate)
                        epi_outputs['notifications' + strain] \
                            = elementwise_list_addition(notifications_increment, epi_outputs['notifications' + strain])

        # mortality
        if 'mortality' in outputs_to_analyse:
            for strain in strains:

                # fixed flows are outside of the health system and so the natural death contribution is reduced
                for from_label, rate in self.models[scenario].fixed_infection_death_rate_flows:
                    if strain in from_label:
                        mortality_increment = self.models[scenario].get_compartment_soln(from_label) \
                                              * rate / total_denominator * 1e5
                        epi_outputs['true_mortality' + strain] \
                            = elementwise_list_addition(mortality_increment, epi_outputs['true_mortality' + strain])
                        epi_outputs['mortality' + strain] \
                            = elementwise_list_addition(
                                mortality_increment * self.models[scenario].params['program_prop_death_reporting'],
                                epi_outputs['mortality' + strain])

                # variable flows are within the health system and so true and reported are dealt with the same way
                for from_label, rate in self.models[scenario].var_infection_death_rate_flows:
                    if strain in from_label:
                        mortality_increment = self.models[scenario].get_compartment_soln(from_label) \
                                              * self.models[scenario].get_var_soln(rate) / total_denominator * 1e5
                        for mortality_type in ['true_mortality', 'mortality']:
                            epi_outputs[mortality_type + strain] \
                                = elementwise_list_addition(mortality_increment, epi_outputs[mortality_type + strain])

        # prevalence
        if 'prevalence' in outputs_to_analyse:
            for strain in strains:
                for label in self.models[scenario].labels:
                    if 'susceptible' not in label and 'latent' not in label and strain in label:
                        prevalence_increment \
                            = self.models[scenario].get_compartment_soln(label) / total_denominator * 1e5
                        epi_outputs['prevalence' + strain] \
                            = elementwise_list_addition(prevalence_increment, epi_outputs['prevalence' + strain])

        # infections (absolute number)
        if 'infections' in outputs_to_analyse:
            for strain in strains:
                for from_label, to_label, rate in self.models[scenario].var_transfer_rate_flows:
                    if 'latent_early' in to_label and strain in to_label:
                        epi_outputs['infections' + strain] \
                            = elementwise_list_addition(self.models[scenario].get_compartment_soln(from_label)
                                                        * self.models[scenario].get_var_soln(rate),
                                                        epi_outputs['infections' + strain])

                # annual risk of infection (as a percentage)
                epi_outputs['annual_risk_infection' + strain] \
                    = elementwise_list_percentage(epi_outputs['infections' + strain], total_denominator)

        ''' stratified outputs (currently not repeated for each strain) '''

        for stratification in stratifications:
            if len(stratification) > 1:
                for stratum in stratification:

                    # initialise lists
                    for output in outputs_to_analyse: epi_outputs[output + stratum] = [0.] * len(epi_outputs['times'])

                    # population
                    for compartment in self.models[scenario].compartments:
                        if stratum in compartment:
                            epi_outputs['population' + stratum] \
                                = elementwise_list_addition(self.models[scenario].get_compartment_soln(compartment),
                                                            epi_outputs['population' + stratum])

                    # the population denominator to be used with zeros replaced with small numbers
                    stratum_denominator = tool_kit.prepare_denominator(epi_outputs['population' + stratum])

                    # incidence
                    if 'incidence' in outputs_to_analyse:
                        for from_label, to_label, rate in self.models[scenario].var_transfer_rate_flows:
                            if 'latent' in from_label and 'active' in to_label and stratum in from_label:
                                incidence_increment = self.models[scenario].get_compartment_soln(from_label) \
                                                      * self.models[scenario].get_var_soln(rate) \
                                                      / stratum_denominator * 1e5
                                epi_outputs['incidence' + stratum] \
                                    = elementwise_list_addition(incidence_increment, epi_outputs['incidence' + stratum])
                        for from_label, to_label, rate in self.models[scenario].fixed_transfer_rate_flows:
                            if 'latent' in from_label and 'active' in to_label and stratum in from_label:
                                incidence_increment = self.models[scenario].get_compartment_soln(from_label) \
                                                      * rate / stratum_denominator * 1e5
                                epi_outputs['incidence' + stratum] \
                                    = elementwise_list_addition(incidence_increment, epi_outputs['incidence' + stratum])

                    # notifications
                    if 'notifications' in outputs_to_analyse:
                        for strain in strains:
                            for from_label, to_label, rate in self.models[scenario].var_transfer_rate_flows:
                                if 'active' in from_label and 'detect' in to_label and strain in to_label \
                                        and stratum in from_label:
                                    notifications_increment \
                                        = self.models[scenario].get_compartment_soln(from_label) \
                                          * self.models[scenario].get_var_soln(rate)
                                    epi_outputs['notifications' + strain + stratum] \
                                        = elementwise_list_addition(
                                            notifications_increment, epi_outputs['notifications' + strain + stratum])

                    # mortality
                    if 'mortality' in outputs_to_analyse:

                        # fixed flows are outside of the health system and so the natural death contribution is reduced
                        for from_label, rate in self.models[scenario].fixed_infection_death_rate_flows:
                            if stratum in from_label:
                                mortality_increment = self.models[scenario].get_compartment_soln(from_label) \
                                                      * rate / stratum_denominator * 1e5
                                epi_outputs['true_mortality' + stratum] \
                                    = elementwise_list_addition(mortality_increment,
                                                                epi_outputs['true_mortality' + stratum])
                                epi_outputs['mortality' + stratum] \
                                    = elementwise_list_addition(
                                        mortality_increment
                                        * self.models[scenario].params['program_prop_death_reporting'],
                                        epi_outputs['mortality' + stratum])

                        # variable flows are within the health system and so dealt with as described above
                        for from_label, rate in self.models[scenario].var_infection_death_rate_flows:
                            if stratum in from_label:
                                mortality_increment = self.models[scenario].get_compartment_soln(from_label) \
                                                      * self.models[scenario].get_var_soln(rate) \
                                                      / stratum_denominator * 1e5
                                for mortality_type in ['true_mortality', 'mortality']:
                                    epi_outputs[mortality_type + stratum] \
                                        = elementwise_list_addition(mortality_increment,
                                                                    epi_outputs[mortality_type + stratum])

                    # prevalence
                    if 'prevalence' in outputs_to_analyse:
                        for label in self.models[scenario].labels:
                            if 'susceptible' not in label and 'latent' not in label and stratum in label:
                                prevalence_increment = self.models[scenario].get_compartment_soln(label) \
                                                       / stratum_denominator * 1e5
                                epi_outputs['prevalence' + stratum] \
                                    = elementwise_list_addition(prevalence_increment,
                                                                epi_outputs['prevalence' + stratum])

                    # infections (absolute number)
                    if 'infections' in outputs_to_analyse:
                        for from_label, to_label, rate in self.models[scenario].var_transfer_rate_flows:
                            if 'latent_early' in to_label and stratum in from_label:
                                epi_outputs['infections' + stratum] \
                                    = elementwise_list_addition(
                                        self.models[scenario].get_compartment_soln(from_label)
                                        * self.models[scenario].get_var_soln(rate),
                                        epi_outputs['infections' + stratum])

                        # annual risk of infection (as a percentage)
                        epi_outputs['annual_risk_infection' + stratum] \
                            = elementwise_list_percentage(epi_outputs['infections' + stratum], stratum_denominator)

        return epi_outputs

    def find_population_fractions(self, scenario_name, stratifications=[]):
        """
        Find the proportion of the population in various strata. The stratifications must apply to the entire
        population, so this method should not be used for strains, health systems, etc.
        """

        for stratification in stratifications:
            if len(stratification) > 1:
                for stratum in stratification:
                    self.epi_outputs[scenario_name]['fraction' + stratum] \
                        = elementwise_list_division(self.epi_outputs[scenario_name]['population' + stratum],
                                                    self.epi_outputs[scenario_name]['population'])

    def find_cost_outputs(self, scenario_name):
        """
        Master method to call methods to find and update costs below.

        Args:
            scenario_name: String for the name of the model being costed.
        """

        self.cost_outputs[scenario_name] = self.find_raw_cost_outputs(scenario_name)
        self.cost_outputs[scenario_name]['raw_cost_all_programs'] = self.find_costs_all_programs(scenario_name)
        self.cost_outputs[scenario_name].update(self.find_adjusted_costs(scenario_name))

    def find_raw_cost_outputs(self, scenario_name):
        """
        Find cost dictionaries to add to cost_outputs attribute.
        """

        cost_outputs = {'times': self.models[scenario_name].cost_times}
        for i, intervention \
                in enumerate(self.interventions_to_cost[tool_kit.find_scenario_number_from_string(scenario_name)]):
            cost_outputs['raw_cost_' + intervention] = self.models[scenario_name].costs[:, i]
        return cost_outputs

    def find_costs_all_programs(self, scenario_name):
        """
        Sum costs across all programs and populate to cost_outputs dictionary for each scenario.
        """

        costs_all_programs \
            = [0.] * len(self.cost_outputs[scenario_name]['raw_cost_' + self.interventions_to_cost[
                tool_kit.find_scenario_number_from_string(scenario_name)][0]])
        for intervention in self.interventions_to_cost[tool_kit.find_scenario_number_from_string(scenario_name)]:
            costs_all_programs \
                = elementwise_list_addition(self.cost_outputs[scenario_name]['raw_cost_' + intervention],
                                            costs_all_programs)
        return costs_all_programs

    def find_adjusted_costs(self, scenario_name):
        """
        Find costs adjusted for inflation, discounting and both.

        Args:
            scenario_name: Scenario being costed
        """

        # get some preliminary parameters
        year_current = self.inputs.model_constants['recent_time']
        current_cpi = self.inputs.scaleup_fns[0]['econ_cpi'](year_current)
        discount_rate = self.inputs.model_constants['econ_discount_rate']

        # loop over interventions for costing and cost types
        cost_outputs = {}
        for intervention in self.interventions_to_cost[tool_kit.find_scenario_number_from_string(scenario_name)] \
                + ['all_programs']:
            for cost_type in self.additional_cost_types:
                cost_outputs[cost_type + '_cost_' + intervention] = []
                for t, time in enumerate(self.cost_outputs[scenario_name]['times']):
                    cost_outputs[cost_type + '_cost_' + intervention].append(
                        autumn.economics.get_adjusted_cost(
                            self.cost_outputs[scenario_name]['raw_cost_' + intervention][t], cost_type, current_cpi,
                            self.inputs.scaleup_fns[0]['econ_cpi'](time), discount_rate,
                            max(0., (time - year_current))))
        return cost_outputs

    ''' epidemiological uncertainty methods '''

    def run_epi_uncertainty(self):
        """
        Main method to run all the uncertainty processes using a Metropolis-Hastings algorithm with normal proposal
        distribution.
        """

        self.add_comment_to_gui_window('Uncertainty analysis commenced')

        param_candidates = {}
        for param in self.inputs.param_ranges_unc:
            param_candidates[param['key']] = [self.inputs.model_constants[param['key']]]

        # find weights for outputs that are being calibrated to
        years_to_compare = range(1990, 2015)
        weights = find_uncertainty_output_weights(years_to_compare, 1, [1., 2.])
        self.add_comment_to_gui_window('"Weights": \n' + str(weights))

        # prepare for uncertainty loop
        n_accepted, prev_log_likelihood = 0, -5e2
        for param in self.inputs.param_ranges_unc:
            self.all_parameters_tried[param['key']] = []
            self.acceptance_dict[param['key']] = {}
            self.rejection_dict[param['key']] = {n_accepted: []}
        for compartment_type in self.inputs.compartment_types:
            if compartment_type in self.inputs.model_constants: self.all_compartment_values_tried[compartment_type] = []
        self.all_other_adjustments_made['program_prop_death_reporting'] = []

        # instantiate uncertainty model objects
        for scenario in self.scenarios:
            scenario_name = 'uncertainty_' + tool_kit.find_scenario_string_from_number(scenario)
            self.models[scenario_name] = model.ConsolidatedModel(scenario, self.inputs, self.gui_inputs)

        # set initial parameter values
        new_param_list = []
        for param in self.inputs.param_ranges_unc:
            new_param_list.append(param_candidates[param['key']][0])
            params = new_param_list
        run, population_adjustment, accepted = 0, 1., 0

        while n_accepted < self.gui_inputs['uncertainty_runs']:

            # set timer
            start_timer_run = datetime.datetime.now()

            # run baseline scenario (includes parameter checking, parameter setting and recording success/failure)
            self.run_with_params(new_param_list, model_object='uncertainty_baseline',
                                 population_adjustment=population_adjustment, accepted=accepted)

            # store outputs regardless of acceptance, provided run was completed successfully
            if self.is_last_run_success:

                # get outputs for calibration and store results
                self.store_uncertainty('uncertainty_baseline', epi_outputs_to_analyse=self.epi_outputs_to_analyse)
                integer_dictionary \
                    = extract_integer_dicts(['uncertainty_baseline'], get_output_dicts_from_lists(
                        models_to_analyse=['uncertainty_baseline'], output_dict_of_lists=self.epi_outputs))

                # calculate prior
                prior_log_likelihood = 0.
                for p, param in enumerate(self.inputs.param_ranges_unc):
                    param_val = new_param_list[p]
                    self.all_parameters_tried[param['key']].append(new_param_list[p])
                    if 'additional_params' not in param: param['additional_params'] = None
                    prior_log_likelihood \
                        += find_log_probability_density(param['distribution'], param_val,  param['bounds'],
                                                        additional_params=param['additional_params'])

                # calculate posterior
                posterior_log_likelihood = 0.
                for output_dict in self.outputs_unc:

                    # the GTB values for the output of interest
                    working_output_dictionary = self.get_fitting_data()[output_dict['key']]
                    for y, year in enumerate(years_to_compare):
                        if year in working_output_dictionary.keys():
                            model_result_for_output = integer_dictionary['uncertainty_baseline']['incidence'][year]
                            mu, sd = working_output_dictionary[year][0], working_output_dictionary[year][1]
                            posterior_log_likelihood += norm.logpdf(model_result_for_output, mu, sd) * weights[y]

                # determine acceptance
                log_likelihood = prior_log_likelihood + posterior_log_likelihood
                accepted = numpy.random.binomial(n=1, p=min(1., numpy.exp(log_likelihood - prev_log_likelihood)))

                # describe progression of likelihood analysis
                self.add_comment_to_gui_window(
                    'Previous log likelihood:\n%4.3f\nLog likelihood this run:\n%4.3f\nAcceptance probability:\n%4.3f'
                    % (log_likelihood, prev_log_likelihood, min(1., numpy.exp(log_likelihood - prev_log_likelihood)))
                    + '\nWhether accepted:\n%s\n________________\n' % str(bool(accepted)))
                self.loglikelihoods.append(log_likelihood)

                # record starting population
                if self.gui_inputs['write_uncertainty_outcome_params']:
                    for compartment_type in self.all_compartment_values_tried:
                        self.all_compartment_values_tried[compartment_type].append(
                            self.inputs.model_constants[compartment_type])

                # record uncertainty calculations for all runs
                if accepted:
                    self.whether_accepted_list.append(True)
                    self.accepted_indices.append(run)
                    n_accepted += 1
                    for p, param in enumerate(self.inputs.param_ranges_unc):
                        self.acceptance_dict[param['key']][n_accepted] = new_param_list[p]
                        self.rejection_dict[param['key']][n_accepted] = []

                    # update likelihood and parameter set for next run
                    prev_log_likelihood = log_likelihood
                    params = new_param_list

                    # run scenarios other than baseline and store uncertainty (only if accepted)
                    for scenario in self.scenarios:
                        if scenario:
                            scenario_name = 'uncertainty_' + tool_kit.find_scenario_string_from_number(scenario)
                            self.prepare_new_model_from_baseline('uncertainty', scenario_name)
                            scenario_name = 'uncertainty_' + tool_kit.find_scenario_string_from_number(scenario)
                            self.run_with_params(new_param_list, model_object=scenario_name)
                            self.store_uncertainty(scenario_name, epi_outputs_to_analyse=self.epi_outputs_to_analyse)

                    # iteratively adjusting proportion of mortality reported
                    ratios = []
                    for year in years_to_compare:
                        if year in self.inputs.original_data['tb']['e_mort_exc_tbhiv_100k']:
                            ratios.append(self.epi_outputs['uncertainty_baseline']['mortality'][
                                             tool_kit.find_first_list_element_above_value(self.epi_outputs[
                                                  'uncertainty_baseline']['times'], float(year))]
                                         / self.inputs.original_data['tb']['e_mort_exc_tbhiv_100k'][year])
                    average_ratio = numpy.mean(ratios)
                    if average_ratio < 1. / self.relative_difference_to_adjust_mortality:
                        self.inputs.model_constants['program_prop_death_reporting'] += self.amount_to_adjust_mortality
                    elif average_ratio > self.relative_difference_to_adjust_mortality:
                        self.inputs.model_constants['program_prop_death_reporting'] -= self.amount_to_adjust_mortality

                else:
                    self.whether_accepted_list.append(False)
                    self.rejected_indices.append(run)
                    for p, param in enumerate(self.inputs.param_ranges_unc):
                        self.rejection_dict[param['key']][n_accepted].append(new_param_list[p])

                # plot parameter progression and report on progress
                self.plot_progressive_parameters()
                self.add_comment_to_gui_window(
                    str(n_accepted) + ' accepted / ' + str(run) + ' candidates. Running time: '
                    + str(datetime.datetime.now() - start_timer_run))

                # find value to adjust starting population by, if a target population specified
                if 'target_population' in self.inputs.model_constants:
                    population_adjustment \
                        = self.inputs.model_constants['target_population'] \
                          / self.epi_outputs['uncertainty_baseline']['population'][
                              tool_kit.find_first_list_element_above_value(
                                  self.epi_outputs['uncertainty_baseline']['times'],
                                  self.inputs.model_constants['current_time'])]

                # record death reporting proportion, which may or may not have been adjusted
                self.all_other_adjustments_made['program_prop_death_reporting'].append(
                    self.inputs.model_constants['program_prop_death_reporting'])

                run += 1

            new_param_list = self.update_params(params)

    def get_fitting_data(self):
        """
        Define the characteristics (mean and standard deviation) of the normal distribution for model outputs
        (incidence, mortality).

        Returns:
            normal_char: Dictionary with keys outputs and values dictionaries. Sub-dictionaries have keys years
                and values lists, with first element of list means and second standard deviations.
        """

        # dictionary storing the characteristics of the normal distributions
        normal_char = {}
        for output_dict in self.inputs.outputs_unc:
            normal_char[output_dict['key']] = {}

            # incidence
            if output_dict['key'] == 'incidence':
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    low = self.inputs.data_to_fit['incidence_low'][year]
                    high = self.inputs.data_to_fit['incidence_high'][year]
                    sd = output_dict['width_multiplier'] * (high - low) / (2. * 1.96)
                    mu = (high + low) / 2.
                    normal_char[output_dict['key']][year] = [mu, sd]

            # mortality
            elif output_dict['key'] == 'mortality':
                sd = output_dict['posterior_width'] / (2. * 1.96)
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    mu = self.inputs.data_to_fit[output_dict['key']][year]
                    normal_char[output_dict['key']][year] = [mu, sd]

        return normal_char

    def run_with_params(self, params, model_object='uncertainty_baseline', population_adjustment=1., accepted=0):
        """
        Integrate the model with the proposed parameter set.

        Args:
            params: The parameters to be set in the model.
        """

        # check whether parameter values are acceptable
        for p, param in enumerate(params):

            # whether the parameter value is valid
            if not is_parameter_value_valid(param):
                print 'Warning: parameter%d=%f is invalid for model' % (p, param)
                self.is_last_run_success = False
                return

            # whether the parameter value is within acceptable ranges
            bounds = self.inputs.param_ranges_unc[p]['bounds']
            if param < bounds[0] or param > bounds[1]:
                print 'Warning: parameter%d=%f is outside of the allowed bounds' % (p, param)
                self.is_last_run_success = False
                return

        param_dict = {names['key']: vals for names, vals in zip(self.inputs.param_ranges_unc, params)}

        # set parameters and run
        self.set_model_with_params(param_dict, model_object, population_adjustment=population_adjustment,
                                   accepted=accepted)
        self.is_last_run_success = True
        try:
            self.models[model_object].integrate()
        except:
            print 'Warning: parameters=%s failed with model' % params
            self.is_last_run_success = False

    def set_model_with_params(self, param_dict, model_object='baseline', population_adjustment=1., accepted=0):
        """
        Populates baseline model with params from uncertainty calculations, including adjusting starting time.
        Also adjusts starting population to better match target population at current time using target_population input
        from country sheet. (Not currently in default sheet.)

        Args:
            param_dict: Dictionary of the parameters to be set within the model (keys parameter name strings and values
                parameter values).
        """

        # adjust starting populations if target_population in sheets (i.e. country sheet, because not in defaults)
        if accepted and population_adjustment != 1.:
            for compartment_type in self.inputs.compartment_types:
                if compartment_type in self.inputs.model_constants:
                    self.inputs.model_constants[compartment_type] *= population_adjustment

        for key in param_dict:

            # start time usually set in instantiation, which has already been done here, so needs to be set separately
            if key == 'start_time': self.models[model_object].start_time = param_dict[key]

            # set parameters
            elif key in self.models[model_object].params:
                self.models[model_object].set_parameter(key, param_dict[key])
            else:
                raise ValueError('%s not in model_object params' % key)

    def store_uncertainty(self, scenario_name, epi_outputs_to_analyse):
        """
        Add model results from one uncertainty run to the appropriate outputs dictionary, vertically stacking
        results on to the previous matrix.

        Args:
            scenario_name: The scenario being run.
            epi_outputs_to_analyse: The epidemiological outputs of interest.
        Updates:
            self.epi_outputs_uncertainty
            self.cost_outputs_uncertainty
        """

        # get outputs
        self.epi_outputs[scenario_name] \
            = self.find_epi_outputs(scenario_name, outputs_to_analyse=self.epi_outputs_to_analyse)
        if self.models[scenario_name].interventions_to_cost: self.find_cost_outputs(scenario_name)

        # initialise dictionaries if needed
        if scenario_name not in self.epi_outputs_uncertainty:
            self.epi_outputs_uncertainty[scenario_name] = {'times': self.epi_outputs[scenario_name]['times']}
            self.cost_outputs_uncertainty[scenario_name] = {'times': self.cost_outputs[scenario_name]['times']}
            for output in epi_outputs_to_analyse:
                self.epi_outputs_uncertainty[scenario_name][output] \
                    = numpy.empty(shape=[0, len(self.epi_outputs[scenario_name]['times'])])
            for output in self.cost_outputs[scenario_name]:
                self.cost_outputs_uncertainty[scenario_name][output] \
                    = numpy.empty(shape=[0, len(self.cost_outputs[scenario_name]['times'])])

        # add uncertainty data to dictionaries
        scenario_for_length = 'manual_baseline'
        if scenario_name == 'manual_scenario_15': scenario_for_length = 'manual_scenario_15'
        for output in epi_outputs_to_analyse:
            new_output = tool_kit.force_list_to_length(self.epi_outputs[scenario_name][output],
                                                       len(self.epi_outputs[scenario_for_length][output]))
            self.epi_outputs_uncertainty[scenario_name][output] \
                = numpy.vstack([self.epi_outputs_uncertainty[scenario_name][output], new_output])
        for output in self.cost_outputs[scenario_name]:
            self.cost_outputs_uncertainty[scenario_name][output] \
                = numpy.vstack([self.cost_outputs_uncertainty[scenario_name][output],
                                self.cost_outputs[scenario_name][output]])

    def update_params(self, old_params):
        """
        Update all the parameter values being used in the uncertainty analysis.

        Args:
            old_params:
        Returns:
            new_params: The new parameters to be used in the next model run.
        """

        new_params = []

        # iterate through the parameters being used
        for p, param_dict in enumerate(self.inputs.param_ranges_unc):
            bounds = param_dict['bounds']
            sd = self.gui_inputs['search_width'] * (bounds[1] - bounds[0]) / (2. * 1.96)
            random = -100.

            # search for new parameters
            while random < bounds[0] or random > bounds[1]: random = norm.rvs(loc=old_params[p], scale=sd, size=1)

            # add them to the dictionary
            new_params.append(random[0])

        return new_params

    ''' other run type methods '''

    def run_intervention_uncertainty(self):
        """
        Master method for running intervention uncertainty. That is, starting from the calibrated baseline simulated,
        project forward scenarios based on varying parameters for the effectiveness of the intervention under
        consideration.

        Args:
            intervention: String for intervention of interest
            n_samples: Number of samples to explore
        """

        # extract relevant intervention parameters from the intervention uncertainty dictionary
        working_param_dict = {}
        for param in self.inputs.intervention_param_dict[self.inputs.uncertainty_intervention]:
            for int_param in range(len(self.inputs.int_ranges_unc)):
                if self.inputs.int_ranges_unc[int_param]['key'] \
                        in self.inputs.intervention_param_dict[self.inputs.uncertainty_intervention]:
                    working_param_dict[param] = self.inputs.int_ranges_unc[int_param]

        # generate samples using latin hypercube design
        sample_values = lhs(len(working_param_dict), samples=self.inputs.n_samples)
        parameter_values = {}
        for p, param in enumerate(working_param_dict):
            parameter_values[param] = []
            for sample in range(self.inputs.n_samples):
                parameter_values[param].append(
                    working_param_dict[param]['bounds'][0]
                    + (working_param_dict[param]['bounds'][1] - working_param_dict[param]['bounds'][0])
                    * sample_values[sample][p])

        # loop through parameter values
        for sample in range(self.inputs.n_samples):

            # prepare for integration of scenario
            self.models['manual_scenario_15'] = model.ConsolidatedModel(15, self.inputs, self.gui_inputs)
            self.prepare_new_model_from_baseline('manual', 'manual_scenario_15')
            self.models['manual_scenario_15'].relevant_interventions.append(
                self.inputs.uncertainty_intervention)
            for param in parameter_values:
                self.models['manual_scenario_15'].set_parameter(param, parameter_values[param][sample])

            # integrate and save
            self.models['manual_scenario_15'].integrate()
            self.store_uncertainty('manual_scenario_15', epi_outputs_to_analyse=self.epi_outputs_to_analyse)

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
        self.models['optimisation'] = model.ConsolidatedModel(0, self.inputs, self.gui_inputs)
        self.prepare_new_model_from_baseline('manual', 'optimisation')
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

            print('Optimisation of the distribution across: ')
            print(dict_optimised_combi['interventions'])

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
                for intervention in self.interventions_considered_for_opti:
                    self.models['optimisation'].available_funding[intervention] = 0.

                # input values from x
                for i in range(len(x)):
                    intervention = self.interventions_considered_for_opti[combination[i]]
                    self.models['optimisation'].available_funding[intervention] = x[i] * self.total_funding
                self.models['optimisation'].distribute_funding_across_years()
                self.models['optimisation'].integrate()
                output_list = self.find_epi_outputs('optimisation',
                                                    outputs_to_analyse=['incidence', 'mortality', 'true_mortality'])
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
        self.models['optimisation'] = model.ConsolidatedModel(0, self.inputs, self.gui_inputs)
        self.prepare_new_model_from_baseline('manual', 'optimisation')
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
        output_list = self.find_epi_outputs('optimisation',
                                            outputs_to_analyse=['incidence', 'mortality', 'true_mortality'])
        del self.models['optimisation']
        return {'best_allocation': self.optimal_allocation, 'incidence': output_list['incidence'][-1],
                'mortality': output_list['mortality'][-1]}

    def load_opti_results(self):
        """
        Load optimisation results if attribute to self is True.
        """

        if self.load_optimisation:
            storage_file_name = os.path.join(self.opti_outputs_dir, 'opti_outputs.pkl')
            self.opti_results = tool_kit.pickle_load(storage_file_name)
            self.add_comment_to_gui_window('Optimisation results loaded')

    def save_opti_results(self):
        """
        Save optimisation results, which is expected to be the usual behaviour for the model runner.
        """

        # save only if optimisation has been run and save requested
        if self.save_opti and self.optimisation:
            filename = os.path.join(self.opti_outputs_dir, 'opti_outputs.pkl')
            tool_kit.pickle_save(self.opti_results, filename)
            self.add_comment_to_gui_window('Optimisation results saved')

    ''' GUI-related methods '''

    def add_comment_to_gui_window(self, comment):

        if self.js_gui:
            self.js_gui('console', {"message": comment})

        else:
            self.runtime_outputs.insert(END, comment + '\n')
            self.runtime_outputs.see(END)

    def plot_progressive_parameters(self):
        """
        Produce real-time parameter plot, according to which GUI is in use.
        """

        if self.js_gui:
            self.plot_progressive_parameters_js()
        else:
            try:
                self.plot_progressive_parameters_tk(from_runner=True)
            except:
                pass

    def plot_progressive_parameters_tk(self, from_runner=True, input_figure=None):

        # initialise plotting
        if from_runner:
            param_tracking_figure = plt.Figure()
            parameter_plots = FigureCanvasTkAgg(param_tracking_figure, master=self.figure_frame)

        else:
            param_tracking_figure = input_figure

        subplot_grid = outputs.find_subplot_numbers(len(self.all_parameters_tried))

        # cycle through parameters with one subplot for each parameter
        for p, param in enumerate(self.all_parameters_tried):

            # extract accepted params from all tried params
            accepted_params = list(p for p, a in zip(self.all_parameters_tried[param], self.whether_accepted_list) if a)

            # plot
            ax = param_tracking_figure.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)
            ax.plot(range(1, len(accepted_params) + 1), accepted_params, linewidth=2, marker='o', markersize=4,
                    mec='b', mfc='b')
            ax.set_xlim((1., len(self.accepted_indices) + 1))

            # find the y-limits from the parameter bounds and the parameter values tried
            for param_number in range(len(self.inputs.param_ranges_unc)):
                if self.inputs.param_ranges_unc[param_number]['key'] == param:
                    bounds = self.inputs.param_ranges_unc[param_number]['bounds']
            ylim_margins = .1
            min_ylimit = min(accepted_params + [bounds[0]])
            max_ylimit = max(accepted_params + [bounds[1]])
            ax.set_ylim((min_ylimit * (1 - ylim_margins), max_ylimit * (1 + ylim_margins)))

            # indicate the prior bounds
            ax.plot([1, len(self.accepted_indices) + 1], [min_ylimit, min_ylimit], color='0.8')
            ax.plot([1, len(self.accepted_indices) + 1], [max_ylimit, max_ylimit], color='0.8')

            # plot rejected parameters
            for run, rejected_params in self.rejection_dict[param].items():
                if self.rejection_dict[param][run]:
                    ax.plot([run + 1] * len(rejected_params), rejected_params, marker='o', linestyle='None',
                            mec='0.5', mfc='0.5', markersize=3)
                    for r in range(len(rejected_params)):
                        ax.plot([run, run + 1], [self.acceptance_dict[param][run], rejected_params[r]], color='0.5',
                                linestyle='--')

            # label
            ax.set_title(tool_kit.find_title_from_dictionary(param))
            if p > len(self.all_parameters_tried) - subplot_grid[1] - 1:
                ax.set_xlabel('Accepted runs')

            if from_runner:
                # output to GUI window
                parameter_plots.show()
                parameter_plots.draw()
                parameter_plots.get_tk_widget().grid(row=1, column=1)

        if not from_runner:
            return param_tracking_figure

        accepted_params = [
            list(p for p, a in zip(self.all_parameters_tried[param], self.whether_accepted_list) if a)[-1]
            for p, param in enumerate(self.all_parameters_tried)]
        names = [tool_kit.find_title_from_dictionary(param) for p, param in
                 enumerate(self.all_parameters_tried)]
        import json
        import os.path
        with open('graph.json', 'wt') as f:
            f.write(json.dumps({
                "all_parameters_tried": self.all_parameters_tried,
                "whether_accepted": self.whether_accepted_list,
                "rejected_dict": self.rejection_dict,
                "names": names,
                "count": self.plot_count
            }, indent=2))
        print('> writing', os.path.abspath('graph.json'))

    def plot_progressive_parameters_js(self):
        """
        Method to shadow previous method in JavaScript GUI.
        """

        names = [tool_kit.find_title_from_dictionary(param) for p, param in
                 enumerate(self.all_parameters_tried)]
        self.js_gui('uncertainty_graph', {
            "all_parameters_tried": self.all_parameters_tried,
            "whether_accepted": self.whether_accepted_list,
            "rejected_dict": self.rejection_dict,
            "names": names,
            "count": self.plot_count
        })
        self.plot_count += 1

