
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
import autumn.economics
import itertools
import time
import eventlet
from flask_socketio import emit


def generate_candidates(n_candidates, param_ranges_unc):

    """
    Function for generating candidate parameters.
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


def elementwise_list_addition(increment, list_to_increment):

    """
    Simple method to element-wise increment a list by the values in another list of the same length.
    """

    assert len(increment) == len(list_to_increment), 'Attempted to add two lists of different lengths'
    return [sum(x) for x in zip(list_to_increment, increment)]


def elementwise_list_division(numerator, denominator):

    """
    Simple method to element-wise increment a list by the values in another list of the same length.
    """

    assert len(numerator) == len(denominator), 'Attempted to divide two lists of different lengths'
    return [n / d for n, d in zip(numerator, denominator)]


def find_integer_dict_from_float_dict(float_dict):

    # Method may be redundant with optimal code

    integer_dict = {}
    times = float_dict.keys()
    times.sort()
    start = numpy.floor(times[0])
    finish = numpy.floor(times[-1])
    float_years = numpy.linspace(start, finish, finish - start + 1.)
    for year in float_years:
        key = [t for t in times if t >= year][0]
        integer_dict[int(key)] = float_dict[key]
    return integer_dict


def extract_integer_dicts(models_to_analyse={}, dict_to_extract_from={}):

    # Method may be redundant with optimal code

    integer_dict = {}
    for scenario in models_to_analyse:
        integer_dict[scenario] = {}
        for output in dict_to_extract_from[scenario]:
            integer_dict[scenario][output] \
                = find_integer_dict_from_float_dict(dict_to_extract_from[scenario][output])
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

    # Linearly scaling weights summing to one
    if method == 1:
        weights = []
        if len(list) == 1:
            weights = [1.]
        else:
            for y in range(len(list)):
                weights.append(relative_weights[0]
                               + (relative_weights[1] - relative_weights[0]) / float(len(list)-1) * y)
            return [i / sum(weights) for i in weights]

    # Equally distributed weights summing to one
    elif method == 2:
        return [1. / float(len(list))] * len(list)

    # All weights equal to one
    elif method == 3:
        return [1.] * len(list)


class ModelRunner:

    def __init__(self, gui_inputs, runtime_outputs, figure_frame, js_gui=False):

        """
        Instantiation method for model runner - currently including many attributes that should be set externally, e.g.
        in the GUI(s).

        Args:
            gui_inputs: Inputs from the off-line Tkinter GUI.
            runtime_outputs: Off-line GUI window for commenting.
            figure_frame: Uncertainty parameter plotting window of Tkinter GUI.
            js_gui: JavaScript GUI inputs.
        """

        # Loading of inputs
        self.gui_inputs = gui_inputs
        self.runtime_outputs = runtime_outputs
        self.figure_frame = figure_frame
        self.inputs = data_processing.Inputs(gui_inputs, runtime_outputs, from_test=True, js_gui=js_gui)
        self.inputs.read_and_load_data()

        # Preparing for basic runs
        self.model_dict = {}
        self.interventions_to_cost = self.inputs.interventions_to_cost

        # Uncertainty-related attributes
        self.is_last_run_success = False
        self.loglikelihoods = []
        self.outputs_unc = [{'key': 'incidence',
                             'posterior_width': None,
                             'width_multiplier': 2.  # Width of normal posterior relative to range of parameter values allowed
                             }]
        self.all_parameters_tried = {}
        self.whether_accepted_list = []
        self.accepted_indices = []
        self.rejected_indices = []
        self.solns_for_extraction = ['compartment_soln', 'fraction_soln']
        self.arrays_for_extraction = ['flow_array', 'fraction_array', 'soln_array', 'var_array', 'costs']
        self.acceptance_dict = {}
        self.rejection_dict = {}
        self.uncertainty_percentiles = {}
        self.percentiles = [2.5, 50, 97.5]
        self.accepted_no_burn_in_indices = []
        self.random_start = False  # Whether to start from a random point, as opposed to the manually calibrated value

        # Optimisation attributes
        self.optimisation = False  # Leave True even if loading optimisation results
        self.indicator_to_minimise = 'incidence'  # Currently must be 'incidence' or 'mortality'
        self.annual_envelope = [25e6, 50e6, 75e6, 100e6, 200e6]  # Size of funding envelope in scenarios to be run
        self.save_opti = True
        self.load_opti = False  # Optimisation will not be run if on
        self.total_funding = None  # Funding for entire period
        self.f_tol = {'incidence': 0.5,
                      'mortality': 0.05}  # Stopping condition for optimisation algorithm (differs by indicator)
        self.year_end_opti = 2035.  # Model is run until that date during optimisation
        self.acceptable_combinations = []  # List of intervention combinations that can be considered with available funding
        self.opti_results = {}  # Store all the results that we need for optimisation
        self.optimised_combinations = []
        self.optimal_allocation = {}
        self.interventions_considered_for_opti \
            = ['xpertacf_ruralpoor', 'xpertacf_prison', 'xpertacf', 'xpert', 'engage_lowquality']
        self.interventions_forced_for_opti \
            = ['xpertacf_ruralpoor', 'engage_lowquality']  # Interventions that must appear in optimal plan

        # Output-related attributes
        self.epi_outputs_to_analyse = ['population', 'incidence', 'true_incidence', 'prevalence', 'true_prevalence',
                                       'mortality', 'true_mortality', 'notifications']
        self.epi_outputs = {}
        self.epi_outputs_dict = {}
        self.epi_outputs_integer_dict = {}
        self.epi_outputs_uncertainty = {}
        self.epi_outputs_uncertainty_centiles = None
        self.cost_outputs = {}
        self.cost_outputs_dict = {}
        self.cost_outputs_integer_dict = {}
        self.cost_outputs_uncertainty = {}
        self.cost_outputs_uncertainty_centiles = None
        self.additional_cost_types = ['inflated', 'discounted', 'discounted_inflated']
        self.cost_types = self.additional_cost_types + ['raw']

        # Saving-related
        self.attributes_to_save = ['epi_outputs', 'epi_outputs_dict', 'epi_outputs_integer_dict',
                                   'epi_outputs_uncertainty', 'cost_outputs', 'cost_outputs_dict',
                                   'cost_outputs_integer_dict', 'cost_outputs_uncertainty', 'accepted_indices',
                                   'rejected_indices', 'all_parameters_tried', 'whether_accepted_list',
                                   'acceptance_dict', 'accepted_no_burn_in_indices', 'rejection_dict', 'loglikelihoods']

        # GUI-related
        self.emit_delay = 0.1
        self.plot_count = 0
        self.js_gui = js_gui
        if self.js_gui: eventlet.monkey_patch()

    ###############################################
    ### Master methods to run all other methods ###
    ###############################################

    def master_runner(self):

        """
        Calls methods to run model with each of the three fundamental approaches.
        """

        # Prepare file for saving
        out_dir = 'saved_uncertainty_analyses'
        if not os.path.isdir(out_dir): os.makedirs(out_dir)
        storage_file_name = os.path.join(out_dir, 'store.pkl')

        # Load a saved simulation
        if self.gui_inputs['pickle_uncertainty'] == 'Load':
            self.add_comment_to_gui_window('Results loading from previous simulation')
            loaded_data = tool_kit.pickle_load(storage_file_name)
            self.add_comment_to_gui_window('Loading finished')
            for attribute in loaded_data:
                setattr(self, attribute, loaded_data[attribute])

        # Or run the manual scenarios as requested by user
        else:
            self.run_manual_calibration()
            if self.gui_inputs['output_uncertainty']: self.run_uncertainty()

        # Save uncertainty if requested
        if self.gui_inputs['pickle_uncertainty'] == 'Save':
            data_to_save = {}
            for attribute in self.attributes_to_save:
                data_to_save[attribute] = getattr(self, attribute)
            tool_kit.pickle_save(data_to_save, storage_file_name)
            self.add_comment_to_gui_window('Uncertainty results saved to disc')

        # Processing methods that are only required for outputs
        self.epi_outputs_uncertainty_centiles = self.find_uncertainty_centiles(self.epi_outputs_uncertainty)
        self.cost_outputs_uncertainty_centiles = self.find_uncertainty_centiles(self.cost_outputs_uncertainty)

        # Master optimisation method
        if self.optimisation and not self.load_opti:
            self.run_optimisation()

        # Notify user that model running has finished
        self.add_comment_to_gui_window('Finished')

    def run_manual_calibration(self):

        """
        Runs the scenarios a single time, starting from baseline with parameter values as specified in spreadsheets.
        """

        for scenario in self.gui_inputs['scenarios_to_run']:

            # Name and initialise model
            scenario_name = 'manual_' + tool_kit.find_scenario_string_from_number(scenario)
            self.model_dict[scenario_name] = model.ConsolidatedModel(scenario, self.inputs, self.gui_inputs)

            # Sort out times for scenario runs
            self.run_scenarios('manual', scenario)

            # Describe model and integrate
            self.add_comment_to_gui_window('Running ' + scenario_name[7:] + ' conditions for '
                                           + self.gui_inputs['country'] + ' using point estimates for parameters.')
            self.model_dict[scenario_name].integrate()

            # Model interpretation for each scenario
            self.epi_outputs[scenario_name] \
                = self.find_epi_outputs(scenario_name,
                                        outputs_to_analyse=self.epi_outputs_to_analyse,
                                        stratifications=[self.model_dict[scenario_name].agegroups,
                                                         self.model_dict[scenario_name].riskgroups])
            self.find_cost_outputs(scenario_name)

        # Model interpretation that applies to baseline run only
        self.find_population_fractions(stratifications=[self.model_dict['manual_baseline'].agegroups,
                                                        self.model_dict['manual_baseline'].riskgroups])

        # If you want some dictionaries based on the lists created above (may not be necessary)
        self.epi_outputs_dict.update(get_output_dicts_from_lists(models_to_analyse=self.model_dict,
                                                                 output_dict_of_lists=self.epi_outputs))
        self.cost_outputs_dict.update(get_output_dicts_from_lists(models_to_analyse=self.model_dict,
                                                                  output_dict_of_lists=self.cost_outputs))

        # If integer-based dictionaries required
        self.epi_outputs_integer_dict.update(extract_integer_dicts(self.model_dict, self.epi_outputs_dict))
        self.cost_outputs_integer_dict.update(extract_integer_dicts(self.model_dict, self.cost_outputs_dict))

    def run_scenarios(self, run_type, scenario):

        """
        Method to prepare a scenario for running - applied to both manual calibration and to uncertainty.

        Args:
            run_type: Whether manual or uncertainty being run
            scenario: The scenario being run
        """

        if scenario is not None:
            scenario_name = run_type + '_' + tool_kit.find_scenario_string_from_number(scenario)
            scenario_start_time_index = \
                self.model_dict[run_type + '_baseline'].find_time_index(self.inputs.model_constants['recent_time'])
            start_time = self.model_dict[run_type + '_baseline'].times[scenario_start_time_index]
            self.model_dict[scenario_name].start_time = start_time
            self.model_dict[scenario_name].next_time_point = start_time
            self.model_dict[scenario_name].loaded_compartments = \
                self.model_dict[run_type + '_baseline'].load_state(scenario_start_time_index)

    ####################################
    ### Model interpretation methods ###
    ####################################

    def find_epi_outputs(self, scenario, outputs_to_analyse, stratifications=[]):

        """
        Method to extract all requested epidemiological outputs from the models. Intended ultimately to be flexible\
        enough for use for analysis of scenarios, uncertainty and optimisation.
        """

        epi_outputs = {'times': self.model_dict[scenario].times}

        # Unstratified outputs______________________________________________________________________________________
        # Initialise lists
        for output in outputs_to_analyse:
            epi_outputs[output] = [0.] * len(epi_outputs['times'])
            for strain in self.model_dict[scenario].strains:
                epi_outputs[output + strain] = [0.] * len(epi_outputs['times'])

        # Population
        if 'population' in outputs_to_analyse:
            for compartment in self.model_dict[scenario].compartments:
                epi_outputs['population'] \
                    = elementwise_list_addition(self.model_dict[scenario].get_compartment_soln(compartment),
                                                epi_outputs['population'])
        # Replace zeroes with small numbers for division
        total_denominator = tool_kit.prepare_denominator(epi_outputs['population'])

        # To allow calculation by strain and the total output
        strains = self.model_dict[scenario].strains + ['']

        # Incidence
        if 'incidence' in outputs_to_analyse:
            for strain in strains:
                # Variable flows
                for from_label, to_label, rate in self.model_dict[scenario].var_transfer_rate_flows:
                    if 'latent' in from_label and 'active' in to_label and strain in to_label:
                        incidence_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                                 * self.model_dict[scenario].get_var_soln(rate) \
                                                 / total_denominator \
                                                 * 1e5
                        epi_outputs['true_incidence' + strain] \
                            = elementwise_list_addition(incidence_increment,
                                                        epi_outputs['true_incidence' + strain])
                        # Reduce paediatric contribution
                        if '_age' in from_label and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                            incidence_increment *= self.inputs.model_constants['program_prop_child_reporting']
                        epi_outputs['incidence' + strain] \
                            = elementwise_list_addition(incidence_increment,
                                                        epi_outputs['incidence' + strain])
                # Fixed flows
                for from_label, to_label, rate in self.model_dict[scenario].fixed_transfer_rate_flows:
                    if 'latent' in from_label and 'active' in to_label and strain in to_label:
                        incidence_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                                 * rate / total_denominator * 1e5
                        epi_outputs['true_incidence' + strain] \
                            = elementwise_list_addition(incidence_increment,
                                                        epi_outputs['incidence' + strain])
                        # Reduce paedatric contribution
                        if '_age' in from_label and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                            incidence_increment *= self.inputs.model_constants['program_prop_child_reporting']
                        epi_outputs['incidence' + strain] \
                            = elementwise_list_addition(incidence_increment,
                                                        epi_outputs['true_incidence' + strain])
            # Find percentage incidence by strain
            if len(self.model_dict[scenario].strains) > 1:
                for strain in self.model_dict[scenario].strains:
                    epi_outputs['perc_incidence' + strain] \
                        = [i / j * 1e2 for i, j in zip(epi_outputs['incidence' + strain],
                                                       tool_kit.prepare_denominator(epi_outputs['incidence']))]

        # Notifications
        if 'notifications' in outputs_to_analyse:
            for strain in strains:
                for from_label, to_label, rate in self.model_dict[scenario].var_transfer_rate_flows:
                    if 'active' in from_label and 'detect' in to_label and strain in to_label:
                        notifications_increment \
                            = self.model_dict[scenario].get_compartment_soln(from_label) \
                              * self.model_dict[scenario].get_var_soln(rate)
                        if '_age' in from_label and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                            notifications_increment *= self.inputs.model_constants['program_prop_child_reporting']
                        epi_outputs['notifications' + strain] \
                            = elementwise_list_addition(notifications_increment, epi_outputs['notifications' + strain])

        # Mortality
        if 'mortality' in outputs_to_analyse:
            for strain in strains:
                # Fixed flows
                for from_label, rate in self.model_dict[scenario].fixed_infection_death_rate_flows:
                    if strain in from_label:
                        mortality_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                              * rate \
                                              / total_denominator \
                                              * 1e5
                        epi_outputs['true_mortality' + strain] \
                            = elementwise_list_addition(mortality_increment,
                                                        epi_outputs['true_mortality' + strain])
                        # Reduce paediatric contribution
                        if '_age' in from_label and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                            mortality_increment *= self.inputs.model_constants['program_prop_child_reporting']
                        # Reduce outside health system contribution
                        epi_outputs['mortality' + strain] \
                            = elementwise_list_addition(mortality_increment
                                                        * self.model_dict[scenario].params['program_prop_death_reporting'],
                                                        epi_outputs['mortality' + strain])
                # Variable flows
                for from_label, rate in self.model_dict[scenario].var_infection_death_rate_flows:
                    if strain in from_label:
                        mortality_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                              * self.model_dict[scenario].get_var_soln(rate) \
                                              / total_denominator \
                                              * 1e5
                        epi_outputs['true_mortality' + strain] \
                            = elementwise_list_addition(mortality_increment,
                                                        epi_outputs['true_mortality' + strain])
                        # Reduce paediatric contribution
                        if '_age' in from_label and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                            mortality_increment *= self.inputs.model_constants['program_prop_child_reporting']
                        epi_outputs['mortality' + strain] \
                            = elementwise_list_addition(mortality_increment,
                                                        epi_outputs['mortality' + strain])

        # Prevalence
        if 'prevalence' in outputs_to_analyse:
            for strain in strains:
                for label in self.model_dict[scenario].labels:
                    if 'susceptible' not in label and 'latent' not in label and strain in label:
                        prevalence_increment = self.model_dict[scenario].get_compartment_soln(label) \
                                               / total_denominator \
                                               * 1e5
                        epi_outputs['true_prevalence' + strain] \
                            = elementwise_list_addition(prevalence_increment,
                                                        epi_outputs['true_prevalence' + strain])
                        # Reduce paediatric contribution
                        if '_age' in label and tool_kit.is_upper_age_limit_at_or_below(label, 15.):
                            prevalence_increment *= self.inputs.model_constants['program_prop_child_reporting']
                        epi_outputs['prevalence' + strain] \
                            = elementwise_list_addition(prevalence_increment,
                                                        epi_outputs['prevalence' + strain])

        # Infections
        if 'infections' in outputs_to_analyse:
            for strain in strains:
                for from_label, to_label, rate in self.model_dict[scenario].var_transfer_rate_flows:
                    if 'latent_early' in to_label and strain in to_label:
                        # Absolute number of infections
                        epi_outputs['infections' + strain] \
                            = elementwise_list_addition(self.model_dict[scenario].get_compartment_soln(from_label)
                                                        * self.model_dict[scenario].get_var_soln(rate),
                                                        epi_outputs['infections' + strain])
                # ARI
                epi_outputs['annual_risk_infection' + strain] \
                    = [i / j * 1e2 for i, j in zip(epi_outputs['infections' + strain], total_denominator)]

        # Stratified outputs________________________________________________________________________________________
        # Currently not bothering to do this for each strain
        for stratification in stratifications:
            if len(stratification) > 1:
                for stratum in stratification:

                    # Initialise lists
                    for output in outputs_to_analyse:
                        epi_outputs[output + stratum] = [0.] * len(epi_outputs['times'])

                    # Population
                    if 'population' in outputs_to_analyse:
                        for compartment in self.model_dict[scenario].compartments:
                            if stratum in compartment:
                                epi_outputs['population' + stratum] \
                                    = elementwise_list_addition(self.model_dict[scenario].get_compartment_soln(compartment),
                                                                epi_outputs['population' + stratum])

                    # The population denominator to be used with zeros replaced with small numbers
                    stratum_denominator \
                        = tool_kit.prepare_denominator(epi_outputs['population' + stratum])

                    # Incidence
                    if 'incidence' in outputs_to_analyse:
                        # Variable flows
                        for from_label, to_label, rate in self.model_dict[scenario].var_transfer_rate_flows:
                            if 'latent' in from_label and 'active' in to_label and stratum in from_label:
                                incidence_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                                      * self.model_dict[scenario].get_var_soln(rate) \
                                                      / stratum_denominator \
                                                      * 1e5
                                epi_outputs['true_incidence' + stratum] \
                                    = elementwise_list_addition(incidence_increment,
                                                                epi_outputs['true_incidence' + stratum])
                                # Reduce paediatric contribution
                                if '_age' in from_label \
                                        and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                                    incidence_increment *= self.inputs.model_constants[
                                        'program_prop_child_reporting']
                                epi_outputs['incidence' + stratum] \
                                    = elementwise_list_addition(incidence_increment,
                                                                epi_outputs['incidence' + stratum])

                        # Fixed flows
                        for from_label, to_label, rate in self.model_dict[scenario].fixed_transfer_rate_flows:
                            if 'latent' in from_label and 'active' in to_label and stratum in from_label:
                                incidence_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                                      * rate \
                                                      / stratum_denominator \
                                                      * 1e5
                                epi_outputs['true_incidence' + stratum] \
                                    = elementwise_list_addition(incidence_increment,
                                                                epi_outputs['true_incidence' + stratum])
                                # Reduce paediatric contribution
                                if '_age' in from_label \
                                        and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                                    incidence_increment \
                                        *= self.inputs.model_constants['program_prop_child_reporting']
                                epi_outputs['incidence' + stratum] \
                                    = elementwise_list_addition(incidence_increment,
                                                                epi_outputs['incidence' + stratum])

                    # Mortality
                    if 'mortality' in outputs_to_analyse:
                        for from_label, rate in self.model_dict[scenario].fixed_infection_death_rate_flows:
                            if stratum in from_label:
                                mortality_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                                      * rate \
                                                      / stratum_denominator \
                                                      * 1e5
                                epi_outputs['true_mortality' + stratum] \
                                    = elementwise_list_addition(mortality_increment,
                                                                epi_outputs['true_mortality' + stratum])
                                # Reduce paediatric contribution
                                if '_age' in from_label \
                                        and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                                    mortality_increment *= self.inputs.model_constants['program_prop_child_reporting']

                                # Reduce outside health system contribution
                                epi_outputs['mortality' + stratum] \
                                    = elementwise_list_addition(mortality_increment
                                                                * self.model_dict[scenario].params[
                                                                    'program_prop_death_reporting'],
                                                                epi_outputs['mortality' + stratum])
                        for from_label, rate in self.model_dict[scenario].var_infection_death_rate_flows:
                            # Variable flows
                            if stratum in from_label:
                                mortality_increment = self.model_dict[scenario].get_compartment_soln(from_label) \
                                                      * self.model_dict[scenario].get_var_soln(rate) \
                                                      / stratum_denominator \
                                                      * 1e5
                                epi_outputs['true_mortality' + stratum] \
                                    = elementwise_list_addition(mortality_increment,
                                                                epi_outputs['true_mortality' + stratum])
                                # Reduce paediatric contribution
                                if '_age' in from_label \
                                        and tool_kit.is_upper_age_limit_at_or_below(from_label, 15.):
                                    mortality_increment *= self.inputs.model_constants[
                                        'program_prop_child_reporting']
                                epi_outputs['mortality' + stratum] \
                                    = elementwise_list_addition(mortality_increment
                                                                * self.model_dict[scenario].params[
                                                                    'program_prop_death_reporting'],
                                                                epi_outputs['mortality' + stratum])

                    # Prevalence
                    if 'prevalence' in outputs_to_analyse:
                        for label in self.model_dict[scenario].labels:
                            if 'susceptible' not in label and 'latent' not in label and stratum in label:
                                prevalence_increment = self.model_dict[scenario].get_compartment_soln(label) \
                                                       / stratum_denominator \
                                                       * 1e5
                                epi_outputs['true_prevalence' + stratum] \
                                    = elementwise_list_addition(prevalence_increment,
                                                                epi_outputs['true_prevalence' + stratum])
                            # Reduce paediatric contribution
                            if '_age' in label and tool_kit.is_upper_age_limit_at_or_below(label, 15.):
                                prevalence_increment *= self.inputs.model_constants['program_prop_child_reporting']
                            epi_outputs['prevalence' + stratum] \
                                = elementwise_list_addition(prevalence_increment, epi_outputs['prevalence' + stratum])

                    # Infections
                    if 'infections' in outputs_to_analyse:
                        for from_label, to_label, rate in self.model_dict[scenario].var_transfer_rate_flows:
                            if 'latent_early' in to_label and stratum in from_label:
                                # Absolute number of infections
                                epi_outputs['infections' + stratum] \
                                    = elementwise_list_addition(self.model_dict[scenario].get_compartment_soln(from_label)
                                                                * self.model_dict[scenario].get_var_soln(rate),
                                                                epi_outputs['infections' + stratum])
                        # ARI
                        epi_outputs['annual_risk_infection' + stratum] \
                            = [i / j * 1e2 for i, j in zip(epi_outputs['infections' + stratum], stratum_denominator)]

        return epi_outputs

    def find_population_fractions(self, stratifications=[]):

        """
        Find the proportion of the population in various stratifications.
        The stratifications must apply to the entire population, so not to be used for strains, etc.
        """

        for scenario in self.model_dict:
            for stratification in stratifications:
                if len(stratification) > 1:
                    for stratum in stratification:
                        self.epi_outputs[scenario]['fraction' + stratum] \
                            = elementwise_list_division(self.epi_outputs[scenario]['population' + stratum],
                                                        self.epi_outputs[scenario]['population'])

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
        Add cost dictionaries to cost_outputs attribute.
        """

        cost_outputs = {'times': self.model_dict[scenario_name].cost_times}
        for i, intervention in enumerate(self.interventions_to_cost):
            cost_outputs['raw_cost_' + intervention] = self.model_dict[scenario_name].costs[:, i]
        return cost_outputs

    def find_costs_all_programs(self, scenario_name):

        """
        Sum costs across all programs and populate to cost_outputs dictionary for each scenario.
        """

        costs_all_programs = [0.] * len(self.cost_outputs[scenario_name]['raw_cost_' + self.interventions_to_cost[0]])
        for i in self.interventions_to_cost:
            costs_all_programs \
                = elementwise_list_addition(self.cost_outputs[scenario_name]['raw_cost_' + i], costs_all_programs)
        return costs_all_programs

    def find_adjusted_costs(self, scenario_name):

        """
        Find costs adjusted for inflation and discounting.

        Args:
            scenario_name: Scenario being costed
        """

        # Get some preliminary parameters
        year_current = self.inputs.model_constants['current_time']
        current_cpi = self.inputs.scaleup_fns[None]['econ_cpi'](year_current)
        discount_rate = self.inputs.model_constants['econ_discount_rate']

        # Loop over interventions to be costed and cost types to calculate costs
        cost_outputs = {}
        for intervention in self.interventions_to_cost + ['all_programs']:
            for cost_type in self.additional_cost_types:
                cost_outputs[cost_type + '_cost_' + intervention] = []
                for t, time in enumerate(self.cost_outputs[scenario_name]['times']):
                    cost_outputs[cost_type + '_cost_' + intervention].append(
                        autumn.economics.get_adjusted_cost(self.cost_outputs[scenario_name]['raw_cost_'
                                                                                            + intervention][t],
                                                           cost_type,
                                                           current_cpi,
                                                           self.inputs.scaleup_fns[None]['econ_cpi'](time),
                                                           discount_rate,
                                                           max(0., (time - year_current))))

        return cost_outputs

    def find_uncertainty_centiles(self, full_uncertainty_outputs):

        """
        Find percentiles from uncertainty dictionaries.

        Updates:
            self.percentiles: Adds all the required percentiles to this dictionary.
        """

        uncertainty_centiles = {}
        self.accepted_no_burn_in_indices = [i for i in self.accepted_indices if i >= self.gui_inputs['burn_in_runs']]

        # Loop through scenarios and outputs
        for scenario in full_uncertainty_outputs:
            uncertainty_centiles[scenario] = {}
            for output in full_uncertainty_outputs[scenario]:
                if output != 'times':

                    # To deal with the fact that we are currently saving all baseline runs but only accepted scenarios:
                    if scenario == 'uncertainty_baseline':
                        matrix_to_analyse = full_uncertainty_outputs[scenario][output][
                                            self.accepted_no_burn_in_indices, :]
                    else:
                        matrix_to_analyse = full_uncertainty_outputs[scenario][output]

                    # Find the actual centiles
                    uncertainty_centiles[scenario][output] \
                        = numpy.percentile(matrix_to_analyse, self.percentiles, axis=0)

        # Return result to make usable in other situations
        return uncertainty_centiles

    ###########################
    ### Uncertainty methods ###
    ###########################

    def run_uncertainty(self):

        """
        Main method to run all the uncertainty processes.
        """

        self.add_comment_to_gui_window('Uncertainty analysis commenced')

        # If not doing an adaptive search, only need to start with a single parameter set
        if self.gui_inputs['adaptive_uncertainty']:
            n_candidates = 1
        else:
            n_candidates = self.gui_inputs['uncertainty_runs'] * 10

        # Decide whether to start analysis from a random point or the manual values of the parameters
        if self.random_start:
            param_candidates = generate_candidates(n_candidates, self.inputs.param_ranges_unc)
        else:
            param_candidates = {}
            for param_dict in self.inputs.param_ranges_unc:
                param_candidates[param_dict['key']] = [self.inputs.model_constants[param_dict['key']]]

        # Find weights for outputs that are being calibrated to
        normal_char = self.get_fitting_data()
        years_to_compare = range(1990, 2015)
        weights = find_uncertainty_output_weights(years_to_compare, 1, [1., 2.])
        self.add_comment_to_gui_window('"Weights": \n' + str(weights))

        # Prepare for uncertainty loop
        n_accepted = 0
        prev_log_likelihood = -1e10
        params = []
        for param_dict in self.inputs.param_ranges_unc:
            self.all_parameters_tried[param_dict['key']] = []
            self.acceptance_dict[param_dict['key']] = {}
            self.rejection_dict[param_dict['key']] = {}
            self.rejection_dict[param_dict['key']][n_accepted] = []

        # Instantiate uncertainty model objects
        for scenario in self.gui_inputs['scenarios_to_run']:
            scenario_name = 'uncertainty_' + tool_kit.find_scenario_string_from_number(scenario)
            self.model_dict[scenario_name] = model.ConsolidatedModel(scenario, self.inputs, self.gui_inputs)

        # Until a sufficient number of parameters are accepted
        run = 0
        while n_accepted < self.gui_inputs['uncertainty_runs']:

            # Set timer
            start_timer_run = datetime.datetime.now()

            # Update parameters
            new_params = []
            if self.gui_inputs['adaptive_uncertainty']:
                if run == 0:
                    for param_dict in self.inputs.param_ranges_unc:
                        new_params.append(param_candidates[param_dict['key']][run])
                        params.append(param_candidates[param_dict['key']][run])
                else:
                    new_params = self.update_params(params)
            else:
                for param_dict in self.inputs.param_ranges_unc:
                    new_params.append(param_candidates[param_dict['key']][run])

            # Run baseline integration (includes parameter checking, parameter setting and recording success/failure)
            self.run_with_params(new_params, model_object='uncertainty_baseline')

            # Now storing regardless of acceptance, provided run was completed successfully
            if self.is_last_run_success:

                # Get outputs for calibration and store results
                self.store_uncertainty('uncertainty_baseline', epi_outputs_to_analyse=self.epi_outputs_to_analyse)
                integer_dictionary \
                    = extract_integer_dicts(['uncertainty_baseline'],
                                            get_output_dicts_from_lists(models_to_analyse=['uncertainty_baseline'],
                                                                        output_dict_of_lists=self.epi_outputs))

                # Calculate prior
                prior_log_likelihood = 0.
                for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                    param_val = new_params[p]
                    self.all_parameters_tried[param_dict['key']].append(new_params[p])

                    # Calculate the density of param_val
                    bound_low, bound_high = param_dict['bounds'][0], param_dict['bounds'][1]

                    # Normalise value and find log of PDF from appropriate distribution
                    if param_dict['distribution'] == 'beta':
                        prior_log_likelihood \
                            += beta.logpdf((param_val - bound_low) / (bound_high - bound_low), 2., 2.)
                    elif param_dict['distribution'] == 'uniform':
                        prior_log_likelihood += numpy.log(1. / (bound_high - bound_low))

                # Calculate posterior
                posterior_log_likelihood = 0.
                for output_dict in self.outputs_unc:

                    # The GTB values for the output of interest
                    working_output_dictionary = normal_char[output_dict['key']]
                    for y, year in enumerate(years_to_compare):
                        if year in working_output_dictionary.keys():
                            model_result_for_output = integer_dictionary['uncertainty_baseline']['incidence'][year]
                            mu, sd = working_output_dictionary[year][0], working_output_dictionary[year][1]
                            posterior_log_likelihood += norm.logpdf(model_result_for_output, mu, sd) * weights[y]

                # Determine acceptance
                log_likelihood = prior_log_likelihood + posterior_log_likelihood
                accepted = numpy.random.binomial(n=1, p=min(1., numpy.exp(log_likelihood - prev_log_likelihood)))

                # Explain progression of likelihood
                self.add_comment_to_gui_window('Previous log likelihood:\n' + str(prev_log_likelihood)
                                               + '\nLog likelihood this run:\n' + str(log_likelihood)
                                               + '\nAcceptance probability:\n'
                                               + str(min(1., numpy.exp(log_likelihood - prev_log_likelihood)))
                                               + '\nWhether accepted:\n' + str(bool(accepted)) + '\n________________\n')
                self.loglikelihoods.append(log_likelihood)

                # Record information for all runs
                if bool(accepted):
                    self.whether_accepted_list.append(True)
                    self.accepted_indices += [run]
                    n_accepted += 1
                    for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                        self.acceptance_dict[param_dict['key']][n_accepted] = new_params[p]
                        self.rejection_dict[param_dict['key']][n_accepted] = []

                    # Update likelihood and parameter set for next run
                    prev_log_likelihood = log_likelihood
                    params = new_params

                    # Run scenarios other than baseline and store uncertainty (only if accepted)
                    for scenario in self.gui_inputs['scenarios_to_run']:
                        self.run_scenarios('uncertainty', scenario)
                        scenario_name = 'uncertainty_' + tool_kit.find_scenario_string_from_number(scenario)
                        self.run_with_params(new_params, model_object=scenario_name)
                        self.store_uncertainty(scenario_name, epi_outputs_to_analyse=self.epi_outputs_to_analyse)

                else:
                    self.whether_accepted_list.append(False)
                    self.rejected_indices += [run]
                    for p, param_dict in enumerate(self.inputs.param_ranges_unc):
                        self.rejection_dict[param_dict['key']][n_accepted].append(new_params[p])

                # Plot parameter progression and report on progress
                self.plot_progressive_parameters()
                self.add_comment_to_gui_window(str(n_accepted) + ' accepted / ' + str(run) +
                                               ' candidates. Running time: '
                                               + str(datetime.datetime.now() - start_timer_run))
                run += 1

            # Generate more candidates if required -
            if not self.gui_inputs['adaptive_uncertainty'] and run >= len(param_candidates.keys()):
                param_candidates = generate_candidates(n_candidates, self.inputs.param_ranges_unc)
                run = 0

    def set_model_with_params(self, param_dict, model_object='baseline'):

        """
        Populates baseline model with params from uncertainty calculations.

        Args:
            param_dict: Dictionary of the parameters to be set within the model (keys parameter name strings and values
                parameter values).
        """

        for key in param_dict:
            if key in self.model_dict[model_object].params:
                self.model_dict[model_object].set_parameter(key, param_dict[key])
            else:
                raise ValueError("%s not in model_object params" % key)

    def convert_param_list_to_dict(self, params):

        """
        Extract parameters from list into dictionary that can be used for setting in the model through the
        set_model_with_params method.

        Args:
            params: The parameter names for extraction.
        Returns:
            param_dict: The dictionary returned in appropriate format.
        """

        param_dict = {}
        for names, vals in zip(self.inputs.param_ranges_unc, params):
            param_dict[names['key']] = vals
        return param_dict

    def get_fitting_data(self):

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
                sd = output_dict['posterior_width'] / (2. * 1.96)
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    mu = self.inputs.data_to_fit[output_dict['key']][year]
                    normal_char[output_dict['key']][year] = [mu, sd]

            # Incidence
            elif output_dict['key'] == 'incidence':
                for year in self.inputs.data_to_fit[output_dict['key']].keys():
                    low = self.inputs.data_to_fit['incidence_low'][year]
                    high = self.inputs.data_to_fit['incidence_high'][year]
                    sd = output_dict['width_multiplier'] * (high - low) / (2. * 1.96)
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

    def run_with_params(self, params, model_object='uncertainty_baseline'):

        """
        Integrate the model with the proposed parameter set.

        Args:
            params: The parameters to be set in the model.
        """

        # Check whether parameter values are acceptable
        for p, param in enumerate(params):

            # Whether the parameter value is valid
            if not tool_kit.is_parameter_value_valid(param):
                print 'Warning: parameter%d=%f is invalid for model' % (p, param)
                self.is_last_run_success = False
                return
            bounds = self.inputs.param_ranges_unc[p]['bounds']

            # Whether the parameter value is within acceptable ranges
            if (param < bounds[0]) or (param > bounds[1]):
                # print 'Warning: parameter%d=%f is outside of the allowed bounds' % (p, param)
                self.is_last_run_success = False
                return

        param_dict = self.convert_param_list_to_dict(params)

        # Set parameters and run
        self.set_model_with_params(param_dict, model_object)
        self.is_last_run_success = True
        try:
            self.model_dict[model_object].integrate()
        except:
            print "Warning: parameters=%s failed with model" % params
            self.is_last_run_success = False

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

        # Get outputs
        self.epi_outputs[scenario_name] \
            = self.find_epi_outputs(scenario_name, outputs_to_analyse=self.epi_outputs_to_analyse)
        self.find_cost_outputs(scenario_name)

        # Initialise dictionaries if needed
        if scenario_name not in self.epi_outputs_uncertainty:
            self.epi_outputs_uncertainty[scenario_name] = {'times': self.epi_outputs[scenario_name]['times']}
            self.cost_outputs_uncertainty[scenario_name] = {'times': self.cost_outputs[scenario_name]['times']}
            for output in epi_outputs_to_analyse:
                self.epi_outputs_uncertainty[scenario_name][output] \
                    = numpy.empty(shape=[0, len(self.epi_outputs[scenario_name]['times'])])
            for output in self.cost_outputs[scenario_name]:
                self.cost_outputs_uncertainty[scenario_name][output] \
                    = numpy.empty(shape=[0, len(self.cost_outputs[scenario_name]['times'])])

        # Add uncertainty data to dictionaries
        for output in epi_outputs_to_analyse:
            self.epi_outputs_uncertainty[scenario_name][output] \
                = numpy.vstack([self.epi_outputs_uncertainty[scenario_name][output],
                                self.epi_outputs[scenario_name][output]])
        for output in self.cost_outputs[scenario_name]:
            self.cost_outputs_uncertainty[scenario_name][output] \
                = numpy.vstack([self.cost_outputs_uncertainty[scenario_name][output],
                                self.cost_outputs[scenario_name][output]])

    ############################
    ### Optimisation methods ###
    ############################

    def run_optimisation(self):

        """
        Triggers optimisation for the different levels of funding defined in self.annual_envelope
        """

        standard_optimisation_attributes = ['best_allocation', 'incidence', 'mortality']

        # Initialise the optimisation output container
        self.opti_results['indicator_to_minimise'] = self.indicator_to_minimise
        self.opti_results['annual_envelope'] = self.annual_envelope
        for attribute in standard_optimisation_attributes:
            self.opti_results[attribute] = []

        # Run optimisation for each envelope
        for env in self.annual_envelope:
            print "Start optimisation for annual total envelope of:" + str(env)
            self.total_funding = env * (2035. - self.inputs.model_constants['scenario_start_time'])
            self.execute_optimisation()
            full_results = self.get_full_results_opti()
            for attribute in standard_optimisation_attributes:
                self.opti_results[attribute].append(full_results[attribute])

    def get_acceptable_combinations(self):

        """
        determines the acceptable combinations of interventions according to the related starting costs and given a total
        ammount of funding
        populates the attribute 'acceptable_combinations' of model_runner.
        """

        self.acceptable_combinations = []

        n_interventions = len(self.interventions_considered_for_opti)
        full_set = range(n_interventions)
        canditate_combinations = list(itertools.chain.from_iterable(itertools.combinations(full_set, n) \
                                                                    for n in range(n_interventions + 1)[1:]))

        for combi in canditate_combinations:
            total_start_cost = 0
            for ind_intervention in combi:
                # Start-up costs apply
                if self.model_dict['manual_baseline'].intervention_startdates[self.interventions_considered_for_opti[ind_intervention]] is None:
                    total_start_cost \
                        += self.inputs.model_constants['econ_startupcost_' +
                                                       self.interventions_considered_for_opti[ind_intervention]]
            if total_start_cost <= self.total_funding:
                self.acceptable_combinations.append(combi)

    def execute_optimisation(self):
        start_timer_opti = datetime.datetime.now()
        self.optimised_combinations = []
        # Initialise a new model that will be run from 'recent_time' for optimisation
        inputs_opti = self.inputs
        inputs_opti.model_constants['scenario_end_time'] = self.year_end_opti

        self.model_dict['optimisation'] = model.ConsolidatedModel(None, inputs_opti, self.gui_inputs)
        start_time_index = \
            self.model_dict['manual_baseline'].find_time_index(self.inputs.model_constants['recent_time'])
        self.model_dict['optimisation'].start_time = \
            self.model_dict['manual_baseline'].times[start_time_index]
        self.model_dict['optimisation'].loaded_compartments = \
            self.model_dict['manual_baseline'].load_state(start_time_index)

        self.model_dict['optimisation'].eco_drives_epi = True
        self.model_dict['optimisation'].interventions_considered_for_opti = self.interventions_considered_for_opti

        self.get_acceptable_combinations()

        def force_presence_intervention(intervention):
            # keeps only combinations including intervention
            if intervention in self.interventions_considered_for_opti:
                ind_intervention = self.interventions_considered_for_opti.index(intervention)
                updated_acceptable_combinations = []
                for combi in self.acceptable_combinations:
                    if ind_intervention in combi:
                        updated_acceptable_combinations.append(combi)
                return updated_acceptable_combinations
            else:
                return self.acceptable_combinations

        for forced_intervention in self.interventions_forced_for_opti:
            self.acceptable_combinations = force_presence_intervention(forced_intervention)

        print "Number of combinations to consider: " + str(len(self.acceptable_combinations))

        for j, combi in enumerate(self.acceptable_combinations): # for each acceptable combination of interventions
            # prepare storage
            dict_optimised_combi = {'interventions': [], 'distribution': [], 'objective': None}

            for i in range(len(combi)):
                intervention = self.interventions_considered_for_opti[combi[i]]
                dict_optimised_combi['interventions'].append(intervention)

            print "Optimisation of the distribution across: "
            print dict_optimised_combi['interventions']

            # function to minimize: incidence in 2035
            def func(x):
                """
                Args:
                    x: defines the resource allocation (as absolute funding over the total period (2015 - 2035))
                Returns:
                    x has same length as combi
                    predicted incidence for 2035
                """
                #initialise funding at 0 for each intervention
                for intervention in self.interventions_considered_for_opti:
                    self.model_dict['optimisation'].available_funding[intervention] = 0.

                # input values from x
                for i in range(len(x)):
                    intervention = self.interventions_considered_for_opti[combi[i]]
                    self.model_dict['optimisation'].available_funding[intervention] = x[i] * self.total_funding
                self.model_dict['optimisation'].distribute_funding_across_years()
                self.model_dict['optimisation'].integrate()
                output_list \
                    = self.find_epi_outputs('optimisation',
                                            outputs_to_analyse=['population', 'incidence', 'true_incidence',
                                                                'mortality', 'true_mortality'])
                return output_list[self.indicator_to_minimise][-1]

            if len(combi) == 1: # the distribution result is obvious
                dict_optimised_combi['distribution'] = [1.]
                dict_optimised_combi['objective'] = func([1.])
            else:
                # initial guess
                x_0 = []
                for i in range(len(combi)):
                    x_0.append(1./len(combi))

                # Equality constraint:  Sum(x)=Total funding
                cons = [{'type': 'ineq',
                         'fun': lambda x: 1 - sum(x),  # if x is proportion
                         'jac': lambda x: -numpy.ones(len(x))}]
                bnds = []
                for i in range(len(combi)):
                    minimal_allocation = 0.
                    if self.model_dict['manual_baseline'].intervention_startdates[
                        self.model_dict['manual_baseline'].interventions_to_cost[
                            combi[i]]] is None:  # start-up costs apply
                        minimal_allocation = self.model_dict['manual_baseline'].inputs.model_constants['econ_startupcost_' + \
                                                self.model_dict['manual_baseline'].interventions_to_cost[combi[i]]] / self.total_funding
                    bnds.append((minimal_allocation, 1.0))
                # Ready to run optimisation
                res = minimize(func, x_0, jac=None, bounds=bnds, constraints=cons, method='SLSQP',
                               options={'disp': False, 'ftol': self.f_tol[self.indicator_to_minimise]})
                dict_optimised_combi['distribution'] = res.x
                dict_optimised_combi['objective'] = res.fun

            self.optimised_combinations.append(dict_optimised_combi)
            print "Combination " + str(j + 1) + "/" + str(len(self.acceptable_combinations)) + " completed."

        # Update self.optimal_allocation
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

        print 'End optimisation after ' + str(datetime.datetime.now() - start_timer_opti)

    def get_full_results_opti(self):
        """
        We need to run the best allocation scenario until 2035 to obtain the final incidence and mortality
        """
        self.model_dict['optimisation'] = model.ConsolidatedModel(None, self.inputs, self.gui_inputs)
        start_time_index = \
            self.model_dict['manual_baseline'].find_time_index(self.inputs.model_constants['recent_time'])
        self.model_dict['optimisation'].start_time = \
            self.model_dict['manual_baseline'].times[start_time_index]
        self.model_dict['optimisation'].loaded_compartments = \
            self.model_dict['manual_baseline'].load_state(start_time_index)
        self.model_dict['optimisation'].eco_drives_epi = True
        self.model_dict['optimisation'].interventions_considered_for_opti = self.interventions_considered_for_opti

        # initialise funding at 0 for each intervention
        for intervention in self.interventions_considered_for_opti:
            self.model_dict['optimisation'].available_funding[intervention] = 0.

        for intervention, prop in self.optimal_allocation.iteritems():
            self.model_dict['optimisation'].available_funding[intervention] = prop * self.total_funding
        self.model_dict['optimisation'].distribute_funding_across_years()
        self.model_dict['optimisation'].integrate()

        output_list = self.find_epi_outputs('optimisation',
                                            outputs_to_analyse=['population', 'incidence', 'true_incidence',
                                                                'mortality', 'true_mortality'])
        del self.model_dict['optimisation']
        return {'best_allocation': self.optimal_allocation, 'incidence': output_list['incidence'][-1], \
                'mortality': output_list['mortality'][-1]}

    ###########################
    ### GUI-related methods ###
    ###########################

    def add_comment_to_gui_window(self, comment, target='console'):

        if self.js_gui:
            emit(target, {"message": comment})
            time.sleep(self.emit_delay)

            print "Emitting:", comment

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
            self.plot_progressive_parameters_tk(from_runner=True)

    def plot_progressive_parameters_tk(self, from_runner=True, input_figure=None):

        # Initialise plotting
        if from_runner:
            param_tracking_figure = plt.Figure()
            parameter_plots = FigureCanvasTkAgg(param_tracking_figure, master=self.figure_frame)

        else:
            param_tracking_figure = input_figure

        subplot_grid = outputs.find_subplot_numbers(len(self.all_parameters_tried))

        # Cycle through parameters with one subplot for each parameter
        for p, param in enumerate(self.all_parameters_tried):

            # Extract accepted params from all tried params
            accepted_params = list(p for p, a in zip(self.all_parameters_tried[param], self.whether_accepted_list) if a)

            # Plot
            ax = param_tracking_figure.add_subplot(subplot_grid[0], subplot_grid[1], p + 1)
            ax.plot(range(1, len(accepted_params) + 1), accepted_params, linewidth=2, marker='o', markersize=4,
                    mec='b', mfc='b')
            ax.set_xlim((1., len(self.accepted_indices) + 1))

            # Find the y-limits from the parameter bounds and the parameter values tried
            for param_number in range(len(self.inputs.param_ranges_unc)):
                if self.inputs.param_ranges_unc[param_number]['key'] == param:
                    bounds = self.inputs.param_ranges_unc[param_number]['bounds']
            ylim_margins = .1
            min_ylimit = min(accepted_params + [bounds[0]])
            max_ylimit = max(accepted_params + [bounds[1]])
            ax.set_ylim((min_ylimit * (1 - ylim_margins), max_ylimit * (1 + ylim_margins)))

            # Indicate the prior bounds
            ax.plot([1, len(self.accepted_indices) + 1], [min_ylimit, min_ylimit], color='0.8')
            ax.plot([1, len(self.accepted_indices) + 1], [max_ylimit, max_ylimit], color='0.8')

            # Plot rejected parameters
            for run, rejected_params in self.rejection_dict[param].items():
                if self.rejection_dict[param][run]:
                    ax.plot([run + 1] * len(rejected_params), rejected_params, marker='o', linestyle='None',
                            mec='0.5', mfc='0.5', markersize=3)
                    for r in range(len(rejected_params)):
                        ax.plot([run, run + 1], [self.acceptance_dict[param][run], rejected_params[r]], color='0.5',
                                linestyle='--')

            # Label
            ax.set_title(tool_kit.find_title_from_dictionary(param))
            if p > len(self.all_parameters_tried) - subplot_grid[1] - 1:
                ax.set_xlabel('Accepted runs')

            if from_runner:

                # Output to GUI window
                parameter_plots.show()
                parameter_plots.draw()
                parameter_plots.get_tk_widget().grid(row=1, column=1)

        if not from_runner:
            return param_tracking_figure

    def plot_progressive_parameters_js(self):

        """
        Method to shadow previous method in JavaScript GUI.
        """

        accepted_params = [
            list(p for p, a in zip(self.all_parameters_tried[param], self.whether_accepted_list) if a)[-1]
            for p, param in enumerate(self.all_parameters_tried)]
        names = [tool_kit.find_title_from_dictionary(param) for p, param in
                 enumerate(self.all_parameters_tried)]
        emit('uncertainty_graph', {"data": accepted_params, "names": names, "count": self.plot_count})
        self.plot_count += 1

