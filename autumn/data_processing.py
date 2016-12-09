
import autumn.spreadsheet as spreadsheet
import copy
import numpy
import tool_kit
from curve import scale_up_function, freeze_curve
from Tkinter import *

import time
import eventlet
from flask_socketio import emit

def calculate_proportion_dict(data, indices, percent=False):

    """
    General method to calculate proportions from absolute values provided as dictionaries.

    Args:
        data: Dicionary containing the absolute values.
        indices: The keys of data from which proportions are to be calculated (generally a list of strings).
        percent: Boolean describing whether the method should return the output as a percent or proportion.

    Returns:
        proportions: A dictionary of the resulting proportions.
    """

    # Calculate multiplier for percentages if requested, otherwise leave as one
    if percent:
        multiplier = 1E2
    else:
        multiplier = 1.

    # Create a list of the years that are common to all indices within data
    lists_of_years = []
    for i in range(len(indices)):
        lists_of_years += [data[indices[i]].keys()]
    common_years = find_common_elements_multiple_lists(lists_of_years)

    # Calculate the denominator by summing the values for which proportions have been requested
    denominator = {}
    for i in common_years:
        for j in indices:
            if j == indices[0]:
                denominator[i] = data[j][i]
            else:
                denominator[i] += data[j][i]

    # Calculate the proportions
    proportions = {}
    for j in indices:
        proportions['prop_' + j] = {}
        for i in common_years:
            if denominator[i] > 0.:
                proportions['prop_' + j][i] = \
                    data[j][i] / denominator[i] \
                    * multiplier

    return proportions


def find_common_elements_multiple_lists(list_of_lists):

    """
    Simple method to find the common elements of any number of lists

    Args:
        list_of_lists: A list whose elements are all the lists we want to find the
            intersection of.

    Returns:
        intersection: Common elements of all lists.
    """

    intersection = list_of_lists[0]
    for i in range(1, len(list_of_lists)):
        intersection = find_common_elements(intersection, list_of_lists[i])
    return intersection


def find_common_elements(list_1, list_2):

    """
    Simple method to find the intersection of two lists

    Args:
        list_1 and list_2: The two lists

    Returns:
        intersection: The common elements of the two lists
    """

    intersection = []
    for i in list_1:
        if i in list_2:
            intersection += [i]
    return intersection


def remove_specific_key(dictionary, key):

    """
    Remove a specific named key from a dictionary

    Args:
        dictionary: The dictionary to have a key removed
        key: The key to be removed

    Returns:
        dictionary: The dictionary with the key removed
    """

    if key in dictionary:
        del dictionary[key]

    return dictionary


def remove_nans(dictionary):

    """
    Takes a dictionary and removes all of the elements for which the value is nan

    Args:
        dictionary: Should typically be the dictionary of programmatic values, usually
                    with time in years as the key.

    Returns:
        dictionary: The dictionary with the nans removed.
    """

    nan_indices = []
    for i in dictionary:
        if type(dictionary[i]) == float and numpy.isnan(dictionary[i]):
            nan_indices += [i]
    for i in nan_indices:
        del dictionary[i]

    return dictionary


class Inputs:

    def __init__(self, gui_inputs, runtime_outputs, from_test=False, js_gui=False):

        self.gui_inputs = gui_inputs
        self.country = gui_inputs['country']
        self.runtime_outputs = runtime_outputs
        self.original_data = None
        self.from_test = from_test
        self.derived_data = {}
        self.time_variants = {}
        self.model_constants = {}
        self.available_strains = [
            '_ds',
            '_mdr',
            '_xdr']
        self.available_organs = [
            '_smearpos',
            '_smearneg',
            '_extrapul']
        self.agegroups = None
        self.irrelevant_time_variants = []
        self.is_organvariation = False
        self.scaleup_data = {}
        self.scaleup_fns = {}
        self.intervention_applied = {}
        self.param_ranges_unc = []
        self.mode = 'uncertainty'
        self.data_to_fit = {}
        # For incidence for ex. Width of Normal posterior relative to CI width in data
        self.outputs_unc = [{'key': 'incidence',
                             'posterior_width': None,
                             'width_multiplier': 2.}]
        #self.potential_interventions_to_cost = ['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf',
        #                                        'ipt_age0to5', 'ipt_age5to15', 'decentralisation']
        self.potential_interventions_to_cost = ['vaccination','xpertacf_ruralpoor', \
                                                  'xpertacf_prison', 'xpertacf', 'xpert', 'engage_lowquality']

        if self.gui_inputs['n_strains'] > 1:
            self.potential_interventions_to_cost += ['shortcourse_mdr']
        if self.gui_inputs['is_lowquality']:
            self.potential_interventions_to_cost += ['engage_lowquality']
        if self.gui_inputs['comorbidity_prison']:
            self.potential_interventions_to_cost += ['xpertacf_prison']
        if self.gui_inputs['comorbidity_indigenous']:
            self.potential_interventions_to_cost += ['xpertacf_indigenous']
        if self.gui_inputs['comorbidity_urbanpoor']:
            self.potential_interventions_to_cost += ['xpertacf_urbanpoor']
        if self.gui_inputs['comorbidity_ruralpoor']:
            self.potential_interventions_to_cost += ['xpertacf_ruralpoor']

        self.interventions_to_cost = []

        self.emit_delay = 0.1
        self.plot_count = 0

        self.js_gui = js_gui

        if self.js_gui:
            eventlet.monkey_patch()

    def read_and_load_data(self):

        # Default keys of sheets to read (ones that should always be read)
        self.add_comment_to_gui_window('Reading Excel sheets with input data.\n')
        keys_of_sheets_to_read = ['bcg', 'rate_birth', 'life_expectancy',
                                  'default_parameters', 'tb', 'notifications', 'outcomes',
                                  'country_constants', 'default_constants',
                                  'country_programs', 'default_programs']

        # Add the optional ones (this is intended to be the standard approach to reading additional
        # data to the data object - currently not that useful as it only applies to diabetes)
        if 'comorbidity_diabetes' in self.gui_inputs:
            keys_of_sheets_to_read += ['diabetes']

        # Read all the original data required
        self.original_data \
            = spreadsheet.read_input_data_xls(self.from_test,
                                              keys_of_sheets_to_read,
                                              self.country)

        ####################################
        # Work through the data processing #
        ####################################

        # Find the proportion of new cases by organ status and start to populate the derived data dictionary
        self.find_organ_proportions()

        # Extract freeze times for scenarios as a separate dictionary
        self.extract_freeze_times()

        # Start to populate the time variants dictionary
        if 'country_programs' in self.original_data:
            self.time_variants.update(self.original_data['country_programs'])

        # Populate time variant dictionary with defaults where not present in country-specific data
        self.add_time_variant_defaults()

        # Add vaccination and case detection time variants to time variant dictionary
        self.update_time_variants()

        # Populate constant model values hierarchically
        self.add_model_constant_defaults()

        # Convert time variants loaded as percentages to proportions
        self.convert_percentages_to_proportions()

        # Populate freeze time dictionary with defaults where unavailable
        self.complete_freeze_time_dictionary()

        # Find outcomes for smear-positive DS-TB patients and populate to derived data dictionary
        self.find_ds_outcomes()
        self.add_treatment_outcomes()

        # Add ds to the naming of the treatment outcomes for multistrain models
        if self.gui_inputs['n_strains'] > 1:
            self.duplicate_ds_outcomes_for_multistrain()

        # Add time variant demographic dictionaries
        self.add_demo_dictionaries_to_timevariants()

        # Add time-variant organ status to time variant parameters
        if self.time_variants['epi_prop_smearpos']['load_data'] == u'yes':
            self.add_organ_status_to_timevariants()

        # Add outcomes for resistant strains - currently using XDR-TB outcomes for inappropriate treatment
        self.add_resistant_strain_outcomes()

        # Add zeroes, remove nans and remove load_data key from time variant dictionaries
        self.tidy_time_variants()

        # Add hard-coded parameters that are universal to all models that require them
        self.add_universal_parameters()

        # Describe and work out age stratification structure for model from the list of age breakpoints
        self.agegroups, _ = tool_kit.get_agegroups_from_breakpoints(self.model_constants['age_breakpoints'])

        # Find ageing rates and age-weighted parameters
        if len(self.agegroups) > 1:
            self.find_ageing_rates()
            self.find_fixed_age_specific_parameters()
            agegroups_to_print = ''
            for a, agegroup in enumerate(self.model_constants['age_breakpoints']):
                if a == len(self.model_constants['age_breakpoints']) - 1:
                    agegroups_to_print += ' and ' + str(agegroup) + '.\n'
                elif a == len(self.model_constants['age_breakpoints']) - 2:
                    agegroups_to_print += str(agegroup)
                else:
                    agegroups_to_print += str(agegroup) + ', '
            self.add_comment_to_gui_window('Age breakpoints are at: %s' % agegroups_to_print)
        else:
            self.add_comment_to_gui_window('Model is not stratified by age.\n')

        # Add treatment time periods for single strain model, as only populated for DS-TB to now
        if self.gui_inputs['n_strains'] == 0:
            self.find_single_strain_timeperiods()

        # Define the structuring of comorbidities for the model
        self.define_comorbidity_structure()
        if len(self.comorbidities) == 1:
            self.add_comment_to_gui_window('Model does not incorporate any additional risk groups.\n')
        elif len(self.comorbidities) == 2:
            self.add_comment_to_gui_window('Model incorporates one additional risk group.\n')
        elif len(self.comorbidities) > 2:
            self.add_comment_to_gui_window('Model incorporates %s additional risk groups.\n'
                                        % str(len(self.comorbidities) - 1))

        # Code to ensure some starting proportion of births go to the comorbidity stratum if value not loaded earlier
        for comorbidity in self.comorbidities:
            if 'comorb_prop' + comorbidity not in self.model_constants:
                self.model_constants['comorb_prop' + comorbidity] = 0.

        # Define the strain structure for the model
        self.define_strain_structure()

        # Define the organ status structure for the model
        self.define_organ_structure()

        # Find the time non-infectious on treatment from the total time on treatment and the time infectious
        self.find_noninfectious_period()

        # List all the time variant parameters that are not relevant to this model structure
        self.list_irrelevant_time_variants()

        # Find comorbidity-specific parameters
        if len(self.comorbidities) > 1:
            self.find_comorb_progressions()

        # Calculate rates of progression to active disease or late latency
        self.find_progression_rates_from_params()

        # Work through whether organ status should be time variant
        self.find_organ_time_variation()

        # Create a scale-up dictionary for resistance amplification if appropriate
        if self.gui_inputs['n_strains'] > 1:
            self.find_amplification_data()
            self.add_comment_to_gui_window('Model simulating %d strains.\n' % self.gui_inputs['n_strains'])
        else:
            self.add_comment_to_gui_window('Model simulating single strain only.\n')

        # Derive some basic parameters for IPT
        self.find_ipt_params()

        # Extract data from time variants dictionary and populate to dictionary with scenario keys
        self.find_data_for_functions_or_params()

        # Find scale-up functions or constant parameters from
        self.find_functions_or_params()

        # Find extrapulmonary proportion if model is stratified by organ type, but there is no time variant organ
        # proportion. Note that this has to be done after find_functions_or_params or the constant parameter
        # won't have been calculated yet.
        if not self.is_organvariation and len(self.organ_status) > 2:
            self.find_constant_extrapulmonary_proportion()

        # Find the proportion of cases that are infectious for models that are unstratified by organ status
        if len(self.organ_status) < 2:
            self.set_fixed_infectious_proportion()

        # Add parameters for IPT, if and where not specified for the age range being implemented
        self.add_missing_economics_for_ipt()

        self.find_interventions_to_cost()

        # Specify the parameters to be used for uncertainty
        if self.gui_inputs['output_uncertainty']:
            self.find_uncertainty_distributions()
            self.get_data_to_fit()

        # Perform checks
        self.checks()

    def extract_freeze_times(self):

        """
        Extract the freeze_times for each scenario, if specified. If not specified, will be populated in
        self.complete_freeze_time_dictionary below.
        """

        if 'country_programs' in self.original_data and 'freeze_times' in self.original_data['country_programs']:
            self.freeze_times = self.original_data['country_programs'].pop('freeze_times')
        else:
            self.freeze_times = {}

    def update_time_variants(self):

        """
        Adds vaccination and case detection time-variants to the manually entered data
        loaded from the spreadsheets.
        Note that the manual inputs always over-ride the loaded data if both are present.
        """

        # Vaccination
        if self.time_variants['program_perc_vaccination']['load_data'] == u'yes':
            for year in self.original_data['bcg']:
                # If not already loaded through the inputs spreadsheet
                if year not in self.time_variants['program_perc_vaccination']:
                    self.time_variants['program_perc_vaccination'][year] \
                        = self.original_data['bcg'][year]

        # Case detection
        if self.time_variants['program_perc_detect']['load_data'] == u'yes':
            for year in self.original_data['tb']['c_cdr']:
                # If not already loaded through the inputs spreadsheet
                if year not in self.time_variants['program_perc_detect']:
                    self.time_variants['program_perc_detect'][year] \
                        = self.original_data['tb']['c_cdr'][year]

    def find_organ_proportions(self):

        """
        Calculates dictionaries with proportion of cases progressing to each organ status by year,
        and adds these to the derived_data attribute of the object.

        """

        self.derived_data.update(calculate_proportion_dict(self.original_data['notifications'],
                                                           ['new_sp', 'new_sn', 'new_ep']))

    def add_time_variant_defaults(self):

        """
        Populates time variant parameters with defaults if those values aren't found
        in the manually entered country-specific data.

        """

        for program_var in self.original_data['default_programs']:

            # If the key isn't in available for the country at all
            if program_var not in self.time_variants:
                self.time_variants[program_var] = \
                    self.original_data['default_programs'][program_var]

            # Otherwise if it's there, populate for the missing years
            else:
                for year in self.original_data['default_programs'][program_var]:
                    if year not in self.time_variants[program_var]:
                        self.time_variants[program_var][year] = \
                            self.original_data['default_programs'][program_var][year]

    def add_model_constant_defaults(self,
                                    other_sheets_with_constants=('diabetes',
                                                                 'country_constants',
                                                                 'default_constants')):

        """
        Populate model_constants with data from control panel, country sheet or default sheet hierarhically
        - such that the control panel is read in preference to the country data in preference to the default back-ups
        Args:
            other_sheets_with_constants: The sheets of original_data which contain model constants
        """

        # Populate from country_constants if available and default_constants if not
        for other_sheet in other_sheets_with_constants:
            if other_sheet in self.original_data:
                for item in self.original_data[other_sheet]:

                    # Only add the item if it hasn't been added yet
                    if item not in self.model_constants:
                        self.model_constants[item] = \
                            self.original_data[other_sheet][item]

    def convert_percentages_to_proportions(self):

        """
        Converts time-variant dictionaries to proportions if they are loaded as percentages in their raw form.

        """

        time_variants_converted_to_prop = {}
        for time_variant in self.time_variants:

            # If the entered data is a percentage
            if 'perc_' in time_variant:
                time_variants_converted_to_prop[time_variant.replace('perc_', 'prop_')] = {}
                for i in self.time_variants[time_variant]:
                    # If it's a year or scenario
                    if type(i) == int or 'scenario' in i:
                        time_variants_converted_to_prop[time_variant.replace('perc_', 'prop_')][i] \
                            = self.time_variants[time_variant][i] / 1e2
                    else:
                        time_variants_converted_to_prop[time_variant.replace('perc_', 'prop_')][i] \
                            = self.time_variants[time_variant][i]

        self.time_variants.update(time_variants_converted_to_prop)

    def complete_freeze_time_dictionary(self):

        """
        Ensure all scenarios have an entry in the self.freeze_times dictionary, if a time hadn't been
        specified in self.extract_freeze_times above.
        """

        for scenario in self.gui_inputs['scenarios_to_run']:

            # Baseline
            if scenario is None:
                self.freeze_times['baseline'] = self.model_constants['current_time']

            # Scenarios with no time specified
            elif 'scenario_' + str(scenario) not in self.freeze_times:
                self.freeze_times['scenario_' + str(scenario)] = self.model_constants['current_time']

    def find_ds_outcomes(self):

        """
        Calculates proportions of patients with each reported outcome for DS-TB,
        then sums cure and completion to obtain treatment success proportion.
        Note that the outcomes are reported differently for resistant strains, so code differs for them.
        """

        # Calculate each proportion
        self.derived_data.update(
            calculate_proportion_dict(
                self.original_data['outcomes'],
                ['new_sp_cmplt', 'new_sp_cur', 'new_sp_def', 'new_sp_died', 'new_sp_fail'],
                percent=False))

        # Sum to get treatment success
        self.derived_data['prop_new_sp_success'] = {}
        for year in self.derived_data['prop_new_sp_cmplt']:
            self.derived_data['prop_new_sp_success'][year] \
                = self.derived_data['prop_new_sp_cmplt'][year] \
                  + self.derived_data['prop_new_sp_cur'][year]

    def add_treatment_outcomes(self):

        """
        Add treatment outcomes for DS-TB to the time variants attribute.
        Use the same approach as above to adding if requested and if data not manually entered.
        """

        # Iterate over success and death outcomes
        for outcome in ['_success', '_death']:

            # Populate data
            if self.time_variants['program_prop_treatment' + outcome]['load_data'] == u'yes':

                # Correct naming GTB report
                if outcome == '_success':
                    report_outcome = '_success'
                elif outcome == '_death':
                    report_outcome = '_died'

                for year in self.derived_data['prop_new_sp' + report_outcome]:
                    if year not in self.time_variants['program_prop_treatment' + outcome]:
                        self.time_variants['program_prop_treatment' + outcome][year] \
                            = self.derived_data['prop_new_sp' + report_outcome][year]

    def duplicate_ds_outcomes_for_multistrain(self):

        """
        Duplicates the treatment outcomes with DS-TB key if it is a multi-strain model.
        """

        for outcome in ['_success', '_death']:
            self.time_variants['program_prop_treatment' + outcome + '_ds'] \
                = copy.copy(self.time_variants['program_prop_treatment' + outcome])

    def add_demo_dictionaries_to_timevariants(self):

        """
        Add epidemiological time variant parameters to time_variants.
        Similarly to previous methods, only performed if requested
        and only populated where absent (either entirely or for that year).
        """

        for demo_parameter in ['life_expectancy', 'rate_birth']:
            if self.time_variants['demo_' + demo_parameter]['load_data'] == u'yes':
                for year in self.original_data[demo_parameter]:
                    if year not in self.time_variants['demo_' + demo_parameter]:
                        self.time_variants['demo_' + demo_parameter][year] \
                            = self.original_data[demo_parameter][year]

    def add_organ_status_to_timevariants(self):

        """
        Populate organ status dictionaries where requested and not already loaded.
        """

        # Iterate over smear positive and negative status
        for outcome in ['_smearpos', '_smearneg']:

            # Correct naming GTB report
            if outcome == '_smearpos':
                report_outcome = '_sp'
            elif outcome == '_smearneg':
                report_outcome = '_sn'

            # Populate data
            for year in self.derived_data['prop_new' + report_outcome]:
                if year not in self.time_variants['epi_prop' + outcome]:
                    self.time_variants['epi_prop' + outcome][year] \
                        = self.derived_data['prop_new' + report_outcome][year]

    def add_resistant_strain_outcomes(self):

        """
        Finds treatment outcomes for the resistant strains (i.e. MDR and XDR-TB).
        As for DS-TB, no need to find default proportion, as it is equal to one minus success minus death.
        **** Inappropriate outcomes are currently set to those for XDR-TB - this is temporary ****
        """

        # Calculate proportions of each outcome for MDR and XDR-TB from GTB
        for strain in ['mdr', 'xdr']:
            self.derived_data.update(
                calculate_proportion_dict(self.original_data['outcomes'],
                                          [strain + '_succ', strain + '_fail', strain + '_died', strain + '_lost'],
                                          percent=False))

            # Populate MDR and XDR data from outcomes dictionary into time variants where requested and not entered
            if self.time_variants['program_prop_treatment_success_' + strain]['load_data'] == u'yes':
                for year in self.derived_data['prop_' + strain + '_succ']:
                    if year not in self.time_variants['program_prop_treatment_success_' + strain]:
                        self.time_variants['program_prop_treatment_success_' + strain][year] \
                            = self.derived_data['prop_' + strain + '_succ'][year]

        # Temporarily assign the same treatment outcomes to XDR-TB as for inappropriate
        for outcome in ['_success', '_death']:
            self.time_variants['program_prop_treatment' + outcome + '_inappropriate'] \
                = copy.copy(self.time_variants['program_prop_treatment' + outcome + '_xdr'])

    def tidy_time_variants(self):

        """
        Perform final rounds of tidying of time-variants
        """

        # Looping over each time variant
        for program in self.time_variants:

            # Add zero at starting time for model run to all programs that are proportions
            if 'program_prop' in program:
                self.time_variants[program][int(self.model_constants['start_time'])] = 0.

            # Remove the load_data keys, as they have been used and are now redundant
            self.time_variants[program] \
                = remove_specific_key(self.time_variants[program], 'load_data')

            # Remove dictionary keys for which values are nan
            self.time_variants[program] \
                = remove_nans(self.time_variants[program])

    def add_universal_parameters(self):

        """
        Sets parameters that should never be changed in any situation,
        i.e. "by definition" parameters (although note that the infectiousness
        of the single infectious compartment for models unstratified by organ
        status is now set in set_fixed_infectious_proportion, because it is
        dependent upon loading some parameters in find_functions_or_params)
        """

        if self.gui_inputs['n_organs'] < 2:
            # Proportion progressing to the only infectious compartment for models unstratified by organ status
            self.model_constants['epi_prop'] = 1.
        else:
            self.model_constants['tb_multiplier_force_smearpos'] \
                = 1.  # Infectiousness of smear-positive patients
            self.model_constants['tb_multiplier_force_extrapul'] \
                = 0.  # Infectiousness of extrapulmonary patients

    def find_fixed_age_specific_parameters(self):

        """
        Find weighted age specific parameters using Romain's age weighting code (now in took_kit)
        """

        # Extract age breakpoints in appropriate form for module
        model_breakpoints = []
        for i in self.model_constants['age_breakpoints']:
            model_breakpoints += [float(i)]

        for param in ['early_progression_age', 'late_progression_age',
                      'tb_multiplier_child_infectiousness_age']:
            # Extract age-stratified parameters in the appropriate form
            prog_param_vals = {}
            prog_age_dict = {}
            for constant in self.model_constants:
                if param in constant:
                    prog_param_string, prog_stem = \
                        tool_kit.find_string_from_starting_letters(constant, '_age')
                    prog_age_dict[prog_param_string], _ = \
                        tool_kit.interrogate_age_string(prog_param_string)
                    prog_param_vals[prog_param_string] = \
                        self.model_constants[constant]

            param_breakpoints = tool_kit.find_age_breakpoints_from_dicts(prog_age_dict)

            # Find and set age-adjusted parameters
            prog_age_adjusted_params = \
                tool_kit.adapt_params_to_stratification(param_breakpoints,
                                                        model_breakpoints,
                                                        prog_param_vals,
                                                        parameter_name=param,
                                                        whether_to_plot=self.gui_inputs['output_age_calculations'])
            for agegroup in self.agegroups:
                self.model_constants[prog_stem + agegroup] = prog_age_adjusted_params[agegroup]

    def find_ageing_rates(self):

        """
        Calculate ageing rates as the reciprocal of the width of the age bracket.
        """

        for agegroup in self.agegroups:
            age_limits, _ = tool_kit.interrogate_age_string(agegroup)
            if 'up' not in agegroup:
                self.model_constants['ageing_rate' + agegroup] \
                    = 1. / (age_limits[1] - age_limits[0])

    def find_single_strain_timeperiods(self):

        """
        If the model isn't stratified by strain, use DS-TB time-periods for the single strain.
        Note that the parameter for the time period infectious on treatment will only be defined
        for DS-TB in this case and not for no strain name.
        """

        for timeperiod in ['tb_timeperiod_infect_ontreatment', 'tb_timeperiod_ontreatment']:
            self.model_constants[timeperiod] \
                = self.model_constants[timeperiod + '_ds']

    def define_comorbidity_structure(self):

        """
        Work out the comorbidity stratification
        """

        # Create list of comorbidity names
        self.comorbidities = []
        for time_variant in self.time_variants:
            if 'comorb_prop_' in time_variant and self.gui_inputs['comorbidity' + time_variant[11:]]:
                self.comorbidities += [time_variant[11:]]
        if len(self.comorbidities) == 0:
            self.comorbidities += ['']
        else:
            self.comorbidities += ['_nocomorb']

    def define_strain_structure(self):

        """
        Finds the strains to be present in the model from a list of available strains and
        the integer value for the number of strains selected.
        """

        # Need a list of an empty string to be iterable for methods iterating by strain
        if self.gui_inputs['n_strains'] == 0:
            self.strains = ['']
        else:
            self.strains = self.available_strains[:self.gui_inputs['n_strains']]

    def define_organ_structure(self):

        """
        Defines the organ status stratification from the number of statuses selected
        Note that "organ" is the simplest single-word term that I can currently think of to
        describe whether patients have smear-positive, smear-negative or extrapulmonary disease.
        """

        if self.gui_inputs['n_organs'] == 0:
            # Need a list of an empty string to be iterable for methods iterating by organ status
            self.organ_status = ['']
        else:
            self.organ_status = self.available_organs[:self.gui_inputs['n_organs']]

    def find_noninfectious_period(self):

        """
        Work out the periods of time spent non-infectious for each strain (plus inappropriate as required)
        by very simple subtraction.
        """

        treatment_outcome_types = copy.copy(self.strains)
        if self.gui_inputs['n_strains'] > 1 and self.gui_inputs['is_misassignment']:
            treatment_outcome_types += ['_inappropriate']

        for strain in treatment_outcome_types:

            # Find the non-infectious periods
            self.model_constants['tb_timeperiod_noninfect_ontreatment' + strain] \
                = self.model_constants['tb_timeperiod_ontreatment' + strain] \
                  - self.model_constants['tb_timeperiod_infect_ontreatment' + strain]

    def find_comorb_progressions(self):

        """
        Code to adjust the progression rates to active disease for various comorbidities - so far diabetes and HIV.

        """

        # Initialise dictionary of additional adjusted parameters to avoid dictionary changing size during iterations
        comorb_adjusted_parameters = {}
        for comorb in self.comorbidities:
            for param in self.model_constants:

                # Start from the assumption that parameter is not being adjusted
                whether_to_adjust = False

                # For age-stratified parameters
                if '_age' in param:

                    # Find the age string, the lower and upper age limits and the parameter name without the age string
                    age_string, _ = tool_kit.find_string_from_starting_letters(param, '_age')
                    age_limits, _ = tool_kit.interrogate_age_string(age_string)
                    param_without_age = param[:-len(age_string)]

                    # Diabetes progression rates only start from age groups with lower limit above the start age
                    # and apply to both early and late progression.
                    if comorb == '_diabetes' and '_progression' in param \
                            and age_limits[0] >= self.model_constants['comorb_startage' + comorb]:
                        whether_to_adjust = True

                    # HIV applies to all age groups, but only late progression
                    elif comorb == '_hiv' and '_late_progression' in param:
                        whether_to_adjust = True

                    # Shouldn't apply this to the multiplier parameters or non-TB-specific parameters
                    if '_multiplier' in param or 'tb_' not in param:
                        whether_to_adjust = False

                    # Now adjust the age-stratified parameter values
                    if whether_to_adjust:
                        comorb_adjusted_parameters[param_without_age + comorb + age_string] \
                            = self.model_constants[param] \
                              * self.model_constants['comorb_multiplier' + comorb + '_progression']
                    elif '_progression' in param:
                        comorb_adjusted_parameters[param_without_age + comorb + age_string] \
                            = self.model_constants[param]

                # Parameters not stratified by age
                else:

                    # Explanation as above
                    if comorb == '_diabetes' and '_progression' in param:
                        whether_to_adjust = True
                    elif comorb == '_hiv' and '_late_progression' in param:
                        whether_to_adjust = True
                    if '_multiplier' in param or 'tb_' not in param:
                        whether_to_adjust = False

                    # Adjustment as above, except age string not included
                    if whether_to_adjust:
                        comorb_adjusted_parameters[param + comorb] \
                            = self.model_constants[param] \
                              * self.model_constants['comorb_multiplier' + comorb + '_progression']
                    elif '_progression' in param:
                        comorb_adjusted_parameters[param + comorb] \
                            = self.model_constants[param]

        self.model_constants.update(comorb_adjusted_parameters)

    def find_progression_rates_from_params(self):

        """
        Find early progression rates by age group and by comorbidity status - i.e. early progression to
        active TB and stabilisation into late latency.

        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:

                # Early progression rate is early progression proportion divided by early time period
                self.model_constants['tb_rate_early_progression' + comorbidity + agegroup] \
                    = self.model_constants['tb_prop_early_progression' + comorbidity + agegroup] \
                      / self.model_constants['tb_timeperiod_early_latent']

                # Stabilisation rate is one minus early progression proportion divided by early time period
                self.model_constants['tb_rate_stabilise' + comorbidity + agegroup] \
                    = (1. - self.model_constants['tb_prop_early_progression' + comorbidity + agegroup]) \
                      / self.model_constants['tb_timeperiod_early_latent']

    def find_organ_time_variation(self):

        """
        Work through whether variation in organ status with time should be implemented,
        according to whether the model is stratified by organ and whether organ stratification is requested
        through the smear-positive time-variant input.
        """

        # If no organ stratification
        if self.gui_inputs['n_organs'] < 2:
            # Leave organ variation as false if no organ stratification and warn if variation requested
            for status in ['pos', 'neg']:
                if self.time_variants['epi_prop_smear' + status]['time_variant'] == u'yes':
                    self.add_comment_to_gui_window(
                                                'Time variant smear-' + status + ' proportion requested, but ' +
                                                'model is not stratified by organ status. ' +
                                                'Therefore, time variant smear-' + status +
                                                ' status has been changed to off.\n')

                    self.time_variants['epi_prop_smear' + status]['time_variant'] = u'no'
        else:

            # Change to organ variation true if organ stratification and smear-positive variation requested
            if self.time_variants['epi_prop_smearpos']['time_variant'] == u'yes':
                self.is_organvariation = True
                # Warn if smear-negative variation not requested
                if self.time_variants['epi_prop_smearneg']['time_variant'] == u'no':
                    self.add_comment_to_gui_window(
                                                'Requested time variant smear-positive status, but ' +
                                                'not time variant smear-negative status. ' +
                                                'Therefore, changed to time variant smear-negative status.\n')

                    self.time_variants['epi_prop_smearneg']['time_variant'] = u'yes'

            # Leave organ variation as false if smear-positive variation not requested
            elif self.time_variants['epi_prop_smearpos']['time_variant'] == u'no':
                # Warn if smear-negative variation requested
                if self.time_variants['epi_prop_smearneg']['time_variant'] == u'yes':
                    self.add_comment_to_gui_window(
                                                'Requested non-time variant smear-positive status, but ' +
                                                'time variant smear-negative status. ' +
                                                'Therefore, changed to non-time variant smear-negative status.\n')

                    self.time_variants['epi_prop_smearneg']['time_variant'] = u'no'

        # Set fixed parameters if no organ status variation
        if not self.is_organvariation:
            for organ in self.organ_status:
                for timing in ['_early', '_late']:
                    for agegroup in self.agegroups:
                        self.model_constants['tb_rate' + timing + '_progression' + organ + agegroup] \
                            = self.model_constants['tb_rate' + timing + '_progression' + agegroup] \
                              * self.model_constants['epi_prop' + organ]

    def find_amplification_data(self):

        """
        Add dictionary for the amplification proportion scale-up, where relevant.
        """

        self.time_variants['epi_prop_amplification'] \
            = {self.model_constants['start_mdr_introduce_time']: 0.,
               self.model_constants['end_mdr_introduce_time']: self.model_constants['tb_prop_amplification'],
               'time_variant': u'yes'}

    def find_ipt_params(self):

        """
        Calculate number of persons eligible for IPT per person commencing treatment and then the number of persons
        who receive effective IPT per person assessed for LTBI.
        """

        self.model_constants['ipt_eligible_per_treatment_start'] = (self.model_constants['demo_household_size'] - 1.) \
                                                                   * self.model_constants['tb_prop_contacts_infected']

        for ipt_type in ['', 'novel_']:
            self.model_constants[ipt_type + 'ipt_effective_per_assessment'] \
                = self.model_constants['tb_prop_ltbi_test_sensitivity'] \
                  * self.model_constants['tb_prop_' + ipt_type + 'ipt_effectiveness']

    def find_data_for_functions_or_params(self):

        """
        Method to load all the dictionaries to be used in generating scale-up functions to
        a single attribute of the class instance (to avoid creating heaps of functions for
        irrelevant programs)

        Returns:
            Creates self.scaleup_data, a dictionary of the relevant scale-up data for creating
             scale-up functions in set_scaleup_functions within the model object. First tier
             of keys is the scenario to be run, next is the time variant parameter to be calculated.

        """

        for scenario in self.gui_inputs['scenarios_to_run']:

            self.scaleup_data[scenario] = {}
            # Find the programs that are relevant and load them to the scaleup_data attribute
            for time_variant in self.time_variants:
                if time_variant not in self.irrelevant_time_variants:
                    self.scaleup_data[scenario][str(time_variant)] = {}
                    for i in self.time_variants[time_variant]:
                        if i == 'scenario_' + str(scenario):
                            self.scaleup_data[scenario][str(time_variant)]['scenario'] = \
                                self.time_variants[time_variant][i]
                        elif type(i) == str:
                            if 'scenario_' not in i:
                                self.scaleup_data[scenario][str(time_variant)][i] \
                                    = self.time_variants[time_variant][i]
                        elif scenario is None or 'program_' not in time_variant:
                            self.scaleup_data[scenario][str(time_variant)][i] \
                                = self.time_variants[time_variant][i]
                        else:
                            self.scaleup_data[scenario][str(time_variant)][i] \
                                = self.time_variants[time_variant][i]

    def list_irrelevant_time_variants(self):

        """
        List all the time-variant parameters that are not relevant to the current model structure.
        """

        for time_variant in self.time_variants:
            if 'perc_' in time_variant:
                self.irrelevant_time_variants += [time_variant]
            for strain in self.available_strains:
                if strain not in self.strains and strain in time_variant and '_dst' not in time_variant:
                    self.irrelevant_time_variants += [time_variant]
            if self.gui_inputs['n_strains'] < 2 and 'line_dst' in time_variant:
                self.irrelevant_time_variants += [time_variant]
            elif '_inappropriate' in time_variant \
                    and (self.gui_inputs['n_strains'] < 2 or not self.gui_inputs['is_misassignment']):
                self.irrelevant_time_variants += [time_variant]
            elif self.gui_inputs['n_strains'] == 2 and 'secondline_dst' in time_variant:
                self.irrelevant_time_variants += [time_variant]
            elif self.gui_inputs['n_organs'] == 1 and 'smearneg' in time_variant:
                self.irrelevant_time_variants += [time_variant]
            if 'lowquality' in time_variant and not self.gui_inputs['is_lowquality']:
                self.irrelevant_time_variants += [time_variant]

    def find_functions_or_params(self):

        """
        Calculate the scale-up functions from the scale-up data attribute and populate to
        a dictionary with keys of the scenarios to be run.

        """

        # For each scenario to be run
        for scenario in self.gui_inputs['scenarios_to_run']:

            # Dictionary of whether interventions are applied or not
            self.intervention_applied[scenario] = {}

            # Need dictionary to track whether each parameter is time variant
            whether_time_variant = {}

            # Initialise the scaleup function dictionary
            self.scaleup_fns[scenario] = {}

            # Define scale-up functions from these datasets
            for param in self.scaleup_data[scenario]:

                # Determine whether the parameter is time variant at the first scenario iteration,
                # because otherwise the code will keep trying to pop off the time variant string
                # from the same scaleup data dictionary.
                if param not in whether_time_variant:
                    whether_time_variant[param] = self.scaleup_data[scenario][param].pop('time_variant')

                # If time variant
                if whether_time_variant[param] == u'yes':

                    # Extract and remove the smoothness parameter from the dictionary
                    if 'smoothness' in self.scaleup_data[scenario][param]:
                        smoothness = self.scaleup_data[scenario][param].pop('smoothness')
                    else:
                        smoothness = self.gui_inputs['default_smoothness']

                    # If the parameter is being modified for the scenario being run
                    self.intervention_applied[scenario][param] = False
                    if 'scenario' in self.scaleup_data[scenario][param]:
                        scenario_for_function = [self.model_constants['scenario_full_time'],
                                    self.scaleup_data[scenario][param].pop('scenario')]
                        self.intervention_applied[scenario][param] = True
                    else:
                        scenario_for_function = None

                    # Upper bound depends on whether the parameter is a proportion
                    if 'prop' in param:
                        upper_bound = 1.
                    else:
                        upper_bound = None

                    # Calculate the scaling function
                    self.scaleup_fns[scenario][param] \
                        = scale_up_function(self.scaleup_data[scenario][param].keys(),
                                            self.scaleup_data[scenario][param].values(),
                                            self.gui_inputs['fitting_method'],
                                            smoothness,
                                            bound_low=0.,
                                            bound_up=upper_bound,
                                            intervention_end=scenario_for_function,
                                            intervention_start_date=self.model_constants['scenario_start_time'])

                    if scenario is not None:
                        freeze_time = self.freeze_times['scenario_' + str(scenario)]
                        if freeze_time < self.model_constants['current_time']:
                            self.scaleup_fns[scenario][param] = freeze_curve(self.scaleup_fns[scenario][param],
                                                                             freeze_time)

                # If no is selected in the time variant column
                elif whether_time_variant[param] == u'no':

                    # Get rid of smoothness, which isn't relevant
                    if 'smoothness' in self.scaleup_data[scenario][param]:
                        del self.scaleup_data[scenario][param]['smoothness']

                    # Set as a constant parameter
                    self.model_constants[param] \
                        = self.scaleup_data[scenario][param][max(self.scaleup_data[scenario][param])]

                    # Note that the 'demo_life_expectancy' parameter has to be given this name
                    # and base.py will then calculate population death rates automatically.

    def find_constant_extrapulmonary_proportion(self):

        """
        Calculate constant proportion progressing to extrapulmonary for models that are stratified by organ status,
        but are not time-variant by organ status.
        """

        self.model_constants['epi_prop_extrapul'] \
            = 1. \
              - self.model_constants['epi_prop_smearpos'] \
              - self.model_constants['epi_prop_smearneg']

    def set_fixed_infectious_proportion(self):

        """
        Find a multiplier for the proportion of all cases infectious for
        models unstructured by organ status.
        """

        self.model_constants['tb_multiplier_force'] \
            = self.model_constants['epi_prop_smearpos'] \
              + self.model_constants['epi_prop_smearneg'] * self.model_constants['tb_multiplier_force_smearneg']

    def add_missing_economics_for_ipt(self):

        """
        To avoid errors because no economic values are available for age-stratified IPT, use the unstratified values
        for each age group for which no value is provided.

        """

        for agegroup in self.agegroups:
            for param in ['_saturation', '_inflectioncost', '_unitcost', '_startupduration', '_startupcost']:
                if 'econ' + param + '_ipt' + agegroup not in self.model_constants:
                    self.model_constants['econ' + param + '_ipt' + agegroup] \
                        = self.model_constants['econ' + param + '_ipt']
                    limits, _ = tool_kit.interrogate_age_string(agegroup)
                    if limits[1] == float('Inf'):
                        self.add_comment_to_gui_window(
                                                    '"' + param[1:] + '" parameter unavailable for ' +
                                                    str(int(limits[0])) + ' and up ' +
                                                    'age-group, so default value used.\n')

                    else:
                        self.add_comment_to_gui_window(
                                                    '"' + param[1:] + '" parameter unavailable for ' +
                                                    str(int(limits[0])) + ' to ' + str(int(limits[1])) +
                                                    ' age-group, so default value used.\n')

    def find_uncertainty_distributions(self):

        """
        Populate a dictionary of uncertainty parameters from the inputs dictionary in a format that matches
        Romain's code for uncertainty.

        """

        for param in self.model_constants:
            if '_uncertainty' in param and type(self.model_constants[param]) == dict:
                self.param_ranges_unc += [{'key': param[:-12],
                                           'bounds': [self.model_constants[param]['lower'],
                                                      self.model_constants[param]['upper']],
                                           'distribution': 'uniform'}]

    def get_data_to_fit(self):

        """
        Extract the data to be used for model fitting. (Choices currently hard-coded above.)

        """

        # Decide whether calibration or uncertainty analysis is being run
        if self.mode == 'calibration':
            var_to_iterate = self.calib_outputs
        elif self.mode == 'uncertainty':
            var_to_iterate = self.outputs_unc

        # Work through vars to be used and populate into the data fitting dictionary
        for output in var_to_iterate:
            if output['key'] == 'incidence':
                self.data_to_fit['incidence'] = self.original_data['tb']['e_inc_100k']
                self.data_to_fit['incidence_low'] = self.original_data['tb']['e_inc_100k_lo']
                self.data_to_fit['incidence_high'] = self.original_data['tb']['e_inc_100k_hi']
            elif output['key'] == 'mortality':
                self.data_to_fit['mortality'] = self.original_data['tb']['e_mort_exc_tbhiv_100k']
                self.data_to_fit['mortality_low'] = self.original_data['tb']['e_mort_exc_tbhiv_100k_lo']
                self.data_to_fit['mortality_high'] = self.original_data['tb']['e_mort_exc_tbhiv_100k_hi']
            else:
                print 'Warning: Calibrated output %s is not directly available from the data' % output['key']

    def find_interventions_to_cost(self):

        """
        Work out which interventions should be costed, selecting from the ones that can be costed in
        self.potential_interventions_to_cost.
        """

        for intervention in self.potential_interventions_to_cost:
            if 'program_prop_' + intervention in self.time_variants and \
                    ('_age' not in intervention or len(self.agegroups) > 1):
                self.interventions_to_cost += [intervention]

    def checks(self):

        """
        Not much in here as yet. However, this function is intended to contain all the data consistency checks for
        data entry.
        """

        # Check that the time to start economics analyses from is earlier than the end time of the model run
        assert self.model_constants['econ_start_time'] \
               <= self.model_constants['scenario_end_time'], \
            'Period_end must be before the end of the model integration time'

        # Check that all entered times occur after the model start time
        for time in self.model_constants:
            if time[-5:] == '_time' and '_step_time' not in time:
                assert self.model_constants[time] >= self.model_constants['start_time'], \
                    '% is before model start time' % self.model_constants[time]

    def add_comment_to_gui_window(self, comment, target='console'):

        if self.js_gui:
            emit(target, {"message": comment})
            time.sleep(self.emit_delay)

            print "Emitting:", comment

        else:
            self.runtime_outputs.insert(END, comment + '\n')
            self.runtime_outputs.see(END)