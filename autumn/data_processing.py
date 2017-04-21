
import autumn.spreadsheet as spreadsheet
import copy
import numpy
import tool_kit
from curve import scale_up_function, freeze_curve
from Tkinter import *
import time
import eventlet
from flask_socketio import emit


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


def calculate_proportion_dict(data, indices, percent=False):

    """
    General method to calculate proportions from absolute values provided as dictionaries.

    Args:
        data: Dictionary containing the absolute values.
        indices: The keys of data from which proportions are to be calculated (generally a list of strings).
        percent: Boolean describing whether the method should return the output as a percent or proportion.
    Returns:
        proportions: A dictionary of the resulting proportions.
    """

    # Calculate multiplier for percentages if requested, otherwise leave as one
    if percent:
        multiplier = 1e2
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


def remove_specific_key(dictionary, key):

    """
    Remove a specific named key from a dictionary.

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
    Takes a dictionary and removes all of the elements for which the value is nan.

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
        self.available_strains = ['_ds', '_mdr', '_xdr']
        self.available_organs = ['_smearpos', '_smearneg', '_extrapul']
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
        self.outputs_unc = [{'key': 'incidence', 'posterior_width': None, 'width_multiplier': 2.}]
        self.freeze_times = {}
        self.treatment_outcome_types = []
        self.relevant_interventions = {}
        self.include_relapse_in_ds_outcomes = True
        self.interventions_to_cost = []
        self.emit_delay = 0.1
        self.plot_count = 0
        self.js_gui = js_gui
        if self.js_gui:
            eventlet.monkey_patch()
        self.intervention_startdates = {}

    ######################
    ### Master methods ###
    ######################

    def read_and_load_data(self):

        """
        Master method of this object, calling all sub-methods to read and process data and define model structure.
        """

        # Read all required data
        keys_of_sheets_to_read = self.find_keys_of_sheets_to_read()
        self.add_comment_to_gui_window('Reading Excel sheets with input data.\n')
        self.original_data = spreadsheet.read_input_data_xls(self.from_test, keys_of_sheets_to_read, self.country)

        # Process constant parameters
        self.process_model_constants()

        # Process time-variant parameters
        self.process_time_variants()

        # Define model structure
        self.define_model_structure()

        # Find parameters that require processing
        self.find_additional_parameters()

        # Find which interventions need to be costed
        self.find_interventions_to_cost()

        # Prepare for uncertainty analysis
        self.process_uncertainty_parameters()

        # Optimisation-related methods
        self.find_intervention_startdates()

        # Perform checks
        self.checks()

    def process_model_constants(self):

        """
        Master method to call methods for processing constant model parameters.
        """

        self.add_model_constant_defaults(['diabetes', 'country_constants', 'default_constants'])
        self.add_universal_parameters()

    def process_time_variants(self):

        """
        Master method to perform all preparation and processing tasks for time-variant parameters.
        Does not actually fit functions, which is done later.
        Note that the order of call is important and can lead to errors if changed.
        """

        # Run first to remove from time-variants before they are processed
        self.extract_freeze_times()
        self.find_organ_proportions()
        if 'country_programs' in self.original_data:  # start with country programs
            self.time_variants.update(self.original_data['country_programs'])
        self.add_time_variant_defaults()  # add any necessary time-variants from defaults if not in country programs
        self.load_vacc_detect_time_variants()
        self.convert_percentages_to_proportions()
        self.find_ds_outcomes()
        self.add_treatment_outcomes()
        if self.gui_inputs['n_strains'] > 1:
            self.duplicate_ds_outcomes_for_multistrain()
        self.add_resistant_strain_outcomes()
        self.add_demo_dictionaries_to_timevariants()
        if self.time_variants['epi_prop_smearpos']['load_data'] == u'yes':
            self.add_organ_status_to_timevariants()
        self.complete_freeze_time_dictionary()
        self.tidy_time_variants()

    def define_model_structure(self):

        """
        Master method to define all aspects of model structure.
        """

        self.define_age_structure()
        self.define_riskgroup_structure()
        self.define_strain_structure()
        self.define_organ_structure()

    def find_additional_parameters(self):

        # Find the time non-infectious on treatment from the total time on treatment and the time infectious
        self.find_noninfectious_period()

        # Find risk group-specific parameters
        if len(self.riskgroups) > 1:
            self.find_riskgroup_progressions()

        # Calculate rates of progression to active disease or late latency
        self.find_progression_rates_from_params()

        # Derive some basic parameters for IPT
        self.find_ipt_params()

        # Work out which programs are relevant
        self.list_irrelevant_time_variants()
        self.find_relevant_programs()

        # Extract data into structures for creating time-variant parameters or constant ones
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

    def find_interventions_to_cost(self):

        """
        Work out which interventions should be costed, selecting from the ones that can be costed in
        self.potential_interventions_to_cost.
        """

        self.find_potential_interventions_to_cost()

        for intervention in self.potential_interventions_to_cost:
            if 'program_prop_' + intervention in self.relevant_interventions and \
                    ('_age' not in intervention or len(self.agegroups) > 1):
                self.interventions_to_cost += [intervention]

    def find_potential_interventions_to_cost(self):

        """
        Creates a list of the interventions that could potentially be costed if they are requested - that is, the ones
        for which model.py has popsize calculations coded.
        """

        self.potential_interventions_to_cost = ['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf',
                                                'ipt_age0to5', 'ipt_age5to15', 'decentralisation', 'improve_dst',
                                                'intensive_screening', 'ipt_age15up']
        if self.gui_inputs['n_strains'] > 1:
            self.potential_interventions_to_cost += ['shortcourse_mdr']
            self.potential_interventions_to_cost += ['food_voucher_ds']
            self.potential_interventions_to_cost += ['food_voucher_mdr']
        if self.gui_inputs['is_lowquality']:
            self.potential_interventions_to_cost += ['engage_lowquality']
        if self.gui_inputs['riskgroup_prison']:
            self.potential_interventions_to_cost += ['xpertacf_prison', 'cxrxpertacf_prison']
        if self.gui_inputs['riskgroup_indigenous']:
            self.potential_interventions_to_cost += ['xpertacf_indigenous']
        if self.gui_inputs['riskgroup_urbanpoor']:
            self.potential_interventions_to_cost += ['xpertacf_urbanpoor', 'cxrxpertacf_urbanpoor']
        if self.gui_inputs['riskgroup_ruralpoor']:
            self.potential_interventions_to_cost += ['xpertacf_ruralpoor', 'cxrxpertacf_ruralpoor']

    def process_uncertainty_parameters(self):

        # Specify the parameters to be used for uncertainty
        if self.gui_inputs['output_uncertainty']:
            self.find_uncertainty_distributions()
            self.get_data_to_fit()

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

    ##########################
    ### Processing methods ###
    ##########################

    def find_keys_of_sheets_to_read(self):

        """
        Find keys of spreadsheets to read.
        """

        keys_of_sheets_to_read = ['bcg', 'rate_birth', 'life_expectancy', 'default_parameters', 'tb', 'notifications',
                                  'outcomes', 'country_constants', 'default_constants', 'country_programs',
                                  'default_programs']

        # Add any optional sheets required for specific model being run
        if 'riskgroup_diabetes' in self.gui_inputs:
            keys_of_sheets_to_read += ['diabetes']

        return keys_of_sheets_to_read

    def add_model_constant_defaults(self, other_sheets_with_constants):

        """
        Populate model_constants with data from control panel, country sheet or default sheet hierarchically
        - such that the control panel is read in preference to the country data in preference to the default back-ups.

        Args:
            other_sheets_with_constants: The sheets of original_data which contain model constants
        """

        # Populate from country_constants if available and default_constants if not
        for other_sheet in other_sheets_with_constants:
            if other_sheet in self.original_data:

                # Only add the item if it hasn't been added yet
                for item in self.original_data[other_sheet]:
                    if item not in self.model_constants:
                        self.model_constants[item] = self.original_data[other_sheet][item]

    def add_universal_parameters(self):

        """
        Sets parameters that should never be changed in any situation, i.e. "by definition" parameters (although note
        that the infectiousness of the single infectious compartment for models unstratified by organ status is now set
        in set_fixed_infectious_proportion, because it's dependent upon loading parameters in find_functions_or_params).
        """

        # Proportion progressing to the only infectious compartment for models unstratified by organ status
        if self.gui_inputs['n_organs'] < 2:
            self.model_constants['epi_prop'] = 1.

        # Infectiousness of smear-positive and extrapulmonary patients
        else:
            self.model_constants['tb_multiplier_force_smearpos'] = 1.
            self.model_constants['tb_multiplier_force_extrapul'] = 0.

    def extract_freeze_times(self):

        """
        Extract the freeze_times for each scenario, if specified. If not specified, will be populated in
        self.complete_freeze_time_dictionary below.
        """

        if 'country_programs' in self.original_data and 'freeze_times' in self.original_data['country_programs']:
            self.freeze_times.update(self.original_data['country_programs'].pop('freeze_times'))

    def find_organ_proportions(self):

        """
        Calculates dictionaries with proportion of cases progressing to each organ status by year, and adds these to
        the derived_data attribute of the object.
        """

        self.derived_data.update(calculate_proportion_dict(self.original_data['notifications'],
                                                           ['new_sp', 'new_sn', 'new_ep']))

    def add_time_variant_defaults(self):

        """
        Populates time-variant parameters with defaults if those values aren't found in the manually entered
        country-specific data.
        """

        for program_var in self.original_data['default_programs']:

            # If the key isn't in available for the country at all
            if program_var not in self.time_variants:
                self.time_variants[program_var] = self.original_data['default_programs'][program_var]

            # Otherwise if it's there and load_data is requested in the country sheet, populate for the missing years
            else:
                for year in self.original_data['default_programs'][program_var]:
                    if year not in self.time_variants[program_var] \
                            and 'load_data' in self.original_data['country_programs'][program_var] \
                            and self.original_data['country_programs'][program_var]['load_data'] == u'yes':
                        self.time_variants[program_var][year] \
                            = self.original_data['default_programs'][program_var][year]

    def load_vacc_detect_time_variants(self):

        """
        Adds vaccination and case detection time-variants to the manually entered data loaded from the spreadsheets.
        Note that the manual inputs over-ride the loaded data if both are present.
        """

        # Vaccination
        if self.time_variants['program_perc_vaccination']['load_data'] == u'yes':
            for year in self.original_data['bcg']:
                if year not in self.time_variants['program_perc_vaccination']:
                    self.time_variants['program_perc_vaccination'][year] = self.original_data['bcg'][year]

        # Case detection
        if self.time_variants['program_perc_detect']['load_data'] == u'yes':
            for year in self.original_data['tb']['c_cdr']:
                if year not in self.time_variants['program_perc_detect']:
                    self.time_variants['program_perc_detect'][year] = self.original_data['tb']['c_cdr'][year]

    def convert_percentages_to_proportions(self):

        """
        Converts time-variant dictionaries to proportions if they are loaded as percentages in their raw form.
        """

        for time_variant in self.time_variants.keys():
            if 'perc_' in time_variant:  # if a percentage
                perc_name = time_variant.replace('perc', 'prop')
                self.time_variants[perc_name] = {}
                for year in self.time_variants[time_variant]:
                    if type(year) == int or 'scenario' in year:  # to exclude load_data, smoothness, etc.
                        self.time_variants[perc_name][year] \
                            = self.time_variants[time_variant][year] / 1e2
                    else:
                        self.time_variants[perc_name][year] \
                            = self.time_variants[time_variant][year]

    def find_ds_outcomes(self):

        """
        Calculates proportions of patients with each reported outcome for DS-TB, then sums cure and completion to obtain
        treatment success proportion. Note that the outcomes are reported differently for resistant strains, so code
        differs for them.
        """

        # Adjusting the original data to add a success number for smear-positive (so technically not still "original")
        self.original_data['outcomes']['new_sp_succ'] \
            = tool_kit.increment_dictionary_with_dictionary(self.original_data['outcomes']['new_sp_cmplt'],
                                                            self.original_data['outcomes']['new_sp_cur'])

        # Similarly, move completion over to represent success for smear-negative, extrapulmonary and retreatment
        for treatment_type in ['new_snep', 'ret']:
            self.original_data['outcomes'][treatment_type + '_succ'] \
                = self.original_data['outcomes'][treatment_type + '_cmplt']

        # And (effectively) rename the outcomes for the years that are pooled
        self.original_data['outcomes']['newrel_def'] = self.original_data['outcomes']['newrel_lost']

        # Sum over smear-positive, smear-negative, extrapulmonary and (if required) retreatment
        for outcome in ['succ', 'def', 'died', 'fail']:
            self.derived_data[outcome] \
                = tool_kit.increment_dictionary_with_dictionary(self.original_data['outcomes']['new_sp_' + outcome],
                                                                self.original_data['outcomes']['new_snep_' + outcome])
            if self.include_relapse_in_ds_outcomes:
                self.derived_data[outcome] \
                    = tool_kit.increment_dictionary_with_dictionary(self.derived_data[outcome],
                                                                    self.original_data['outcomes']['ret_' + outcome])

            # Update with newer pooled outcomes
            self.derived_data[outcome].update(self.original_data['outcomes']['newrel_' + outcome])

        # Calculate default rates from 'def' and 'fail' reported outcomes
        self.derived_data['default'] \
            = tool_kit.increment_dictionary_with_dictionary(self.derived_data['def'], self.derived_data['fail'])

        # Calculate the proportions for use in creating the treatment scale-up functions
        self.derived_data.update(calculate_proportion_dict(self.derived_data,
                                                           ['succ', 'died', 'default'], percent=False))

    def add_treatment_outcomes(self):

        """
        Add treatment outcomes for DS-TB to the time variants attribute.
        Use the same approach as above to adding if requested and data not manually entered.
        """

        name_conversion_dict = {'_success': 'succ', '_death': 'died'}
        for outcome in ['_success', '_death']:  # only for success and death because default is derived from these
            if self.time_variants['program_prop_treatment' + outcome]['load_data'] == u'yes':
                for year in self.derived_data['prop_' + name_conversion_dict[outcome]]:
                    if year not in self.time_variants['program_prop_treatment' + outcome]:
                        self.time_variants['program_prop_treatment' + outcome][year] \
                            = self.derived_data['prop_' + name_conversion_dict[outcome]][year]

    def duplicate_ds_outcomes_for_multistrain(self):

        """
        Duplicates the treatment outcomes with DS-TB key if it is a multi-strain model.
        """

        for outcome in ['_success', '_death']:
            self.time_variants['program_prop_treatment' + outcome + '_ds'] \
                = copy.copy(self.time_variants['program_prop_treatment' + outcome])

    def add_resistant_strain_outcomes(self):

        """
        Finds treatment outcomes for the resistant strains (i.e. MDR and XDR-TB).
        As for DS-TB, no need to find default proportion, as it is equal to one minus success minus death.
        Inappropriate outcomes are currently set to those for XDR-TB - intended to be temporary.
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

    def add_demo_dictionaries_to_timevariants(self):

        """
        Add epidemiological time variant parameters to time_variants.
        Similarly to previous methods, only performed if requested and only populated where absent.
        """

        for demo_parameter in ['life_expectancy', 'rate_birth']:
            if self.time_variants['demo_' + demo_parameter]['load_data'] == u'yes':
                for year in self.original_data[demo_parameter]:
                    if year not in self.time_variants['demo_' + demo_parameter]:
                        self.time_variants['demo_' + demo_parameter][year] = self.original_data[demo_parameter][year]

    def add_organ_status_to_timevariants(self):

        """
        Populate organ status dictionaries where requested and not already loaded.
        """

        name_conversion_dict = {'_smearpos': '_sp', '_smearneg': '_sn'}
        for outcome in ['_smearpos', '_smearneg']:
            for year in self.derived_data['prop_new' + name_conversion_dict[outcome]]:
                if year not in self.time_variants['epi_prop' + outcome]:
                    self.time_variants['epi_prop' + outcome][year] \
                        = self.derived_data['prop_new' + name_conversion_dict[outcome]][year]

    def complete_freeze_time_dictionary(self):

        """
        Ensure all scenarios have an entry in the self.freeze_times dictionary, if a time hadn't been specified in
        self.extract_freeze_times above.
        """

        for scenario in self.gui_inputs['scenarios_to_run']:
            if scenario is None:  # baseline
                self.freeze_times['baseline'] = self.model_constants['recent_time']
            elif 'scenario_' + str(scenario) not in self.freeze_times:  # scenarios with no freeze time specified
                self.freeze_times['scenario_' + str(scenario)] = self.model_constants['recent_time']

    def tidy_time_variants(self):

        """
        Final tidying of time-variants.
        """

        for program in self.time_variants:

            # Add zero at starting time for model run to all program proportions
            if 'program_prop' in program: self.time_variants[program][int(self.model_constants['start_time'])] = 0.

            # Remove the load_data keys, as they have been used and are now redundant
            self.time_variants[program] = remove_specific_key(self.time_variants[program], 'load_data')

            # Remove keys for which values are nan
            self.time_variants[program] = remove_nans(self.time_variants[program])

    def define_age_structure(self):

        """
        Define the model's age structure based on the breakpoints provided in spreadsheets.
        """

        # Describe and work out age stratification structure for model from the list of age breakpoints
        self.agegroups, _ = tool_kit.get_agegroups_from_breakpoints(self.model_constants['age_breakpoints'])

        # Find ageing rates and age-weighted parameters
        if len(self.agegroups) > 1:
            self.find_ageing_rates()
            self.find_fixed_age_specific_parameters()

    def define_riskgroup_structure(self):

        """
        Work out the risk group stratification.
        """

        # Create list of risk group names
        self.riskgroups = []
        for time_variant in self.time_variants:
            if 'riskgroup_prop_' in time_variant and self.gui_inputs['riskgroup' + time_variant[14:]]:
                self.riskgroups += [time_variant[14:]]

        # Add the null group
        if len(self.riskgroups) == 0:
            self.riskgroups += ['']
        else:
            self.riskgroups += ['_norisk']

        # Ensure some starting proportion of births go to the risk group stratum if value not loaded earlier
        for riskgroup in self.riskgroups:
            if 'riskgroup_prop' + riskgroup not in self.model_constants:
                self.model_constants['riskgroup_prop' + riskgroup] = 0.

    def define_strain_structure(self):

        """
        Finds the strains to be present in the model from a list of available strains and the integer value for the
        number of strains selected.
        """

        # Need a list of an empty string to be iterable for methods iterating by strain
        if self.gui_inputs['n_strains'] == 0:
            self.find_single_strain_timeperiods()
            self.strains = ['']
        else:
            self.strains = self.available_strains[:self.gui_inputs['n_strains']]
            if self.gui_inputs['is_amplification']:
                self.find_amplification_data()
            self.treatment_outcome_types = copy.copy(self.strains)
            if self.gui_inputs['is_misassignment']:
                for strain in self.strains[1:]:
                    for treated_as in self.strains:  # for each strain
                        if treated_as != strain:  # misassigned strain has to be different from the actual strain
                            if self.strains.index(treated_as) < self.strains.index(
                                    strain):  # if treated with weaker regimen
                                self.treatment_outcome_types += [strain + '_as' + treated_as[1:]]

    def define_organ_structure(self):

        """
        Defines the organ status stratification from the number of statuses selected
        Note that "organ" is the simplest single-word term that I can currently think of to
        describe whether patients have smear-positive, smear-negative or extrapulmonary disease.
        """

        if self.gui_inputs['n_organs'] == 0:
            self.organ_status = ['']
        else:
            self.organ_status = self.available_organs[:self.gui_inputs['n_organs']]

        # Work through whether organ status should be time variant
        self.find_organ_time_variation()

    def find_noninfectious_period(self):

        """
        Work out the periods of time spent non-infectious for each strain (plus inappropriate if required).
        """

        for strain in self.strains:
            self.model_constants['tb_timeperiod_noninfect_ontreatment' + strain] \
                = self.model_constants['tb_timeperiod_ontreatment' + strain] \
                  - self.model_constants['tb_timeperiod_infect_ontreatment' + strain]

    def find_riskgroup_progressions(self):

        """
        Code to adjust the progression rates to active disease for various risk groups - so far diabetes and HIV.
        """

        # Initialise dictionary of additional adjusted parameters to avoid dictionary changing size during iterations
        risk_adjusted_parameters = {}
        for riskgroup in self.riskgroups:
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
                    if riskgroup == '_diabetes' and '_progression' in param \
                            and age_limits[0] >= self.model_constants['riskgroup_startage' + riskgroup]:
                        whether_to_adjust = True

                    # HIV applies to all age groups, but only late progression
                    elif riskgroup == '_hiv' and '_late_progression' in param:
                        whether_to_adjust = True

                    # Shouldn't apply this to the multiplier parameters or non-TB-specific parameters
                    if '_multiplier' in param or 'tb_' not in param:
                        whether_to_adjust = False

                    # Now adjust the age-stratified parameter values
                    if whether_to_adjust:
                        risk_adjusted_parameters[param_without_age + riskgroup + age_string] \
                            = self.model_constants[param] \
                              * self.model_constants['riskgroup_multiplier' + riskgroup + '_progression']
                    elif '_progression' in param:
                        risk_adjusted_parameters[param_without_age + riskgroup + age_string] \
                            = self.model_constants[param]

                # Parameters not stratified by age
                else:

                    # Explanation as above
                    if riskgroup == '_diabetes' and '_progression' in param:
                        whether_to_adjust = True
                    elif riskgroup == '_hiv' and '_late_progression' in param:
                        whether_to_adjust = True
                    if '_multiplier' in param or 'tb_' not in param:
                        whether_to_adjust = False

                    # Adjustment as above, except age string not included
                    if whether_to_adjust:
                        risk_adjusted_parameters[param + riskgroup] \
                            = self.model_constants[param] \
                              * self.model_constants['riskgroup_multiplier' + riskgroup + '_progression']
                    elif '_progression' in param:
                        risk_adjusted_parameters[param + riskgroup] \
                            = self.model_constants[param]

        self.model_constants.update(risk_adjusted_parameters)

    def find_progression_rates_from_params(self):

        """
        Find early progression rates by age group and by risk group status - i.e. early progression to active TB and
        stabilisation into late latency.
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:

                # Early progression rate is early progression proportion divided by early time period
                self.model_constants['tb_rate_early_progression' + riskgroup + agegroup] \
                    = self.model_constants['tb_prop_early_progression' + riskgroup + agegroup] \
                      / self.model_constants['tb_timeperiod_early_latent']

                # Stabilisation rate is one minus early progression proportion divided by early time period
                self.model_constants['tb_rate_stabilise' + riskgroup + agegroup] \
                    = (1. - self.model_constants['tb_prop_early_progression' + riskgroup + agegroup]) \
                      / self.model_constants['tb_timeperiod_early_latent']

    def find_ipt_params(self):

        """
        Calculate number of persons eligible for IPT per person commencing treatment and then the number of persons
        who receive effective IPT per person assessed for LTBI.
        """

        self.model_constants['ipt_eligible_per_treatment_start'] \
            = (self.model_constants['demo_household_size'] - 1.) * self.model_constants['tb_prop_contacts_infected']

        for ipt_type in ['', 'novel_']:
            self.model_constants[ipt_type + 'ipt_effective_per_assessment'] \
                = self.model_constants['tb_prop_ltbi_test_sensitivity'] \
                  * self.model_constants['tb_prop_' + ipt_type + 'ipt_effectiveness']

    def find_functions_or_params(self):

        """
        Calculate the scale-up functions from the scale-up data attribute and populate to a dictionary with keys of the
        scenarios to be run.
        Note that the 'demo_life_expectancy' parameter has to be given this name and base.py will then calculate
        population death rates automatically.
        """

        for scenario in self.gui_inputs['scenarios_to_run']:

            # Dictionary of whether interventions are applied or not
            self.intervention_applied[scenario] = {}

            # Initialise the scale-up function dictionary
            self.scaleup_fns[scenario] = {}

            # Define scale-up functions from these datasets
            for param in self.scaleup_data[scenario]:

                # Determine whether the parameter is time variant at the first scenario iteration,
                # because otherwise the code will keep trying to pop off the time variant string
                # from the same scaleup data dictionary.
                whether_time_variant = self.scaleup_data[scenario][param].pop('time_variant') == u'yes'

                # If time variant
                if whether_time_variant:

                    # Extract and remove the smoothness parameter from the dictionary
                    if 'smoothness' in self.scaleup_data[scenario][param]:
                        smoothness = self.scaleup_data[scenario][param].pop('smoothness')
                    else:
                        smoothness = self.gui_inputs['default_smoothness']

                    # If the parameter is being modified for the scenario being run
                    self.intervention_applied[scenario][param] = False
                    scenario_for_function = None
                    if 'scenario' in self.scaleup_data[scenario][param]:
                        scenario_for_function = [self.model_constants['scenario_full_time'],
                                    self.scaleup_data[scenario][param].pop('scenario')]
                        self.intervention_applied[scenario][param] = True

                    # Upper bound depends on whether the parameter is a proportion
                    upper_bound = None
                    if 'prop_' in param: upper_bound = 1.

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
                        if freeze_time < self.model_constants['recent_time']:
                            self.scaleup_fns[scenario][param] = freeze_curve(self.scaleup_fns[scenario][param],
                                                                             freeze_time)

                # If no is selected in the time variant column
                elif whether_time_variant:

                    # Smoothness no longer relevant
                    if 'smoothness' in self.scaleup_data[scenario][param]:
                        del self.scaleup_data[scenario][param]['smoothness']

                    # Set as a constant parameter
                    self.model_constants[param] \
                        = self.scaleup_data[scenario][param][max(self.scaleup_data[scenario][param])]

    def find_constant_extrapulmonary_proportion(self):

        """
        Calculate constant proportion progressing to extrapulmonary for models that are stratified but not time-variant
        by organ status.
        """

        self.model_constants['epi_prop_extrapul'] \
            = 1. - self.model_constants['epi_prop_smearpos'] - self.model_constants['epi_prop_smearneg']

    def set_fixed_infectious_proportion(self):

        """
        Find a multiplier for the proportion of all cases infectious for models unstructured by organ status.
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
        Populate a dictionary of uncertainty parameters from the inputs dictionary in a format that matches code for
        uncertainty.
        """

        for param in self.model_constants:
            if '_uncertainty' in param and type(self.model_constants[param]) == dict:
                self.param_ranges_unc += [{'key': param[:-12],
                                           'bounds': [self.model_constants[param]['lower'],
                                                      self.model_constants[param]['upper']],
                                           'distribution': 'uniform'}]

    def get_data_to_fit(self):

        """
        Extract data for model fitting. Choices currently hard-coded above.
        """

        # Decide whether calibration or uncertainty analysis is being run
        if self.mode == 'calibration':
            var_to_iterate = self.calib_outputs
        elif self.mode == 'uncertainty':
            var_to_iterate = self.outputs_unc

        inc_conversion_dict = {'incidence': 'e_inc_100k',
                               'incidence_low': 'e_inc_100k_lo',
                               'incidence_high': 'e_inc_100k_hi'}
        mort_conversion_dict = {'mortality': 'e_mort_exc_tbhiv_100k',
                                'mortality_low': 'e_mort_exc_tbhiv_100k_lo',
                                'mortality_high': 'e_mort_exc_tbhiv_100k_hi'}

        # Work through vars to be used and populate into the data fitting dictionary
        for output in var_to_iterate:
            if output['key'] == 'incidence':
                for key in inc_conversion_dict:
                    self.data_to_fit[key] = self.original_data['tb'][inc_conversion_dict[key]]
            elif output['key'] == 'mortality':
                for key in mort_conversion_dict:
                    self.data_to_fit[key] = self.original_data['tb'][mort_conversion_dict[key]]
            else:
                self.add_comment_to_gui_window(
                    'Warning: Calibrated output %s is not directly available from the data' % output['key'])

    def find_intervention_startdates(self):

        """
        Find the dates when the different interventions start and populate self.intervention_startdates
        """

        for scenario in self.gui_inputs['scenarios_to_run']:
            self.intervention_startdates[scenario] = {}
            for intervention in self.interventions_to_cost:
                self.intervention_startdates[scenario][intervention] = None
                years_pos_coverage \
                    = [key for (key, value) in
                       self.scaleup_data[scenario]['program_prop_' + intervention].items()
                       if value > 0.]  # Years from start
                if len(years_pos_coverage) > 0:  # i.e. some coverage present from start
                    self.intervention_startdates[scenario][intervention] = min(years_pos_coverage)

    ###########################
    ### Second tier methods ###
    ###########################

    def find_ageing_rates(self):

        """
        Calculate ageing rates as the reciprocal of the width of the age bracket.
        """

        for agegroup in self.agegroups:
            age_limits, _ = tool_kit.interrogate_age_string(agegroup)
            if 'up' not in agegroup:
                self.model_constants['ageing_rate' + agegroup] = 1. / (age_limits[1] - age_limits[0])

    def find_fixed_age_specific_parameters(self):

        """
        Find weighted age specific parameters using age weighting code from took_kit.
        """

        # Extract age breakpoints in appropriate form for module
        model_breakpoints = []
        for i in self.model_constants['age_breakpoints']:
            model_breakpoints += [float(i)]

        for param in ['early_progression_age', 'late_progression_age', 'tb_multiplier_child_infectiousness_age']:

            # Extract age-stratified parameters in the appropriate form
            prog_param_vals = {}
            prog_age_dict = {}
            for constant in self.model_constants:
                if param in constant:
                    prog_param_string, prog_stem = tool_kit.find_string_from_starting_letters(constant, '_age')
                    prog_age_dict[prog_param_string], _ = tool_kit.interrogate_age_string(prog_param_string)
                    prog_param_vals[prog_param_string] = self.model_constants[constant]

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

    def find_single_strain_timeperiods(self):

        """
        If the model isn't stratified by strain, use DS-TB time-periods for the single strain.
        Note that the parameter for the time period infectious on treatment will only be defined for DS-TB in this case
        and not for no strain name.
        """

        for timeperiod in ['tb_timeperiod_infect_ontreatment', 'tb_timeperiod_ontreatment']:
            self.model_constants[timeperiod] = self.model_constants[timeperiod + '_ds']

    def find_amplification_data(self):

        """
        Add dictionary for the amplification proportion scale-up.
        """

        self.time_variants['epi_prop_amplification'] \
            = {self.model_constants['start_mdr_introduce_time']: 0.,
               self.model_constants['end_mdr_introduce_time']: self.model_constants['tb_prop_amplification'],
               'time_variant': u'yes'}

    def find_organ_time_variation(self):

        """
        Work through whether variation in organ status with time should be implemented, according to whether the model
        is stratified by organ and whether organ stratification is requested through the smear-positive time-variant
        input.
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

    def find_data_for_functions_or_params(self):

        """
        Method to load all the dictionaries to be used in generating scale-up functions to a single attribute of the
        class instance (to avoid creating heaps of functions for irrelevant programs).

        Creates: self.scaleup_data, a dictionary of the relevant scale-up data for creating scale-up functions in
            set_scaleup_functions within the model object. First tier of keys is the scenario to be run, next is the
            time variant parameter to be calculated.
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
                                self.scaleup_data[scenario][str(time_variant)][i] = self.time_variants[time_variant][i]
                        else:
                            self.scaleup_data[scenario][str(time_variant)][i] = self.time_variants[time_variant][i]

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
            if 'program_prop_shortcourse_mdr' in time_variant and self.gui_inputs['n_strains'] < 2:
                self.irrelevant_time_variants += [time_variant]
                print('program_prop_shortcourse_mdr requested, but not implemented during to insufficient strains')

    def find_relevant_programs(self):

        """
        Code to create lists of the programmatic interventions that are relevant to a particular scenario being run.

        Creates:
            self.relevant_interventions: A dict with keys scenarios and values lists of programs relevant to that scenario
        """

        for scenario in self.gui_inputs['scenarios_to_run']:
            self.relevant_interventions[scenario] = []
            for time_variant in self.time_variants:
                for key in self.time_variants[time_variant]:
                    if time_variant not in self.irrelevant_time_variants and 'program_' in time_variant \
                            and time_variant not in self.relevant_interventions[scenario]:
                        if type(key) == int and self.time_variants[time_variant][key] > 0.:
                            self.relevant_interventions[scenario] += [time_variant]
                        elif type(key) == str and key == tool_kit.find_scenario_string_from_number(scenario):
                            self.relevant_interventions[scenario] += [time_variant]

    ############################
    ### Miscellaneous method ###
    ############################

    def add_comment_to_gui_window(self, comment, target='console'):

        """
        Output message to either JavaScript or Tkinter GUI.
        """

        if self.js_gui:
            emit(target, {"message": comment})
            time.sleep(self.emit_delay)
            print "Emitting:", comment
        else:
            self.runtime_outputs.insert(END, comment + '\n')
            self.runtime_outputs.see(END)
