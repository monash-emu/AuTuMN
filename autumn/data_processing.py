
import autumn.spreadsheet as spreadsheet
import copy
import tool_kit
from curve import scale_up_function, freeze_curve
from Tkinter import *
import time
import eventlet
from flask_socketio import emit


def make_constant_function(value):
    """
    Function that returns a function of constant returned value with a deliberately irrelevant argument,
    to maintain consistency with the number of arguments to other functions that take time as an argument.

    Args:
        value: The value for the created function to return
        time: Irrelevant but necessary argument to the returned function
    Returns:
        constant: The constant function
    """

    def constant(time):
        return value
    return constant


def find_latest_value_from_year_dict(dictionary, ceiling):
    """
    Finds the value corresponding to the latest key that is not into the future.

    Args:
        dictionary: The dictionary to be interrogated
        ceiling: Upper limit of time to prevent entries into the future being included
    Returns:
        The value corresponding to the key found through this process
    """

    return dictionary[max([i for i in dictionary if i <= int(ceiling)])]


class Inputs:

    def __init__(self, gui_inputs, runtime_outputs, js_gui=False):

        # GUI inputs
        self.gui_inputs = gui_inputs
        self.country = gui_inputs['country']
        self.scenarios = self.gui_inputs['scenarios_to_run']

        # parameter structures
        self.original_data = None
        self.derived_data = {}
        self.time_variants = {}
        self.model_constants = {}
        self.scaleup_data = {}
        self.scaleup_fns = {}
        self.param_ranges_unc = []
        self.data_to_fit = {}
        # for incidence for ex, width of normal posterior relative to CI width in data
        self.outputs_unc = [{'key': 'incidence', 'posterior_width': None, 'width_multiplier': 2.}]

        # model structure
        self.available_strains = ['_ds', '_mdr', '_xdr']
        self.available_organs = ['_smearpos', '_smearneg', '_extrapul']
        self.agegroups = None
        self.vary_force_infection_by_riskgroup = self.gui_inputs['is_vary_force_infection_by_riskgroup']
        self.mixing = {}
        self.compartment_types \
            = ['susceptible_fully', 'susceptible_vac', 'susceptible_treated', 'latent_early', 'latent_late', 'active',
               'detect', 'missed', 'treatment_infect', 'treatment_noninfect']

        # interventions
        self.irrelevant_time_variants = []
        self.relevant_interventions = {}
        self.interventions_to_cost = {}
        self.intervention_startdates = {}
        self.potential_interventions_to_cost \
            = ['vaccination', 'xpert', 'treatment_support', 'smearacf', 'xpertacf', 'ipt_age0to5', 'ipt_age5to15',
               'decentralisation', 'improve_dst', 'bulgaria_improve_dst', 'intensive_screening', 'ipt_age15up']
        self.freeze_times = {}

        # miscellaneous
        self.runtime_outputs = runtime_outputs
        self.mode = 'uncertainty'
        self.js_gui = js_gui
        if self.js_gui:
            eventlet.monkey_patch()
        self.plot_count = 0
        self.emit_delay = .1
        self.treatment_outcome_types = []
        self.include_relapse_in_ds_outcomes = True

    #####################
    ### Master method ###
    #####################

    def read_and_load_data(self):
        """
        Master method of this object, calling all sub-methods to read and process data and define model structure.
        """

        # read all required data
        self.add_comment_to_gui_window('Reading Excel sheets with input data.\n')
        self.original_data = spreadsheet.read_input_data_xls(True, self.find_keys_of_sheets_to_read(), self.country)

        # process constant parameters
        self.process_model_constants()

        # process time-variant parameters
        self.process_time_variants()

        # define model structure
        self.define_model_structure()

        # find parameters that require processing
        self.find_additional_parameters()

        # classify interventions as to whether they apply and are to be costed
        self.classify_interventions()

        # calculate time-variant functions
        self.find_scaleup_functions()

        # create mixing matrix (has to be run after scale-up functions, so can't go in model structure method)
        if self.vary_force_infection_by_riskgroup: self.create_mixing_matrix()

        # define compartmental structure
        self.define_compartment_structure()

        # uncertainty-related analysis
        self.process_uncertainty_parameters()

        # optimisation-related methods
        self.find_intervention_startdates()  # currently sitting with intervention classification methods, though

        # perform checks (undeveloped still)
        self.checks()

    #############################################
    ### Constant parameter processing methods ###
    #############################################

    # populate with first round of unprocessed parameters
    def process_model_constants(self):
        """
        Master method to call methods for processing constant model parameters.
        """

        # note ordering to list of sheets to be worked through is important for hierarchical loading of constants
        sheets_with_constants = ['country_constants', 'default_constants']
        if self.gui_inputs['riskgroup_diabetes']: sheets_with_constants += ['diabetes']
        self.add_model_constant_defaults(sheets_with_constants)

        # add "by definition" hard-coded parameters
        self.add_universal_parameters()

    def add_model_constant_defaults(self, other_sheets_with_constants):
        """
        Populate model_constants with data from control panel, country sheet or default sheet hierarchically
        - such that the control panel is read in preference to the country data in preference to the default back-ups.

        Args:
            other_sheets_with_constants: The sheets of original_data which contain model constants
        """

        # populate hierarchically from the earliest sheet in the list as available
        for other_sheet in other_sheets_with_constants:
            if other_sheet in self.original_data:
                for item in self.original_data[other_sheet]:
                    if item not in self.model_constants:
                        self.model_constants[item] = self.original_data[other_sheet][item]

    def add_universal_parameters(self):
        """
        Sets parameters that should never be changed in any situation, i.e. "by definition" parameters (although note
        that the infectiousness of the single infectious compartment for models unstratified by organ status is now set
        in set_fixed_infectious_proportion, because it's dependent upon loading parameters in find_functions_or_params).
        """

        # proportion progressing to the only infectious compartment for models unstratified by organ status
        if self.gui_inputs['n_organs'] < 2:
            self.model_constants['epi_prop'] = 1.

        # infectiousness of smear-positive and extrapulmonary patients
        else:
            self.model_constants['tb_multiplier_force_smearpos'] = 1.
            self.model_constants['tb_multiplier_force_extrapul'] = 0.

    # derive further parameters
    def find_additional_parameters(self):
        """
        Find additional parameters.
        Includes methods that require the model structure to be defined,
        so that this can't be run with process_model_constants.
        """

        # find risk group-specific parameters
        if len(self.riskgroups) > 1: self.find_riskgroup_progressions()

        # calculate rates of progression to active disease or late latency
        self.find_latency_progression_rates()

        # find the time non-infectious on treatment from the total time on treatment and the time infectious
        self.find_noninfectious_period()

        # derive some basic parameters for IPT
        self.find_ipt_params()

    def find_latency_progression_rates(self):
        """
        Find early progression rates by age group and by risk group status - i.e. early progression to active TB and
        stabilisation into late latency.
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:

                prop_early = self.model_constants['tb_prop_early_progression' + riskgroup + agegroup]
                time_early = self.model_constants['tb_timeperiod_early_latent']

                # early progression rate is early progression proportion divided by early time period
                self.model_constants['tb_rate_early_progression' + riskgroup + agegroup] = prop_early / time_early

                # stabilisation rate is one minus early progression proportion divided by early time period
                self.model_constants['tb_rate_stabilise' + riskgroup + agegroup] = (1. - prop_early) / time_early

    def find_riskgroup_progressions(self):
        """
        Code to adjust the progression rates to active disease for various risk groups - so far diabetes and HIV.
        """

        # initialise dictionary of additional adjusted parameters to avoid dictionary changing size during iterations
        risk_adjusted_parameters = {}
        for riskgroup in self.riskgroups:
            for param in self.model_constants:

                # start from the assumption that parameter is not being adjusted
                whether_to_adjust = False

                # for age-stratified parameters
                if '_age' in param:

                    # find the age string, the lower and upper age limits and the parameter name without the age string
                    age_string, _ = tool_kit.find_string_from_starting_letters(param, '_age')
                    age_limits, _ = tool_kit.interrogate_age_string(age_string)
                    param_without_age = param[:-len(age_string)]

                    # diabetes progression rates only start from age groups with lower limit above the start age
                    # and apply to both early and late progression.
                    if riskgroup == '_diabetes' and '_progression' in param \
                            and age_limits[0] >= self.model_constants['riskgroup_startage' + riskgroup]:
                        whether_to_adjust = True

                    # HIV applies to all age groups, but only late progression
                    elif riskgroup == '_hiv' and '_late_progression' in param:
                        whether_to_adjust = True

                    # shouldn't apply this to the multiplier parameters or non-TB-specific parameters
                    if '_multiplier' in param or 'tb_' not in param:
                        whether_to_adjust = False

                    # now adjust the age-stratified parameter values
                    if whether_to_adjust:
                        risk_adjusted_parameters[param_without_age + riskgroup + age_string] \
                            = self.model_constants[param] \
                              * self.model_constants['riskgroup_multiplier' + riskgroup + '_progression']
                    elif '_progression' in param:
                        risk_adjusted_parameters[param_without_age + riskgroup + age_string] \
                            = self.model_constants[param]

                # parameters not stratified by age
                else:

                    # explanation as above
                    if riskgroup == '_diabetes' and '_progression' in param:
                        whether_to_adjust = True
                    elif riskgroup == '_hiv' and '_late_progression' in param:
                        whether_to_adjust = True
                    if '_multiplier' in param or 'tb_' not in param:
                        whether_to_adjust = False

                    # adjustment as above, except age string not included
                    if whether_to_adjust:
                        risk_adjusted_parameters[param + riskgroup] \
                            = self.model_constants[param] \
                              * self.model_constants['riskgroup_multiplier' + riskgroup + '_progression']
                    elif '_progression' in param:
                        risk_adjusted_parameters[param + riskgroup] \
                            = self.model_constants[param]

        self.model_constants.update(risk_adjusted_parameters)

    def find_noninfectious_period(self):
        """
        Work out the periods of time spent non-infectious for each strain (plus inappropriate if required).
        """

        for strain in self.strains:
            self.model_constants['tb_timeperiod_noninfect_ontreatment' + strain] \
                = self.model_constants['tb_timeperiod_ontreatment' + strain] \
                  - self.model_constants['tb_timeperiod_infect_ontreatment' + strain]

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

    #########################################
    ### Methods to define model structure ###
    #########################################

    def define_model_structure(self):
        """
        Master method to define all aspects of model structure.
        """

        self.define_age_structure()
        self.define_riskgroup_structure()
        self.define_strain_structure()
        self.define_organ_structure()

    def define_age_structure(self):
        """
        Define the model's age structure based on the breakpoints provided in spreadsheets.
        """

        # describe and work out age stratification structure for model from the list of age breakpoints
        self.agegroups, _ = tool_kit.get_agegroups_from_breakpoints(self.model_constants['age_breakpoints'])

        # find ageing rates and age-weighted parameters
        if len(self.agegroups) > 1:
            self.find_ageing_rates()
            self.find_fixed_age_specific_parameters()

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
        Find weighted age specific parameters using age weighting code from tool_kit.
        """

        model_breakpoints = [float(i) for i in self.model_constants['age_breakpoints']]  # convert list of ints to float
        for param in ['early_progression_age', 'late_progression_age', 'tb_multiplier_child_infectiousness_age']:

            # extract age-stratified parameters in the appropriate form
            prog_param_vals = {}
            prog_age_dict = {}
            for constant in self.model_constants:
                if param in constant:
                    prog_param_string, prog_stem = tool_kit.find_string_from_starting_letters(constant, '_age')
                    prog_age_dict[prog_param_string], _ = tool_kit.interrogate_age_string(prog_param_string)
                    prog_param_vals[prog_param_string] = self.model_constants[constant]

            param_breakpoints = tool_kit.find_age_breakpoints_from_dicts(prog_age_dict)

            # find and set age-adjusted parameters
            prog_age_adjusted_params = \
                tool_kit.adapt_params_to_stratification(param_breakpoints,
                                                        model_breakpoints,
                                                        prog_param_vals,
                                                        parameter_name=param,
                                                        whether_to_plot=self.gui_inputs['output_age_calculations'])
            for agegroup in self.agegroups:
                self.model_constants[prog_stem + agegroup] = prog_age_adjusted_params[agegroup]

    def define_riskgroup_structure(self):
        """
        Work out the risk group stratification.
        """

        # create list of risk group names
        self.riskgroups = []
        for time_variant in self.time_variants:
            if 'riskgroup_prop_' in time_variant and self.gui_inputs['riskgroup' + time_variant[14:]]:
                self.riskgroups += [time_variant[14:]]

        # add the null group
        if len(self.riskgroups) == 0:
            self.riskgroups += ['']
        else:
            self.riskgroups += ['_norisk']

        # ensure some starting proportion of births go to the risk group stratum if value not loaded earlier
        for riskgroup in self.riskgroups:
            if 'riskgroup_prop' + riskgroup not in self.model_constants:
                self.model_constants['riskgroup_prop' + riskgroup] = 0.

    def define_strain_structure(self):
        """
        Finds the strains to be present in the model from a list of available strains and the integer value for the
        number of strains selected.
        """

        # unstratified by strain
        if self.gui_inputs['n_strains'] == 0:

            # if the model isn't stratified by strain, use DS-TB time-periods for the single strain
            for timeperiod in ['tb_timeperiod_infect_ontreatment', 'tb_timeperiod_ontreatment']:
                self.model_constants[timeperiod] = self.model_constants[timeperiod + '_ds']
            # need a list of an empty string to be iterable for methods iterating by strain
            self.strains = ['']

        # stratified
        else:
            self.strains = self.available_strains[:self.gui_inputs['n_strains']]
            if self.gui_inputs['is_amplification']:
                self.time_variants['epi_prop_amplification'] \
                    = {self.model_constants['start_mdr_introduce_time']: 0.,
                       self.model_constants['end_mdr_introduce_time']: self.model_constants['tb_prop_amplification']}
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
        Defines the organ status stratification from the number of statuses selected.
        Note that "organ" is the simplest single-word term that I can currently think of to describe whether patients
        have smear-positive, smear-negative or extrapulmonary disease.
        """

        if self.gui_inputs['n_organs'] == 0:
            self.organ_status = ['']
        else:
            self.organ_status = self.available_organs[:self.gui_inputs['n_organs']]

    def create_mixing_matrix(self):
        """
        Creates model attribute for mixing between population risk groups, for use in calculate_force_infection_vars
        method below only.
        """

        # create mixing matrix separately for each scenario, just in case risk groups being managed differently
        for scenario in self.scenarios:
            self.mixing[scenario] = {}

            # next tier of dictionary is the "to" risk group that is being infected
            for to_riskgroup in self.riskgroups:
                self.mixing[scenario][to_riskgroup] = {}

                # last tier of dictionary is the "from" risk group describing the make up of contacts
                for from_riskgroup in self.riskgroups:
                    if from_riskgroup != '_norisk':

                        # use parameters for risk groups other than "_norisk" if available
                        if 'prop_mix' + to_riskgroup + '_from' + from_riskgroup in self.model_constants:
                            self.mixing[scenario][to_riskgroup][from_riskgroup] \
                                = self.model_constants['prop_mix' + to_riskgroup + '_from' + from_riskgroup]

                        # otherwise use the latest value for the proportion of the population with that risk factor
                        else:
                            self.mixing[scenario][to_riskgroup][from_riskgroup] \
                                = find_latest_value_from_year_dict(
                                self.scaleup_data[scenario]['riskgroup_prop' + from_riskgroup],
                                self.model_constants['current_time'])

                # give the remainder to the "_norisk" group without any risk factors
                self.mixing[scenario][to_riskgroup][from_riskgroup] \
                    = 1. - sum(self.mixing[scenario][to_riskgroup].values())

    def define_compartment_structure(self):
        """
        Determines the compartment types required for model run,
        not including stratifications by age and risk groups, etc.
        """

        # add elaboration compartments to default list of mandatory compartments
        if self.gui_inputs['is_lowquality']: self.compartment_types += ['lowquality']
        if 'int_prop_novel_vaccination' in self.relevant_interventions:
            self.compartment_types += ['susceptible_novelvac']

    #################################################
    ### Time variant parameter processing methods ###
    #################################################

    def process_time_variants(self):
        """
        Master method to perform all preparation and processing tasks for time-variant parameters.
        Does not perform the fitting of functions to the data, which is done later in find_scaleup_functions.
        Note that the order of call is important and can lead to errors if changed.
        """

        self.extract_freeze_times()  # goes first to remove from time-variants before they are processed
        self.find_organ_proportions()
        if 'country_programs' in self.original_data: self.time_variants.update(self.original_data['country_programs'])
        self.add_time_variant_defaults()  # add any necessary time-variants from defaults if not in country programs
        self.load_vacc_detect_time_variants()
        self.convert_percentages_to_proportions()
        self.find_ds_outcomes()
        self.add_treatment_outcomes()
        if self.gui_inputs['n_strains'] > 1: self.duplicate_ds_outcomes_for_multistrain()
        self.add_resistant_strain_outcomes()
        self.add_demo_dictionaries_to_timevariants()
        if self.gui_inputs['is_timevariant_organs']: self.add_organ_status_to_timevariants()
        self.tidy_time_variants()
        self.adjust_param_for_reporting('program_prop_detect', 'Bulgaria', 0.95)  # Bulgaria thought over-estimated CDR

    def extract_freeze_times(self):
        """
        Extract the freeze_times for each scenario, if specified.
        """

        if 'country_programs' in self.original_data and 'freeze_times' in self.original_data['country_programs']:
            self.freeze_times.update(self.original_data['country_programs'].pop('freeze_times'))

    def find_organ_proportions(self):
        """
        Calculates dictionaries with proportion of cases progressing to each organ status by year, and adds these to
        the derived_data attribute of the object.
        """

        self.derived_data.update(tool_kit.calculate_proportion_dict(self.original_data['notifications'],
                                                                    ['new_sp', 'new_sn', 'new_ep']))

    def add_time_variant_defaults(self):
        """
        Populates time-variant parameters with defaults if those values aren't found in the manually entered
        country-specific data.
        """

        for program_var in self.original_data['default_programs']:

            # if the key isn't in available for the country
            if program_var not in self.time_variants:
                self.time_variants[program_var] = self.original_data['default_programs'][program_var]

            # otherwise if it's there and load_data is requested in the country sheet, populate for the missing years
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

        # vaccination
        if self.time_variants['int_perc_vaccination']['load_data'] == u'yes':
            for year in self.original_data['bcg']:
                if year not in self.time_variants['int_perc_vaccination']:
                    self.time_variants['int_perc_vaccination'][year] = self.original_data['bcg'][year]

        # case detection
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
                        self.time_variants[perc_name][year] = self.time_variants[time_variant][year] / 1e2
                    else:
                        self.time_variants[perc_name][year] = self.time_variants[time_variant][year]

    def find_ds_outcomes(self):
        """
        Calculates proportions of patients with each reported outcome for DS-TB, then sums cure and completion to obtain
        treatment success proportion. Note that the outcomes are reported differently for resistant strains, so code
        differs for them.
        """

        # adjusting the original data to add a success number for smear-positive (so technically not still "original")
        self.original_data['outcomes']['new_sp_succ'] \
            = tool_kit.increment_dictionary_with_dictionary(
            self.original_data['outcomes']['new_sp_cmplt'],
            self.original_data['outcomes']['new_sp_cur'])

        # similarly, move completion over to represent success for smear-negative, extrapulmonary and retreatment
        for treatment_type in ['new_snep', 'ret']:
            self.original_data['outcomes'][treatment_type + '_succ'] \
                = self.original_data['outcomes'][treatment_type + '_cmplt']

        # and (effectively) rename the outcomes for the years that are pooled
        self.original_data['outcomes']['newrel_def'] = self.original_data['outcomes']['newrel_lost']

        # sum over smear-positive, smear-negative, extrapulmonary and (if required) retreatment
        for outcome in ['succ', 'def', 'died', 'fail']:
            self.derived_data[outcome] \
                = tool_kit.increment_dictionary_with_dictionary(self.original_data['outcomes']['new_sp_' + outcome],
                                                                self.original_data['outcomes']['new_snep_' + outcome])
            if self.include_relapse_in_ds_outcomes:
                self.derived_data[outcome] \
                    = tool_kit.increment_dictionary_with_dictionary(self.derived_data[outcome],
                                                                    self.original_data['outcomes']['ret_' + outcome])

            # update with newer pooled outcomes
            self.derived_data[outcome].update(self.original_data['outcomes']['newrel_' + outcome])

        # calculate default rates from 'def' and 'fail' reported outcomes
        self.derived_data['default'] \
            = tool_kit.increment_dictionary_with_dictionary(self.derived_data['def'], self.derived_data['fail'])

        # calculate the proportions for use in creating the treatment scale-up functions
        self.derived_data.update(tool_kit.calculate_proportion_dict(self.derived_data,
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

        # calculate proportions of each outcome for MDR and XDR-TB from GTB
        for strain in ['mdr', 'xdr']:
            self.derived_data.update(tool_kit.calculate_proportion_dict(self.original_data['outcomes'],
                                                                        [strain + '_succ', strain + '_fail', strain
                                                                         + '_died', strain + '_lost'], percent=False))

            # populate MDR and XDR data from outcomes dictionary into time variants where requested and not entered
            if self.time_variants['program_prop_treatment_success_' + strain]['load_data'] == u'yes':
                for year in self.derived_data['prop_' + strain + '_succ']:
                    if year not in self.time_variants['program_prop_treatment_success_' + strain]:
                        self.time_variants['program_prop_treatment_success_' + strain][year] \
                            = self.derived_data['prop_' + strain + '_succ'][year]

        # temporarily assign the same treatment outcomes to XDR-TB as for inappropriate
        for outcome in ['_success', '_death']:
            self.time_variants['program_prop_treatment' + outcome + '_inappropriate'] \
                = copy.copy(self.time_variants['program_prop_treatment' + outcome + '_xdr'])

    def add_demo_dictionaries_to_timevariants(self):
        """
        Add epidemiological time variant parameters to time_variants.
        Similarly to previous methods, only performed if requested and only populated where absent.
        """

        # for the two types of demographic parameters
        for demo_parameter in ['life_expectancy', 'rate_birth']:

            # if there are data available from the user-derived sheets and loading external data is requested
            if 'demo_' + demo_parameter in self.time_variants \
                    and self.time_variants['demo_' + demo_parameter]['load_data'] == u'yes':
                for year in self.original_data[demo_parameter]:
                    if year not in self.time_variants['demo_' + demo_parameter]:
                        self.time_variants['demo_' + demo_parameter][year] = self.original_data[demo_parameter][year]

            # if there are no data available from the user sheets
            else:
                self.time_variants['demo_' + demo_parameter] = self.original_data[demo_parameter]

    def add_organ_status_to_timevariants(self):
        """
        Populate organ status dictionaries where requested and not already loaded.
        """

        # conversion from GTB code to AuTuMN code
        name_conversion_dict = {'_smearpos': '_sp', '_smearneg': '_sn'}

        # for the time variant progression parameters that are used (extrapulmonary just calculated as a complement)
        for organ in ['_smearpos', '_smearneg']:

            # populate absent values from derived data if input data available
            if 'epi_prop' + organ in self.time_variants:
                for year in self.derived_data['prop_new' + name_conversion_dict[organ]]:
                    if year not in self.time_variants['epi_prop' + organ]:
                        self.time_variants['epi_prop' + organ][year] \
                            = self.derived_data['prop_new' + name_conversion_dict[organ]][year]

            # otherwise if no input data available, just take the derived data straight from the loaded sheets
            else:
                self.time_variants['epi_prop' + organ] = self.derived_data['prop_new' + name_conversion_dict[organ]]

    def tidy_time_variants(self):
        """
        Final tidying of time-variants, as described in comments to each line of code below.
        """

        for time_variant in self.time_variants:

            # add zero at starting time for model run to all program proportions
            if 'program_prop' in time_variant or 'int_prop' in time_variant:
                self.time_variants[time_variant][int(self.model_constants['start_time'])] = 0.

            # remove the load_data keys, as they have been used and are now redundant
            self.time_variants[time_variant] \
                = tool_kit.remove_specific_key(self.time_variants[time_variant], 'load_data')

            # remove keys for which values are nan
            self.time_variants[time_variant] = tool_kit.remove_nans(self.time_variants[time_variant])

    def adjust_param_for_reporting(self, param, country, adjustment_factor):
        """
        Adjust a parameter that is thought to be mis-reported by the country by a constant factor across the estimates
        for all years.

        Args:
            param: The string for the parameter to be adjusted
            country: The country to which this applies
            adjustment_factor: A float to multiply the reported values by to get the adjusted values
        """

        if self.country == country:
            for year in self.time_variants[param]:
                if type(year) == int:
                    self.time_variants[param][year] *= adjustment_factor

    # called later than the methods above - after interventions have been classified
    def find_scaleup_functions(self):
        """
        Master method for calculation of time-variant parameters/scale-up functions.
        """

        # extract data into structures for creating time-variant parameters or constant ones
        self.find_data_for_functions_or_params()

        # find scale-up functions or constant parameters
        self.find_constant_functions()
        self.find_scaleups()

        # find the proportion of cases that are infectious for models that are unstratified by organ status
        if len(self.organ_status) < 2:
            self.set_fixed_infectious_proportion()

        # add parameters for IPT, if and where not specified for the age range being implemented
        self.add_missing_economics_for_ipt()

    def find_data_for_functions_or_params(self):

        """
        Method to load all the dictionaries to be used in generating scale-up functions to a single attribute of the
        class instance (to avoid creating heaps of functions for irrelevant programs).

        Creates: self.scaleup_data, a dictionary of the relevant scale-up data for creating scale-up functions in
            set_scaleup_functions within the model object. First tier of keys is the scenario to be run, next is the
            time variant parameter to be calculated.
        """

        for scenario in self.scenarios:
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

    def find_constant_functions(self):
        """
        Method that can be used to set some variables that might usually be time-variant to be constant instead,
        by creating a function that is just a single constant value (through the static make_constant_function method
        above).
        """

        for scenario in self.scenarios:

            # initialise the scale-up function dictionary for the scenario
            self.scaleup_fns[scenario] = {}

            # set constant functions for proportion smear-positive and negative
            if not self.gui_inputs['is_timevariant_organs']:
                for organ in ['pos', 'neg']:
                    self.scaleup_fns[scenario]['epi_prop_smear' + organ] \
                        = make_constant_function(self.model_constants['epi_prop_smear' + organ])

            # set constant function for effective contact rate
            if not self.gui_inputs['is_timevariant_contactrate']:
                self.scaleup_fns[scenario]['tb_n_contact'] \
                    = make_constant_function(self.model_constants['tb_n_contact'])

    def find_scaleups(self):
        """
        Calculate the scale-up functions from the scale-up data attribute and populate to a dictionary with keys of the
        scenarios to be run.
        Note that the 'demo_life_expectancy' parameter has to be given this name and base.py will then calculate
        population death rates automatically.
        """

        for scenario in self.scenarios:
            scenario_name = tool_kit.find_scenario_string_from_number(scenario)

            # define scale-up functions from these datasets
            for param in self.scaleup_data[scenario]:
                if param not in self.scaleup_fns[scenario]:  # if not already set as constant previously

                    # extract and remove the smoothness parameter from the dictionary
                    smoothness = self.gui_inputs['default_smoothness']
                    if 'smoothness' in self.scaleup_data[scenario][param]:
                        smoothness = self.scaleup_data[scenario][param].pop('smoothness')

                    # if the parameter is being modified for the scenario being run
                    scenario_for_function = None
                    if 'scenario' in self.scaleup_data[scenario][param]:
                        scenario_for_function = [self.model_constants['scenario_full_time'],
                                                 self.scaleup_data[scenario][param].pop('scenario')]

                    # upper bound depends on whether the parameter is a proportion
                    upper_bound = None
                    if 'prop_' in param: upper_bound = 1.

                    # calculate the scaling function
                    self.scaleup_fns[scenario][param] \
                        = scale_up_function(self.scaleup_data[scenario][param].keys(),
                                            self.scaleup_data[scenario][param].values(),
                                            self.gui_inputs['fitting_method'], smoothness,
                                            bound_low=0., bound_up=upper_bound,
                                            intervention_end=scenario_for_function,
                                            intervention_start_date=self.model_constants[
                                                'scenario_start_time'])

                    # freeze at point in time if necessary
                    if scenario_name in self.freeze_times \
                            and self.freeze_times[scenario_name] < self.model_constants['recent_time']:
                        self.scaleup_fns[scenario][param] \
                            = freeze_curve(self.scaleup_fns[scenario][param], self.freeze_times[scenario_name])

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
                    self.add_comment_to_gui_window('"' + param[1:] + '" parameter unavailable for "' + agegroup +
                                                   '" age-group, so default value used.\n')

    ##############################
    ### Classify interventions ###
    ##############################

    def classify_interventions(self):
        """
        Classify the interventions as to whether they are generally relevant, whether they apply to specific scenarios
        being run and whether they are to be costed.
        """

        self.find_irrelevant_time_variants()
        self.find_relevant_interventions()
        self.determine_organ_detection_variation()
        self.determine_riskgroup_detection_variation()
        self.find_potential_interventions_to_cost()
        self.find_interventions_to_cost()

    def find_irrelevant_time_variants(self):
        """
        List all the time-variant parameters that are not relevant to the current model structure (unstratified by
        the scenario being run).
        """

        for time_variant in self.time_variants:
            for strain in self.available_strains:

                # exclude programs relevant to strains that aren't included in the model
                if strain not in self.strains and strain in time_variant:
                    self.irrelevant_time_variants += [time_variant]

            # exclude time-variants that are percentages, irrelevant drug-susceptibility testing programs, inappropriate
            # treatment time-variants for single strain models, smear-negative parameters for unstratified models,
            # low-quality care sector interventions for models not including this.
            if 'perc_' in time_variant \
                    or (len(self.strains) < 2 and 'line_dst' in time_variant) \
                    or (len(self.strains) < 3 and 'secondline_dst' in time_variant) \
                    or ('_inappropriate' in time_variant
                                and (len(self.strains) < 2 or not self.gui_inputs['is_misassignment'])) \
                    or (len(self.organ_status) == 1 and 'smearneg' in time_variant) \
                    or ('lowquality' in time_variant and not self.gui_inputs['is_lowquality']) \
                    or (len(self.strains) > 1 and 'treatment_' in time_variant and 'timeperiod_' not in time_variant
                        and ('_ds' not in time_variant and 'dr' not in time_variant)):
                self.irrelevant_time_variants += [time_variant]

    def find_relevant_interventions(self):
        """
        Code to create lists of the programmatic interventions that are relevant to a particular scenario being run.

        Creates:
            self.relevant_interventions: A dict with keys scenarios and values lists of scenario-relevant programs
        """

        for scenario in self.scenarios:
            self.relevant_interventions[scenario] = []
            for time_variant in self.time_variants:
                for key in self.time_variants[time_variant]:
                    # if 1) not irrelevant to structure, 2) it is a programmatic time variant,
                    # 3) it hasn't been added yet, 4) it has a non-zero entry for any year or scenario value
                    if time_variant not in self.irrelevant_time_variants \
                            and ('program_' in time_variant or 'int_' in time_variant) \
                            and time_variant not in self.relevant_interventions[scenario] \
                            and ((type(key) == int and self.time_variants[time_variant][key] > 0.)
                                 or (type(key) == str and key == tool_kit.find_scenario_string_from_number(scenario))):
                        self.relevant_interventions[scenario] += [time_variant]

        # add terms for the IPT interventions to the list that refer to its general type without the specific age string
        for scenario in self.scenarios:
            for intervention in self.relevant_interventions[scenario]:
                if 'int_prop_ipt_age' in intervention:
                    self.relevant_interventions[scenario] += ['agestratified_ipt']
                elif 'int_prop_ipt' in intervention and 'community_ipt' not in intervention:
                    self.relevant_interventions[scenario] += ['ipt']

            # similarly, add universal terms for ACF interventions, regardless of the risk-group applied to
            for riskgroup in self.riskgroups:
                for intervention in ['_xpertacf', '_cxrxpertacf']:
                    if 'int_prop' + intervention + riskgroup in self.relevant_interventions[scenario]:
                        self.relevant_interventions[scenario] += ['acf']

    def determine_organ_detection_variation(self):
        """
        Work out what we're doing with variation of detection rates by organ status (consistently for all scenarios).
        """

        # start with request
        self.vary_detection_by_organ = self.gui_inputs['is_vary_detection_by_organ']

        # turn off and warn if model unstratified by organ status
        if len(self.organ_status) == 1 and self.vary_detection_by_organ:
            self.vary_detection_by_organ = False
            print('Requested variation by organ status turned off, as model is unstratified by organ status.')

        # turn on and warn if Xpert requested but variation not requested
        for scenario in self.scenarios:
            if len(self.organ_status) > 1 and 'int_prop_xpert' in self.relevant_interventions[scenario] \
                    and not self.vary_detection_by_organ:
                self.vary_detection_by_organ = True
                print('Variation in detection by organ status added for Xpert implementation, although not requested.')
            elif len(self.organ_status) == 1 and 'int_prop_xpert' in self.relevant_interventions[scenario]:
                print('Effect of Xpert on smear-negative detection not simulated as model unstratified by organ status.')

        # set relevant attributes
        self.organs_for_detection = ['']
        if self.vary_detection_by_organ: self.organs_for_detection = self.organ_status

    def determine_riskgroup_detection_variation(self):
        """
        Set variation in detection by risk-group according to whether ACF or intensive screening implemented (in any of
        the scenarios).
        """

        self.vary_detection_by_riskgroup = False
        for scenario in self.scenarios:
            for intervention in self.relevant_interventions[scenario]:
                if 'acf' in intervention or 'intensive_screening' in intervention:
                    self.vary_detection_by_riskgroup = True
        self.riskgroups_for_detection = ['']
        if self.vary_detection_by_riskgroup: self.riskgroups_for_detection = self.riskgroups

    def find_potential_interventions_to_cost(self):
        """
        Creates a list of the interventions that could potentially be costed if they are requested - that is, the ones
        for which model.py has popsize calculations coded.
        """

        if len(self.strains) > 1:
            self.potential_interventions_to_cost += ['shortcourse_mdr']
            self.potential_interventions_to_cost += ['food_voucher_ds']
            self.potential_interventions_to_cost += ['food_voucher_mdr']
        for organ in self.organ_status:
            self.potential_interventions_to_cost += ['ambulatorycare' + organ]
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

    def find_interventions_to_cost(self):
        """
        Work out which interventions should be costed, selecting from the ones that can be costed in
        self.potential_interventions_to_cost.
        """

        for scenario in self.scenarios:
            self.interventions_to_cost[scenario] = []
            for intervention in self.potential_interventions_to_cost:
                if 'int_prop_' + intervention in self.relevant_interventions[scenario]:
                    self.interventions_to_cost[scenario] += [intervention]

    # actually has to be called later and is just required for optimisation
    def find_intervention_startdates(self):
        """
        Find the dates when the different interventions start and populate self.intervention_startdates
        """

        for scenario in self.scenarios:
            self.intervention_startdates[scenario] = {}
            for intervention in self.interventions_to_cost[scenario]:
                self.intervention_startdates[scenario][intervention] = None
                years_pos_coverage \
                    = [key for (key, value) in
                       self.scaleup_data[scenario]['int_prop_' + intervention].items()
                       if value > 0.]
                if len(years_pos_coverage) > 0:  # i.e. some coverage present from start
                    self.intervention_startdates[scenario][intervention] = min(years_pos_coverage)

    ###################################
    ### Uncertainty-related methods ###
    ###################################

    def process_uncertainty_parameters(self):
        """
        Master method to uncertainty processing, calling other relevant methods.
        """

        # specify the parameters to be used for uncertainty
        if self.gui_inputs['output_uncertainty']:
            self.find_uncertainty_distributions()
            self.get_data_to_fit()

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

        # decide whether calibration or uncertainty analysis is being run
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

        # work through vars to be used and populate into the data fitting dictionary
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

    #############################
    ### Miscellaneous methods ###
    #############################

    def find_keys_of_sheets_to_read(self):
        """
        Find keys of spreadsheets to read. Pretty simplistic at this stage, but expected to get more complicated as
        other sheets (like diabetes) are added as optional.
        """

        keys_of_sheets_to_read = ['bcg', 'rate_birth', 'life_expectancy', 'default_parameters', 'tb', 'notifications',
                                  'outcomes', 'country_constants', 'default_constants', 'country_programs',
                                  'default_programs']

        # add any optional sheets required for specific model being run (currently just diabetes)
        if 'riskgroup_diabetes' in self.gui_inputs: keys_of_sheets_to_read += ['diabetes']

        return keys_of_sheets_to_read

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

    def checks(self):
        """
        Not much in here as yet. However, this function is intended to contain all the data consistency checks for
        data entry.
        """

        # Check that all entered times occur after the model start time
        for time in self.model_constants:
            if time[-5:] == '_time' and '_step_time' not in time:
                assert self.model_constants[time] >= self.model_constants['start_time'], \
                    '% is before model start time' % self.model_constants[time]
