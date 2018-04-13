
# external imports
import copy
import numpy
import itertools

# AuTuMN imports
import spreadsheet
import tool_kit
from curve import scale_up_function


def make_constant_function(value):
    """
    Function that returns a function of constant returned value with a deliberately irrelevant argument,
    to maintain consistency with the number of arguments to other functions that take time as an argument.
    Note that "time" is an irrelevant argument to the returned function, but necessary for consistency with other
    functions.

    Args:
        value: The value for the created function to return
    Returns:
        constant: The constant function
    """

    def constant(_):
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
    def __init__(self, gui_inputs, gui_console_fn=None):
        """
        As much data processing occurs here as possible, with GUI modules intended to be restricted to just loading of
        data in its rawest form, while as little processing is done in the model runner and model objects as possible.
        As most of this module is about processing the data and creating attributes to this object, instantiation does
        the minimum amount possible - mostly just setting the necessary attributes to empty versions of themselves.

        Args:
            gui_inputs: Raw inputs from GUI
            gui_console_fn: callback function to handle model diagnostics output,
                function should accept parameters (output_type, data) where
                    output_type: string of type of message
                    data: dictionary literal
        """

        # most general inputs, including attributes that need to be converted from keys to input dictionary
        self.gui_inputs = gui_inputs
        self.gui_console_fn = gui_console_fn
        if self.gui_console_fn:
            self.gui_console_fn('init')

        # initialising attributes by data type now, rather than purpose
        (self.plot_count, self.n_organs, self.n_strains, self.fitting_method, self.n_samples) \
            = [0 for _ in range(5)]
        self.emit_delay = .1
        (self.uncertainty_intervention, self.comorbidity_to_increment, self.run_mode, self.original_data,
         self.agegroups, self.country) \
            = [None for _ in range(6)]
        (self.param_ranges_unc, self.int_ranges_unc, self.outputs_unc, self.riskgroups, self.treatment_outcome_types,
         self.irrelevant_time_variants, self.organ_status, self.scenarios, self.histories,
         self.inappropriate_regimens) \
            = [[] for _ in range(10)]
        (self.original_data, self.derived_data, self.time_variants, self.model_constants, self.scaleup_data,
         self.scaleup_fns, self.intervention_param_dict, self.comorbidity_prevalences,
         self.alternative_distribution_dict, self.data_to_fit, self.mixing, self.relevant_interventions,
         self.interventions_to_cost, self.intervention_startdates, self.freeze_times) \
            = [{} for _ in range(15)]
        (self.riskgroups_for_detection, self.organs_for_detection, self.strains) \
            = [[''] for _ in range(3)]
        (self.is_vary_detection_by_organ, self.is_vary_detection_by_riskgroup, self.is_include_relapse_in_ds_outcomes,
         self.is_vary_force_infection_by_riskgroup, self.is_include_hiv_treatment_outcomes, self.is_adjust_population) \
            = [False for _ in range(6)]

        # set some attributes direct from GUI inputs
        for attribute in \
                ['country', 'is_vary_detection_by_organ', 'is_vary_detection_by_riskgroup',
                 'is_include_relapse_in_ds_outcomes', 'is_vary_force_infection_by_riskgroup', 'fitting_method',
                 'uncertainty_intervention', 'is_include_hiv_treatment_outcomes', 'is_adjust_population',
                 'n_centiles_for_shading']:
            setattr(self, attribute, gui_inputs[attribute])

        # various lists of strings for features available to the models or running modes
        self.compartment_types \
            = ['susceptible_fully', 'susceptible_immune', 'latent_early', 'latent_late', 'active',
               'detect', 'missed', 'treatment_infect', 'treatment_noninfect']
        self.interventions_available_for_costing \
            = ['vaccination', 'xpert', 'treatment_support_relative', 'treatment_support_absolute', 'smearacf',
               'xpertacf', 'ipt_age0to5', 'ipt_age5to15', 'decentralisation', 'improve_dst', 'bulgaria_improve_dst',
               'firstline_dst', 'intensive_screening', 'ipt_age15up', 'dot_groupcontributor', 'awareness_raising']
        self.available_strains = ['_ds', '_mdr', '_xdr']
        self.intervention_param_dict \
            = {'int_prop_treatment_support_relative': ['int_prop_treatment_support_improvement'],
               'int_prop_decentralisation': ['int_ideal_detection'],
               'int_prop_xpert': ['int_prop_xpert_smearneg_sensitivity', 'int_prop_xpert_sensitivity_mdr',
                                  'int_timeperiod_await_treatment_smearneg_xpert'],
               'int_prop_ipt': ['int_prop_ipt_effectiveness', 'int_prop_ltbi_test_sensitivity',
                                'int_prop_infections_in_household'],
               'int_prop_acf': ['int_prop_acf_detections_per_round'],
               'int_prop_awareness_raising': ['int_multiplier_detection_with_raised_awareness'],
               'int_perc_shortcourse_mdr': ['int_prop_treatment_success_shortcoursemdr'],
               'int_perc_firstline_dst': [],
               'int_perc_treatment_support_relative_ds': ['int_prop_treatment_support_improvement_ds'],
               'int_perc_dots_contributor': ['int_prop_detection_dots_contributor'],
               'int_perc_dots_groupcontributor': ['int_prop_detection_dots_contributor',
                                                  'int_prop_detection_ngo_ruralpoor']}

    ''' master method '''

    def process_inputs(self):
        """
        Master method of this object, calling all sub-methods to read and process data and define model structure.
        Effectively the initialisation of the model and the model runner objects occurs through here.
        """

        # get inputs from GUI
        self.find_user_inputs()

        # read data from sheets
        self.read_data()

        # process constant parameters
        self.process_model_constants()

        # process time-variant parameters
        self.process_time_variants()

        # has to go after time variants so that the starting proportion in each risk group can be specified
        self.define_riskgroup_structure()

        # find parameters that require processing after stratified model structure has been fully defined
        self.find_stratified_parameters()

        # classify interventions as to whether they apply and are to be costed
        self.classify_interventions()

        # add compartment for IPT, low-quality health care and novel vaccination if implemented
        self.extend_model_structure_for_interventions()

        # calculate time-variant functions
        self.find_scaleup_functions()

        # perform checks (undeveloped)
        self.checks()

    ''' user input-related methods and defining model structure '''

    def find_user_inputs(self):
        """
        Decide on run mode, basic model strata and scenarios to be run.
        """

        self.add_comment_to_gui_window('Preparing inputs for model run.\n')
        self.define_run_mode()
        self.define_model_strata()
        self.find_scenarios_to_run()
        self.reconcile_user_inputs()

    def define_run_mode(self):
        """
        Defines a few basic structures specific to the mode of running for the model runner object.
        """

        # running mode
        run_mode_conversion \
            = {'Scenario analysis': 'scenario',
               'Epidemiological uncertainty': 'epi_uncertainty',
               'Intervention uncertainty': 'int_uncertainty',
               'Increment comorbidity': 'increment_comorbidity',
               'Rapid calibration': 'rapid_calibration'}
        self.run_mode = run_mode_conversion[self.gui_inputs['run_mode']]

        # uncertainty
        if self.run_mode == 'epi_uncertainty':

            # for incidence for ex, width of normal posterior relative to CI width in data
            self.outputs_unc \
                = [{'key': 'incidence', 'posterior_width': None, 'width_multiplier': 0.5}]
            self.alternative_distribution_dict \
                = {'tb_prop_casefatality_untreated_smearpos': ['beta_mean_stdev', .7, .15],
                   'tb_timeperiod_activeuntreated': ['gamma_mean_stdev', 3., .5],
                   'tb_multiplier_treated_protection': ['gamma_mean_stdev', 1., .6]}

        # intervention uncertainty
        elif self.run_mode == 'int_uncertainty':
            self.scenarios.append(15)
            self.gui_inputs['output_by_scenario'] = True

        # increment comorbidity
        elif self.run_mode == 'increment_comorbidity':
            self.comorbidity_to_increment = self.gui_inputs['comorbidity_to_increment'].lower()
            prevalences = [0.05] + list(numpy.linspace(.1, .5, 5))
            self.comorbidity_prevalences = {i: prevalences[i] for i in range(len(prevalences))}

        elif self.run_mode == 'rapid_calibration':
            self.outputs_unc = [{'key': 'incidence', 'posterior_width': None, 'width_multiplier': 0.5}]

    def define_model_strata(self):
        """
        Method to group the methods defining strata setting.
        """

        self.define_age_structure()
        self.define_organ_structure()
        self.define_strain_structure()
        self.define_treatment_history_structure()

    def define_age_structure(self):
        """
        Define the model's age structure based on the breakpoints provided in spreadsheets.
        """

        self.model_constants['age_breakpoints'] = self.gui_inputs['age_breakpoints']
        self.add_comment_to_gui_window('GUI breakpoints ' + str(self.gui_inputs['age_breakpoints']) + '.\n')
        self.agegroups = tool_kit.get_agegroups_from_breakpoints(self.gui_inputs['age_breakpoints'])[0]

    def define_organ_structure(self):
        """
        Define the organ status strata for the model. Convert from GUI input structure to number of strata. Then, create
        list of organ states. If unstratified, need list of an empty string for iteration.
        """

        organ_stratification_keys \
            = {'Unstratified': 0,
               'Pos / Neg': 2,
               'Pos / Neg / Extra': 3}
        self.n_organs = organ_stratification_keys[self.gui_inputs['organ_strata']]
        available_organs = ['_smearpos', '_smearneg', '_extrapul']
        self.organ_status = available_organs[:self.n_organs] if self.n_organs > 1 else ['']

    def define_strain_structure(self):
        """
        Find the number of strata present for TB strains being simulated.
        """

        strain_stratification_keys \
            = {'Single strain': 0,
               'DS / MDR': 2,
               'DS / MDR / XDR': 3}
        self.n_strains = strain_stratification_keys[self.gui_inputs['strains']]
        if self.n_strains:
            self.strains = self.available_strains[:self.n_strains]
            if self.gui_inputs['is_misassignment']:
                self.treatment_outcome_types = copy.copy(self.strains)

                # if misassigned strain treated with weaker regimen
                for strain in self.strains[1:]:
                    for treated_as in self.strains:
                        if self.strains.index(treated_as) < self.strains.index(strain):
                            self.treatment_outcome_types.append(strain + '_as' + treated_as[1:])

        self.inappropriate_regimens = [regimen for regimen in self.treatment_outcome_types if '_as' in regimen]

    def define_treatment_history_structure(self):
        """
        Define the structure for tracking patients' treatment histories (i.e. whether they are treatment naive "_new"
        patients, or whether they are previously treated "_treated" patients. Note that the list was set to an empty
        string (for no stratification) in initialisation of this object.
        """

        self.histories = ['_new', '_treated'] if self.gui_inputs['is_treatment_history'] else ['']

    def find_scenarios_to_run(self):
        """
        Find the scenarios to be run. Only applicable to certain run modes, but code is currently run for any mode.
        """

        # scenario processing
        self.scenarios = [0]
        self.scenarios \
            += [tool_kit.find_scenario_number_from_string(key) for key in self.gui_inputs
                if key.startswith('scenario_') and len(key) < 12 and self.gui_inputs[key]]

    def reconcile_user_inputs(self):
        """
        Method to ensure that user inputs make sense within the model, including that elaborations that are specific to
        particular ways of structuring the model are turned off with a warning if the model doesn't have the structure
        to allow for those elaborations.
        """

        if self.n_organs <= 1 and self.gui_inputs['is_timevariant_organs']:
            self.add_comment_to_gui_window(
                'Time-variant organ status requested, but not implemented as no stratification by organ status')
            self.gui_inputs['is_timevariant_organs'] = False
        if len(self.organ_status) == 1 and self.is_vary_detection_by_organ:
            self.is_vary_detection_by_organ = False
            self.add_comment_to_gui_window(
                'Requested variation by organ status turned off, as model is unstratified by organ status.')
        if self.gui_inputs['is_amplification'] and self.n_strains <= 1:
            self.add_comment_to_gui_window(
                'Resistance amplification requested, but not implemented as single strain model only')
            self.gui_inputs['is_amplification'] = False
        if self.gui_inputs['is_misassignment'] and self.n_strains <= 1:
            self.add_comment_to_gui_window(
                'Misassignment requested, but not implemented as single strain model only')
            self.gui_inputs['is_misassignment'] = False
        # test presence of riskgroups
        is_any_riskgroup = False
        for gui_input_name in self.gui_inputs.keys():
            if 'riskgroup' in gui_input_name:
                if self.gui_inputs[gui_input_name]:
                    is_any_riskgroup = True
                    break
        if not is_any_riskgroup and self.is_vary_force_infection_by_riskgroup:
            self.add_comment_to_gui_window(
                'Heterogeneous mixing requested, but not implemented as no risk groups are present')
            self.is_vary_force_infection_by_riskgroup = False

    # second category of structure methods must come after spreadsheet reading

    def define_riskgroup_structure(self):
        """
        Work out the risk group stratification.
        """

        # create list of risk group names
        for item in self.gui_inputs:
            if item.startswith('riskgroup_'):
                riskgroup = item.replace('riskgroup', '')
                if self.gui_inputs[item] and 'riskgroup_prop' + riskgroup in self.time_variants:
                    self.riskgroups.append(riskgroup)
                elif self.gui_inputs[item]:
                    self.add_comment_to_gui_window(
                        'Stratification requested for %s risk group, but proportions not specified'
                        % tool_kit.find_title_from_dictionary(riskgroup))

        # add the null group according to whether there are any risk groups
        norisk_string = '_norisk' if self.riskgroups else ''
        self.riskgroups.append(norisk_string)

        # ensure some starting proportion of births go to the risk group stratum if value not loaded earlier
        self.model_constants.update({'riskgroup_prop' + riskgroup: 0. for riskgroup in self.riskgroups
                                     if 'riskgroup_prop' + riskgroup not in self.model_constants})

        # create mixing matrix (has to be run after scale-up data collation, so can't go in model structure method)
        self.mixing = self.create_mixing_matrix() if self.is_vary_force_infection_by_riskgroup else {}

    def create_mixing_matrix(self):
        """
        Creates model attribute for mixing between population risk groups, for use in calculate_force_infection_vars
        method below only.
        """

        # create mixing matrix separately for each scenario, just in case risk groups being managed differently
        mixing = {}

        # next tier of dictionary is the "to" risk group that is being infected
        for to_riskgroup in self.riskgroups:
            mixing[to_riskgroup] = {}

            # last tier of dictionary is the "from" risk group describing the make up of contacts
            for from_riskgroup in self.riskgroups:
                if from_riskgroup != '_norisk':

                    # use parameters for risk groups other than "_norisk" if available
                    if 'prop_mix' + to_riskgroup + '_from' + from_riskgroup in self.model_constants:
                        mixing[to_riskgroup][from_riskgroup] \
                            = self.model_constants['prop_mix' + to_riskgroup + '_from' + from_riskgroup]

                    # otherwise use the latest value for the proportion of the population with that risk factor
                    else:
                        mixing[to_riskgroup][from_riskgroup] \
                            = find_latest_value_from_year_dict(self.time_variants['riskgroup_prop' + from_riskgroup],
                                                               self.model_constants['current_time'])

            # give the remainder to the "_norisk" group without any risk factors
            if sum(mixing[to_riskgroup].values()) >= 1.:
                self.add_comment_to_gui_window(
                    'Total of proportions of contacts for risk group %s greater than one. Model invalid.'
                    % to_riskgroup)
            mixing[to_riskgroup]['_norisk'] = 1. - sum(mixing[to_riskgroup].values())
        return mixing

    # last category of model structure methods must come after interventions classified and time-variants defined

    def extend_model_structure_for_interventions(self):
        """
        Add any additional elaborations to the model structure for extra processes - specifically, IPT, low-quality
        health care and novel vaccinations.
        """

        for scenario in self.scenarios:
            if 'agestratified_ipt' in self.relevant_interventions[scenario] \
                    or 'ipt' in self.relevant_interventions[scenario]:
                self.compartment_types.append('onipt')
        if self.gui_inputs['is_lowquality']:
            self.compartment_types += ['lowquality']
        if 'int_prop_novel_vaccination' in self.relevant_interventions:
            self.compartment_types += ['susceptible_novelvac']

    ''' spreadsheet-related methods '''

    def read_data(self):
        """
        Simple method to read spreadsheet inputs.
        """

        self.add_comment_to_gui_window('Reading Excel sheets with input data.\n')
        self.original_data = spreadsheet.read_input_data_xls(True, self.find_keys_of_sheets_to_read(), self.country,
                                                             self.gui_console_fn)
        self.add_comment_to_gui_window('Spreadsheet reading complete.\n')

    def find_keys_of_sheets_to_read(self):
        """
        Find keys of spreadsheets to read. Pretty simplistic at this stage, but expected to get more complicated as
        other sheets (like diabetes) are added as optional.
        """

        # where sheets are available from multiple years, use _ and the year to choose the sheet, which will be dropped
        keys_of_sheets_to_read \
            = ['bcg_2016', 'rate_birth_2015', 'life_expectancy_2015', 'gtb_2015', 'gtb_2016', 'notifications_2016',
               'outcomes_2015', 'default_parameters', 'country_constants', 'default_constants', 'country_programs',
               'default_programs']

        # add any optional sheets required for specific model being run (currently just diabetes)
        if 'riskgroup_diabetes' in self.gui_inputs:
            keys_of_sheets_to_read.append('diabetes')

        return keys_of_sheets_to_read

    ''' constant parameter processing methods '''

    # populate with first round of unprocessed parameters, before model structure defined

    def process_model_constants(self):
        """
        Master method to call methods for processing constant model parameters.
        """

        # note ordering to list of sheets to be worked through is important for hierarchical loading of constants
        sheets_with_constants = ['country_constants', 'default_constants']
        if self.gui_inputs['riskgroup_diabetes']:
            sheets_with_constants.append('diabetes')
        self.add_model_constant_defaults(sheets_with_constants)

        self.add_universal_parameters()
        self.process_uncertainty_parameters()

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
        if self.n_organs < 2:
            self.model_constants['epi_prop'] = 1.

        # infectiousness of smear-positive and extrapulmonary patients
        else:
            self.model_constants['tb_multiplier_force_smearpos'] = 1.
            self.model_constants['tb_multiplier_force_extrapul'] = 0.

        # no additional protection for new patients (tb_multiplier_treated_protection is used for additional immunity)
        if len(self.histories) > 1:
            self.model_constants['tb_multiplier_new_protection'] = 1.
        else:
            self.model_constants['tb_multiplier_protection'] = 1.

        # add a time period to treatment for models unstratified by organ status
        if len(self.organ_status) == 1:
            self.model_constants['program_timeperiod_await_treatment'] \
                = self.model_constants['program_timeperiod_await_treatment_smearpos']

        # reference group for susceptibility
        self.model_constants['tb_multiplier_fully_protection'] = 1.

    def process_uncertainty_parameters(self):
        """
        Master method to uncertainty processing, calling other relevant methods.
        """

        # specify the parameters to be used for uncertainty
        if self.run_mode == 'epi_uncertainty' or self.run_mode == 'int_uncertainty':
            self.find_uncertainty_distributions()
        if self.run_mode == 'epi_uncertainty' or self.run_mode == 'rapid_calibration':
            self.get_data_to_fit()

    def find_uncertainty_distributions(self):
        """
        Populate a dictionary of uncertainty parameters from the inputs dictionary in a format that matches code for
        uncertainty.
        """
        for param in self.model_constants:
            if ('tb_' in param or 'start_time' in param) \
                    and '_uncertainty' in param and type(self.model_constants[param]) == dict:
                self.param_ranges_unc += [{'key': param[:-12],
                                           'bounds': [self.model_constants[param]['lower'],
                                                      self.model_constants[param]['upper']],
                                           'distribution': 'uniform'}]
            elif 'int_' in param and '_uncertainty' in param and type(self.model_constants[param]) == dict:
                self.int_ranges_unc += [{'key': param[:-12],
                                         'bounds': [self.model_constants[param]['lower'],
                                                    self.model_constants[param]['upper']],
                                         'distribution': 'uniform'}]

        # change distributions for parameters hard-coded to alternative distributions in instantiation above
        for n_param in range(len(self.param_ranges_unc)):
            if self.param_ranges_unc[n_param]['key'] in self.alternative_distribution_dict:
                self.param_ranges_unc[n_param]['distribution'] \
                    = self.alternative_distribution_dict[self.param_ranges_unc[n_param]['key']][0]
                if len(self.alternative_distribution_dict[self.param_ranges_unc[n_param]['key']]) > 1:
                    self.param_ranges_unc[n_param]['additional_params'] \
                        = self.alternative_distribution_dict[self.param_ranges_unc[n_param]['key']][1:]

    def get_data_to_fit(self):
        """
        Extract data for model fitting. Choices currently hard-coded above.
        """

        inc_conversion_dict \
            = {'incidence': 'e_inc_100k',
               'incidence_low': 'e_inc_100k_lo',
               'incidence_high': 'e_inc_100k_hi'}
        mort_conversion_dict \
            = {'mortality': 'e_mort_exc_tbhiv_100k',
               'mortality_low': 'e_mort_exc_tbhiv_100k_lo',
               'mortality_high': 'e_mort_exc_tbhiv_100k_hi'}

        # work through vars to be used and populate into the data fitting dictionary
        for output in self.outputs_unc:
            if output['key'] == 'incidence':
                for key in inc_conversion_dict:
                    self.data_to_fit[key] = self.original_data['gtb'][inc_conversion_dict[key]]
            elif output['key'] == 'mortality':
                for key in mort_conversion_dict:
                    self.data_to_fit[key] = self.original_data['gtb'][mort_conversion_dict[key]]
            else:
                self.add_comment_to_gui_window(
                    'Warning: Calibrated output %s is not directly available from the data' % output['key'])

    # derive parameters specific to stratification, after model structure fully defined

    def find_stratified_parameters(self):
        """
        Find additional parameters. Includes methods that require the model structure to be defined, so that this can't
        be run with process_model_constants.
        """

        if len(self.agegroups) > 1:
            self.find_ageing_rates()
            self.find_fixed_age_specific_parameters()

        # if the model isn't stratified by strain, use DS-TB time-periods for the single strain
        if not self.n_strains:
            for timeperiod in ['tb_timeperiod_infect_ontreatment', 'tb_timeperiod_ontreatment']:
                self.model_constants[timeperiod] = self.model_constants[timeperiod + '_ds']

        # find risk group-specific parameters
        if len(self.riskgroups) > 1:
            self.find_riskgroup_progressions()

        # calculate rates of progression to active disease or late latency
        self.find_latency_progression_rates()

        # find the time non-infectious on treatment from the total time on treatment and the time infectious
        self.find_noninfectious_period()

    def find_ageing_rates(self):
        """
        Calculate ageing rates as the reciprocal of the width of the age bracket.
        """

        for agegroup in self.agegroups:
            age_limits = tool_kit.interrogate_age_string(agegroup)[0]
            if 'up' not in agegroup:
                self.model_constants['ageing_rate' + agegroup] = 1. / (age_limits[1] - age_limits[0])

    def find_fixed_age_specific_parameters(self):
        """
        Find weighted age-specific parameters using age weighting code from tool_kit.
        """

        model_breakpoints = [float(i) for i in self.model_constants['age_breakpoints']]  # convert list of ints to float
        for param_type in ['early_progression_age', 'late_progression_age', 'tb_multiplier_child_infectiousness_age']:

            # extract age-stratified parameters in the appropriate form
            param_vals, age_breaks, stem = {}, {}, None
            for param in self.model_constants:
                if param_type in param:
                    age_string, stem = tool_kit.find_string_from_starting_letters(param, '_age')
                    age_breaks[age_string] = tool_kit.interrogate_age_string(age_string)[0]
                    param_vals[age_string] = self.model_constants[param]
            param_breakpoints = tool_kit.find_age_breakpoints_from_dicts(age_breaks)

            # find and set age-adjusted parameters
            age_adjusted_values = \
                tool_kit.adapt_params_to_stratification(param_breakpoints, model_breakpoints, param_vals,
                                                        parameter_name=param_type,
                                                        gui_console_fn=self.gui_console_fn)
            for agegroup in self.agegroups:
                self.model_constants[stem + agegroup] = age_adjusted_values[agegroup]

    def find_riskgroup_progressions(self):
        """
        Adjust the progression parameters from latency to active disease for various risk groups. Early progression
        parameters currently still proportions rather than rates.
        """

        for riskgroup in self.riskgroups:

            # find age above which adjustments should be made, with default assumption of applying to all age-groups
            start_age = self.model_constants['riskgroup_startage' + riskgroup] \
                if 'riskgroup_startage' + riskgroup in self.model_constants else -1.

            # make adjustments for each age group if required
            for agegroup in self.agegroups:
                riskgroup_modifier = self.model_constants['riskgroup_multiplier' + riskgroup + '_progression'] \
                    if 'riskgroup_multiplier' + riskgroup + '_progression' in self.model_constants \
                       and tool_kit.interrogate_age_string(agegroup)[0][0] >= start_age else 1.
                self.model_constants['tb_rate_late_progression' + riskgroup + agegroup] \
                    = self.model_constants['tb_rate_late_progression' + agegroup] * riskgroup_modifier
                self.model_constants['tb_prop_early_progression' + riskgroup + agegroup] \
                    = tool_kit.apply_odds_ratio_to_proportion(
                    self.model_constants['tb_prop_early_progression' + agegroup], riskgroup_modifier)

    def find_latency_progression_rates(self):
        """
        Find early progression rates by age group and by risk group status - i.e. early progression to active TB and
        stabilisation into late latency.
        """

        time_early = self.model_constants['tb_timeperiod_early_latent']
        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                prop_early = self.model_constants['tb_prop_early_progression' + riskgroup + agegroup]

                # early progression rate is early progression proportion divided by early time period
                self.model_constants['tb_rate_early_progression' + riskgroup + agegroup] = prop_early / time_early

                # stabilisation rate is one minus early progression proportion divided by early time period
                self.model_constants['tb_rate_stabilise' + riskgroup + agegroup] = (1. - prop_early) / time_early

    def find_noninfectious_period(self):
        """
        Very simple calculation to work out the periods of time spent non-infectious for each strain (plus inappropriate
        if required).
        """

        for strain in self.strains:
            self.model_constants['tb_timeperiod_noninfect_ontreatment' + strain] \
                = self.model_constants['tb_timeperiod_ontreatment' + strain] \
                - self.model_constants['tb_timeperiod_infect_ontreatment' + strain]

    ''' time variant parameter processing methods '''

    def process_time_variants(self):
        """
        Master method to perform all preparation and processing tasks for time-variant parameters.
        Does not perform the fitting of functions to the data, which is done later in find_scaleup_functions.
        Note that the order of call is important and can lead to errors if changed.
        """

        if 'country_programs' in self.original_data:
            self.time_variants.update(self.original_data['country_programs'])
        self.add_time_variant_defaults()  # add any necessary time-variants from defaults if not in country programs
        self.load_vacc_detect_time_variants()
        self.convert_percentages_to_proportions()
        self.find_treatment_outcomes()
        self.find_irrelevant_treatment_timevariants()
        self.add_demo_dictionaries_to_timevariants()
        self.find_organ_proportions()
        if self.gui_inputs['is_timevariant_organs']:
            self.add_organ_status_to_timevariants()
        else:
            self.find_average_organ_status()
        self.tidy_time_variants()
        self.adjust_param_for_reporting('program_prop_detect', 'Bulgaria', 0.95)  # Bulgaria thought CDR over-estimated

    # general and demographic methods

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
            self.time_variants['int_perc_vaccination'] \
                = dict(self.original_data['bcg'], **self.time_variants['int_perc_vaccination'])

        # case detection
        if self.time_variants['program_perc_detect']['load_data'] == u'yes':
            self.time_variants['program_perc_detect'] \
                = dict(self.original_data['gtb']['c_cdr'], **self.time_variants['program_perc_detect'])

    def convert_percentages_to_proportions(self):
        """
        Converts time-variant dictionaries to proportions if they are loaded as percentages in their raw form.
        """

        for time_variant in self.time_variants.keys():
            if 'perc_' in time_variant:  # if a percentage
                prop_name = time_variant.replace('perc', 'prop')
                self.time_variants[prop_name] = {}
                for year in self.time_variants[time_variant]:
                    if type(year) == int or 'scenario' in year:  # to exclude load_data, smoothness, etc.
                        self.time_variants[prop_name][year] = self.time_variants[time_variant][year] / 1e2
                    else:
                        self.time_variants[prop_name][year] = self.time_variants[time_variant][year]

    # treatment outcome methods

    def find_treatment_outcomes(self):
        """
        Master method for working through all the processes for finding treatment outcome functions.
        """

        self.aggregate_treatment_outcomes()
        self.calculate_treatment_outcome_proportions()
        self.add_treatment_outcomes_to_timevariants()

    def aggregate_treatment_outcomes(self):
        """
        Sums the treatment outcome numbers from the Global TB Report to get aggregate values for the number of patients
        achieving 1) success, 2) death on treatment, 3) unfavourable outcomes other than death on treatment (termed
        default here.
        """

        ''' up to 2011 fields for DS-TB '''

        # create string conversion structures for communication between GTB report and AuTuMN
        hiv_statuses_to_include = ['']
        if self.is_include_hiv_treatment_outcomes:
            hiv_statuses_to_include.append('hiv_')
        pre2011_map_gtb_to_autumn \
            = {'_cmplt': '_success',
               '_cur': '_success',
               '_def': '_default',
               '_fail': '_default',
               '_died': '_death'}

        # by each outcome, find total number of patients achieving that outcome (up to 2011, with or without HIV)
        for outcome in pre2011_map_gtb_to_autumn:
            for status in ['_new', '_treated']:
                self.derived_data[self.strains[0] + status + pre2011_map_gtb_to_autumn[outcome]] = {}

        # needs another loop to prevent the default dictionaries being blanked after working out default
        for outcome in pre2011_map_gtb_to_autumn:
            for hiv_status in hiv_statuses_to_include:

                # new outcomes are disaggregated by organ involvement and hiv status up to 2011
                for organ in ['sp', 'snep']:

                    # for smear-negative/extrapulmonary where cure isn't an outcome
                    if organ != 'snep' or outcome != '_cur':
                        self.derived_data[self.strains[0] + '_new' + pre2011_map_gtb_to_autumn[outcome]] \
                            = tool_kit.increment_dictionary_with_dictionary(
                            self.derived_data[self.strains[0] + '_new' + pre2011_map_gtb_to_autumn[outcome]],
                            self.original_data['outcomes'][hiv_status + 'new_' + organ + outcome])

                # re-treatment outcomes are only disaggregated by hiv status pre-2011
                self.derived_data[self.strains[0] + '_treated' + pre2011_map_gtb_to_autumn[outcome]] \
                    = tool_kit.increment_dictionary_with_dictionary(
                    self.derived_data[self.strains[0] + '_treated' + pre2011_map_gtb_to_autumn[outcome]],
                    self.original_data['outcomes'][hiv_status + 'ret' + outcome])

        ''' post-2011 fields for DS-TB '''

        # create string conversion structures
        hiv_statuses_to_include = ['newrel']
        if self.is_include_hiv_treatment_outcomes:
            hiv_statuses_to_include.append('tbhiv')
        post2011_map_gtb_to_autumn \
            = {'_succ': '_success',
               '_fail': '_default',
               '_lost': '_default',
               '_died': '_death'}

        # by each outcome, find total number of patients achieving that outcome
        for outcome in post2011_map_gtb_to_autumn:

            # new outcomes are disaggregated by hiv status post-2011
            for hiv_status in hiv_statuses_to_include:
                self.derived_data[self.strains[0] + '_new' + post2011_map_gtb_to_autumn[outcome]] \
                    = tool_kit.increment_dictionary_with_dictionary(
                    self.derived_data[self.strains[0] + '_new' + post2011_map_gtb_to_autumn[outcome]],
                    self.original_data['outcomes'][hiv_status + outcome])

            # previously treated outcomes (now excluding relapse) are not disaggregated post-2011
            self.derived_data[self.strains[0] + '_treated' + post2011_map_gtb_to_autumn[outcome]] \
                = tool_kit.increment_dictionary_with_dictionary(
                self.derived_data[self.strains[0] + '_treated' + post2011_map_gtb_to_autumn[outcome]],
                self.original_data['outcomes']['ret_nrel' + outcome])

        # add re-treatment rates on to new if the model is not stratified by treatment history
        if not self.gui_inputs['is_treatment_history']:
            for outcome in ['_success', '_default', '_death']:
                self.derived_data[self.strains[0] + outcome] = {}
                for history in ['_new', '_treated']:
                    self.derived_data[self.strains[0] + outcome] \
                        = tool_kit.increment_dictionary_with_dictionary(
                        self.derived_data[self.strains[0] + outcome],
                        self.derived_data[self.strains[0] + history + outcome])

        ''' MDR and XDR-TB '''

        # simpler because unaffected by 2011 changes
        for strain in self.strains[1:]:
            for outcome in post2011_map_gtb_to_autumn:
                self.derived_data[strain + post2011_map_gtb_to_autumn[outcome]] = {}
                self.derived_data[strain + post2011_map_gtb_to_autumn[outcome]] \
                    = tool_kit.increment_dictionary_with_dictionary(
                        self.derived_data[strain + post2011_map_gtb_to_autumn[outcome]],
                        self.original_data['outcomes'][strain[1:] + outcome])

        # duplicate outcomes by treatment history because not provided as disaggregated for resistant strains
        for history in self.histories:
            for outcome in ['_success', '_default', '_death']:
                for strain in self.strains[1:]:
                    self.derived_data[strain + history + outcome] \
                        = self.derived_data[strain + outcome]

    def calculate_treatment_outcome_proportions(self):
        """
        Find proportions by each outcome for later use in creating the treatment scale-up functions.
        """

        for history in self.histories:
            for strain in self.strains:
                overall_outcomes = tool_kit.calculate_proportion_dict(
                    self.derived_data,
                    [strain + history + '_success', strain + history + '_death', strain + history + '_default'],
                    percent=False, floor=self.model_constants['tb_n_outcome_minimum'], underscore=False)
                self.derived_data['prop_treatment' + strain + history + '_success'] \
                    = overall_outcomes['prop' + strain + history + '_success']
                nonsuccess_outcomes = tool_kit.calculate_proportion_dict(
                    self.derived_data,
                    [strain + history + '_death', strain + history + '_default'],
                    percent=False, floor=0., underscore=False)
                self.derived_data['prop_nonsuccess' + strain + history + '_death'] \
                    = nonsuccess_outcomes['prop' + strain + history + '_death']

    def add_treatment_outcomes_to_timevariants(self):
        """
        Add treatment outcomes for all strains and treatment histories to the time variants attribute. Use the same
        approach as elsewhere to adding if requested and data not manually entered. Only done for success and death
        because default is derived from these.
        """

        converter = {'_success': 'treatment', '_death': 'nonsuccess'}
        for strata in itertools.product(self.strains, ['_success', '_death'], self.histories):
            strain, outcome, history = strata
            if self.time_variants['program_prop_' + converter[outcome] + strain + history + outcome]['load_data'] \
                    == u'yes':
                for year in self.derived_data['prop_' + converter[outcome] + strain + history + outcome]:
                    if year not in self.time_variants[
                            'program_prop_' + converter[outcome] + strain + history + outcome]:
                        self.time_variants['program_prop_' + converter[outcome] + strain + history + outcome][year] \
                            = self.derived_data['prop_' + converter[outcome] + strain + history + outcome][year]

    # miscellaneous time-variant parameter methods

    def find_irrelevant_treatment_timevariants(self):
        """
        Find treatment time-variant functions that are irrelevant to requested model structure (such as those specific
        to treatment history in models that are unstratified by treatment history, percentages and outcomes for strains
        not implemented in the current version of the model).
        """

        keep = {}
        for time_variant in self.time_variants:
            keep[time_variant] = True
            remove_on_strain = True
            strains = copy.copy(self.strains)
            if self.gui_inputs['is_misassignment']:
                strains.append('_inappropriate')
            for strain in strains:
                if strain in time_variant:
                    remove_on_strain = False
            remove_on_history = True
            for history in self.histories:
                if history in time_variant:
                    remove_on_history = False
            if len(self.histories) == 1:
                if '_new' in time_variant or '_treated' in time_variant:
                    remove_on_history = True
            if 'program_prop_treatment' in time_variant and (remove_on_strain or remove_on_history):
                keep[time_variant] = False
            if 'program_perc_treatment' in time_variant:
                keep[time_variant] = False
            if not keep[time_variant]:
                self.irrelevant_time_variants += [time_variant]

    def add_demo_dictionaries_to_timevariants(self):
        """
        Add epidemiological time variant parameters to time_variants.
        Similarly to previous methods, only performed if requested and only populated where absent.
        """

        # for each type of demographic parameters
        for demo_parameter in ['life_expectancy', 'rate_birth']:

            # if there are data available from the user-derived sheets and loading external data is requested
            if 'demo_' + demo_parameter in self.time_variants \
                    and self.time_variants['demo_' + demo_parameter]['load_data'] == u'yes':
                self.time_variants['demo_' + demo_parameter] \
                    = dict(self.original_data[demo_parameter], **self.time_variants['demo_' + demo_parameter])

            # otherwise
            else:
                self.time_variants['demo_' + demo_parameter] = self.original_data[demo_parameter]

    def find_organ_proportions(self):
        """
        Calculates dictionaries with proportion of cases progressing to each organ status by year, and adds these to
        the derived_data attribute of the object.
        """

        self.derived_data.update(tool_kit.calculate_proportion_dict(self.original_data['notifications'],
                                                                    ['new_sp', 'new_sn', 'new_ep']))

    def add_organ_status_to_timevariants(self):
        """
        Populate organ status dictionaries where requested and not already loaded.
        """

        # conversion from GTB terminology to AuTuMN
        name_conversion_dict = {'_smearpos': '_sp', '_smearneg': '_sn'}

        # for the time variant progression parameters that are used (extrapulmonary just calculated as a complement)
        for organ in ['_smearpos', '_smearneg']:

            # populate absent values from derived data if input data available
            if 'epi_prop' + organ in self.time_variants:
                self.time_variants['epi_prop' + organ] \
                    = dict(self.derived_data['prop_new' + name_conversion_dict[organ]],
                           **self.time_variants['epi_prop' + organ])

            # otherwise if no input data available, just take the derived data straight from the loaded sheets
            else:
                self.time_variants['epi_prop' + organ] = self.derived_data['prop_new' + name_conversion_dict[organ]]

    def find_average_organ_status(self):
        """
        Determine proportion of incident cases that go to each organ status. If specified in input sheets, take the
        user-requested value. However if not, use the proportions of total notifications for the country being
        simulated.
        """

        name_conversion_dict = {'_smearpos': '_sp', '_smearneg': '_sn', '_extrapul': '_ep'}

        # if specific values requested by user
        if 'epi_prop_smearpos' in self.model_constants and 'epi_prop_smearneg' in self.model_constants:
            self.model_constants['epi_prop_extrapul'] \
                = 1. - self.model_constants['epi_prop_smearpos'] - self.model_constants['epi_prop_smearneg']

        # otherwise use aggregate notifications
        else:

            # count totals notified by each organ status and find denominator
            count_by_organ_status = {}
            for organ in name_conversion_dict.values():
                count_by_organ_status[organ] = numpy.sum(self.original_data['notifications']['new' + organ].values())
            total = numpy.sum(count_by_organ_status.values())

            # calculate proportions from totals
            for organ in name_conversion_dict:
                self.model_constants['epi_prop' + organ] = count_by_organ_status[name_conversion_dict[organ]] / total

    def tidy_time_variants(self):
        """
        Final tidying of time-variants, as described in comments to each line of code below.
        """

        for time_variant in self.time_variants:

            # add zero at starting time for model run to all program proportions
            if ('program_prop' in time_variant or 'int_prop' in time_variant) and '_death' not in time_variant \
                    and '_dots_contributor' not in time_variant \
                    and '_dot_groupcontributor' not in time_variant:
                self.time_variants[time_variant][int(self.model_constants['early_time'])] = 0.

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

    ''' classify interventions '''

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
            # treatment time-variants for single strain models, smear-negative parameters for unstratified models and
            # low-quality care sector interventions for models not including this
            if 'perc_' in time_variant \
                    or (len(self.strains) < 2 and 'line_dst' in time_variant) \
                    or (len(self.strains) < 3 and 'secondline_dst' in time_variant) \
                    or ('_inappropriate' in time_variant and not self.gui_inputs['is_misassignment']) \
                    or (len(self.organ_status) == 1 and 'smearneg' in time_variant) \
                    or ('lowquality' in time_variant and not self.gui_inputs['is_lowquality']) \
                    or (len(self.strains) > 1 and 'treatment_' in time_variant and 'timeperiod_' not in time_variant
                        and '_support' not in time_variant
                        and ('_ds' not in time_variant and 'dr' not in time_variant
                             and '_inappropriate' not in time_variant)):
                self.irrelevant_time_variants += [time_variant]

    def find_relevant_interventions(self):
        """
        Create lists of the programmatic interventions that are relevant to a particular scenario being run.

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
            riskgroups_to_loop = copy.copy(self.riskgroups)
            if '' not in riskgroups_to_loop:
                riskgroups_to_loop.append('')
            for riskgroup in riskgroups_to_loop:
                for acf_type in ['smear', 'xpert']:
                    for whether_cxr_screen in ['', 'cxr']:
                        intervention = 'int_prop_' + whether_cxr_screen + acf_type + 'acf' + riskgroup
                        if intervention in self.relevant_interventions[scenario]:
                            if '_smearpos' in self.organ_status:
                                self.relevant_interventions[scenario] += ['acf']
                            else:
                                self.add_comment_to_gui_window(
                                    intervention + ' not implemented as insufficient organ stratification structure')
                            if '_smearneg' not in self.organ_status and acf_type == 'xpert':
                                self.add_comment_to_gui_window(
                                    'Effect of ' + intervention
                                    + ' on smear-negatives not incorporated, as absent from model')

    def determine_organ_detection_variation(self):
        """
        Work out what we're doing with variation of detection rates by organ status (consistently for all scenarios).
        Note that self.is_vary_detection_by_organ is set to False by default in instantiation.
        """

        for scenario in self.scenarios:
            # turn on and warn if Xpert requested but variation not requested
            if len(self.organ_status) > 1 and 'int_prop_xpert' in self.relevant_interventions[scenario] \
                    and not self.is_vary_detection_by_organ:
                self.is_vary_detection_by_organ = True
                self.add_comment_to_gui_window(
                    'Variation in detection by organ status added for Xpert implementation, although not requested.')

            # leave effect of Xpert on improved diagnosis of smear-negative disease turned off if no organ strata
            elif len(self.organ_status) == 1 and 'int_prop_xpert' in self.relevant_interventions[scenario]:
                self.add_comment_to_gui_window(
                    'Effect of Xpert on smear-negative detection not simulated as model unstratified by organ status.')

        # set relevant attributes
        self.organs_for_detection = self.organ_status if self.is_vary_detection_by_organ else ['']

    def determine_riskgroup_detection_variation(self):
        """
        Set variation in detection by risk-group according to whether ACF or intensive screening implemented (in any of
        the scenarios). Note that self.is_vary_detection_by_riskgroup is set to False by default in instantiation.
        """

        for scenario in self.scenarios:
            for intervention in self.relevant_interventions[scenario]:
                if 'acf' in intervention or 'intensive_screening' in intervention or 'groupcontributor' in intervention:
                    self.is_vary_detection_by_riskgroup = True
        self.riskgroups_for_detection = self.riskgroups if self.is_vary_detection_by_riskgroup else ['']

    def find_potential_interventions_to_cost(self):
        """
        Creates a list of the interventions that could potentially be costed if they are requested - that is, the ones
        for which model.py has popsize calculations coded.
        """

        if len(self.strains) > 1:
            self.interventions_available_for_costing \
                += ['shortcourse_mdr', 'treatment_support_relative_ds', 'treatment_support_relative_mdr']
        for organ in self.organ_status:
            self.interventions_available_for_costing += ['ambulatorycare' + organ]
        if self.gui_inputs['is_lowquality']:
            self.interventions_available_for_costing += ['engage_lowquality']
        for riskgroup in ['_prison', '_indigenous', '_urbanpoor', '_ruralpoor']:
            if self.gui_inputs['riskgroup' + riskgroup]:
                self.interventions_available_for_costing += ['xpertacf' + riskgroup, 'cxrxpertacf' + riskgroup]

    def find_interventions_to_cost(self):
        """
        Work out which interventions should be costed, selecting from the ones that can be costed in
        self.potential_interventions_to_cost.
        """

        for scenario in self.scenarios:
            self.interventions_to_cost[scenario] = []
            for intervention in self.interventions_available_for_costing:
                if 'int_prop_' + intervention in self.relevant_interventions[scenario]:
                    self.interventions_to_cost[scenario] += [intervention]

        if self.run_mode == 'int_uncertainty':
            self.interventions_to_cost[15] = self.interventions_to_cost[0]

    # def find_intervention_startdates(self):
    #     """
    #     Find the dates when the different interventions start and populate self.intervention_startdates
    #     """
    #
    #     for scenario in self.scenarios:
    #         self.intervention_startdates[scenario] = {}
    #         for intervention in self.interventions_to_cost[scenario]:
    #             self.intervention_startdates[scenario][intervention] = None
    #             years_pos_coverage \
    #                 = [key for (key, value)
    #                    in self.scaleup_data[scenario]['int_prop_' + intervention].items() if value > 0.]
    #             if years_pos_coverage:  # i.e. some coverage present from start
    #                 self.intervention_startdates[scenario][intervention] = min(years_pos_coverage)

    ''' finding scale-up functions and related methods '''

    def find_scaleup_functions(self):
        """
        Master method for calculation of time-variant parameters/scale-up functions.
        """

        # extract data into structures for creating time-variant parameters or constant ones
        self.find_data_for_functions_or_params()

        # find scale-up functions or constant parameters
        if self.run_mode == 'increment_comorbidity':
            self.create_comorbidity_scaleups()
        self.find_constant_functions()
        self.find_scaleups()

        # find the proportion of cases that are infectious for models that are unstratified by organ status
        if len(self.organ_status) < 2:
            self.set_fixed_infectious_proportion()

        # add parameters for IPT and treatment support
        self.add_missing_economics()

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

            # find the programs that are relevant and load them to the scaleup_data attribute
            for time_variant in self.time_variants:
                if time_variant not in self.irrelevant_time_variants:
                    self.scaleup_data[scenario][str(time_variant)] = {}
                    for scaleup_key in self.time_variants[time_variant]:
                        if type(scaleup_key) == str and scaleup_key == 'scenario_' + str(scenario):
                            self.scaleup_data[scenario][str(time_variant)]['scenario'] \
                                = self.time_variants[time_variant][scaleup_key]
                        elif scaleup_key == 'smoothness' or type(scaleup_key) == int:
                            self.scaleup_data[scenario][str(time_variant)][scaleup_key] \
                                = self.time_variants[time_variant][scaleup_key]

    def create_comorbidity_scaleups(self):
        """
        Another method that is hard-coded and not elegantly embedded with the GUI, but aiming towards creating better
        appearing outputs when we want to look at what varying levels of comorbidities do over time.
        """

        for scenario in self.comorbidity_prevalences:
            self.scenarios.append(scenario)
            for attribute in ['scaleup_data', 'interventions_to_cost', 'relevant_interventions']:
                getattr(self, attribute)[scenario] = copy.deepcopy(getattr(self, attribute)[0])
            self.scaleup_data[scenario]['riskgroup_prop_' + self.comorbidity_to_increment]['scenario'] \
                = self.comorbidity_prevalences[scenario]

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

    def find_scaleups(self):
        """
        Calculate the scale-up functions from the scale-up data attribute and populate to a dictionary with keys of the
        scenarios to be run.
        Note that the 'demo_life_expectancy' parameter has to be given this name and base.py will then calculate
        population death rates automatically.
        """

        for scenario in self.scenarios:

            # define scale-up functions from these datasets
            for param in self.scaleup_data[scenario]:
                if param not in self.scaleup_fns[scenario]:  # if not already set as constant previously

                    # extract and remove the smoothness parameter from the dictionary
                    smoothness = self.scaleup_data[scenario][param].pop('smoothness') \
                        if 'smoothness' in self.scaleup_data[scenario][param] else self.gui_inputs['default_smoothness']

                    # if the parameter is being modified for the scenario being run
                    scenario_for_function = [self.model_constants['scenario_full_time'],
                                             self.scaleup_data[scenario][param].pop('scenario')] \
                        if 'scenario' in self.scaleup_data[scenario][param] else None

                    # upper bound depends on whether the parameter is a proportion
                    upper_bound = 1. if 'prop_' in param else None

                    # calculate the scaling function
                    self.scaleup_fns[scenario][param] \
                        = scale_up_function(self.scaleup_data[scenario][param].keys(),
                                            self.scaleup_data[scenario][param].values(),
                                            int(self.gui_inputs['fitting_method'][-1]), smoothness,
                                            bound_low=0., bound_up=upper_bound,
                                            intervention_end=scenario_for_function,
                                            intervention_start_date=self.model_constants['scenario_start_time'])

                    # freeze at point in time if necessary
                    # if scenario_name in self.freeze_times \
                    #         and self.freeze_times[scenario_name] < self.model_constants['recent_time']:
                    #     self.scaleup_fns[scenario][param] \
                    #         = freeze_curve(self.scaleup_fns[scenario][param],
                    #                        self.freeze_times[scenario_name])

    def set_fixed_infectious_proportion(self):
        """
        Find a multiplier for the proportion of all cases infectious for models unstructured by organ status.
        """

        self.model_constants['tb_multiplier_force'] \
            = self.model_constants['epi_prop_smearpos'] \
            + self.model_constants['epi_prop_smearneg'] * self.model_constants['tb_multiplier_force_smearneg']

    def add_missing_economics(self):
        """
        To avoid errors because no economic values are available for age-stratified IPT, use the unstratified values
        for each age group for which no value is provided.
        Also need to reproduce economics parameters for relative and absolute treatment support, so that only single set
        of parameters need to be entered.
        """

        for param in ['_saturation', '_inflectioncost', '_unitcost', '_startupduration', '_startupcost']:

            # ipt
            for agegroup in self.agegroups:
                if 'econ' + param + '_ipt' + agegroup not in self.model_constants:
                    self.model_constants['econ' + param + '_ipt' + agegroup] \
                        = self.model_constants['econ' + param + '_ipt']
                    self.add_comment_to_gui_window('"' + param[1:] + '" parameter unavailable for "' + agegroup +
                                                   '" age-group, so default value used.\n')

            # treatment support
            for treatment_support_type in ['_relative', '_absolute']:
                self.model_constants['econ' + param + '_treatment_support' + treatment_support_type] \
                    = self.model_constants['econ' + param + '_treatment_support']

    ''' miscellaneous methods '''

    def add_comment_to_gui_window(self, comment):
        """
        Output message to either JavaScript or Tkinter GUI.
        """

        if self.gui_console_fn:
            self.gui_console_fn('console', {'message': comment})

    def checks(self):
        """
        Not much in here as yet. However, this function is intended to contain all the data consistency checks for
        data entry.
        """

        # check that all entered times occur after the model start time
        for time_param in self.model_constants:
            if time_param[-5:] == '_time' and '_step_time' not in time_param:
                assert self.model_constants[time_param] >= self.model_constants['start_time'], \
                    '% is before model start time' % self.model_constants[time_param]
