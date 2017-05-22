
"""
All TB-specific model (or models) should be coded in this file, including the interventions applied to them.

Time unit throughout: years
Compartment unit throughout: patients

Nested inheritance from BaseModel and StratifiedModel in base.py - the former sets some fundamental methods for
creating intercompartmental flows, costs, etc., while the latter sets down the approach to population stratification.
"""


from scipy import exp, log
from autumn.base import BaseModel, StratifiedModel
import copy


def label_intersects_tags(label, tags):

    """
    Primarily for use in force of infection calculation to determine whether a compartment is infectious.

    Args:
        label: Generally a compartment label.
        tags: Tag for whether label is to be counted.
    Returns:
        Boolean for whether any of the tags are in the label.
    """

    for tag in tags:
        if tag in label: return True
    return False


def find_outcome_proportions_by_period(proportion, early_period, total_period):

    """
    Split one treatment outcome proportion (e.g. default, death) over multiple time periods.

    Args:
        proportion: Total proportion to be split
        early_period: Early time period
        total_period: Late time period
    Returns:
        early_proportion: Proportion allocated to early time period
        late_proportion: Proportion allocated to late time period
    """

    if proportion > 1. or proportion < 0.:
        raise Exception('Proportion parameter not between zero and one.')
    # To avoid errors where the proportion is exactly one (although the function isn't really intended for this):
    elif proportion > .99:
        early_proportion = 0.99
    else:
        early_proportion = 1. - exp(log(1. - proportion) * early_period / total_period)
    late_proportion = proportion - early_proportion
    return early_proportion, late_proportion


class ConsolidatedModel(StratifiedModel):

    """
    The transmission dynamic model to underpin all AuTuMN analyses.
    Inherits from BaseModel and then StratifiedModel, which is intended to be general to any infectious disease.
    All TB-specific methods and structures are contained in this model.
    Methods are written to be adaptable to any model structure selected (through the __init__ arguments).
    Time variant parameters that are optional (which mostly consists of optional interventions that will be required in
    some countries and not others) are written as "plug-ins" wherever possible (meaning that the model should still run
    if the parameter isn't included in inputs).

    The workflow of the simulation is structured in the following order:
        1. Defining the model structure
        2. Initialising compartments
        3. Setting parameters
        4. Calculating derived parameters
        5. Assigning names to intercompartmental flows
        6. The main loop over simulation time-points:
                a. Extracting scale-up variables
                b. Calculating variable flow rates
                c. Calculating diagnostic variables
                    (b. and  c. can be calculated from a compartment values,
                    parameters, scale-up functions or a combination of these)
        7. Calculating the diagnostic solutions
    """

    def __init__(self, scenario=None, inputs=None, gui_inputs=None):
        """
        Instantiation, partly inherited from the lower level model objects through nested inheritance.

        Args:
            scenario: Single number for the scenario to run (with None meaning baseline)
            inputs: Non-GUI inputs from data_processing
            gui_inputs: GUI inputs from Tkinter or JS GUI
        """

        # Inherited initialisations
        BaseModel.__init__(self)
        StratifiedModel.__init__(self)

        # Model attributes to be set directly to attributes of the inputs object
        self.inputs = inputs
        for attribute in \
                ['organ_status', 'strains', 'riskgroups', 'agegroups']:
            setattr(self, attribute, getattr(inputs, attribute))

        # Model attributes to set to just the relevant scenario key from an inputs dictionary
        self.scenario = scenario
        for attribute in \
                ['relevant_interventions', 'scaleup_fns', 'interventions_to_cost']:
            setattr(self, attribute, getattr(inputs, attribute)[scenario])

        # start_time can't be left as a model constant as it needs to be set for each scenario through the model runner
        self.start_time = inputs.model_constants['start_time']

        # Model attributes to be set directly to attributes from the GUI object
        for attribute in \
                ['is_lowquality', 'is_amplification', 'is_misassignment', 'is_timevariant_organs',
                 'country', 'time_step', 'integration_method']:
            setattr(self, attribute, gui_inputs[attribute])
        if self.is_misassignment: assert self.is_amplification, 'Misassignment requested without amplification'

        # Set fixed parameters from inputs object
        for key, value in inputs.model_constants.items():
            if type(value) == float: self.set_parameter(key, value)

        for timevariant in self.relevant_interventions:
            if 'int_prop_ipt_age' in timevariant:
                self.relevant_interventions += ['agestratified_ipt']
            elif 'int_prop_ipt' in timevariant and 'community_ipt' not in timevariant:
                self.relevant_interventions += ['ipt']
        for riskgroup in self.riskgroups:
            for program in ['_xpertacf', '_cxrxpertacf']:
                if 'int_prop' + program + riskgroup in self.scaleup_fns:
                    self.relevant_interventions += ['int_prop' + program + riskgroup]

        # Define model compartmental structure (note that compartment initialisation is in base.py)
        self.define_model_structure()

        # Treatment outcomes
        self.outcomes = ['_success', '_death', '_default']
        self.treatment_stages = ['_infect', '_noninfect']

        # Intervention and economics-related initialisiations
        if self.eco_drives_epi: self.distribute_funding_across_years()

        # Work out what we're doing with organ status and variation of detection rates by organ status
        # ** This should probably be moved to the data processing module
        self.vary_detection_by_organ = gui_inputs['is_vary_detection_by_organ']
        if len(self.organ_status) == 1 and self.vary_detection_by_organ:
            self.vary_detection_by_organ = False
            print('Requested variation by organ status turned off, as model is unstratified by organ status.')
        if len(self.organ_status) > 1 and 'int_prop_xpert' in self.relevant_interventions \
                and not self.vary_detection_by_organ:
            self.vary_detection_by_organ = True
            print('Variation in detection by organ status added although not requested, for Xpert implementation.')
        elif len(self.organ_status) == 1 and 'int_prop_xpert' in self.relevant_interventions:
            print('Effect of Xpert on smear-negative detection not simulated as model unstratified by organ status.')

        self.detection_algorithm_ceiling = .95
        self.organs_for_detection = ['']
        if self.vary_detection_by_organ:
            self.organs_for_detection = self.organ_status

        # Boolean automatically set according to whether any form of ACF or intensive screening is being implemented
        self.vary_detection_by_riskgroup = False
        for timevariant in self.scaleup_fns:
            if 'acf' in timevariant or 'intensive_screening' in timevariant: self.vary_detection_by_riskgroup = True

        self.riskgroups_for_detection = ['']
        if self.vary_detection_by_riskgroup:
            self.riskgroups_for_detection = self.riskgroups

        self.ngo_groups = ['_ruralpoor']  # list of risk groups affected by ngo activities for detection

        # Whether short-course MDR-TB regimen improves outcomes
        self.shortcourse_improves_outcomes = False

        # Mixing matrix
        self.vary_force_infection_by_riskgroup = True  # whether to incorporate heterogeneous mixing by risk group
        if self.vary_force_infection_by_riskgroup:
            self.mixing = {}
            self.create_mixing_matrix()

        # Create time ticker
        self.next_time_point = copy.copy(self.start_time)

    def define_model_structure(self):

        # All compartmental disease stages
        self.compartment_types = ['susceptible_fully', 'susceptible_vac', 'susceptible_treated', 'latent_early',
                                  'latent_late', 'active', 'detect', 'missed', 'treatment_infect',
                                  'treatment_noninfect']
        if self.is_lowquality: self.compartment_types += ['lowquality']
        if 'int_prop_novel_vaccination' in self.relevant_interventions:
            self.compartment_types += ['susceptible_novelvac']

        # Compartments that contribute to force of infection calculations
        self.infectious_tags = ['active', 'missed', 'detect', 'treatment_infect', 'lowquality']

        # Initialise compartments
        self.initial_compartments = {}
        for compartment in self.compartment_types:
            if compartment in self.inputs.model_constants:
                self.initial_compartments[compartment] \
                    = self.inputs.model_constants[compartment]

    def initialise_compartments(self):

        """
        Initialise all compartments to zero and then populate with the requested values.
        """

        # Initialise to zero
        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for compartment in self.compartment_types:

                    # Replicate susceptible for age and risk groups
                    if 'susceptible' in compartment: self.set_compartment(compartment + riskgroup + agegroup, 0.)

                    # Replicate latent classes for age-groups, risk groups and strains
                    elif 'latent' in compartment:
                        for strain in self.strains:
                            self.set_compartment(compartment + strain + riskgroup + agegroup, 0.)

                    # Replicate active classes for age-groups, risk groups, strains and organs
                    elif 'active' in compartment or 'missed' in compartment or 'lowquality' in compartment:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                self.set_compartment(compartment + organ + strain + riskgroup + agegroup, 0.)

                    # Replicate treatment classes for age-groups, risk groups, strains, organs and assigned strains
                    else:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                if self.is_misassignment:
                                    for assigned_strain in self.strains:
                                        self.set_compartment(compartment + organ + strain + '_as' + assigned_strain[1:]
                                                             + riskgroup + agegroup,
                                                             0.)
                                else:
                                    self.set_compartment(compartment + organ + strain + riskgroup + agegroup, 0.)

        # Find starting proportions for risk groups
        if len(self.riskgroups) == 1:
            start_risk_prop = {'': 1.}
        else:
            start_risk_prop = {'_norisk': 1.}
            for riskgroup in self.riskgroups:
                if riskgroup != '_norisk':
                    start_risk_prop[riskgroup] \
                        = self.scaleup_fns['riskgroup_prop' + riskgroup](self.inputs.model_constants['start_time'])
                    start_risk_prop['_norisk'] -= start_risk_prop[riskgroup]

        # Find starting strain for compartment initialisation
        if self.strains == ['']:
            default_start_strain = ''
        else:
            default_start_strain = '_ds'

        # Arbitrarily split equally by age-groups and organ status, but avoid starting with any resistant strains
        for compartment in self.compartment_types:
            if compartment in self.initial_compartments:
                for agegroup in self.agegroups:
                    for riskgroup in self.riskgroups:
                        if 'susceptible_fully' in compartment:
                            self.set_compartment(compartment + riskgroup + agegroup,
                                                 self.initial_compartments[compartment] * start_risk_prop[riskgroup]
                                                 / len(self.agegroups))
                        elif 'latent' in compartment:
                            self.set_compartment(compartment + default_start_strain + riskgroup + agegroup,
                                                 self.initial_compartments[compartment] * start_risk_prop[riskgroup]
                                                 / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(compartment + organ + default_start_strain + riskgroup
                                                     + agegroup,
                                                     self.initial_compartments[compartment]
                                                     * start_risk_prop[riskgroup]
                                                     / len(self.organ_status) / len(self.agegroups))

    def create_mixing_matrix(self):

        """
        Creates model attribute for mixing between population risk groups, for use in calculate_force_infection_vars
        method below only.

        *** Would be nice to make this more general - currently dependent on all inter-group mixing proportions being
        defined in the inputs spreadsheet. ***
        """

        for from_riskgroup in self.riskgroups:
            self.mixing[from_riskgroup] = {}
            for to_riskgroup in self.riskgroups:
                if 'prop' + from_riskgroup + '_mix' + to_riskgroup in self.params:
                    self.mixing[from_riskgroup][to_riskgroup] \
                        = self.params['prop' + from_riskgroup + '_mix' + to_riskgroup]
            self.mixing[from_riskgroup][from_riskgroup] = 1. - sum(self.mixing[from_riskgroup].values())

    #######################################################
    ### Single method to process uncertainty parameters ###
    #######################################################

    def process_uncertainty_params(self):
        """
        Perform some simple parameter processing - just for those that are used as uncertainty parameters and so can't
        be processed in the data_processing module.
        """

        # find the case fatality of smear-negative TB using the relative case fatality
        self.params['tb_prop_casefatality_untreated_smearneg'] = \
            self.params['tb_prop_casefatality_untreated_smearpos'] \
            * self.params['tb_relative_casefatality_untreated_smearneg']

        # add the extrapulmonary case fatality (currently not entered from the spreadsheets)
        self.params['tb_prop_casefatality_untreated_extrapul'] \
            = self.params['tb_prop_casefatality_untreated_smearneg']

        # calculate the rates of death and recovery from the above parameters
        for organ in self.organ_status:
            self.params['tb_rate_death' + organ] \
                = self.params['tb_prop_casefatality_untreated' + organ] \
                  / self.params['tb_timeperiod_activeuntreated']
            self.params['tb_rate_recover' + organ] \
                = (1. - self.params['tb_prop_casefatality_untreated' + organ]) \
                  / self.params['tb_timeperiod_activeuntreated']

    ########################################################################
    ### Methods that calculate variables to be used in calculating flows ###
    ### (Note that all scaleup_fns have already been calculated.)        ###
    ########################################################################

    def calculate_vars(self):

        """
        The master method that calls all the other methods for the calculations of variable rates
        """

        self.ticker()
        # The parameter values are calculated from the costs, but only in the future
        if self.eco_drives_epi and self.time > self.inputs.model_constants['recent_time']: self.update_vars_from_cost()
        self.calculate_populations()
        self.calculate_birth_rates_vars()
        self.calculate_progression_vars()
        if 'int_prop_opendoors_activities' in self.relevant_interventions \
                or 'int_prop_ngo_activities' in self.relevant_interventions:
            self.adjust_case_detection_and_ipt_for_opendoors()
        if 'int_prop_decentralisation' in self.relevant_interventions:
            self.adjust_case_detection_for_decentralisation()
        if self.vary_detection_by_organ:
            self.calculate_case_detection_by_organ()
            if 'int_prop_xpert' in self.relevant_interventions:
                self.adjust_smearneg_detection_for_xpert()
        self.calculate_detect_missed_vars()
        if self.vary_detection_by_riskgroup:
            self.calculate_acf_rate()
            self.calculate_intensive_screening_rate()
            self.adjust_case_detection_for_acf()
            self.adjust_case_detection_for_intensive_screening()
        self.calculate_misassignment_detection_vars()
        if self.is_lowquality: self.calculate_lowquality_detection_vars()
        self.calculate_await_treatment_var()
        self.calculate_treatment_rates_vars()
        self.calculate_prop_infections_reachable_with_ipt()
        if 'agestratified_ipt' in self.relevant_interventions or 'ipt' in self.relevant_interventions:
            self.calculate_ipt_effect()
        self.calculate_force_infection_vars()
        self.calculate_population_sizes()

    def ticker(self):

        """
        Prints time every ten years to give a sense of progress through integration.
        """

        if self.time > self.next_time_point:
            print(int(self.time))
            self.next_time_point += 10.

    def calculate_populations(self):

        """
        Find the total absolute numbers of people in each population sub-group.
        """

        for riskgroup in self.riskgroups + ['']:
            self.vars['population' + riskgroup] = 0.
            for label in self.labels:
                if riskgroup in label:
                    self.vars['population' + riskgroup] += self.compartments[label]

    def calculate_birth_rates_vars(self):

        """
        Calculate birth rates into vaccinated and unvaccinated compartments.
        """

        # Calculate total births (also for tracking for for interventions)
        self.vars['births_total'] = self.vars['demo_rate_birth'] / 1e3 \
                                    * self.vars['population']

        # Determine vaccinated and unvaccinated proportions
        vac_props = {'vac': self.vars['int_prop_vaccination']}
        vac_props['unvac'] = 1. - vac_props['vac']
        if 'int_prop_novel_vaccination' in self.relevant_interventions:
            vac_props['novelvac'] = self.vars['int_prop_vaccination'] \
                                    * self.vars['int_prop_novel_vaccination']
            vac_props['vac'] -= vac_props['novelvac']

        # Calculate birth rates
        for riskgroup in self.riskgroups:
            for vac_status in vac_props:
                self.vars['births_' + vac_status + riskgroup] \
                    = vac_props[vac_status] * self.vars['births_total'] * self.target_risk_props[riskgroup][-1]

    def calculate_progression_vars(self):

        """
        Calculate vars for the remainder of progressions. The vars for the smear-positive and smear-negative proportions
        have already been calculated, but as all progressions have to go somewhere, we need to calculate the remainder.
        """

        # Unstratified (self.organ_status should really have length 0, but length 1 OK)
        if len(self.organ_status) < 2:
            self.vars['epi_prop'] = 1.

        # Stratified into smear-positive and smear-negative
        elif len(self.organ_status) == 2:
            self.vars['epi_prop_smearneg'] = 1. - self.vars['epi_prop_smearpos']

        # Fully stratified into smear-positive, smear-negative and extra-pulmonary
        else:
            self.vars['epi_prop_extrapul'] = 1. - self.vars['epi_prop_smearpos'] - self.vars['epi_prop_smearneg']

        # Determine variable progression rates
        for organ in self.organ_status:
            for agegroup in self.agegroups:
                for riskgroup in self.riskgroups:
                    for timing in ['_early', '_late']:
                        if riskgroup == '_diabetes':
                            self.vars['tb_rate' + timing + '_progression' + organ + riskgroup + agegroup] \
                                = self.vars['epi_prop' + organ] \
                                  * self.params['tb_rate' + timing + '_progression' + '_norisk' + agegroup] \
                                  * self.params['riskgroup_multiplier_diabetes_progression']
                        else:
                            self.vars['tb_rate' + timing + '_progression' + organ + riskgroup + agegroup] \
                                = self.vars['epi_prop' + organ] \
                                  * self.params['tb_rate' + timing + '_progression' + riskgroup + agegroup]

    def adjust_case_detection_for_decentralisation(self):

        """
        Implement the decentralisation intervention, which narrows the case detection gap between the current values
        and the idealised estimated value.
        """

        assert self.params['program_ideal_detection'] >= self.vars['program_prop_detect'], \
            'program_prop_detect should not be greater than program_ideal_detection'
        self.vars['program_prop_detect'] \
            += self.vars['int_prop_decentralisation'] \
               * (self.params['program_ideal_detection'] - self.vars['program_prop_detect'])

    def calculate_case_detection_by_organ(self):

        """
        Method to perform simple weighting on the assumption that the smear-negative and extra-pulmonary rates are less
        than the smear-positive rate by a proportion specified in program_prop_snep_relative_algorithm.
        Places a ceiling on these values, to prevent the smear-positive one going too close to (or above) one.
        """

        for parameter in ['_detect', '_algorithm_sensitivity']:
            self.vars['program_prop' + parameter + '_smearpos'] \
                = min(self.vars['program_prop' + parameter + '']
                      / (self.vars['epi_prop_smearpos']
                         + self.params['program_prop_snep_relative_algorithm']
                         * (1. - self.vars['epi_prop_smearpos'])), self.detection_algorithm_ceiling)
            for organ in ['_smearneg', '_extrapul']:
                if organ in self.organ_status:
                    self.vars['program_prop' + parameter + organ] \
                        = self.vars['program_prop' + parameter + '_smearpos'] \
                          * self.params['program_prop_snep_relative_algorithm']

    def adjust_smearneg_detection_for_xpert(self):
        """
        Adjust case case detection and algorithm sensitivity for Xpert (will only work with weighting and so is grouped
        with the previous method in calculate_vars).
        """

        for parameter in ['_detect', '_algorithm_sensitivity']:
            self.vars['program_prop' + parameter + '_smearneg'] \
                += (self.vars['program_prop' + parameter + '_smearpos'] -
                    self.vars['program_prop' + parameter + '_smearneg']) \
                   * self.params['tb_prop_xpert_smearneg_sensitivity'] * self.vars['int_prop_xpert']

    def adjust_case_detection_and_ipt_for_opendoors(self):

        # if opendoors programs are stopped, case detection and IPT coverage are reduced
        # this applies to the whole population as opposed to NGOs activities that apply to specific risk groups

        if 'int_prop_opendoors_activities' in self.relevant_interventions and \
                self.vars['int_prop_opendoors_activities'] < 1.:
            self.vars['program_prop_detect'] *= (1. - self.params['program_prop_detection_from_opendoors'])

            # adjust IPT coverage
            for agegroup in self.agegroups:
                if 'int_prop_ipt' + agegroup in self.vars:
                    self.vars['int_prop_ipt' + agegroup] *= (1. - self.params['program_prop_ipt_from_opendoors'])

    def calculate_detect_missed_vars(self):

        """"
        Calculate rates of detection and failure of detection from the programmatic report of the case detection "rate"
        (which is actually a proportion and referred to as program_prop_detect here).

        Derived by solving the following simultaneous equations:

          algorithm sensitivity = detection rate / (detection rate + missed rate)
              - and -
          detection proportion = detection rate
                / (detection rate + spont recover rate + tb death rate + natural death rate)
        """

        organs = copy.copy(self.organs_for_detection)
        if self.vary_detection_by_organ:
            organs += ['']

        for organ in organs:
            for riskgroup in [''] + self.riskgroups_for_detection:

                # Detected
                self.vars['program_rate_detect' + organ + riskgroup] \
                    = self.vars['program_prop_detect' + organ] \
                      * (1. / self.params['tb_timeperiod_activeuntreated']
                         + 1. / self.vars['demo_life_expectancy']) \
                      / (1. - self.vars['program_prop_detect' + organ])

                # Adjust detection rates for NGOs activities in specific risk groups
                if 'int_prop_ngo_activities' in self.relevant_interventions and \
                                self.vars['int_prop_ngo_activities'] < 1. and \
                                riskgroup in self.ngo_groups:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        *= 1 - self.params['program_prop_detection_from_ngo']

            # Missed (no need to loop by risk-group as ACF is the only difference here, which is applied next)
            self.vars['program_rate_missed' + organ] \
                = self.vars['program_rate_detect' + organ] \
                  * (1. - self.vars['program_prop_algorithm_sensitivity' + organ]) \
                  / max(self.vars['program_prop_algorithm_sensitivity' + organ], 1e-6)

            # Adjust for awareness raising
            if 'int_prop_awareness_raising' in self.vars:
                case_detection_ratio_with_awareness \
                    = (self.params['int_ratio_case_detection_with_raised_awareness'] - 1.) \
                      * self.vars['int_prop_awareness_raising'] + 1.
                self.vars['program_rate_missed' + organ] *= case_detection_ratio_with_awareness
                for riskgroup in [''] + self.riskgroups_for_detection:
                    self.vars['program_rate_detect' + organ + riskgroup] *= case_detection_ratio_with_awareness

    def calculate_acf_rate(self):

        """
        Calculates rates of ACF from the proportion of programmatic coverage of both smear-based and Xpert-based ACF
        (both presuming symptom-based screening before this, as in the studies on which this is based).
        Smear-based screening only detects smear-positive disease, while Xpert-based screening detects some
        smear-negative disease (incorporating a multiplier for the sensitivity of Xpert for smear-negative disease).
        Extrapulmonary disease can't be detected through ACF.
        Creates vars for both ACF in specific risk groups and for ACF in the general community (which uses '').
        """

        # Loop covers risk groups and community-wide ACF
        for riskgroup in [''] + self.riskgroups:

            # Decide whether to use the general detection proportion, or a risk-group specific one
            if 'int_prop_acf_detections_per_round' + riskgroup in self.params:
                int_prop_acf_detections_per_round \
                    = self.params['int_prop_acf_detections_per_round' + riskgroup]
            else:
                int_prop_acf_detections_per_round = self.params['int_prop_acf_detections_per_round']

            # Implement intervention
            if 'int_prop_smearacf' + riskgroup in self.relevant_interventions \
                    or 'int_prop_xpertacf' + riskgroup in self.relevant_interventions:

                # The following can't be written as self.organ_status, as it won't work for non-fully-stratified models
                for organ in ['', '_smearpos', '_smearneg', '_extrapul']:
                    self.vars['int_rate_acf' + organ + riskgroup] = 0.

                # Smear-based ACF rate
                if 'int_prop_smearacf' + riskgroup in self.relevant_interventions:
                    self.vars['int_rate_acf_smearpos' + riskgroup] \
                        += self.vars['int_prop_smearacf' + riskgroup] \
                           * int_prop_acf_detections_per_round / self.params['int_timeperiod_acf_rounds']

                # Xpert-based ACF rate for smear-positives and smear-negatives - with or without pre-screening with CXR
                for acf_type in ['xpert', 'cxrxpert']:
                    if acf_type == 'xpert':
                        cxr_sensitivity = 1.
                    else:
                        cxr_sensitivity = self.params['tb_sensitivity_cxr']
                    if 'int_prop_' + acf_type + 'acf' + riskgroup in self.relevant_interventions:
                        for organ in ['_smearpos', '_smearneg']:
                            self.vars['int_rate_acf' + organ + riskgroup] \
                                += self.vars['int_prop_' + acf_type + 'acf' + riskgroup] \
                                   * int_prop_acf_detections_per_round \
                                   / self.params['int_timeperiod_acf_rounds'] \
                                   * cxr_sensitivity

                        # Adjust smear-negative detections for Xpert's sensitivity
                        self.vars['int_rate_acf_smearneg' + riskgroup] \
                            *= self.params['tb_prop_xpert_smearneg_sensitivity'] \
                               * cxr_sensitivity

    def calculate_intensive_screening_rate(self):
        """
            Calculates rates of intensive screening from the proportion of programmatic coverage.
            Intensive screening detects smear-positive disease, and some
            smear-negative disease (incorporating a multiplier for the sensitivity of Xpert for smear-negative disease).
            Extrapulmonary disease can't be detected through intensive screening.
        """

        if 'int_prop_intensive_screening' in self.relevant_interventions:
            screened_subgroups = ['_diabetes', '_hiv'] # may be incorporated into the GUI
            # Loop covers risk groups
            for riskgroup in screened_subgroups:
                # The following can't be written as self.organ_status, as it won't work for non-fully-stratified models
                for organ in ['', '_smearpos', '_smearneg', '_extrapul']:
                    self.vars['int_rate_intensive_screening' + organ + riskgroup] = 0.

                for organ in ['_smearpos', '_smearneg']:
                    self.vars['int_rate_intensive_screening' + organ + riskgroup] \
                        += self.vars['int_prop_intensive_screening'] * \
                           self.params['int_prop_attending_clinics' + riskgroup]

                # Adjust smear-negative detections for Xpert's sensitivity
                self.vars['int_rate_intensive_screening_smearneg' + riskgroup] \
                    *= self.params['tb_prop_xpert_smearneg_sensitivity']

    def adjust_case_detection_for_acf(self):

        """
        Add ACF detection rates to previously calculated passive case detection rates, creating vars for case detection
        that are specific for organs.
        """

        for organ in self.organs_for_detection:
            for riskgroup in self.riskgroups:

                # ACF in risk groups
                if 'int_prop_smearacf' + riskgroup in self.relevant_interventions \
                        or 'int_prop_xpertacf' + riskgroup in self.relevant_interventions:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        += self.vars['int_rate_acf' + organ + riskgroup]

                # ACF in the general community
                if 'int_prop_smearacf' in self.relevant_interventions \
                        or 'int_prop_xpertacf' in self.relevant_interventions:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        += self.vars['int_rate_acf' + organ]

    def adjust_case_detection_for_intensive_screening(self):

        if 'int_prop_intensive_screening' in self.relevant_interventions:
            for organ in self.organs_for_detection:
                screened_subgroups = ['_diabetes', '_hiv']  # may be incorporated into the GUI
                for riskgroup in screened_subgroups:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        += self.vars['int_rate_intensive_screening' + organ + riskgroup]

    def calculate_misassignment_detection_vars(self):

        """
        Calculate the proportions of patients assigned to each strain. (Note that second-line DST availability refers to
        the proportion of those with first-line DST who also have second-line DST available.)
        """

        # With misassignment:
        for organ in self.organs_for_detection:
            for riskgroup in self.riskgroups_for_detection:
                if self.is_misassignment:

                    prop_firstline = self.vars['program_prop_firstline_dst']

                    # Add effect of improve_dst program
                    if 'int_prop_improve_dst' in self.relevant_interventions:
                        prop_firstline += (1. - prop_firstline) * self.vars['int_prop_improve_dst']

                    # Add effect of Xpert on identification, assuming that independent distribution to conventional DST
                    if 'int_prop_xpert' in self.relevant_interventions:
                        prop_firstline += (1. - prop_firstline) * self.vars['int_prop_xpert']


                    # Determine rates of identification/misidentification as each strain
                    self.vars['program_rate_detect' + organ + riskgroup + '_ds_asds'] \
                        = self.vars['program_rate_detect' + organ + riskgroup]
                    self.vars['program_rate_detect' + organ + riskgroup + '_ds_asmdr'] = 0.
                    self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asds'] \
                        = (1. - prop_firstline) * self.vars['program_rate_detect' + organ + riskgroup]
                    self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asmdr'] \
                        = prop_firstline * self.vars['program_rate_detect' + organ + riskgroup]

                    # If a third strain is present
                    if len(self.strains) > 2:
                        prop_secondline = self.vars['program_prop_secondline_dst']
                        self.vars['program_rate_detect' + organ + riskgroup + '_ds_asxdr'] = 0.
                        self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asxdr'] = 0.
                        self.vars['program_rate_detect' + organ + riskgroup + '_xdr_asds'] \
                            = (1. - prop_firstline) * self.vars['program_rate_detect' + organ + riskgroup]
                        self.vars['program_rate_detect' + organ + riskgroup + '_xdr_asmdr'] \
                            = prop_firstline \
                              * (1. - prop_secondline) * self.vars['program_rate_detect' + organ + riskgroup]
                        self.vars['program_rate_detect' + organ + riskgroup + '_xdr_asxdr'] \
                            = prop_firstline * prop_secondline * self.vars['program_rate_detect' + organ + riskgroup]

                # Without misassignment, everyone is correctly allocated
                else:
                    for strain in self.strains:
                        self.vars['program_rate_detect' + organ + riskgroup + strain + '_as' + strain[1:]] \
                            = self.vars['program_rate_detect' + organ + riskgroup]

    def calculate_lowquality_detection_vars(self):

        """
        Calculate rate of entry to low-quality care ffom the proportion of treatment administered in low-quality sector.
        Note that this now means that the case detection proportion only applies to those with access to care, so
        that proportion of all cases isn't actually detected.
        """

        prop_lowqual = self.vars['program_prop_lowquality']
        if 'int_prop_engage_lowquality' in self.relevant_interventions:
            prop_lowqual *= (1. - self.vars['int_prop_engage_lowquality'])

        # Note that there is still a program_rate_detect var even if detection is varied by organ and/or risk group
        self.vars['program_rate_enterlowquality'] \
            = self.vars['program_rate_detect'] * prop_lowqual / (1. - prop_lowqual)

    def calculate_await_treatment_var(self):

        """
        Take the reciprocal of the waiting times to calculate the flow rate to start treatment after detection.
        Note that the default behaviour for a single strain model is to use the waiting time for smear-positives.
        Also weight the time period
        """

        # If only one organ stratum
        if len(self.organ_status) == 1:
            self.vars['program_rate_start_treatment'] \
                = 1. / self.vars['program_timeperiod_await_treatment_smearpos']

        # Organ stratification
        else:
            for organ in self.organ_status:

                # Adjust smear-negative for Xpert coverage
                if organ == '_smearneg' and 'int_prop_xpert' in self.relevant_interventions:
                    prop_xpert = self.vars['int_prop_xpert']
                    self.vars['program_rate_start_treatment_smearneg'] = \
                        1. / (self.vars['program_timeperiod_await_treatment_smearneg'] * (1. - prop_xpert)
                              + self.params['int_timeperiod_await_treatment_smearneg_xpert'] * prop_xpert)

                # Do other organ stratifications (including smear-negative if Xpert not an intervention)
                else:
                    self.vars['program_rate_start_treatment' + organ] = \
                        1. / self.vars['program_timeperiod_await_treatment' + organ]

    def calculate_treatment_rates_vars(self):

        """
        Work out rates of progression through treatment by stage of treatment from the proportions provided for success
        and death.
        """

        treatments = copy.copy(self.strains)
        for strain in treatments:

            # Find baseline treatment period for total duration and for period infectious
            for treatment_stage in ['', '_infect']:
                self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                    = self.params['tb_timeperiod' + treatment_stage + '_ontreatment' + strain]

            # Adapt treatment periods for short course regimen
            if strain == '_mdr' and 'int_prop_shortcourse_mdr' in self.relevant_interventions:
                relative_treatment_duration_mdr \
                    = 1. - self.vars['int_prop_shortcourse_mdr'] \
                           * (1. - self.params['int_prop_shortcourse_mdr_relativeduration'])
                for treatment_stage in ['', '_infect']:
                    self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                        *= relative_treatment_duration_mdr

                # Adapt treatment outcomes for short course regimen
                if self.shortcourse_improves_outcomes:
                    for outcome in ['_success', '_death']:
                        self.vars['program_prop_treatment' + outcome + '_mdr'] \
                            += (self.params['program_prop_treatment' + outcome + '_shortcoursemdr']
                                - self.vars['program_prop_treatment' + outcome + '_mdr']) \
                               * self.vars['int_prop_shortcourse_mdr']

            # Add some extra treatment success if the treatment support int is active
            if 'int_prop_treatment_support' in self.relevant_interventions:
                self.vars['program_prop_treatment_success' + strain] \
                    += (1. - self.vars['program_prop_treatment_success' + strain]) \
                       * self.params['int_prop_treatment_support_improvement'] \
                       * self.vars['int_prop_treatment_support']
                self.vars['program_prop_treatment_death' + strain] \
                    -= self.vars['program_prop_treatment_death' + strain] \
                       * self.params['int_prop_treatment_support_improvement'] \
                       * self.vars['int_prop_treatment_support']

            # Add some extra treatment success if food vouchers are provided to treated cases
            if 'int_prop_food_voucher' + strain in self.relevant_interventions:
                self.vars['program_prop_treatment_success' + strain] \
                    += (1. - self.vars['program_prop_treatment_success' + strain]) \
                       * self.params['int_prop_food_voucher_improvement'] \
                       * self.vars['int_prop_food_voucher' + strain]
                self.vars['program_prop_treatment_death' + strain] \
                    -= self.vars['program_prop_treatment_death' + strain] \
                       * self.params['int_prop_food_voucher_improvement'] \
                       * self.vars['int_prop_food_voucher' + strain]

            # Calculate the default proportion as the remainder from success and death
            self.vars['program_prop_treatment_default' + strain] \
                = 1. - self.vars['program_prop_treatment_success' + strain] \
                  - self.vars['program_prop_treatment_death' + strain]

            # Find non-infectious period from infectious and total
            self.vars['tb_timeperiod_noninfect_ontreatment' + strain] \
                = self.vars['tb_timeperiod_ontreatment' + strain] \
                  - self.vars['tb_timeperiod_infect_ontreatment' + strain]

            # Find the proportion of deaths/defaults during the infectious and non-infectious stages
            props = {}
            for outcome in self.outcomes[1:]:
                props['_infect'], props['_noninfect'] \
                    = find_outcome_proportions_by_period(self.vars['program_prop_treatment' + outcome + strain],
                                                         self.vars['tb_timeperiod_infect_ontreatment' + strain],
                                                         self.vars['tb_timeperiod_ontreatment' + strain])
                for treatment_stage in props:
                    self.vars['program_prop_treatment' + outcome + treatment_stage + strain] = props[treatment_stage]

            for treatment_stage in self.treatment_stages:

                # Find the success proportions
                self.vars['program_prop_treatment_success' + treatment_stage + strain] \
                    = 1. - self.vars['program_prop_treatment_default' + treatment_stage + strain] \
                      - self.vars['program_prop_treatment_death' + treatment_stage + strain]

                # Find the corresponding rates from the proportions
                for outcome in self.outcomes:
                    self.vars['program_rate' + outcome + treatment_stage + strain] \
                        = 1. / self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                          * self.vars['program_prop_treatment' + outcome + treatment_stage + strain]

                # Split default according to whether amplification occurs (if not the most resistant strain)
                if self.is_amplification:
                    self.vars['program_rate_default' + treatment_stage + '_amplify' + strain] \
                        = self.vars['program_rate_default' + treatment_stage + strain] \
                          * self.vars['epi_prop_amplification']
                    self.vars['program_rate_default' + treatment_stage + '_noamplify' + strain] \
                        = self.vars['program_rate_default' + treatment_stage + strain] \
                          * (1. - self.vars['epi_prop_amplification'])

            # create new vars for misassigned individuals
            if len(self.strains) > 1 and self.is_misassignment and strain != '_ds':
                for treated_as in treatments:  # for each strain
                    if treated_as != strain:  # misassigned strain has to be different from the actual strain
                        if self.strains.index(treated_as) < self.strains.index(
                                strain):  # if treated with weaker regimen
                            props = {}
                            for outcome in self.outcomes[1:]:
                                treatment_type = strain + '_as' + treated_as[1:]
                                if outcome == '_default':
                                    self.params['program_prop_treatment' + outcome + treatment_type] = 1. - \
                                        self.params['program_prop_treatment_success' + treatment_type] - \
                                        self.params['program_prop_treatment_death' + treatment_type]
                                props['_infect'], props['_noninfect'] \
                                    = find_outcome_proportions_by_period(
                                    self.params['program_prop_treatment' + outcome + treatment_type],
                                    self.vars['tb_timeperiod_infect_ontreatment' + treated_as],
                                    self.vars['tb_timeperiod_ontreatment' + treated_as])
                                for treatment_stage in props:
                                    self.vars[
                                        'program_prop_treatment' + outcome + treatment_stage + treatment_type] = \
                                    props[treatment_stage]

                            for treatment_stage in self.treatment_stages:
                                # Find the success proportions
                                self.vars['program_prop_treatment_success' + treatment_stage + treatment_type] \
                                    = 1. - self.vars[
                                    'program_prop_treatment_default' + treatment_stage + treatment_type] \
                                      - self.vars['program_prop_treatment_death' + treatment_stage + treatment_type]

                                # Find the corresponding rates from the proportions
                                for outcome in self.outcomes:
                                    self.vars['program_rate' + outcome + treatment_stage + treatment_type] \
                                        = 1. / self.vars[
                                        'tb_timeperiod' + treatment_stage + '_ontreatment' + treated_as] \
                                          * self.vars[
                                              'program_prop_treatment' + outcome + treatment_stage + treatment_type]

                                # Split default according to whether amplification occurs (if not the most resistant strain)
                                if self.is_amplification:
                                    self.vars[
                                        'program_rate_default' + treatment_stage + '_amplify' + treatment_type] \
                                        = self.vars['program_rate_default' + treatment_stage + treatment_type] \
                                          * self.vars['epi_prop_amplification']
                                    self.vars[
                                        'program_rate_default' + treatment_stage + '_noamplify' + treatment_type] \
                                        = self.vars['program_rate_default' + treatment_stage + treatment_type] \
                                          * (1. - self.vars['epi_prop_amplification'])

    def calculate_prop_infections_reachable_with_ipt(self):

        """
        Calculates the proportion of new infections that could potentially be targeted with IPT.
        Obtained by multiplying the proportion of active cases that are detected with the proportion of infections
        that occur within the household
        """

        self.vars['tb_prop_infections_reachable_with_ipt'] = \
            self.calculate_aggregate_outgoing_proportion('active', 'detect') * \
            self.params['tb_prop_infections_in_household']

    def calculate_ipt_effect(self):
        """
        Method to estimate the proportion of infections averted through the IPT program - as the proportion of all cases
        detected by the high quality sector, multiplied by the proportion of infections in the household (giving the
        proportion of all infections we can target), multiplied by the coverage of the program (in each age-group) and
        the effectiveness of treatment.
        """

        for agegroup in self.agegroups:
            self.vars['prop_infections_averted_ipt' + agegroup] = 0.
            if 'int_prop_ipt' + agegroup in self.vars:
                self.vars['prop_infections_averted_ipt' + agegroup] \
                    = self.vars['tb_prop_infections_reachable_with_ipt'] \
                      * self.vars['int_prop_ipt' + agegroup] \
                      * self.params['tb_prop_ipt_effectiveness']
            else:
                self.vars['prop_infections_averted_ipt' + agegroup] = 0.

    def calculate_force_infection_vars(self):
        """
        Calculate force of infection independently for each strain, incorporating partial immunity and infectiousness.
        First calculate the effective infectious population (incorporating infectiousness by organ involvement), then
        calculate the raw force of infection, then adjust for various levels of susceptibility.
        """

        # find the effective infectious population for each strain
        for strain in self.strains:

            # initialise infectious population vars as needed
            if self.vary_force_infection_by_riskgroup:
                for riskgroup in self.riskgroups:
                    self.vars['effective_infectious_population' + strain + riskgroup] = 0.
            else:
                self.vars['effective_infectious_population' + strain] = 0.

            # loop through compartments, skipping on as soon as possible if not relevant
            for label in self.labels:
                if strain not in label and strain != '':
                    continue
                for organ in self.organ_status:
                    if organ not in label and organ != '':
                        continue
                    for agegroup in self.agegroups:
                        if agegroup not in label and agegroup != '':
                            continue

                        # calculate effective infectious population with stratification for risk-group
                        if self.vary_force_infection_by_riskgroup:
                            for riskgroup in self.riskgroups:
                                if riskgroup not in label:
                                    continue
                                elif label_intersects_tags(label, self.infectious_tags):
                                    for source_riskgroup in self.riskgroups:

                                        # adjustment for increased infectiousness of risk groups as required
                                        if 'riskgroup_multiplier_force_infection' + source_riskgroup in self.params:
                                            riskgroup_multiplier_force_infection \
                                                = self.params['riskgroup_multiplier_force_infection' + source_riskgroup]
                                        else:
                                            riskgroup_multiplier_force_infection = 1.

                                        # calculate effective infectious population for each risk group
                                        self.vars['effective_infectious_population' + strain + riskgroup] \
                                            += self.params['tb_multiplier_force' + organ] \
                                               * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                               * self.compartments[label] \
                                               * self.mixing[riskgroup][source_riskgroup] \
                                               * riskgroup_multiplier_force_infection

                        # now without risk-group stratification
                        else:
                            if label_intersects_tags(label, self.infectious_tags):
                                self.vars['effective_infectious_population' + strain] \
                                    += self.params['tb_multiplier_force' + organ] \
                                       * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                       * self.compartments[label]

            # to loop over all risk groups if needed, or otherwise to just run once
            if self.vary_force_infection_by_riskgroup:
                force_riskgroups = copy.copy(self.riskgroups)
            else:
                force_riskgroups = ['']

            # calculate force of infection unadjusted for immunity/susceptibility
            for riskgroup in force_riskgroups:
                for agegroup in self.agegroups:
                    if 'prop_infections_averted_ipt' + agegroup in self.vars and 'dr' not in strain:
                        coverage_multiplier_ngo_stopped = 1.
                        if 'int_prop_ngo_activities' in self.relevant_interventions and \
                                        self.vars['int_prop_ngo_activities'] < 1. and \
                                        riskgroup in self.ngo_groups:
                            coverage_multiplier_ngo_stopped = (1. - self.params['int_prop_ipt_from_ngo'])
                        ipt_infection_modifier = 1. - coverage_multiplier_ngo_stopped * \
                                                      self.vars['prop_infections_averted_ipt' + agegroup]

                    else:
                        ipt_infection_modifier = 1.
                    self.vars['rate_force' + strain + riskgroup + agegroup] \
                        = self.params['tb_n_contact'] \
                          * self.vars['effective_infectious_population' + strain + riskgroup] \
                          / self.vars['population' + riskgroup] \
                          * ipt_infection_modifier

                    # if any modifications to transmission parameter to be made over time
                    if 'transmission_modifier' in self.vars:
                        self.vars['rate_force' + strain + riskgroup + agegroup] *= self.vars['transmission_modifier']

                    # adjust for immunity in various groups
                    force_types = ['_vac', '_latent']
                    if 'int_prop_novel_vaccination' in self.relevant_interventions:
                        force_types += ['_novelvac']
                    for force_type in force_types:
                        self.vars['rate_force' + force_type + strain + riskgroup + agegroup] \
                            = self.params['tb_multiplier' + force_type + '_protection'] \
                              * self.vars['rate_force' + strain + riskgroup + agegroup]

    def calculate_population_sizes(self):
        """
        Calculate the size of the populations to which each intervention is applicable, for use in generating
        cost-coverage curves.
        """

        # treatment support
        if 'int_prop_treatment_support' in self.relevant_interventions:
            self.vars['popsize_treatment_support'] = 0.
            for compartment in self.compartments:
                if 'treatment_' in compartment:
                    self.vars['popsize_treatment_support'] += self.compartments[compartment]

        # food vouchers
        for strain in self.strains:
            if 'int_prop_food_voucher' + strain in self.relevant_interventions:
                self.vars['popsize_food_voucher' + strain] = 0.
                for compartment in self.compartments:
                    if 'treatment_' in compartment and strain + '_' in compartment:
                        self.vars['popsize_food_voucher' + strain] += self.compartments[compartment]

        # IPT: popsize defined as the household contacts of active cases identified by the high-quality sector
        for agegroup in self.agegroups:
            self.vars['popsize_ipt' + agegroup] = 0.
            for strain in self.strains:
                for from_label, to_label, rate in self.var_transfer_rate_flows:
                    if 'latent_early' in to_label and strain in to_label and agegroup in to_label:
                        self.vars['popsize_ipt' + agegroup] += self.compartments[from_label] \
                                              * self.vars[rate]

        # BCG (So simple that it's almost unnecessary, but needed for loops over int names)
        self.vars['popsize_vaccination'] = self.vars['births_total']

        # Xpert and improve DST - all presentations for assessment for active TB
        for active_tb_presentations_intervention in ['xpert', 'improve_dst']:
            if 'int_prop_' + active_tb_presentations_intervention in self.relevant_interventions:
                self.vars['popsize_' + active_tb_presentations_intervention] = 0.
                for agegroup in self.agegroups:
                    for riskgroup in self.riskgroups:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                if self.vary_detection_by_organ:
                                    detection_organ = organ
                                else:
                                    detection_organ = ''
                                self.vars['popsize_' + active_tb_presentations_intervention] \
                                    += self.vars['program_rate_detect' + detection_organ + riskgroup]\
                                       * self.compartments['active' + organ + strain + riskgroup + agegroup] \
                                       * (self.params['int_number_tests_per_tb_presentation'] + 1.)

        # improve DST in Bulgaria - the number of culture-positive cases
        if 'int_prop_bulgaria_improve_dst' in self.relevant_interventions:
            self.vars['popsize_bulgaria_improve_dst'] = 0.
            for agegroup in self.agegroups:
                for riskgroup in self.riskgroups:
                    for strain in self.strains:
                        detection_organ = ''
                        if self.vary_detection_by_organ:
                            detection_organ = '_smearpos'
                        self.vars['popsize_bulgaria_improve_dst'] \
                            += self.vars['program_rate_detect' + detection_organ + riskgroup] \
                               * self.compartments['active_smearpos' + strain + riskgroup + agegroup]
                        if self.vary_detection_by_organ:
                            detection_organ = '_smearneg'
                        self.vars['popsize_bulgaria_improve_dst'] \
                            += self.vars['program_rate_detect' + detection_organ + riskgroup] \
                               * self.compartments['active_smearneg' + strain + riskgroup + agegroup] \
                               * self.params['tb_prop_smearneg_culturepos']

        # ACF
        for riskgroup in [''] + self.riskgroups:

            # Allow proportion of persons actually receiving the screening to vary by risk-group, as per user inputs
            if 'int_prop_population_screened' + riskgroup in self.params:
                int_prop_population_screened = self.params['int_prop_population_screened' + riskgroup]
            else:
                int_prop_population_screened = self.params['int_prop_population_screened']
            for acf_type in ['_smearacf', '_xpertacf','_cxrxpertacf']:
                if 'int_prop' + acf_type + riskgroup in self.relevant_interventions:
                    self.vars['popsize' + acf_type + riskgroup] = 0.
                    for compartment in self.compartments:
                        if riskgroup == '' or riskgroup in compartment:
                            self.vars['popsize' + acf_type + riskgroup] \
                                += self.compartments[compartment] \
                                   / self.params['int_timeperiod_acf_rounds'] \
                                   * int_prop_population_screened

        # intensive_screening
        # popsize includes all active TB cases of targeted groups (HIV and diabetes) that attend specific clinics
        if 'int_prop_intensive_screening' in self.relevant_interventions:
            self.vars['popsize_intensive_screening'] = 0.
            screened_subgroups = ['_diabetes', '_hiv']  # may be incorporated into the GUI
            # Loop covers risk groups
            for riskgroup in screened_subgroups:
                for compartment in self.compartments:
                    if riskgroup in compartment and 'active' in compartment:
                        self.vars['popsize_intensive_screening'] \
                            += self.compartments[compartment] \
                               * self.params['int_prop_attending_clinics' + riskgroup]

        # decentralisation and engage low-quality sector
        adjust_lowquality = True
        all_actives_popsize = 0.
        for compartment in self.compartments:
            if 'susceptible_' not in compartment and 'latent_' not in compartment:
                all_actives_popsize += self.compartments[compartment]
        if 'decentralisation' in self.interventions_to_cost:
            self.vars['popsize_decentralisation'] = all_actives_popsize
        if 'engage_lowquality' in self.interventions_to_cost:
            self.vars['popsize_engage_lowquality'] = all_actives_popsize
            if adjust_lowquality: self.vars['popsize_engage_lowquality'] *= self.vars['program_prop_lowquality']

        # shortcourse MDR-TB regimen
        if 'int_prop_shortcourse_mdr' in self.relevant_interventions:
            self.vars['popsize_shortcourse_mdr'] = 0.
            for compartment in self.compartments:
                if 'treatment' in compartment and '_mdr' in compartment:
                    self.vars['popsize_shortcourse_mdr'] += self.compartments[compartment]




    ################################################################
    ### Methods that calculate the flows of all the compartments ###
    ################################################################

    def set_flows(self):

        """
        Call all the intercompartmental flow setting methods in turn.
        """

        self.set_birth_flows()
        if len(self.agegroups) > 0: self.set_ageing_flows()
        self.set_infection_flows()
        self.set_progression_flows()
        self.set_natural_history_flows()
        self.set_fixed_programmatic_flows()
        self.set_variable_programmatic_flows()
        self.set_detection_flows()
        self.set_treatment_flows()
        # if 'agestratified_ipt' in self.relevant_interventions or 'ipt' in self.relevant_interventions:
        #     self.set_ipt_flows()

    def set_birth_flows(self):

        """
        Set birth (or recruitment) flows by vaccination status (including novel vaccination if implemented).
        """

        for riskgroup in self.riskgroups:
            self.set_var_entry_rate_flow(
                'susceptible_fully' + riskgroup + self.agegroups[0], 'births_unvac' + riskgroup)
            self.set_var_entry_rate_flow(
                'susceptible_vac' + riskgroup + self.agegroups[0], 'births_vac' + riskgroup)
            if 'int_prop_novel_vaccination' in self.relevant_interventions:
                self.set_var_entry_rate_flow('susceptible_novelvac'
                                             + riskgroup + self.agegroups[0], 'births_novelvac' + riskgroup)

    def set_infection_flows(self):

        """
        Set force of infection flows that were estimated by strain in calculate_force_infection_vars above.
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:

                force_riskgroup = ''
                if self.vary_force_infection_by_riskgroup:
                    force_riskgroup = riskgroup

                for strain in self.strains:

                    # Set infection rates according to susceptibility status
                    self.set_var_transfer_rate_flow(
                        'susceptible_fully' + riskgroup + agegroup,
                        'latent_early' + strain + riskgroup + agegroup,
                        'rate_force' + strain + force_riskgroup + agegroup)
                    self.set_var_transfer_rate_flow(
                        'susceptible_vac' + riskgroup + agegroup,
                        'latent_early' + strain + riskgroup + agegroup,
                        'rate_force_vac' + strain + force_riskgroup + agegroup)
                    self.set_var_transfer_rate_flow(
                        'susceptible_treated' + riskgroup + agegroup,
                        'latent_early' + strain + riskgroup + agegroup,
                        'rate_force_vac' + strain + force_riskgroup + agegroup)
                    self.set_var_transfer_rate_flow(
                        'latent_late' + strain + riskgroup + agegroup,
                        'latent_early' + strain + riskgroup + agegroup,
                        'rate_force_latent' + strain + force_riskgroup + agegroup)

                    # Novel vaccination
                    if 'int_prop_novel_vaccination' in self.relevant_interventions:
                        self.set_var_transfer_rate_flow(
                            'susceptible_novelvac' + riskgroup + agegroup,
                            'latent_early' + strain + riskgroup + agegroup,
                            'rate_force_novelvac' + strain + force_riskgroup + agegroup)

    def set_progression_flows(self):

        """
        Set rates of progression from latency to active disease, with rates differing by organ status.
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for strain in self.strains:

                        # Stabilisation
                        self.set_fixed_transfer_rate_flow(
                            'latent_early' + strain + riskgroup + agegroup,
                            'latent_late' + strain + riskgroup + agegroup,
                            'tb_rate_stabilise' + riskgroup + agegroup)

                        for organ in self.organ_status:

                            # Now smear-pos/smear-neg is always a var, even when it's constant
                            self.set_var_transfer_rate_flow(
                                'latent_early' + strain + riskgroup + agegroup,
                                'active' + organ + strain + riskgroup + agegroup,
                                'tb_rate_early_progression' + organ + riskgroup + agegroup)
                            self.set_var_transfer_rate_flow(
                                'latent_late' + strain + riskgroup + agegroup,
                                'active' + organ + strain + riskgroup + agegroup,
                                'tb_rate_late_progression' + organ + riskgroup + agegroup)

    def set_natural_history_flows(self):

        """
        Set flows for progression through active disease to either recovery or death.
        """

        # Determine the compartments to which natural history flows apply
        active_compartments = ['active', 'missed']
        if self.is_lowquality:
            active_compartments += ['lowquality']
        if not self.is_misassignment:
            active_compartments += ['detect']

        # Apply flows
        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for strain in self.strains:
                    for organ in self.organ_status:
                        for compartment in active_compartments:

                            # Recovery
                            self.set_fixed_transfer_rate_flow(compartment + organ + strain + riskgroup + agegroup,
                                                              'latent_late' + strain + riskgroup + agegroup,
                                                              'tb_rate_recover' + organ)

                            # Death
                            self.set_fixed_infection_death_rate_flow(compartment + organ + strain + riskgroup
                                                                     + agegroup,
                                                                     'tb_rate_death' + organ)

                        # Detected, with misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                self.set_fixed_infection_death_rate_flow('detect' + organ + strain + '_as'
                                                                         + assigned_strain[1:] + riskgroup + agegroup,
                                                                         'tb_rate_death' + organ)
                                self.set_fixed_transfer_rate_flow('detect' + organ + strain + '_as'
                                                                  + assigned_strain[1:] + riskgroup + agegroup,
                                                                  'latent_late' + strain + riskgroup + agegroup,
                                                                  'tb_rate_recover' + organ)

    def set_fixed_programmatic_flows(self):

        """
        Set rates of return to active disease for patients who presented for health care and were missed and for
        patients who were in the low-quality health care sector.
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Re-start presenting after a missed diagnosis
                        self.set_fixed_transfer_rate_flow(
                            'missed' + organ + strain + riskgroup + agegroup,
                            'active' + organ + strain + riskgroup + agegroup,
                            'program_rate_restart_presenting')

                        # Giving up on the hopeless low-quality health system
                        if self.is_lowquality:
                            self.set_fixed_transfer_rate_flow(
                                'lowquality' + organ + strain + riskgroup + agegroup,
                                'active' + organ + strain + riskgroup + agegroup,
                                'program_rate_leavelowquality')

    def set_detection_flows(self):

        """
        Set previously calculated detection rates (either assuming everyone is correctly identified if misassignment
        not permitted or with proportional misassignment).
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups_for_detection:
                for organ in self.organ_status:
                    organ_for_detection = organ
                    if not self.vary_detection_by_organ:
                        organ_for_detection = ''

                    for strain_number, strain in enumerate(self.strains):

                        # With misassignment
                        if self.is_misassignment:
                            for assigned_strain_number in range(len(self.strains)):
                                as_assigned_strain = '_as' + self.strains[assigned_strain_number][1:]
                                # If the strain is equally or more resistant than its assignment
                                if strain_number >= assigned_strain_number:
                                    self.set_var_transfer_rate_flow(
                                        'active' + organ + strain + riskgroup + agegroup,
                                        'detect' + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                        'program_rate_detect' + organ_for_detection + riskgroup
                                        + strain + as_assigned_strain)

                        # Without misassignment
                        else:
                            self.set_var_transfer_rate_flow(
                                'active' + organ + strain + riskgroup + agegroup,
                                'detect' + organ + strain + riskgroup + agegroup,
                                'program_rate_detect' + organ_for_detection + riskgroup)

    def set_variable_programmatic_flows(self):

        """
        Set rate of missed diagnosis (which is variable as the algorithm sensitivity typically will be), rate of
        presentation to low quality health care (which is variable as the extent of this health system typically will
        be) and rate of treatment commencement (which is variable and depends on the diagnostics available).
        """

        # Set rate of missed diagnoses and entry to low-quality health care
        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for strain in self.strains:
                    for organ in self.organ_status:
                        detection_organ = ''
                        if self.vary_detection_by_organ: detection_organ = organ
                        self.set_var_transfer_rate_flow('active' + organ + strain + riskgroup + agegroup,
                                                        'missed' + organ + strain + riskgroup + agegroup,
                                                        'program_rate_missed' + detection_organ)

                        # Treatment commencement, with and without misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                self.set_var_transfer_rate_flow('detect' + organ + strain + '_as' + assigned_strain[1:]
                                                                + riskgroup + agegroup,
                                                                'treatment_infect' + organ + strain + '_as'
                                                                + assigned_strain[1:] + riskgroup + agegroup,
                                                                'program_rate_start_treatment' + organ)
                        else:
                            self.set_var_transfer_rate_flow('detect' + organ + strain + riskgroup + agegroup,
                                                            'treatment_infect' + organ + strain + riskgroup
                                                            + agegroup,
                                                            'program_rate_start_treatment' + organ)

                        # Enter the low quality health care system
                        if self.is_lowquality:
                            self.set_var_transfer_rate_flow('active' + organ + strain + riskgroup + agegroup,
                                                            'lowquality' + organ + strain + riskgroup + agegroup,
                                                            'program_rate_enterlowquality')

    def set_treatment_flows(self):

        """
        Set rates of progression through treatment stages - dealing with amplification, as well as misassignment if
        either or both are implemented.
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for organ in self.organ_status:
                    for strain_number, strain in enumerate(self.strains):

                        # Which strains to loop over for strain assignment
                        assignment_strains = ['']
                        if self.is_misassignment:
                            assignment_strains = self.strains
                        for assigned_strain_number, assigned_strain in enumerate(assignment_strains):
                            as_assigned_strain = ''
                            strain_or_inappropriate = strain
                            if self.is_misassignment:
                                as_assigned_strain = '_as' + assigned_strain[1:]

                                # Which treatment parameters to use - for the strain or for inappropriate treatment
                                strain_or_inappropriate = assigned_strain
                                if strain_number > assigned_strain_number:
                                    strain_or_inappropriate = strain + '_as' + assigned_strain[1:]

                            # Success by treatment stage
                            self.set_var_transfer_rate_flow(
                                'treatment_infect' + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                'treatment_noninfect' + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                'program_rate_success_infect' + strain_or_inappropriate)
                            self.set_var_transfer_rate_flow(
                                'treatment_noninfect' + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                'susceptible_treated' + riskgroup + agegroup,
                                'program_rate_success_noninfect' + strain_or_inappropriate)

                            # Death on treatment
                            for treatment_stage in self.treatment_stages:
                                self.set_var_infection_death_rate_flow('treatment' +
                                    treatment_stage + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                    'program_rate_death' + treatment_stage + strain_or_inappropriate)

                            # Default
                            for treatment_stage in self.treatment_stages:

                                # If it's either the most resistant strain available or amplification is not active:
                                if strain == self.strains[-1] or not self.is_amplification:
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                        'active' + organ + strain + riskgroup + agegroup,
                                        'program_rate_default' + treatment_stage + strain_or_inappropriate)

                                # Otherwise with amplification
                                else:
                                    amplify_to_strain = self.strains[strain_number + 1]
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                        'active' + organ + strain + riskgroup + agegroup,
                                        'program_rate_default' + treatment_stage + '_noamplify'
                                        + strain_or_inappropriate)
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + riskgroup + agegroup,
                                        'active' + organ + amplify_to_strain + riskgroup + agegroup,
                                        'program_rate_default' + treatment_stage + '_amplify' + strain_or_inappropriate)

    def set_ipt_flows(self):

        """
        Sets a flow from the early latent compartment to the partially immune susceptible compartment that is determined
        by report_numbers_starting_treatment above and is not linked to the 'from_label' compartment.
        """

        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for strain in self.strains:
                    if 'agestratified_ipt' in self.relevant_interventions or 'ipt' in self.relevant_interventions \
                            and 'dr' not in strain:
                        self.set_linked_transfer_rate_flow('latent_early' + strain + riskgroup + agegroup,
                                                           'susceptible_vac' + riskgroup + agegroup,
                                                           'ipt_effective_treatments' + agegroup)
                    if 'int_prop_community_ipt' in self.relevant_interventions \
                            and 'dr' not in strain:
                        self.set_var_transfer_rate_flow('latent_early' + strain + riskgroup + agegroup,
                                                        'susceptible_vac' + riskgroup + agegroup,
                                                        'rate_community_ipt')
                        self.set_var_transfer_rate_flow('latent_late' + strain + riskgroup + agegroup,
                                                        'susceptible_vac' + riskgroup + agegroup,
                                                        'rate_community_ipt')

