
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
import warnings


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

        # Fundamental attributes of and inputs to model
        self.scenario = scenario

        # Get organ stratification, strains and starting time from inputs objects
        self.inputs = inputs
        for attribute in ['organ_status', 'strains', 'comorbidities', 'is_organvariation', 'agegroups']:
            setattr(self, attribute, getattr(inputs, attribute))
        self.scaleup_fns = inputs.scaleup_fns[self.scenario]
        self.start_time = inputs.model_constants['start_time']

        # Set model characteristics directly from GUI inputs
        for attribute in \
                ['is_lowquality', 'is_amplification', 'is_misassignment', 'country', 'time_step', 'integration_method']:
            setattr(self, attribute, gui_inputs[attribute])
        if self.is_misassignment: assert self.is_amplification, 'Misassignment requested without amplification'

        # Set fixed parameters
        for key, value in inputs.model_constants.items():
            if type(value) == float: self.set_parameter(key, value)

        # Track list of included optional parameters (mostly interventions)
        self.optional_timevariants = []
        for timevariant in \
                ['program_prop_novel_vaccination', 'transmission_modifier', 'program_prop_smearacf',
                 'program_prop_xpertacf', 'program_prop_decentralisation', 'program_prop_xpert',
                 'program_prop_treatment_support', 'program_prop_community_ipt', 'program_prop_xpertacf_indigenous',
                 'program_prop_xpertacf_prison', 'program_prop_xpertacf_indigenous', 'program_prop_xpertacf_urbanpoor',
                 'program_prop_xpertacf_ruralpoor']:
            if timevariant in self.scaleup_fns: self.optional_timevariants += [timevariant]
        if 'program_prop_shortcourse_mdr' in self.scaleup_fns and len(self.strains) > 1:
            self.optional_timevariants += ['program_prop_shortcourse_mdr']

        for timevariant in self.scaleup_fns:
            if '_ipt_age' in timevariant:
                self.optional_timevariants += ['agestratified_ipt']
            elif '_ipt' in timevariant and 'community_ipt' not in timevariant:
                self.optional_timevariants += ['ipt']

        # Define model compartmental structure (compartment initialisation is now in base.py)
        self.define_model_structure()

        # Treatment outcomes
        self.outcomes = ['_success', '_death', '_default']
        self.treatment_stages = ['_infect', '_noninfect']

        # Intervention and economics-related initialisiations
        self.interventions_to_cost = inputs.interventions_to_cost
        self.find_intervention_startdates()
        if self.eco_drives_epi: self.distribute_funding_across_years()

        # Work out what we're doing with organ status and variation of detection rates by organ status
        # ** This should probably be moved to the data processing module
        self.vary_detection_by_organ = gui_inputs['is_vary_detection_by_organ']
        if len(self.organ_status) == 1 and self.vary_detection_by_organ:
            self.vary_detection_by_organ = False
            print('Requested variation by organ status turned off, as model is unstratified by organ status.')
        if len(self.organ_status) > 1 and 'program_prop_xpert' in self.optional_timevariants \
                and not self.vary_detection_by_organ:
            self.vary_detection_by_organ = True
            print('Variation in detection by organ status added although not requested, for Xpert implementation.')
        elif len(self.organ_status) == 1 and 'program_prop_xpert' in self.optional_timevariants:
            print('Effect of Xpert on smear-negative detection not simulated as model unstratified by organ status.')

        self.detection_algorithm_ceiling = .85
        self.organs_for_detection = ['']
        if self.vary_detection_by_organ:
            self.organs_for_detection = self.organ_status

        # Boolean is automatically set according to whether any form of ACF is being implemented
        self.vary_detection_by_comorbidity = False
        for timevariant in self.scaleup_fns:
            if 'acf' in timevariant: self.vary_detection_by_comorbidity = True
        self.comorbidities_for_detection = ['']
        if self.vary_detection_by_comorbidity:
            self.comorbidities_for_detection = self.comorbidities

        # Temporarily hard coded option for short course MDR-TB regimens to improve outcomes
        self.shortcourse_improves_outcomes = False

        # Temporarily hard coded option to vary force of infection across risk groups
        self.vary_force_infection_by_comorbidity = False
        if self.vary_force_infection_by_comorbidity:
            self.mixing = {}
            self.create_mixing_matrix()

        # Add time ticker
        self.next_time_point = copy.copy(self.start_time)

    def define_model_structure(self):

        # All compartmental disease stages
        self.compartment_types = ['susceptible_fully', 'susceptible_vac', 'susceptible_treated', 'latent_early',
                                  'latent_late', 'active', 'detect', 'missed', 'treatment_infect',
                                  'treatment_noninfect']
        if self.is_lowquality: self.compartment_types += ['lowquality']
        if 'program_prop_novel_vaccination' in self.optional_timevariants:
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
            for comorbidity in self.comorbidities:
                for compartment in self.compartment_types:

                    # Replicate susceptible for age-groups and comorbidities
                    if 'susceptible' in compartment: self.set_compartment(compartment + comorbidity + agegroup, 0.)

                    # Replicate latent classes for age-groups, comorbidities and strains
                    elif 'latent' in compartment:
                        for strain in self.strains:
                            self.set_compartment(compartment + strain + comorbidity + agegroup, 0.)

                    # Replicate active classes for age-groups, comorbidities, strains and organs
                    elif 'active' in compartment or 'missed' in compartment or 'lowquality' in compartment:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                self.set_compartment(compartment + organ + strain + comorbidity + agegroup, 0.)

                    # Replicate treatment classes for age-groups, comorbidities, strains, organs and assigned strains
                    else:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                if self.is_misassignment:
                                    for assigned_strain in self.strains:
                                        self.set_compartment(compartment + organ + strain + '_as' + assigned_strain[1:]
                                                             + comorbidity + agegroup,
                                                             0.)
                                else:
                                    self.set_compartment(compartment + organ + strain + comorbidity + agegroup, 0.)

        # Find starting proportions for risk groups
        if len(self.comorbidities) == 1:
            start_comorb_prop = {'': 1.}
        else:
            start_comorb_prop = {'_nocomorb': 1.}
            for comorbidity in self.comorbidities:
                if comorbidity != '_nocomorb':
                    start_comorb_prop[comorbidity] \
                        = self.scaleup_fns['comorb_prop' + comorbidity](self.inputs.model_constants['start_time'])
                    start_comorb_prop['_nocomorb'] -= start_comorb_prop[comorbidity]

        # Find starting strain for compartment initialisation
        if self.strains == ['']:
            default_start_strain = ''
        else:
            default_start_strain = '_ds'

        # Arbitrarily split equally by age-groups and organ status, but avoid starting with any resistant strains
        for compartment in self.compartment_types:
            if compartment in self.initial_compartments:
                for agegroup in self.agegroups:
                    for comorbidity in self.comorbidities:
                        if 'susceptible_fully' in compartment:
                            self.set_compartment(compartment + comorbidity + agegroup,
                                                 self.initial_compartments[compartment] * start_comorb_prop[comorbidity]
                                                 / len(self.agegroups))
                        elif 'latent' in compartment:
                            self.set_compartment(compartment + default_start_strain + comorbidity + agegroup,
                                                 self.initial_compartments[compartment] * start_comorb_prop[comorbidity]
                                                 / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(compartment + organ + default_start_strain + comorbidity
                                                     + agegroup,
                                                     self.initial_compartments[compartment]
                                                     * start_comorb_prop[comorbidity]
                                                     / len(self.organ_status) / len(self.agegroups))

    def create_mixing_matrix(self):

        """
        Creates model attribute for mixing between population risk groups, for use in calculate_force_infection_vars
        method below only.

        *** Would be nice to make this more general - currently dependent on all inter-group mixing proportions being
        defined in the inputs spreadsheet. ***
        """

        # Initialise first tier of dictionaries fully so that parameters can be either way round
        for comorbidity in self.comorbidities:
            self.mixing[comorbidity] = {}

        # Populate symmetric matrices outside of diagonal
        for comorbidity in self.comorbidities:
            for other_comorbidity in self.comorbidities:
                if 'prop' + comorbidity + '_mix' + other_comorbidity in self.params:
                    self.mixing[comorbidity][other_comorbidity] \
                        = self.params['prop' + comorbidity + '_mix' + other_comorbidity]
                    self.mixing[other_comorbidity][comorbidity] \
                        = self.params['prop' + comorbidity + '_mix' + other_comorbidity]

        # Populate diagonal
        for comorbidity in self.comorbidities:
            self.mixing[comorbidity][comorbidity] = 1. - sum(self.mixing[comorbidity].values())

    #######################################################
    ### Single method to process uncertainty parameters ###
    #######################################################

    def process_uncertainty_params(self):

        """
        Perform some simple parameter processing - just for those that are used as uncertainty parameters and so can't
        be processed in the data_processing module.
        """

        # Find the case fatality of smear-negative TB using the relative case fatality
        self.params['tb_prop_casefatality_untreated_smearneg'] = \
            self.params['tb_prop_casefatality_untreated_smearpos'] \
            * self.params['tb_relative_casefatality_untreated_smearneg']

        # Add the extrapulmonary case fatality (currently not entered from the spreadsheets)
        self.params['tb_prop_casefatality_untreated_extrapul'] \
            = self.params['tb_prop_casefatality_untreated_smearneg']

        # Calculate the rates of death and recovery from the above parameters
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
        if self.eco_drives_epi and self.time > self.inputs.model_constants['current_time']: self.update_vars_from_cost()
        self.vars['population'] = sum(self.compartments.values())
        self.calculate_birth_rates_vars()
        self.calculate_force_infection_vars()
        if self.is_organvariation: self.calculate_progression_vars()
        if 'program_prop_decentralisation' in self.optional_timevariants: self.adjust_case_detection_for_decentralisation()
        if self.vary_detection_by_organ:
            self.calculate_case_detection_by_organ()
            if 'program_prop_xpert' in self.optional_timevariants:
                self.adjust_smearneg_detection_for_xpert()
        self.calculate_detect_missed_vars()
        if self.vary_detection_by_comorbidity:
            self.calculate_acf_rate()
            self.adjust_case_detection_for_acf()
        self.calculate_misassignment_detection_vars()
        if self.is_lowquality: self.calculate_lowquality_detection_vars()
        self.calculate_await_treatment_var()
        self.calculate_treatment_rates_vars()
        self.calculate_population_sizes()
        if 'agestratified_ipt' in self.optional_timevariants or 'ipt' in self.optional_timevariants:
            self.calculate_ipt_rate()
        if 'program_prop_community_ipt' in self.optional_timevariants: self.calculate_community_ipt_rate()

    def ticker(self):

        """
        Prints time every ten years to give a sense of progress through integration.
        """

        if self.time > self.next_time_point:
            print(int(self.time))
            self.next_time_point += 10.

    def calculate_birth_rates_vars(self):

        """
        Calculate birth rates into vaccinated and unvaccinated compartments.
        """

        # Calculate total births (also for tracking for for interventions)
        self.vars['births_total'] = self.get_constant_or_variable_param('demo_rate_birth') / 1e3 \
                                    * self.vars['population']

        # Determine vaccinated and unvaccinated proportions
        vac_props = {'vac': self.get_constant_or_variable_param('program_prop_vaccination')}
        vac_props['unvac'] = 1. - vac_props['vac']
        if 'program_prop_novel_vaccination' in self.optional_timevariants:
            vac_props['novelvac'] = self.get_constant_or_variable_param('program_prop_vaccination') \
                                    * self.vars['program_prop_novel_vaccination']
            vac_props['vac'] -= vac_props['novelvac']

        # Calculate birth rates
        for comorbidity in self.comorbidities:
            for vac_status in vac_props:
                self.vars['births_' + vac_status + comorbidity] \
                    = vac_props[vac_status] * self.vars['births_total'] * self.target_comorb_props[comorbidity][-1]

    def calculate_force_infection_vars(self):

        """
        Calculate force of infection independently for each strain, incorporating partial immunity and infectiousness.
        First calculate the effective infectious population (incorporating infectiousness by organ involvement), then
        calculate the raw force of infection, then adjust for various levels of susceptibility.
        """

        # Find the effective infectious population for each strain
        for strain in self.strains:

            # Initialise infectious population vars as needed
            if self.vary_force_infection_by_comorbidity:
                for comorbidity in self.comorbidities:
                    self.vars['effective_infectious_population' + strain + comorbidity] = 0.
            else:
                self.vars['effective_infectious_population' + strain] = 0.

            # Loop through compartments, skipping on as soon as possible if not relevant
            for label in self.labels:
                if strain not in label and strain != '':
                    continue
                for organ in self.organ_status:
                    if organ not in label and organ != '':
                        continue
                    for agegroup in self.agegroups:
                        if agegroup not in label and agegroup != '':
                            continue

                        # Calculate effective infectious population without stratification for risk group
                        if not self.vary_force_infection_by_comorbidity:
                            if label_intersects_tags(label, self.infectious_tags):
                                self.vars['effective_infectious_population' + strain] \
                                    += self.params['tb_multiplier_force' + organ] \
                                       * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                       * self.compartments[label]

                        else:
                            for comorbidity in self.comorbidities:
                                if comorbidity not in label:
                                    continue

                                # Adjustment for increased transmission in risk groups if needed
                                comorb_multiplier_force_infection = 1.
                                if 'comorb_multiplier_force_infection' + comorbidity in self.params:
                                    comorb_multiplier_force_infection \
                                        = self.params['comorb_multiplier_force_infection' + comorbidity]

                                # Calculate effective infectious population for each risk group
                                if label_intersects_tags(label, self.infectious_tags):
                                    for source_comorbidity in self.comorbidities:
                                        self.vars['effective_infectious_population' + strain + comorbidity] \
                                            += self.params['tb_multiplier_force' + organ] \
                                               * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                               * self.compartments[label] \
                                               * self.mixing[comorbidity][source_comorbidity] \
                                               * comorb_multiplier_force_infection

            # To loop over all comorbidities if needed, or otherwise to just run once
            force_comorbidities = ['']
            if self.vary_force_infection_by_comorbidity:
                force_comorbidities = copy.copy(self.comorbidities)

            # Calculate force of infection unadjusted for immunity/susceptibility
            for comorbidity in force_comorbidities:
                self.vars['rate_force' + strain + comorbidity] = \
                    self.params['tb_n_contact'] \
                    * self.vars['effective_infectious_population' + strain + comorbidity] \
                    / self.vars['population']

                # If any modifications to transmission parameter to be made over time
                if 'transmission_modifier' in self.optional_timevariants:
                    self.vars['rate_force' + strain + comorbidity] *= self.vars['transmission_modifier']

                # Adjust for immunity in various groups
                for force_type in ['_vac', '_latent', '_novelvac']:
                    self.vars['rate_force' + force_type + strain + comorbidity] \
                        = self.params['tb_multiplier' + force_type + '_protection'] \
                          * self.vars['rate_force' + strain + comorbidity]

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
                for comorbidity in self.comorbidities:
                    for timing in ['_early', '_late']:
                        if comorbidity == '_diabetes':
                            self.vars['tb_rate' + timing + '_progression' + organ + comorbidity + agegroup] \
                                = self.vars['epi_prop' + organ] \
                                  * self.params['tb_rate' + timing + '_progression' + '_nocomorb' + agegroup] \
                                  * self.params['comorb_multiplier_diabetes_progression']
                        else:
                            self.vars['tb_rate' + timing + '_progression' + organ + comorbidity + agegroup] \
                                = self.vars['epi_prop' + organ] \
                                  * self.params['tb_rate' + timing + '_progression' + comorbidity + agegroup]

    def adjust_case_detection_for_decentralisation(self):

        """
        Implement the decentralisation intervention, which narrows the case detection gap between the current values
        and the idealised estimated value.
        """

        self.vars['program_prop_detect'] \
            += self.vars['program_prop_decentralisation'] \
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
                   * self.params['tb_prop_xpert_smearneg_sensitivity'] * self.vars['program_prop_xpert']

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
            for comorbidity in [''] + self.comorbidities_for_detection:

                # Detected
                self.vars['program_rate_detect' + organ + comorbidity] \
                    = - self.vars['program_prop_detect' + organ] \
                      * (1. / self.params['tb_timeperiod_activeuntreated']
                         + 1. / self.vars['demo_life_expectancy']) \
                      / (self.vars['program_prop_detect' + organ] - 1.)

            # Missed (no need to loop by comorbidity as ACF is the only difference here, which is applied next)
            self.vars['program_rate_missed' + organ] \
                = self.vars['program_rate_detect' + organ] \
                  * (1. - self.vars['program_prop_algorithm_sensitivity' + organ]) \
                  / max(self.vars['program_prop_algorithm_sensitivity' + organ], 1e-6)

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
        for comorbidity in [''] + self.comorbidities:
            if 'program_prop_smearacf' + comorbidity in self.optional_timevariants \
                    or 'program_prop_xpertacf' + comorbidity in self.optional_timevariants:

                # The following can't be written as self.organ_status, as it won't work for non-fully-stratified models
                for organ in ['', '_smearpos', '_smearneg', '_extrapul']:
                    self.vars['program_rate_acf' + organ + comorbidity] = 0.

                # Smear-based ACF rate
                if 'program_prop_smearacf' + comorbidity in self.optional_timevariants:
                    self.vars['program_rate_acf_smearpos' + comorbidity] \
                        += self.vars['program_prop_smearacf' + comorbidity] \
                           * self.params['program_prop_acf_detections_per_round'] \
                           / self.params['program_timeperiod_acf_rounds']

                # Xpert-based ACF rate for smear-positives and smear-negatives
                if 'program_prop_xpertacf' + comorbidity in self.optional_timevariants:
                    for organ in ['_smearpos', '_smearneg']:
                        self.vars['program_rate_acf' + organ + comorbidity] \
                            += self.vars['program_prop_xpertacf' + comorbidity] \
                               * self.params['program_prop_acf_detections_per_round'] \
                               / self.params['program_timeperiod_acf_rounds']

                    # Adjust smear-negative detections for Xpert's sensitivity
                    self.vars['program_rate_acf_smearneg' + comorbidity] \
                        *= self.params['tb_prop_xpert_smearneg_sensitivity']

    def adjust_case_detection_for_acf(self):

        """
        Add ACF detection rates to previously calculated passive case detection rates, creating vars for case detection
        that are specific for organs.
        """

        for organ in self.organs_for_detection:
            for comorbidity in self.comorbidities:

                # ACF in risk groups
                if 'program_prop_smearacf' + comorbidity in self.optional_timevariants \
                        or 'program_prop_xpertacf' + comorbidity in self.optional_timevariants:
                    self.vars['program_rate_detect' + organ + comorbidity] \
                        += self.vars['program_rate_acf' + organ + comorbidity]

                # ACF in the general community
                if 'program_prop_smearacf' in self.optional_timevariants \
                        or 'program_prop_xpertacf' in self.optional_timevariants:
                    self.vars['program_rate_detect' + organ + comorbidity] \
                        += self.vars['program_rate_acf' + organ]

    def calculate_misassignment_detection_vars(self):

        """
        Calculate the proportions of patients assigned to each strain. (Note that second-line DST availability refers to
        the proportion of those with first-line DST who also have second-line DST available.)
        """

        # With misassignment:
        for organ in self.organs_for_detection:
            for comorbidity in self.comorbidities_for_detection:
                if self.is_misassignment:

                    prop_firstline = self.get_constant_or_variable_param('program_prop_firstline_dst')

                    # Add effect of Xpert on identification, assuming that independent distribution to conventional DST
                    if 'program_prop_xpert' in self.optional_timevariants:
                        prop_firstline += (1. - prop_firstline) * self.vars['program_prop_xpert']

                    # Determine rates of identification/misidentification as each strain
                    self.vars['program_rate_detect' + organ + comorbidity + '_ds_asds'] \
                        = self.vars['program_rate_detect' + organ + comorbidity]
                    self.vars['program_rate_detect' + organ + comorbidity + '_ds_asmdr'] = 0.
                    self.vars['program_rate_detect' + organ + comorbidity + '_mdr_asds'] \
                        = (1. - prop_firstline) * self.vars['program_rate_detect' + organ + comorbidity]
                    self.vars['program_rate_detect' + organ + comorbidity + '_mdr_asmdr'] \
                        = prop_firstline * self.vars['program_rate_detect' + organ + comorbidity]

                    # If a third strain is present
                    if len(self.strains) > 2:
                        prop_secondline = self.get_constant_or_variable_param('program_prop_secondline_dst')
                        self.vars['program_rate_detect' + organ + comorbidity + '_ds_asxdr'] = 0.
                        self.vars['program_rate_detect' + organ + comorbidity + '_mdr_asxdr'] = 0.
                        self.vars['program_rate_detect' + organ + comorbidity + '_xdr_asds'] \
                            = (1. - prop_firstline) * self.vars['program_rate_detect' + organ + comorbidity]
                        self.vars['program_rate_detect' + organ + comorbidity + '_xdr_asmdr'] \
                            = prop_firstline \
                              * (1. - prop_secondline) * self.vars['program_rate_detect' + organ + comorbidity]
                        self.vars['program_rate_detect' + organ + comorbidity + '_xdr_asxdr'] \
                            = prop_firstline * prop_secondline * self.vars['program_rate_detect' + organ + comorbidity]

                # Without misassignment, everyone is correctly allocated
                else:
                    for strain in self.strains:
                        self.vars['program_rate_detect' + organ + comorbidity + strain + '_as' + strain[1:]] \
                            = self.vars['program_rate_detect' + organ + comorbidity]

    def calculate_await_treatment_var(self):

        """
        Take the reciprocal of the waiting times to calculate the flow rate to start treatment after detection.
        Note that the default behaviour for a single strain model is to use the waiting time for smear-positives.
        Also weight the time period
        """

        # If only one organ stratum
        if len(self.organ_status) == 1:
            self.vars['program_rate_start_treatment'] \
                = 1. / self.get_constant_or_variable_param('program_timeperiod_await_treatment_smearpos')

        # Organ stratification
        else:
            for organ in self.organ_status:

                # Adjust smear-negative for Xpert coverage
                if organ == '_smearneg' and 'program_prop_xpert' in self.optional_timevariants:
                    prop_xpert = self.get_constant_or_variable_param('program_prop_xpert')
                    self.vars['program_rate_start_treatment_smearneg'] = \
                        1. / (self.vars['program_timeperiod_await_treatment_smearneg'] * (1. - prop_xpert)
                              + self.params['program_timeperiod_await_treatment_smearneg_xpert'] * prop_xpert)

                # Do other organ stratifications (including smear-negative if Xpert not an intervention)
                else:
                    self.vars['program_rate_start_treatment' + organ] = \
                        1. / self.get_constant_or_variable_param('program_timeperiod_await_treatment' + organ)

    def calculate_lowquality_detection_vars(self):

        """
        Calculate rate of entry to low-quality care ffom the proportion of treatment administered in low-quality sector.
        Note that this now means that the case detection proportion only applies to those with access to care, so
        that proportion of all cases isn't actually detected.
        """

        prop_lowqual = self.get_constant_or_variable_param('program_prop_lowquality')
        prop_lowqual *= (1. - self.vars['program_prop_engage_lowquality'])

        # Note that there is still a program_rate_detect var even if detection is varied by organ and/or comorbidity
        self.vars['program_rate_enterlowquality'] \
            = self.vars['program_rate_detect'] * prop_lowqual / (1. - prop_lowqual)

    def calculate_treatment_rates_vars(self):

        """
        Work out rates of progression through treatment by stage of treatment from the proportions provided for success
        and death.
        """

        # Add inappropriate treatments to strains to create a list of all outcomes of interest
        treatments = copy.copy(self.strains)
        if len(self.strains) > 1 and self.is_misassignment:
            treatments += ['_inappropriate']
        for strain in treatments:

            # Find baseline treatment period for total duration and for period infectious
            for treatment_stage in ['', '_infect']:
                self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                    = self.params['tb_timeperiod' + treatment_stage + '_ontreatment' + strain]

            # Adapt treatment periods for short course regimen
            if strain == '_mdr' and 'program_prop_shortcourse_mdr' in self.optional_timevariants:
                relative_treatment_duration_mdr \
                    = 1. - self.vars['program_prop_shortcourse_mdr'] \
                           * (1. - self.params['program_prop_shortcourse_mdr_relativeduration'])
                for treatment_stage in ['', '_infect']:
                    self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                        *= relative_treatment_duration_mdr

                # Adapt treatment outcomes for short course regimen
                if self.shortcourse_improves_outcomes:
                    for outcome in ['_success', '_death']:
                        self.vars['program_prop_treatment' + outcome + '_mdr'] \
                            += (self.params['program_prop_treatment' + outcome + '_shortcoursemdr']
                                - self.vars['program_prop_treatment' + outcome + '_mdr']) \
                               * self.vars['program_prop_shortcourse_mdr']

            # Add some extra treatment success if the treatment support program is active
            if 'program_prop_treatment_support' in self.optional_timevariants:
                self.vars['program_prop_treatment_success' + strain] \
                    += (1. - self.vars['program_prop_treatment_success' + strain]) \
                       * self.params['program_prop_treatment_support_improvement'] \
                       * self.vars['program_prop_treatment_support']
                self.vars['program_prop_treatment_death' + strain] \
                    -= self.vars['program_prop_treatment_death' + strain] \
                       * self.params['program_prop_treatment_support_improvement'] \
                       * self.vars['program_prop_treatment_support']

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

    def calculate_population_sizes(self):

        """
        Calculate the size of the populations to which each intervention is applicable, for use in generating
        cost-coverage curves.
        """

        # Treatment support
        if 'program_prop_treatment_support' in self.optional_timevariants:
            self.vars['popsize_treatment_support'] = 0.
            for compartment in self.compartments:
                if 'treatment_' in compartment:
                    self.vars['popsize_treatment_support'] += self.compartments[compartment]

        # IPT
        for agegroup in self.agegroups:
            self.vars['popsize_ipt' + agegroup] = 0.
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:
                        if '_smearpos' in organ and 'dr' not in strain:
                            if self.is_misassignment:
                                for assigned_strain in self.strains:
                                    self.vars['popsize_ipt' + agegroup] \
                                        += self.vars['program_rate_start_treatment' + organ] \
                                           * self.compartments['detect' + organ + strain + '_as' + assigned_strain[1:] \
                                                               + comorbidity + agegroup] \
                                           * self.inputs.model_constants['ipt_eligible_per_treatment_start']
                            else:
                                self.vars['popsize_ipt' + agegroup] \
                                    += self.vars['program_rate_start_treatment' + organ] \
                                       * self.compartments['detect' + organ + strain + comorbidity + agegroup] \
                                       * self.inputs.model_constants['ipt_eligible_per_treatment_start']

        # BCG (So simple that it's almost unnecessary, but needed for loops over program names)
        self.vars['popsize_vaccination'] = self.vars['births_total']

        # Xpert - all presentations with active TB
        if 'program_prop_xpert' in self.optional_timevariants:
            self.vars['popsize_xpert'] = 0.
            for agegroup in self.agegroups:
                for comorbidity in self.comorbidities:
                    for strain in self.strains:
                        for organ in self.organ_status:
                            if self.vary_detection_by_organ:
                                detection_organ = organ
                            else:
                                detection_organ = ''
                            self.vars['popsize_xpert'] \
                                += self.vars['program_rate_detect' + detection_organ + comorbidity]\
                                   * self.compartments['active' + organ + strain + comorbidity + agegroup] \
                                   * (self.params['program_number_tests_per_tb_presentation'] + 1.)

        # ACF
        for comorbidity in [''] + self.comorbidities:
            if 'program_prop_smearacf' + comorbidity in self.optional_timevariants:
                self.vars['popsize_smearacf' + comorbidity] = 0.
                for compartment in self.compartments:
                    if 'active_' in compartment and '_smearpos' in compartment \
                            and (comorbidity == '' or comorbidity in compartment):
                        self.vars['popsize_smearacf' + comorbidity] \
                            += self.compartments[compartment] * self.params['program_nns_smearacf']
            if 'program_prop_xpertacf' + comorbidity in self.optional_timevariants:
                self.vars['popsize_xpertacf' + comorbidity] = 0.
                for compartment in self.compartments:
                    if 'active_' in compartment and '_smearpos' in compartment \
                            and (comorbidity == '' or comorbidity in compartment):
                        self.vars['popsize_xpertacf' + comorbidity] \
                            += self.compartments[compartment] * self.params['program_nns_xpertacf_smearpos']
                    elif 'active_' in compartment and '_smearneg' in compartment \
                            and (comorbidity == '' or comorbidity in compartment):
                        self.vars['popsize_xpertacf' + comorbidity] \
                            += self.compartments[compartment] * self.params['program_nns_xpertacf_smearneg']

        # Decentralisation and engage low quality sector
        all_actives_popsize = 0.
        for compartment in self.compartments:
            if 'susceptible_' not in compartment and 'latent_' not in compartment:
                all_actives_popsize += self.compartments[compartment]
        all_actives_interventions = ['decentralisation', 'engage_lowquality']
        for intervention in all_actives_interventions:
            if intervention in self.inputs.potential_interventions_to_cost:
                self.vars['popsize_' + intervention] = all_actives_popsize

        # Shortcourse MDR-TB regimen
        if 'program_prop_shortcourse_mdr' in self.optional_timevariants:
            self.vars['popsize_shortcourse_mdr'] = 0.
            for compartment in self.compartments:
                if 'treatment' in compartment and '_mdr' in compartment:
                    self.vars['popsize_shortcourse_mdr'] += self.compartments[compartment]

    def calculate_ipt_rate(self):

        """
        Uses the popsize to which IPT is applicable, which was calculated in calculate_population_sizes
        to determine the actual number of persons who should be shifted across compartments.
        """

        prop_ipt = {}
        for agegroup in self.agegroups:

            # Find IPT coverage for the age group as the maximum of the coverage in that age group and overall coverage.
            for ipt_type in ['ipt', 'novel_ipt']:
                prop_ipt[ipt_type] = 0.
                if 'program_prop_' + ipt_type + agegroup in self.vars:
                    prop_ipt[ipt_type] = self.vars['program_prop_' + ipt_type + agegroup]
                if 'program_prop_' + ipt_type in self.vars:
                    prop_ipt[ipt_type] = max([self.vars['program_prop_' + ipt_type], prop_ipt[ipt_type]])

                # Calculate the number of effective treatments
                self.vars[ipt_type + '_effective_treatments' + agegroup] \
                    = prop_ipt[ipt_type] \
                      * self.vars['popsize_ipt' + agegroup] \
                      * self.inputs.model_constants[ipt_type + '_effective_per_assessment']

            # Check size of latency compartments
            latent_early_pop = 0.
            for compartment in self.compartments:
                if 'latent_early' in compartment and agegroup in compartment:
                    latent_early_pop += self.compartments[compartment]

            # Calculate the total number of effective treatments across both forms of IPT, limiting at all early latents
            self.vars['ipt_effective_treatments' + agegroup] \
                = min([max([self.vars['ipt_effective_treatments' + agegroup],
                            self.vars['novel_ipt_effective_treatments' + agegroup]]),
                       latent_early_pop])

    def calculate_community_ipt_rate(self):

        """
        Implements the community IPT intervention, which is not limited to contacts of persons starting treatment for
        active disease, but rather involves screening of an entire population.
        """

        self.vars['rate_community_ipt'] \
            = self.vars['program_prop_community_ipt'] \
              * self.inputs.model_constants['ipt_effective_per_assessment'] \
              / self.inputs.model_constants['program_timeperiod_community_ipt_round']

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
        if 'agestratified_ipt' in self.optional_timevariants or 'ipt' in self.optional_timevariants:
            self.set_ipt_flows()

    def set_birth_flows(self):

        """
        Set birth (or recruitment) flows by vaccination status (including novel vaccination if implemented).
        """

        for comorbidity in self.comorbidities:
            self.set_var_entry_rate_flow(
                'susceptible_fully' + comorbidity + self.agegroups[0], 'births_unvac' + comorbidity)
            self.set_var_entry_rate_flow(
                'susceptible_vac' + comorbidity + self.agegroups[0], 'births_vac' + comorbidity)
            if 'program_prop_novel_vaccination' in self.optional_timevariants:
                self.set_var_entry_rate_flow('susceptible_novelvac'
                                             + comorbidity + self.agegroups[0], 'births_novelvac' + comorbidity)

    def set_infection_flows(self):

        """
        Set force of infection flows that were estimated by strain in calculate_force_infection_vars above.
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:

                force_comorbidity = ''
                if self.vary_force_infection_by_comorbidity:
                    force_comorbidity = comorbidity

                for strain in self.strains:

                    # Set infection rates according to susceptibility status
                    self.set_var_transfer_rate_flow(
                        'susceptible_fully' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force' + strain + force_comorbidity)
                    self.set_var_transfer_rate_flow(
                        'susceptible_vac' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_vac' + strain + force_comorbidity)
                    self.set_var_transfer_rate_flow(
                        'susceptible_treated' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_vac' + strain + force_comorbidity)
                    self.set_var_transfer_rate_flow(
                        'latent_late' + strain + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_latent' + strain + force_comorbidity)

                    # Novel vaccination
                    if 'program_prop_novel_vaccination' in self.optional_timevariants:
                        self.set_var_transfer_rate_flow(
                            'susceptible_novelvac' + comorbidity + agegroup,
                            'latent_early' + strain + comorbidity + agegroup,
                            'rate_force_novelvac' + strain + force_comorbidity)

    def set_progression_flows(self):

        """
        Set rates of progression from latency to active disease, with rates differing by organ status.
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:

                        # Stabilisation
                        self.set_fixed_transfer_rate_flow(
                            'latent_early' + strain + comorbidity + agegroup,
                            'latent_late' + strain + comorbidity + agegroup,
                            'tb_rate_stabilise' + comorbidity + agegroup)

                        for organ in self.organ_status:

                            # If organ scale-ups available, set flows as variable (if epi_prop_smearpos is in
                            # self.scaleup_fns, then epi_prop_smearneg should be too)
                            if self.is_organvariation:
                                self.set_var_transfer_rate_flow(
                                    'latent_early' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_early_progression' + organ + comorbidity + agegroup)
                                self.set_var_transfer_rate_flow(
                                    'latent_late' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_late_progression' + organ + comorbidity + agegroup)

                            # Otherwise, set fixed flows
                            else:
                                # Early progression
                                self.set_fixed_transfer_rate_flow(
                                    'latent_early' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_early_progression' + organ + comorbidity + agegroup)

                                # Late progression
                                self.set_fixed_transfer_rate_flow(
                                    'latent_late' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_late_progression' + organ + comorbidity + agegroup)

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
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:
                        for compartment in active_compartments:

                            # Recovery
                            self.set_fixed_transfer_rate_flow(compartment + organ + strain + comorbidity + agegroup,
                                                              'latent_late' + strain + comorbidity + agegroup,
                                                              'tb_rate_recover' + organ)

                            # Death
                            self.set_fixed_infection_death_rate_flow(compartment + organ + strain + comorbidity
                                                                     + agegroup,
                                                                     'tb_rate_death' + organ)

                        # Detected, with misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                self.set_fixed_infection_death_rate_flow('detect' + organ + strain + '_as'
                                                                         + assigned_strain[1:] + comorbidity + agegroup,
                                                                         'tb_rate_death' + organ)
                                self.set_fixed_transfer_rate_flow('detect' + organ + strain + '_as'
                                                                  + assigned_strain[1:] + comorbidity + agegroup,
                                                                  'latent_late' + strain + comorbidity + agegroup,
                                                                  'tb_rate_recover' + organ)

    def set_fixed_programmatic_flows(self):

        """
        Set rates of return to active disease for patients who presented for health care and were missed and for
        patients who were in the low-quality health care sector.
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Re-start presenting after a missed diagnosis
                        self.set_fixed_transfer_rate_flow(
                            'missed' + organ + strain + comorbidity + agegroup,
                            'active' + organ + strain + comorbidity + agegroup,
                            'program_rate_restart_presenting')

                        # Giving up on the hopeless low-quality health system
                        if self.is_lowquality:
                            self.set_fixed_transfer_rate_flow(
                                'lowquality' + organ + strain + comorbidity + agegroup,
                                'active' + organ + strain + comorbidity + agegroup,
                                'program_rate_leavelowquality')

    def set_detection_flows(self):

        """
        Set previously calculated detection rates (either assuming everyone is correctly identified if misassignment
        not permitted or with proportional misassignment).
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities_for_detection:
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
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'detect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'program_rate_detect' + organ_for_detection + comorbidity
                                        + strain + as_assigned_strain)

                        # Without misassignment
                        else:
                            self.set_var_transfer_rate_flow(
                                'active' + organ + strain + comorbidity + agegroup,
                                'detect' + organ + strain + comorbidity + agegroup,
                                'program_rate_detect' + organ_for_detection + comorbidity)

    def set_variable_programmatic_flows(self):

        """
        Set rate of missed diagnosis (which is variable as the algorithm sensitivity typically will be), rate of
        presentation to low quality health care (which is variable as the extent of this health system typically will
        be) and rate of treatment commencement (which is variable and depends on the diagnostics available).
        """

        # Set rate of missed diagnoses and entry to low-quality health care
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:
                        detection_organ = ''
                        if self.vary_detection_by_organ: detection_organ = organ
                        self.set_var_transfer_rate_flow('active' + organ + strain + comorbidity + agegroup,
                                                        'missed' + organ + strain + comorbidity + agegroup,
                                                        'program_rate_missed' + detection_organ)

                        # Treatment commencement, with and without misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                self.set_var_transfer_rate_flow('detect' + organ + strain + '_as' + assigned_strain[1:]
                                                                + comorbidity + agegroup,
                                                                'treatment_infect' + organ + strain + '_as'
                                                                + assigned_strain[1:] + comorbidity + agegroup,
                                                                'program_rate_start_treatment' + organ)
                        else:
                            self.set_var_transfer_rate_flow('detect' + organ + strain + comorbidity + agegroup,
                                                            'treatment_infect' + organ + strain + comorbidity
                                                            + agegroup,
                                                            'program_rate_start_treatment' + organ)

                        # Enter the low quality health care system
                        if self.is_lowquality:
                            self.set_var_transfer_rate_flow('active' + organ + strain + comorbidity + agegroup,
                                                            'lowquality' + organ + strain + comorbidity + agegroup,
                                                            'program_rate_enterlowquality')

    def set_treatment_flows(self):

        """
        Set rates of progression through treatment stages - dealing with amplification, as well as misassignment if
        either or both are implemented.
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
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
                                    strain_or_inappropriate = '_inappropriate'

                            # Success by treatment stage
                            self.set_var_transfer_rate_flow(
                                'treatment_infect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'treatment_noninfect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'program_rate_success_infect' + strain_or_inappropriate)
                            self.set_var_transfer_rate_flow(
                                'treatment_noninfect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'susceptible_treated' + comorbidity + agegroup,
                                'program_rate_success_noninfect' + strain_or_inappropriate)

                            # Death on treatment
                            for treatment_stage in self.treatment_stages:
                                self.set_var_infection_death_rate_flow('treatment' +
                                    treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                    'program_rate_death' + treatment_stage + strain_or_inappropriate)

                            # Default
                            for treatment_stage in self.treatment_stages:

                                # If it's either the most resistant strain available or amplification is not active:
                                if strain == self.strains[-1] or not self.is_amplification:
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + strain_or_inappropriate)

                                # Otherwise with amplification
                                else:
                                    amplify_to_strain = self.strains[strain_number + 1]
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + '_noamplify'
                                        + strain_or_inappropriate)
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + amplify_to_strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + '_amplify' + strain_or_inappropriate)

    def set_ipt_flows(self):

        """
        Sets a flow from the early latent compartment to the partially immune susceptible compartment that is determined
        by report_numbers_starting_treatment above and is not linked to the 'from_label' compartment.
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    if 'agestratified_ipt' in self.optional_timevariants or 'ipt' in self.optional_timevariants \
                            and 'dr' not in strain:
                        self.set_linked_transfer_rate_flow('latent_early' + strain + comorbidity + agegroup,
                                                           'susceptible_vac' + comorbidity + agegroup,
                                                           'ipt_effective_treatments' + agegroup)
                    if 'program_prop_community_ipt' in self.optional_timevariants \
                            and 'dr' not in strain:
                        self.set_var_transfer_rate_flow('latent_early' + strain + comorbidity + agegroup,
                                                        'susceptible_vac' + comorbidity + agegroup,
                                                        'rate_community_ipt')
                        self.set_var_transfer_rate_flow('latent_late' + strain + comorbidity + agegroup,
                                                        'susceptible_vac' + comorbidity + agegroup,
                                                        'rate_community_ipt')

