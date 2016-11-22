
"""

The TB-specific model (or models) should be coded in this file.

Time unit throughout: years
Compartment unit throughout: patients

Nested inheritance from BaseModel and StratifiedModel in base.py - the former sets some fundamental methods for
creating intercompartmental flows, costs, etc., while the latter sets down the approach to population stratification.

"""


from scipy import exp, log
from autumn.base import BaseModel, StratifiedModel
import copy


def label_intersects_tags(label, tags):
    for tag in tags:
        if tag in label:
            return True
    return False


def find_outcome_proportions_by_period(
        proportion, early_period, total_period):

    """
    Split one outcome proportion (e.g. default, death) over multiple
    periods
    Args:
        proportion: Total proportion to be split
        early_period: Early time period
        total_period: Late time period
    Returns:
        early_proportion: Proportion allocated to early time period
        late_proportion: Proportion allocated to late time period
    """

    if proportion > 1. or proportion < 0.:
        raise Exception('Proportion greater than one or less than zero')
    elif proportion == 1.:
        # This is just to avoid warnings or errors where the proportion
        # is one. However, this function isn't really intended for this situation.
        early_proportion = 0.5
    else:
        early_proportion \
            = 1. - exp(log(1. - proportion) * early_period / total_period)
    late_proportion = proportion - early_proportion
    return early_proportion, late_proportion


class ConsolidatedModel(StratifiedModel):

    """
    The transmission dynamic model to underpin all AuTuMN analyses
    Inherits from BaseModel, which is intended to be general to any infectious disease
    All TB-specific methods and structures are contained in this model
    Methods are written to be adaptable to any model structure selected through the __init__ arguments

    The work-flow of the simulation is structured in the following order:
        1. Defining the model structure
        2. Initialising the compartments
        3. Setting parameters (needs a lot more work)
        4. Calculating derived parameters and setting scale-up functions
        5. Assigning flows from either parameters or functions
        6. Main loop over simulation time-points:
                a. Extracting scale-up variables
                b. Calculating variable flow rates
                c. Calculating diagnostic variables
                    (b. and  c. can be calculated from a compartment values,
                    parameters, scale-up functions or a combination of these)
        7. Calculating the diagnostic solutions
    """

    def __init__(self, scenario=None, inputs=None, gui_inputs=None):

        # Inherited initialisations
        BaseModel.__init__(self)
        StratifiedModel.__init__(self)

        # Fundamental attributes of model
        self.scenario = scenario
        self.inputs = inputs
        self.gui_inputs = gui_inputs
        self.start_time = self.inputs.model_constants['start_time']

        # Set some model characteristics directly from the GUI inputs
        for attribute in ['is_lowquality', 'is_amplification', 'is_misassignment', 'country']:
            setattr(self, attribute, self.gui_inputs[attribute])
        if self.is_misassignment: assert self.is_amplification, 'Misassignment requested without amplification'

        # Set fixed parameters
        for key, value in self.inputs.model_constants.items():
            if type(value) == float:
                self.set_parameter(key, value)

        # Track list of included additional interventions
        self.optional_timevariants = []
        for timevariant in ['program_prop_novel_vaccination', 'transmission_modifier',
                            'program_prop_smearacf', 'program_prop_xpertacf',
                            'program_prop_decentralisation', 'program_prop_xpert', 'program_prop_treatment_support']:
            if timevariant in self.inputs.scaleup_fns[scenario]:
                self.optional_timevariants += [timevariant]

        # Define model compartmental structure (compartment initialisation is now in base.py)
        self.define_model_structure()

        # Treatment outcomes
        self.outcomes = ['_success', '_death', '_default']
        self.non_success_outcomes = self.outcomes[1:]

        # Get scaleup functions from input object
        self.scaleup_fns = self.inputs.scaleup_fns[self.scenario]

        # Intervention and economics-related initialisiations
        self.interventions_to_cost = self.inputs.interventions_to_cost
        self.find_intervention_startdates()
        if self.eco_drives_epi: self.distribute_funding_across_years()

    def define_model_structure(self):

        # All compartmental disease stages
        self.compartment_types = ['susceptible_fully', 'susceptible_vac', 'susceptible_treated', 'latent_early',
                                  'latent_late', 'active', 'detect', 'missed', 'treatment_infect',
                                  'treatment_noninfect']
        if self.is_lowquality: self.compartment_types += ['lowquality']
        if 'program_prop_novel_vaccination' in self.optional_timevariants:
            self.compartment_types += ['susceptible_novelvac']

        # Stages in progression through treatment
        self.treatment_stages = ['_infect', '_noninfect']

        # Compartments that contribute to force of infection calculations
        self.infectious_tags = ['active', 'missed', 'detect', 'treatment_infect', 'lowquality']

        # Get organ stratification and strains from inputs objects
        for attribute in ['organ_status', 'strains', 'comorbidities', 'is_organvariation', 'agegroups']:
            setattr(self, attribute, getattr(self.inputs, attribute))

        # Initialise compartments
        self.initial_compartments = {}
        for compartment in self.compartment_types:
            if compartment in self.inputs.model_constants:
                self.initial_compartments[compartment] \
                    = self.inputs.model_constants[compartment]

    def initialise_compartments(self):

        # First initialise all compartments to zero
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for compartment in self.compartment_types:

                    # Replicate susceptible for age-groups and comorbidities
                    if 'susceptible' in compartment:
                        self.set_compartment(compartment + comorbidity + agegroup, 0.)

                    # Replicate latent classes for age-groups, comorbidities and strains
                    elif 'latent' in compartment:
                        for strain in self.strains:
                            self.set_compartment(compartment + strain + comorbidity + agegroup, 0.)

                    # Replicate active classes for age-groups, comorbidities, strains and organ status
                    elif 'active' in compartment or 'missed' in compartment or 'lowquality' in compartment:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                self.set_compartment(compartment + organ + strain + comorbidity + agegroup, 0.)

                    # Replicate treatment classes for
                    # age-groups, comorbidities, strains, organ status and assigned strain
                    else:
                        for strain in self.strains:
                            for organ in self.organ_status:
                                if self.is_misassignment:
                                    for assigned_strain in self.strains:
                                        self.set_compartment(compartment + organ + strain +
                                                             '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                                             0.)
                                else:
                                    self.set_compartment(compartment +
                                                         organ + strain + comorbidity + agegroup, 0.)

        # Populate input_compartments
        # Initialise to DS-TB or no strain if single-strain model
        default_start_strain = '_ds'
        if self.strains == ['']: default_start_strain = ''

        # The equal splits will need to be adjusted, but the important thing is not to
        # initialise multiple strains too early, so that MDR-TB doesn't overtake the model

        if len(self.comorbidities) == 1:
            start_comorb_prop = {'': 1.}
        else:
            start_comorb_prop = {'_nocomorb': 1.}
            for comorbidity in self.comorbidities:
                if comorbidity != '_nocomorb':
                    start_comorb_prop[comorbidity] \
                        = self.scaleup_fns['comorb_prop' + comorbidity](self.inputs.model_constants['start_time'])
                    start_comorb_prop['_nocomorb'] \
                        -= start_comorb_prop[comorbidity]

        for compartment in self.compartment_types:
            if compartment in self.initial_compartments:
                for agegroup in self.agegroups:
                    for comorbidity in self.comorbidities:
                        if 'susceptible_fully' in compartment:
                            # Split equally by age-groups
                            self.set_compartment(compartment + comorbidity + agegroup,
                                                 self.initial_compartments[compartment]
                                                 * start_comorb_prop[comorbidity]
                                                 / len(self.agegroups))
                        elif 'latent' in compartment:
                            # Assign all to DS-TB, split equally by age-groups
                            self.set_compartment(compartment + default_start_strain + comorbidity + agegroup,
                                                 self.initial_compartments[compartment]
                                                 * start_comorb_prop[comorbidity]
                                                 / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(compartment +
                                                     organ + default_start_strain + comorbidity + agegroup,
                                                     self.initial_compartments[compartment]
                                                     / len(self.organ_status)
                                                     * start_comorb_prop[comorbidity]
                                                     / len(self.agegroups))

    #######################################################
    ### Single method to process uncertainty parameters ###
    #######################################################

    def process_uncertainty_params(self):

        """
        Perform some parameter processing - just for those that are used as uncertainty parameters and so can't be
        processed in the data_processing module.
        """

        # Find the case fatality of smear-negative TB using the relative case fatality
        # (previously the parameter was entered as the absolute case fatality)
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
                = (1 - self.params['tb_prop_casefatality_untreated' + organ]) \
                  / self.params['tb_timeperiod_activeuntreated']

    ##################################################################
    # Methods that calculate variables to be used in calculating flows
    # Note: all scaleup_fns are calculated and put into self.vars before
    # calculate_vars
    # I think we have to put any calculations that are dependent upon vars
    # into this section

    def calculate_vars(self):

        """
        The master method that calls all the other methods for the calculations of variable rates
        """

        # The parameter values are calculated from the costs, but only in the future
        if self.eco_drives_epi:
            if self.time > self.inputs.model_constants['current_time']:
                self.update_vars_from_cost()

        self.vars['population'] = sum(self.compartments.values())

        self.calculate_birth_rates_vars()

        self.calculate_force_infection_vars()

        if self.is_organvariation: self.calculate_progression_vars()

        self.calculate_acf_rate()

        self.calculate_detect_missed_vars()

        self.calculate_misassignment_detection_vars()

        if self.is_lowquality: self.calculate_lowquality_detection_vars()

        self.calculate_await_treatment_var()

        self.calculate_treatment_rates_vars()

        self.calculate_population_sizes()

        self.calculate_ipt_rate()

        self.calculate_community_ipt_rate()

    def calculate_birth_rates_vars(self):

        """
        Calculate birth rates into vaccinated and unvaccinated compartments.
        """

        # Calculate total births first, so that it can be tracked for interventions as well
        self.vars['births_total'] = self.get_constant_or_variable_param('demo_rate_birth') / 1e3 \
                                    * self.vars['population']

        # Get the parameters depending on whether constant or time variant
        vac_props = {'vac': self.get_constant_or_variable_param('program_prop_vaccination')}
        vac_props['unvac'] = 1. - vac_props['vac']

        if 'program_prop_novel_vaccination' in self.optional_timevariants:
            vac_props['novelvac'] = self.get_constant_or_variable_param('program_prop_vaccination') \
                                    * self.vars['program_prop_novel_vaccination']
            vac_props['vac'] -= vac_props['novelvac']

        # Calculate the birth rates by compartment
        for comorbidity in self.comorbidities:
            for vac_status in vac_props:
                self.vars['births_' + vac_status + comorbidity] \
                    = vac_props[vac_status] \
                      * self.vars['births_total'] \
                      * self.target_comorb_props[comorbidity][-1]

    def calculate_force_infection_vars(self):

        """
        Calculate force of infection for each strain, incorporating partial immunity and infectiousness.
        First calculates the effective infectious population (incorporating infectiousness by organ involvement), then
        calculates the raw force of infection, then adjusts for various levels of susceptibility.
        """

        for strain in self.strains:

            # Initialise infectious population to zero
            self.vars['infectious_population' + strain] = 0.
            for organ in self.organ_status:
                for label in self.labels:

                    # If model is organ-stratified, but we haven't yet reached the organ of interest
                    if organ not in label and organ != '':
                        continue

                    # If model is strain-stratified, but we haven't yet reached the strain of interest
                    if strain not in label and strain != '':
                        continue

                    # If the compartment is infectious
                    if label_intersects_tags(label, self.infectious_tags):

                        # Allow modification for infectiousness by age
                        for agegroup in self.agegroups:
                            if agegroup in label:

                                # Add to the effective infectious population, adjusting for organ involvement and age
                                self.vars['infectious_population' + strain] += \
                                    self.params['tb_multiplier_force' + organ] \
                                    * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                    * self.compartments[label]

            # Calculate force of infection unadjusted for immunity/susceptibility
            self.vars['rate_force' + strain] = \
                self.params['tb_n_contact'] * self.vars['infectious_population' + strain] / self.vars['population']

            # If any modifications to transmission parameter to be made over time
            if 'transmission_modifier' in self.optional_timevariants:
                self.vars['rate_force' + strain] *= self.vars['transmission_modifier']

            # Adjust for immunity in various groups
            self.vars['rate_force_vacc' + strain] \
                = self.params['tb_multiplier_bcg_protection'] * self.vars['rate_force' + strain]
            self.vars['rate_force_latent' + strain] \
                = self.params['tb_multiplier_latency_protection'] * self.vars['rate_force' + strain]
            self.vars['rate_force_novelvacc' + strain] \
                = self.params['tb_multiplier_novelvac_protection'] * self.vars['rate_force' + strain]

    def calculate_progression_vars(self):

        """
        Calculate vars for the remainder of progressions.
        Note that the vars for the smear-positive and smear-negative proportions
        have already been calculated. However, all progressions have to go somewhere,
        so need to calculate the remaining proportions.
        """

        # If unstratified (self.organ_status should have length 0, but length 1 OK) - ??
        if len(self.organ_status) < 2:
            self.vars['epi_prop'] = 1.

        # Stratified into smear-positive and smear-negative
        elif len(self.organ_status) == 2:
            self.vars['epi_prop_smearneg'] = 1. - self.vars['epi_prop_smearpos']

        # Fully stratified into smear-positive, smear-negative and extra-pulmonary
        elif len(self.organ_status) > 2:
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

    def calculate_acf_rate(self):

        """
        Calculates rates of ACF from the proportion of programmatic coverage of both
        smear-based and Xpert-based ACF (both presuming symptom-based screening before this,
        as in the studies on which this is based).
        Smear-based screening only detects smear-positive disease, while Xpert-based screening
        detects some smear-negative disease, with a multiplier for the sensitivity of Xpert
        for smear-negative disease. (Extrapulmonary disease can't be detected through ACF.
        """

        for organ in self.organ_status:
            self.vars['program_rate_acf' + organ] = 0.

        if 'program_prop_xpertacf' in self.optional_timevariants:
            self.vars['program_rate_acf_smearpos'] \
                += self.vars['program_prop_smearacf'] \
                   * self.params['program_prop_acf_detections_per_round'] \
                   / self.params['program_timeperiod_acf_rounds']
            self.vars['program_rate_acf_smearpos'] \
                += self.vars['program_prop_xpertacf'] \
                   * self.params['program_prop_acf_detections_per_round'] \
                   / self.params['program_timeperiod_acf_rounds']
        if 'program_prop_smear_acf' in self.optional_timevariants:
            self.vars['program_rate_acf_smearneg'] \
                += self.vars['program_prop_xpertacf'] \
                   * self.params['tb_prop_xpert_smearneg_sensitivity'] \
                   * self.params['program_prop_acf_detections_per_round'] \
                   / self.params['program_timeperiod_acf_rounds']

    def calculate_detect_missed_vars(self):

        """"
        Calculate rates of detection and failure of detection
        from the programmatic report of the case detection "rate"
        (which is actually a proportion and referred to as program_prop_detect here)

        Derived from original formulas of by solving the simultaneous equations:
          algorithm sensitivity = detection rate / (detection rate + missed rate)
          - and -
          detection proportion = detection rate
                / (detection rate + spont recover rate + tb death rate + natural death rate)
        """

        vary_by_organ = False

        # Detection
        # Note that all organ types are assumed to have the same untreated active
        # sojourn time, so any organ status can be arbitrarily selected (here the first, or smear-positive)

        alg_sens = self.get_constant_or_variable_param('program_prop_algorithm_sensitivity')
        life_expectancy = self.get_constant_or_variable_param('demo_life_expectancy')

        # Calculate detection proportion, allowing for decentralisation coverage if being implemented
        detect_prop = self.get_constant_or_variable_param('program_prop_detect')
        if 'program_prop_decentralisation' in self.optional_timevariants:
            detect_prop += self.vars['program_prop_decentralisation'] \
                           * (self.params['program_ideal_detection']
                              - self.get_constant_or_variable_param('program_prop_detect'))

        # Weighting detection and algorithm sensitivity rates by organ status
        if vary_by_organ:

            def weight_by_organ_status(baseline_value):

                weighted_dict = {}
                weighted_dict['_smearpos'] \
                    = baseline_value \
                      / (self.vars['epi_prop_smearpos']
                         + self.params['program_prop_snep_relative_algorithm'] * (1. - self.vars['epi_prop_smearpos']))
                weighted_dict['_smearneg'] \
                    = weighted_dict['_smearpos'] * self.params['program_prop_snep_relative_algorithm']
                weighted_dict['_extrapul'] = weighted_dict['_smearneg']
                return weighted_dict

            detect_prop_by_organ = weight_by_organ_status(detect_prop)
            alg_sens_by_organ = weight_by_organ_status(alg_sens)

            for organ in self.organ_status:
                self.vars['program_rate_detect' + organ] \
                    = - detect_prop_by_organ[organ] \
                      * (1. / self.params['tb_timeperiod_activeuntreated'] + 1. / life_expectancy) \
                      / (detect_prop_by_organ[organ] - 1.)
                self.vars['program_rate_missed' + organ] \
                    = self.vars['program_rate_detect' + organ] \
                      * (1. - alg_sens_by_organ[organ]) / max(alg_sens_by_organ[organ], 1e-6)

        # Without weighting
        self.vars['program_rate_detect'] \
            = - detect_prop \
              * (1. / self.params['tb_timeperiod_activeuntreated'] + 1. / life_expectancy) \
              / (detect_prop - 1.)
        # Missed (avoid division by zero alg_sens with max)
        self.vars['program_rate_missed'] = self.vars['program_rate_detect'] * (1. - alg_sens) / max(alg_sens, 1e-6)

        # Calculate detection rates by organ stratum (should be the same for each strain)
        for organ in self.organ_status:
            if not vary_by_organ:
                for programmatic_rate in ['_detect', '_missed']:
                    self.vars['program_rate' + programmatic_rate + organ] \
                        = self.vars['program_rate' + programmatic_rate]

            # Add active case finding rate to standard DOTS-based detection rate
            if len(self.organ_status) > 1 \
                and ('program_prop_smearacf' in self.optional_timevariants
                     or 'program_prop_xpertacf' in self.optional_timevariants):
                self.vars['program_rate_detect' + organ] \
                    += self.vars['program_rate_acf' + organ]

    def calculate_await_treatment_var(self):

        """
        Take the reciprocal of the waiting times to calculate the flow rate to start
        treatment after detection.
        Note that the default behaviour for a single strain model is to use the
        waiting time for smear-positive patients.
        Also weight the time period
        """

        # If only one organ stratum
        if len(self.organ_status) == 1:
            self.vars['program_timeperiod_await_treatment'] = \
                self.get_constant_or_variable_param('program_timeperiod_await_treatment_smearpos')
            self.vars['program_rate_start_treatment'] = 1. / self.vars['program_timeperiod_await_treatment']

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
        Calculate rate of entry to low-quality care,
        form the proportion of treatment administered in low-quality sector

        Note that this now means that the case detection proportion only
        applies to those with access to care and so that proportion of all
        cases isn't actually detected
        """

        prop_lowqual = self.get_constant_or_variable_param('program_prop_lowquality')

        self.vars['program_rate_enterlowquality'] = \
            self.vars['program_rate_detect'] \
            * prop_lowqual \
            / (1. - prop_lowqual)

    def calculate_misassignment_detection_vars(self):

        """
        Calculate the proportions of patients assigned to each strain. (Note that second-line DST availability refers to
        the proportion of those with first-line DST who also have second-line DST available.)
        """

        # With misassignment:
        if self.is_misassignment:

            # If there are exactly two strains (DS and MDR)
            prop_firstline = self.get_constant_or_variable_param('program_prop_firstline_dst')

            # Add effect of Xpert on identification, assuming that it is distributed independently to conventional DST
            if 'program_prop_xpert' in self.optional_timevariants:
                prop_firstline += (1. - prop_firstline) * self.vars['program_prop_xpert']

            self.vars['program_rate_detect_ds_asds'] = self.vars['program_rate_detect']
            self.vars['program_rate_detect_ds_asmdr'] = 0.
            self.vars['program_rate_detect_mdr_asds'] = (1. - prop_firstline) * self.vars['program_rate_detect']
            self.vars['program_rate_detect_mdr_asmdr'] = prop_firstline * self.vars['program_rate_detect']

            # If a third strain is present
            if len(self.strains) > 2:
                prop_secondline = self.get_constant_or_variable_param('program_prop_secondline_dst')
                self.vars['program_rate_detect_ds_asxdr'] = 0.
                self.vars['program_rate_detect_mdr_asxdr'] = 0.
                self.vars['program_rate_detect_xdr_asds'] = (1. - prop_firstline) * self.vars['program_rate_detect']
                self.vars['program_rate_detect_xdr_asmdr'] = prop_firstline \
                                                             * (1. - prop_secondline) * self.vars['program_rate_detect']
                self.vars['program_rate_detect_xdr_asxdr'] = prop_firstline \
                                                             * prop_secondline * self.vars['program_rate_detect']

        # Without misassignment, everyone is correctly allocated
        else:
            for strain in self.strains:
                self.vars['program_rate_detect' + strain + '_as'+strain[1:]] = self.vars['program_rate_detect']

    def calculate_treatment_rates_vars(self):

        # May need to adjust this - a bit of a patch for now
        treatments = copy.copy(self.strains)
        if len(self.strains) > 1 and self.gui_inputs['is_misassignment']:
            treatments += ['_inappropriate']

        for strain in treatments:

            # Get treatment success proportion from vars if possible and from params if not
            for outcome in ['_success', '_death']:
                if 'program_prop_treatment' + outcome + strain in self.vars:
                    pass
                elif 'program_prop_treatment' + outcome + strain in self.params:
                    self.vars['program_prop_treatment' + outcome] = self.params['program_prop_treatment' + outcome]
                else:
                    raise NameError('program_prop_treatment' + outcome + strain + ' not found in vars or params')

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

            self.vars['program_prop_treatment_default' + strain] \
                = 1. \
                  - self.vars['program_prop_treatment_success' + strain] \
                  - self.vars['program_prop_treatment_death' + strain]

            # Find the proportion of deaths/defaults during the infectious and non-infectious stages
            for outcome in self.non_success_outcomes:
                early_proportion, late_proportion = find_outcome_proportions_by_period(
                    self.vars['program_prop_treatment' + outcome + strain],
                    self.params['tb_timeperiod_infect_ontreatment' + strain],
                    self.params['tb_timeperiod_treatment' + strain])
                self.vars['program_prop_treatment' + outcome + '_infect' + strain] = early_proportion
                self.vars['program_prop_treatment' + outcome + '_noninfect' + strain] = late_proportion

            for treatment_stage in self.treatment_stages:

                # Find the success proportions
                self.vars['program_prop_treatment_success' + treatment_stage + strain] = \
                    1. - self.vars['program_prop_treatment_default' + treatment_stage + strain] \
                    - self.vars['program_prop_treatment_death' + treatment_stage + strain]

                # Find the corresponding rates from the proportions
                for outcome in self.outcomes:
                    self.vars['program_rate' + outcome + treatment_stage + strain] = \
                        1. / self.params['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                        * self.vars['program_prop_treatment' + outcome + treatment_stage + strain]
                if self.is_amplification:
                    self.vars['program_rate_default' + treatment_stage + '_amplify' + strain] = \
                        self.vars['program_rate_default' + treatment_stage + strain] \
                        * self.vars['epi_prop_amplification']
                    self.vars['program_rate_default' + treatment_stage + '_noamplify' + strain] = \
                        self.vars['program_rate_default' + treatment_stage + strain] \
                        * (1. - self.vars['epi_prop_amplification'])

    def calculate_population_sizes(self):

        """
        Calculate the size of the populations to which each intervention is applicable
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

        # BCG (So simple that it's possibly unnecessary, but may be needed for loops over programs)
        self.vars['popsize_vaccination'] = self.vars['births_total']

        # Xpert - all presentations with active TB
        if 'program_prop_xpert' in self.optional_timevariants:
            self.vars['popsize_xpert'] = 0.
            for agegroup in self.agegroups:
                for comorbidity in self.comorbidities:
                    for strain in self.strains:
                        self.vars['popsize_xpert'] += (self.vars['program_rate_detect']
                                                       + self.vars['program_rate_missed']) \
                                                      * self.compartments['active'
                                                                          + organ + strain + comorbidity + agegroup] \
                                                      * (self.params['program_number_tests_per_tb_presentation'] + 1.)

        # ACF
        if 'program_prop_smearacf' in self.optional_timevariants:
            self.vars['popsize_smearacf'] = 0.
            for compartment in self.compartments:
                if 'active_' in compartment and '_smearpos' in compartment:
                    self.vars['popsize_smearacf'] \
                        += self.compartments[compartment] \
                           * self.params['program_nns_smearacf']
        if 'program_prop_xpertacf' in self.optional_timevariants:
            self.vars['popsize_xpertacf'] = 0.
            for compartment in self.compartments:
                if 'active_' in compartment and '_smearpos' in compartment:
                    self.vars['popsize_xpertacf'] \
                        += self.compartments[compartment] \
                           * self.params['program_nns_xpertacf_smearpos']
                elif 'active_' in compartment and '_smearneg' in compartment:
                    self.vars['popsize_xpertacf'] \
                        += self.compartments[compartment] \
                           * self.params['program_nns_xpertacf_smearneg']

        # Decentralisation
        if 'program_prop_decentralisation' in self.optional_timevariants:
            self.vars['popsize_decentralisation'] = 0.
            for compartment in self.compartments:
                if 'susceptible_' not in compartment and 'latent_' not in compartment:
                    self.vars['popsize_decentralisation'] \
                        += self.compartments[compartment]

    def calculate_ipt_rate(self):

        """
        Uses the popsize to which IPT is applicable, which was calculated in calculate_population_sizes
        to determine the actual number of persons who should be shifted across compartments.
        """

        for agegroup in self.agegroups:

            # Find IPT coverage for the age group as the maximum of the coverage in that age group
            # and the overall coverage.
            prop_ipt = 0.
            if 'program_prop_ipt' + agegroup in self.vars:
                prop_ipt += self.vars['program_prop_ipt' + agegroup]
            if 'program_prop_ipt' in self.vars:
                prop_ipt = max([self.vars['program_prop_ipt'], prop_ipt])

            # Then for the "game-changing" novel version of IPT
            prop_novel_ipt = 0.
            if 'program_prop_novel_ipt' + agegroup in self.vars:
                prop_novel_ipt += self.vars['program_prop_novel_ipt' + agegroup]
            elif 'program_prop_novel_ipt' in self.vars:
                prop_novel_ipt = max([self.vars['program_prop_novel_ipt'], prop_novel_ipt])

            # Calculate the number of effective treatments
            self.vars['standard_ipt_effective_treatments' + agegroup] = prop_ipt \
                                                                        * self.vars['popsize_ipt' + agegroup] \
                                                                        * self.inputs.model_constants[
                                                                            'ipt_effective_per_assessment']
            self.vars['novel_ipt_effective_treatments' + agegroup] = prop_novel_ipt \
                                                                     * self.vars['popsize_ipt' + agegroup] \
                                                                     * self.inputs.model_constants[
                                                                         'novel_ipt_effective_per_assessment']

            # Check size of latency compartments
            latent_early = 0.
            for compartment in self.compartments:
                if 'latent_early' in compartment and agegroup in compartment:
                    latent_early += self.compartments[compartment]

            # Calculate the total number of effective treatments across both forms of IPT, limiting at all latents
            self.vars['ipt_effective_treatments' + agegroup]\
                = min([max([self.vars['novel_ipt_effective_treatments' + agegroup],
                            self.vars['standard_ipt_effective_treatments' + agegroup]]),
                       latent_early])

    def calculate_community_ipt_rate(self):

        if 'program_prop_community_ipt' in self.scaleup_fns:
            self.vars['rate_community_ipt'] \
                = self.vars['program_prop_community_ipt'] \
                  * self.inputs.model_constants['ipt_effective_per_assessment'] \
                  / self.inputs.model_constants['program_timeperiod_community_ipt_round']

    ##################################################################
    # Methods that calculate the flows of all the compartments

    def set_flows(self):

        """
        Call all the rate setting methods
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

        self.set_ipt_flows()

    def set_birth_flows(self):

        # Set birth flows
        for comorbidity in self.comorbidities:
            self.set_var_entry_rate_flow(
                'susceptible_fully' + comorbidity + self.agegroups[0], 'births_unvac' + comorbidity)
            self.set_var_entry_rate_flow(
                'susceptible_vac' + comorbidity + self.agegroups[0], 'births_vac' + comorbidity)
            if 'program_prop_novel_vaccination' in self.optional_timevariants:
                self.set_var_entry_rate_flow(
                    'susceptible_novelvac' + comorbidity + self.agegroups[0], 'births_novelvac' + comorbidity)

    def set_infection_flows(self):

        # Set force of infection flows
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:

                    # Fully susceptible
                    self.set_var_transfer_rate_flow(
                        'susceptible_fully' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force' + strain)

                    # Partially immune
                    self.set_var_transfer_rate_flow(
                        'susceptible_vac' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_vacc' + strain)
                    self.set_var_transfer_rate_flow(
                        'susceptible_treated' + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_vacc' + strain)
                    self.set_var_transfer_rate_flow(
                        'latent_late' + strain + comorbidity + agegroup,
                        'latent_early' + strain + comorbidity + agegroup,
                        'rate_force_latent' + strain)

                    # For novel vaccination
                    if 'program_prop_novel_vaccination' in self.optional_timevariants:
                        self.set_var_transfer_rate_flow(
                            'susceptible_novelvac' + comorbidity + agegroup,
                            'latent_early' + strain + comorbidity + agegroup,
                            'rate_force_novelvacc' + strain)

    def set_progression_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:

                        # Stabilisation
                        self.set_fixed_transfer_rate_flow(
                            'latent_early' + strain + comorbidity + agegroup,
                            'latent_late' + strain + comorbidity + agegroup,
                            'tb_rate_stabilise' + comorbidity + agegroup)

                        for organ in self.organ_status:

                            # If organ scale-ups available, set flows as variable
                            # (if epi_prop_smearpos is in self.scaleup_fns, then epi_prop_smearneg
                            # should be too)
                            if self.is_organvariation:
                                # Early progression
                                self.set_var_transfer_rate_flow(
                                    'latent_early' + strain + comorbidity + agegroup,
                                    'active' + organ + strain + comorbidity + agegroup,
                                    'tb_rate_early_progression' + organ + comorbidity + agegroup)

                                # Late progression
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

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Recovery, base compartments
                        self.set_fixed_transfer_rate_flow(
                            'active' + organ + strain + comorbidity + agegroup,
                            'latent_late' + strain + comorbidity + agegroup,
                            'tb_rate_recover' + organ)
                        self.set_fixed_transfer_rate_flow(
                            'missed' + organ + strain + comorbidity + agegroup,
                            'latent_late' + strain + comorbidity + agegroup,
                            'tb_rate_recover' + organ)

                        # Death, base compartments
                        self.set_fixed_infection_death_rate_flow(
                            'active' + organ + strain + comorbidity + agegroup,
                            'tb_rate_death' + organ)
                        self.set_fixed_infection_death_rate_flow(
                            'missed' + organ + strain + comorbidity + agegroup,
                            'tb_rate_death' + organ)

                        # Extra low-quality compartments
                        if self.is_lowquality:
                            self.set_fixed_transfer_rate_flow(
                                'lowquality' + organ + strain + comorbidity + agegroup,
                                'latent_late' + strain + comorbidity + agegroup,
                                'tb_rate_recover' + organ)
                            self.set_fixed_infection_death_rate_flow(
                                'lowquality' + organ + strain + comorbidity + agegroup,
                                'tb_rate_death' + organ)

                        # Detected, with misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                self.set_fixed_infection_death_rate_flow(
                                    'detect' +
                                    organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                    'tb_rate_death' + organ)
                                self.set_fixed_transfer_rate_flow(
                                    'detect' +
                                    organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                    'latent_late' + strain + comorbidity + agegroup,
                                    'tb_rate_recover' + organ)

                        # Detected, without misassignment
                        else:
                            self.set_fixed_transfer_rate_flow(
                                'detect' + organ + strain + comorbidity + agegroup,
                                'latent_late' + strain + comorbidity + agegroup,
                                'tb_rate_recover' + organ)
                            self.set_fixed_infection_death_rate_flow(
                                'detect' + organ + strain + comorbidity + agegroup,
                                'tb_rate_death' + organ)

    def set_fixed_programmatic_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Re-start presenting after a missed diagnosis
                        self.set_fixed_transfer_rate_flow(
                            'missed' + organ + strain + comorbidity + agegroup,
                            'active' + organ + strain + comorbidity + agegroup,
                            'program_rate_restart_presenting')

                        # Give up on the hopeless low-quality health system
                        if self.is_lowquality:
                            self.set_fixed_transfer_rate_flow(
                                'lowquality' + organ + strain + comorbidity + agegroup,
                                'active' + organ + strain + comorbidity + agegroup,
                                'program_rate_leavelowquality')

    def set_detection_flows(self):

        """
        Set previously calculated detection rates
        Either assuming everyone is correctly identified if misassignment not permitted
        or with proportional misassignment
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for organ in self.organ_status:
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
                                        'program_rate_detect' + strain + as_assigned_strain + organ)

                        # Without misassignment - everyone is correctly identified by strain
                        else:
                            self.set_var_transfer_rate_flow(
                                'active' + organ + strain + comorbidity + agegroup,
                                'detect' + organ + strain + comorbidity + agegroup,
                                'program_rate_detect' + organ)

    def set_variable_programmatic_flows(self):

        # Set rate of missed diagnoses and entry to low-quality health care
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:
                        self.set_var_transfer_rate_flow(
                            'active' + organ + strain + comorbidity + agegroup,
                            'missed' + organ + strain + comorbidity + agegroup,
                            'program_rate_missed')
                        # Detection, with and without misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                # Following line only for models incorporating mis-assignment
                                self.set_var_transfer_rate_flow(
                                    'detect' +
                                    organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                    'treatment_infect' +
                                    organ + strain + '_as' + assigned_strain[1:] + comorbidity + agegroup,
                                    'program_rate_start_treatment' + organ)
                        else:
                            # Following line is the currently running line (without mis-assignment)
                            self.set_var_transfer_rate_flow(
                                'detect' + organ + strain + comorbidity + agegroup,
                                'treatment_infect' + organ + strain + comorbidity + agegroup,
                                'program_rate_start_treatment' + organ)

                        if self.is_lowquality:
                            self.set_var_transfer_rate_flow(
                                'active' + organ + strain + comorbidity + agegroup,
                                'lowquality' + organ + strain + comorbidity + agegroup,
                                'program_rate_enterlowquality')

    def set_treatment_flows(self):

        """
        Set rates of progression through treatment stages
        Accommodates with and without amplification, and with and without misassignment
        """
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for organ in self.organ_status:

                    for actual_strain_number in range(len(self.strains)):
                        strain = self.strains[actual_strain_number]
                        if self.is_misassignment:
                            assignment_strains = range(len(self.strains))
                        else:
                            assignment_strains = ['']
                        for assigned_strain_number in assignment_strains:
                            if self.is_misassignment:
                                as_assigned_strain = '_as' + self.strains[assigned_strain_number][1:]

                                # Which treatment parameters to use - for the strain or for inappropriate treatment
                                if actual_strain_number > assigned_strain_number:
                                    strain_or_inappropriate = '_inappropriate'
                                else:
                                    strain_or_inappropriate = self.strains[assigned_strain_number]

                            else:
                                as_assigned_strain = ''
                                strain_or_inappropriate = self.strains[actual_strain_number]

                            # Success at either treatment stage
                            self.set_var_transfer_rate_flow(
                                'treatment_infect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'treatment_noninfect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'program_rate_success_infect' + strain_or_inappropriate)
                            self.set_var_transfer_rate_flow(
                                'treatment_noninfect' + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                'susceptible_treated' + comorbidity + agegroup,
                                'program_rate_success_noninfect' + strain_or_inappropriate)

                            # Rates of death on treatment
                            for treatment_stage in self.treatment_stages:
                                self.set_var_infection_death_rate_flow(
                                    'treatment' +
                                    treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                    'program_rate_death' + treatment_stage + strain_or_inappropriate)

                            # Default
                            for treatment_stage in self.treatment_stages:

                                # If it's either the most resistant strain available or amplification is not active:
                                if actual_strain_number == len(self.strains) - 1 or not self.is_amplification:
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + strain_or_inappropriate)

                                # Otherwise
                                else:
                                    amplify_to_strain = self.strains[actual_strain_number + 1]
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + '_noamplify' + strain_or_inappropriate)
                                    self.set_var_transfer_rate_flow(
                                        'treatment' +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        'active' + organ + amplify_to_strain + comorbidity + agegroup,
                                        'program_rate_default' + treatment_stage + '_amplify' + strain_or_inappropriate)

    def set_ipt_flows(self):

        """
        Sets a flow from the early latent compartment to the partially immune susceptible compartment
        that is determined by report_numbers_starting_treatment above and is not linked to the
        'from_label' compartment.
        """

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    self.set_linked_transfer_rate_flow('latent_early' + strain + comorbidity + agegroup,
                                                       'susceptible_vac' + comorbidity + agegroup,
                                                       'ipt_effective_treatments' + agegroup)
                    if 'program_prop_community_ipt' in self.scaleup_fns:
                        self.set_var_transfer_rate_flow('latent_early' + strain + comorbidity + agegroup,
                                                        'susceptible_vac' + comorbidity + agegroup,
                                                        'rate_community_ipt')
                        self.set_var_transfer_rate_flow('latent_late' + strain + comorbidity + agegroup,
                                                        'susceptible_vac' + comorbidity + agegroup,
                                                        'rate_community_ipt')

