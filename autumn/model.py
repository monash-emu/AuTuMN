# -*- coding: utf-8 -*-


"""

The TB-specific model, or models, should be coded in this file

Time unit throughout: years
Compartment unit throughout: patients

"""

from scipy import exp, log
from autumn.base import BaseModel


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


class ConsolidatedModel(BaseModel):

    """
    The transmission dynamic model to underpin all AuTuMN analyses
    Inherits from BaseModel, which is intended to be general to any infectious disease
    All TB-specific methods and structures are contained in this model
    Methods are written to be adaptable to any model structure selected through the __init__ arguments

    The work-flow of the simulation is structured into the following parts:
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

    def __init__(self,
                 scenario=None,
                 inputs=None,
                 gui_inputs=None):

        BaseModel.__init__(self)

        self.inputs = inputs
        self.gui_inputs = gui_inputs
        self.country = self.gui_inputs['country']

        # Needed in base.py to work out whether to load a previous model state
        self.loaded_compartments = None

        # Set Boolean conditionals for model structure and additional diagnostics
        self.is_lowquality = self.gui_inputs['is_lowquality']
        self.is_amplification = self.gui_inputs['is_amplification']
        self.is_misassignment = self.gui_inputs['is_misassignment']
        if self.is_misassignment:
            assert self.is_amplification, 'Misassignment requested without amplification'

        self.scenario = scenario

        # Define model compartmental structure
        # (note that compartment initialisation has now been shifted to base.py)
        self.define_model_structure()

        # Set other fixed parameters
        for key, value in self.inputs.model_constants.items():
            if type(value) == float:
                self.set_parameter(key, value)

        # Treatment outcomes that will be universal to all models
        # Global TB outcomes of "completion" and "cure" can be considered "_success",
        # "death" is "_death" (of course), "failure" and "default" are considered "_default"
        # and "transfer out" is removed from denominator calculations.
        self.outcomes = ['_success', '_death', '_default']
        self.non_success_outcomes = self.outcomes[1: 3]

        # Get scaleup functions from input object
        self.scaleup_fns = self.inputs.scaleup_fns[self.scenario]

        self.check_list_of_interventions()
        self.find_intervention_startdates()
        if self.eco_drives_epi:
            self.distribute_funding_across_years()
        self.target_comorb_props = {}

    def define_model_structure(self):

        """
        Args:
            All arguments are set through __init__
            Please refer to __init__ method comments above
        """

        # All compartmental disease stages
        self.compartment_types = [
            'susceptible_fully',
            'susceptible_vac',
            'susceptible_treated',
            'latent_early',
            'latent_late',
            'active',
            'detect',
            'missed',
            'treatment_infect',
            'treatment_noninfect']
        if self.is_lowquality: self.compartment_types += ['lowquality']

        # Broader stages for calculating outputs later
        # Now thinking this should eventually go into the plotting module

        # Stages in progression through treatment
        self.treatment_stages = [
            '_infect',
            '_noninfect']

        # Compartments that contribute to force of infection calculations
        self.infectious_tags = [
            'active',
            'missed',
            'detect',
            'treatment_infect']
        if self.is_lowquality: self.infectious_tags += ['lowquality']

        # Get organ stratification and strains from inputs objects
        self.organ_status = self.inputs.organ_status
        self.is_organvariation = self.inputs.is_organvariation
        self.strains = self.inputs.strains

        self.comorbidities = self.inputs.comorbidities

        # Age stratification
        self.agegroups = self.inputs.agegroups

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
                        for actual_strain_number in range(len(self.strains)):
                            strain = self.strains[actual_strain_number]
                            for organ in self.organ_status:
                                if self.is_misassignment:
                                    for assigned_strain_number in range(len(self.strains)):
                                        self.set_compartment(
                                            compartment + organ + strain +
                                            '_as' + self.strains[assigned_strain_number][1:] +
                                            comorbidity + agegroup, 0.)
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
                            # Split equally by comorbidities and age-groups
                            self.set_compartment(compartment + comorbidity + agegroup,
                                                 self.initial_compartments[compartment]
                                                 * start_comorb_prop[comorbidity]
                                                 / len(self.agegroups))
                        elif 'latent' in compartment:
                            # Assign all to DS-TB, split equally by comorbidities and age-groups
                            self.set_compartment(compartment + default_start_strain + comorbidity + agegroup,
                                                 self.initial_compartments[compartment]
                                                 * start_comorb_prop[comorbidity]
                                                 / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(compartment +
                                                     organ + default_start_strain + comorbidity + agegroup,
                                                     self.initial_compartments[compartment]
                                                     / len(self.organ_status)  # Split equally by organ statuses,
                                                     * start_comorb_prop[comorbidity]
                                                     / len(self.agegroups))  # and split equally by age-groups

    ##################################################################
    # Methods that calculate variables to be used in calculating flows
    # Note: all scaleup_fns are calculated and put into self.vars before
    # calculate_vars
    # I think we have to put any calculations that are dependent upon vars
    # into this section

    def find_target_comorb_props(self):

        if len(self.comorbidities) > 1:
            # Find target comorbidity proportions
            self.target_comorb_props['_nocomorb'] = 1.
            for comorbidity in self.comorbidities:
                if comorbidity != '_nocomorb':
                    self.target_comorb_props[comorbidity] \
                        = self.get_constant_or_variable_param('comorb_prop' + comorbidity)
                    self.target_comorb_props['_nocomorb'] \
                        -= self.target_comorb_props[comorbidity]
        else:
            self.target_comorb_props[''] = 1.

    def calculate_vars(self):

        """
        The master method that calls all the other methods for the calculations of
        variable rates
        """

        # the parameter values are calculated from the costs, but only in the future
        if self.eco_drives_epi:
            if self.time > self.inputs.model_constants['current_time']:
                self.update_vars_from_cost()

        self.vars['population'] = sum(self.compartments.values())

        self.calculate_birth_rates_vars()

        self.calculate_force_infection_vars()

        self.calculate_progression_vars()

        self.calculate_acf_rate()

        self.calculate_detect_missed_vars()

        self.calculate_proportionate_detection_vars()

        if self.is_lowquality: self.calculate_lowquality_detection_vars()

        self.calculate_await_treatment_var()

        self.calculate_treatment_rates_vars()

        self.calculate_population_sizes()

        self.calculate_ipt_rate()

    def calculate_birth_rates_vars(self):

        """
        Calculate birth rates into vaccinated and unvaccinated compartments
        """

        # Get the parameters depending on whether constant or time variant
        rate_birth = self.get_constant_or_variable_param('demo_rate_birth') / 1e3
        prop_vacc = self.get_constant_or_variable_param('program_prop_vaccination')

        # Calculate total births first, so that it can be tracked for interventions as well
        self.vars['births_total'] = \
            rate_birth \
            * self.vars['population']

        # Calculate the birth rates by compartment
        for comorbidity in self.comorbidities:

            # Then split for model implementation
            self.vars['births_unvac' + comorbidity] = \
                (1. - prop_vacc) \
                * self.vars['births_total'] \
                * self.target_comorb_props[comorbidity]
            self.vars['births_vac' + comorbidity] = \
                prop_vacc \
                * self.vars['births_total'] \
                * self.target_comorb_props[comorbidity]

    def calculate_force_infection_vars(self):

        # Calculate force of infection by strain,
        # incorporating partial immunity and infectiousness

        for strain in self.strains:

            # Initialise infectious population to zero
            self.vars['infectious_population' + strain] = 0.
            for organ in self.organ_status:
                for label in self.labels:

                    # If we haven't reached the part of the model divided by organ status
                    # and we're not doing the organ status of interest.
                    if organ not in label and organ != '':
                        continue

                    # If we haven't reached the part of the model divided by strain
                    # and we're not doing the strain of interest.
                    if strain not in label and strain != '':
                        continue

                    # If the compartment is non-infectious
                    if label_intersects_tags(label, self.infectious_tags):

                        # Allow modification for infectiousness by age
                        for agegroup in self.agegroups:
                            if agegroup in label:

                                # Calculate the effective infectious population
                                self.vars['infectious_population' + strain] += \
                                    self.params['tb_multiplier_force' + organ] \
                                    * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                    * self.compartments[label]

            # Calculate non-immune force of infection
            self.vars['rate_force' + strain] = \
                self.params['tb_n_contact'] \
                * self.vars['infectious_population' + strain] \
                / self.vars['population']

            # Adjust for immunity
            self.vars['rate_force_vacc' + strain] = \
                self.params['tb_multiplier_bcg_protection'] \
                * self.vars['rate_force' + strain]
            self.vars['rate_force_latent' + strain] = \
                self.params['tb_multiplier_latency_protection'] \
                * self.vars['rate_force' + strain]

    def calculate_progression_vars(self):

        """
        Calculate vars for the remainder of progressions.
        Note that the vars for the smear-positive and smear-negative proportions
        have already been calculated. However, all progressions have to go somewhere,
        so need to calculate the remaining proportions.
        """

        if self.is_organvariation:

            # If unstratified (self.organ_status should have length 0, but length 1 OK)
            if len(self.organ_status) < 2:
                self.vars['epi_prop'] = 1.

            # Stratified into smear-positive and smear-negative
            elif len(self.organ_status) == 2:
                self.vars['epi_prop_smearneg'] = \
                    1. - self.vars['epi_prop_smearpos']

            # Fully stratified into smear-positive, smear-negative and extra-pulmonary
            elif len(self.organ_status) > 2:
                self.vars['epi_prop_extrapul'] = \
                    1. - self.vars['epi_prop_smearpos'] - self.vars['epi_prop_smearneg']

            # Determine variable progression rates
            for organ in self.organ_status:
                for agegroup in self.agegroups:
                    for comorbidity in self.comorbidities:
                        for timing in ['_early', '_late']:
                            if comorbidity == '_diabetes':
                                rr_diabetes = self.inputs.scaleup_fns[None]['epi_rr_diabetes'](self.time)
                                self.vars['tb_rate' + timing + '_progression' + organ + comorbidity + agegroup] \
                                    = self.vars['epi_prop' + organ] \
                                      * self.params[
                                          'tb_rate' + timing + '_progression' + '_nocomorb' + agegroup] * rr_diabetes
                            else:
                                self.vars['tb_rate' + timing + '_progression' + organ + comorbidity + agegroup] \
                                    = self.vars['epi_prop' + organ] \
                                      * self.params['tb_rate' + timing + '_progression' + comorbidity + agegroup]

                            # self.vars['tb_rate' + timing + '_progression' + organ + comorbidity + agegroup] \
                            #     = self.vars['epi_prop' + organ] \
                            #       * self.params['tb_rate' + timing + '_progression' + comorbidity + agegroup]

    def calculate_acf_rate(self):

        """
        Calculates rates of ACF from the proportion of programmatic coverage of both
        smear-based and Xpert-based ACF (both presuming symptom-based screening before this,
        as in the studies on which this is based).
        Smear-based screening only detects smear-positive disease, while Xpert-based screening
        detects some smear-negative disease, with a multiplier for the sensitivity of Xpert
        for smear-negative disease. (Extrapulmonary disease can't be detected through ACF.
        """

        # Additional detection rate for smear-positive TB
        self.vars['program_rate_acf_smearpos'] \
            = (self.vars['program_prop_smearacf'] + self.vars['program_prop_xpertacf']) \
              * self.params['program_prop_acf_detections_per_round'] \
              / self.params['program_timeperiod_acf_rounds']

        # Additional detection rate for smear-negative TB
        self.vars['program_rate_acf_smearneg'] \
            = self.vars['program_prop_xpertacf'] \
              * self.params['tb_prop_ltbi_test_sensitivity'] \
              * self.params['program_prop_acf_detections_per_round'] \
              / self.params['program_timeperiod_acf_rounds']

        # No additional detection rate for extra-pulmonary TB, but add a var to allow loops to operate
        self.vars['program_rate_acf_extrapul'] \
            = 0.

    def calculate_detect_missed_vars(self):

        """"
        Calculate rates of detection and failure of detection
        from the programmatic report of the case detection "rate"
        (which is actually a proportion and referred to as program_prop_detect here)

        Derived from original formulas of by solving the simultaneous equations:
          algorithm sensitivity = detection rate / (detection rate + missed rate)
          - and -
          detection proportion = detection rate / (detection rate + spont recover rate + tb death rate + natural death rate)
        """

        # Detection
        # Note that all organ types are assumed to have the same untreated active
        # sojourn time, so any organ status can be arbitrarily selected (here the first, or smear-positive)

        alg_sens = self.get_constant_or_variable_param('program_prop_algorithm_sensitivity')
        life_expectancy = self.get_constant_or_variable_param('demo_life_expectancy')

        # Calculate detection proportion, allowing for decentralisation coverage if being implemented
        if self.vars['program_prop_decentralisation'] > 0.:
            detect_prop = self.get_constant_or_variable_param('program_prop_detect') \
                          + self.vars['program_prop_decentralisation'] \
                            * (self.params['program_ideal_detection']
                               - self.get_constant_or_variable_param('program_prop_detect'))
        else:
            detect_prop = self.get_constant_or_variable_param('program_prop_detect')

        # If no division by zero
        if alg_sens > 0.:

            # Detections
            self.vars['program_rate_detect'] = \
                - detect_prop \
                * (self.params['tb_rate_recover' + self.organ_status[0]] +
                   self.params['tb_rate_death' + self.organ_status[0]] +
                   1. / life_expectancy) \
                / (detect_prop - 1.)

            # Missed
            self.vars['program_rate_missed'] = \
                self.vars['program_rate_detect'] \
                * (1. - alg_sens) \
                / alg_sens

        # Otherwise just assign detection and missed rates to zero
        else:
            self.vars['program_rate_detect'] = 0.
            self.vars['program_rate_missed'] = 0.

        # Repeat for each strain
        for strain in self.strains:
            for organ in self.organ_status:
                for programmatic_rate in ['_detect', '_missed']:
                    self.vars['program_rate' + programmatic_rate + strain + organ] \
                        = self.vars['program_rate' + programmatic_rate]

                    # Add active case finding rate to standard DOTS-based detection rate
                    if programmatic_rate == '_detect' and len(self.organ_status) >= 2:
                        self.vars['program_rate' + programmatic_rate + strain + organ] \
                            += self.vars['program_rate_acf' + organ]

    def calculate_await_treatment_var(self):

        """
        Take the reciprocal of the waiting times to calculate the flow rate to start
        treatment after detection.
        Note that the default behaviour for a single strain model is to use the
        waiting time for smear-positive patients.
        Also weight the time period
        """

        # If not stratified by organ status, use the smear-positive value
        if len(self.organ_status) < 2:
            self.vars['program_timeperiod_await_treatment'] = \
                self.get_constant_or_variable_param('program_timeperiod_await_treatment_smearpos')

        # Otherwise, use the organ-specific value
        else:
            for organ in self.organ_status:
                self.vars['program_timeperiod_await_treatment' + organ] = \
                    self.get_constant_or_variable_param('program_timeperiod_await_treatment' + organ)

        prop_xpert = self.get_constant_or_variable_param('program_prop_xpert')

        # If only one organ stratum
        if len(self.organ_status) < 2:
            self.vars['program_rate_start_treatment'] = \
                1. / self.vars['program_timeperiod_await_treatment']
        else:
            for organ in self.organ_status:
                if organ == '_smearneg':
                    self.vars['program_rate_start_treatment_smearneg'] = \
                        1. / (self.vars['program_timeperiod_await_treatment_smearneg'] * (1. - prop_xpert)
                            + self.params['program_timeperiod_await_treatment_smearneg_xpert'] * prop_xpert)
                else:
                    self.vars['program_rate_start_treatment' + organ] = \
                        1. / self.vars['program_timeperiod_await_treatment' + organ]

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

    def calculate_proportionate_detection_vars(self):

        """
        Calculate the proportions of patients assigned to each strain
        """

        # With misassignment:

        # Note that second-line DST availability refers to the proportion
        # of those with first-line DST who also have second-line DST available
        # (rather than the proportion of all presentations with second line
        # treatment available)
        if self.is_misassignment:

            prop_firstline = self.get_constant_or_variable_param('program_prop_firstline_dst')
            prop_secondline = self.get_constant_or_variable_param('program_prop_secondline_dst')

            # DS-TB
            self.vars['program_rate_detect_ds_asds'] = \
                self.vars['program_rate_detect']
            self.vars['program_rate_detect_ds_asmdr'] = 0.
            self.vars['program_rate_detect_ds_asxdr'] = 0.

            # MDR-TB
            self.vars['program_rate_detect_mdr_asds'] = \
                (1. - prop_firstline) \
                * self.vars['program_rate_detect']
            self.vars['program_rate_detect_mdr_asmdr'] = \
                prop_firstline \
                * self.vars['program_rate_detect']
            self.vars['program_rate_detect_mdr_asxdr'] = 0.

            # XDR-TB
            self.vars['program_rate_detect_xdr_asds'] = \
                (1. - prop_firstline) \
                 * self.vars['program_rate_detect']
            self.vars['program_rate_detect_xdr_asmdr'] = \
                prop_firstline \
                * (1. - prop_secondline)\
                * self.vars['program_rate_detect']
            self.vars['program_rate_detect_xdr_asxdr'] = \
                prop_firstline \
                * prop_secondline \
                * self.vars['program_rate_detect']

        # Without misassignment:
        else:
            for strain in self.strains:
                self.vars['program_rate_detect' + strain + '_as'+strain[1:]] = \
                    self.vars['program_rate_detect']

    def calculate_treatment_rates_vars(self):

        # May need to adjust this - a bit of a patch for now
        treatments = self.strains
        if len(self.strains) > 1:
            treatments += ['_inappropriate']

        for strain in treatments:

            # Get treatment success proportion from vars if possible and from params if not
            for outcome in ['_success', '_death']:
                if 'program_prop_treatment'+outcome+strain in self.vars:
                    pass
                elif 'program_prop_treatment'+outcome+strain in self.params:
                    self.vars['program_prop_treatment' + outcome] = \
                        self.params['program_prop_treatment' + outcome]
                else:
                    raise NameError('program_prop_treatment' + outcome + strain + ' not found in vars or params')

            # Add some extra treatment success if the treatment support program is active
            if 'program_prop_treatment_support' in self.vars and self.vars['program_prop_treatment_support'] > 0.:
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

            # Find the success proportions
            for treatment_stage in self.treatment_stages:
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
        self.vars['popsize_smearacf'] = 0.
        for compartment in self.compartments:
            if 'active_' in compartment and '_smearpos' in compartment:
                self.vars['popsize_smearacf'] \
                    += self.compartments[compartment] \
                       * self.params['program_nns_smearacf']
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
            elif 'program_prop_ipt' in self.vars and prop_ipt < self.vars['program_prop_ipt']:
                prop_ipt += self.vars['program_prop_ipt']

            # Multiply by the eligible population and by the number of effective treatments per person assessed
            self.vars['ipt_effective_treatments' + agegroup] = prop_ipt \
                                                               * self.vars['popsize_ipt' + agegroup] \
                                                               * self.inputs.model_constants[
                                                                   'ipt_effective_per_assessment']

    ##################################################################
    # Methods that calculate the flows of all the compartments

    def set_flows(self):

        """
        Call all the rate setting methods
        """

        self.set_birth_flows()

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
                    for actual_strain_number in range(len(self.strains)):
                        strain = self.strains[actual_strain_number]

                        # With misassignment
                        if self.is_misassignment:
                            for assigned_strain_number in range(len(self.strains)):
                                as_assigned_strain = '_as' + self.strains[assigned_strain_number][1:]
                                # If the strain is equally or more resistant than its assignment
                                if actual_strain_number >= assigned_strain_number:
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
                                                       'susceptible_vac' + strain + comorbidity + agegroup,
                                                       'ipt_effective_treatments' + agegroup)

    ##################################################################
    # Methods that calculate the output vars and diagnostic properties

    def calculate_output_vars(self):

        """
        Method similarly structured to calculate_output_vars, just replicated by strains

        """

        # Initialise dictionaries
        rate_incidence = {}
        rate_mortality = {}
        rate_notifications = {}

        # By strain
        for strain in self.strains:

            # Initialise scalars
            rate_incidence[strain] = 0.
            rate_mortality[strain] = 0.
            rate_notifications[strain] = 0.

            # Incidence
            for from_label, to_label, rate in self.var_transfer_rate_flows:
                if 'latent' in from_label and 'active' in to_label and strain in to_label:
                    rate_incidence[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            for from_label, to_label, rate in self.fixed_transfer_rate_flows:
                if 'latent' in from_label and 'active' in to_label and strain in to_label:
                    rate_incidence[strain] \
                        += self.compartments[from_label] * rate

            self.vars['incidence' + strain] \
                = rate_incidence[strain] / self.vars['population'] * 1E5

            # Notifications
            for from_label, to_label, rate in self.var_transfer_rate_flows:
                if 'active' in from_label and 'detect' in to_label and strain in from_label:
                    rate_notifications[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            self.vars['notifications' + strain] \
                = rate_notifications[strain]

            # Mortality
            for from_label, rate in self.fixed_infection_death_rate_flows:
                # Under-reporting factor included for those deaths not occurring on treatment
                if strain in from_label:
                    rate_mortality[strain] \
                        += self.compartments[from_label] * rate \
                            * self.params['program_prop_death_reporting']
            for from_label, rate in self.var_infection_death_rate_flows:
                if strain in from_label:
                    rate_mortality[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            self.vars['mortality' + strain] \
                = rate_mortality[strain] / self.vars['population'] * 1E5

        # Prevalence
        for strain in self.strains:
            self.vars['prevalence' + strain] = 0.
            for label in self.labels:
                if 'susceptible' not in label and \
                                'latent' not in label and strain in label:
                    self.vars['prevalence' + strain] \
                        += (self.compartments[label]
                        / self.vars['population'] * 1E5)

        # Summing MDR and XDR to get the total of all MDRs
        if len(self.strains) > 1:
            rate_incidence['all_mdr_strains'] = 0.
            if len(self.strains) > 1:
                for actual_strain_number in range(len(self.strains)):
                    strain = self.strains[actual_strain_number]
                    if actual_strain_number > 0:
                        rate_incidence['all_mdr_strains'] \
                            += rate_incidence[strain]
            self.vars['all_mdr_strains'] \
                = rate_incidence['all_mdr_strains'] / self.vars['population'] * 1E5
            # Convert to percentage
            self.vars['proportion_mdr'] \
                = self.vars['all_mdr_strains'] / self.vars['incidence'] * 1E2

        # By age group
        if len(self.agegroups) > 1:
            # Calculate outputs by age group - note that this code is fundamentally different
            # to the code above even though it looks similar, because the denominator
            # changes for age group, whereas it remains the whole population for strain calculations
            # (although should be able to use this code for comorbidities).
            for agegroup in self.agegroups:

                # Find age group denominator
                self.vars['population' + agegroup] = 0.
                for compartment in self.compartments:
                    if agegroup in compartment:
                        self.vars['population' + agegroup] \
                            += self.compartments[compartment]

                # Initialise scalars
                rate_incidence[agegroup] = 0.
                rate_mortality[agegroup] = 0.

                # Incidence
                for from_label, to_label, rate in self.var_transfer_rate_flows:
                    if 'latent' in from_label and 'active' in to_label and agegroup in to_label:
                        rate_incidence[agegroup] \
                            += self.compartments[from_label] * self.vars[rate]
                for from_label, to_label, rate in self.fixed_transfer_rate_flows:
                    if 'latent' in from_label and 'active' in to_label and agegroup in to_label:
                        rate_incidence[agegroup] \
                            += self.compartments[from_label] * rate
                self.vars['incidence' + agegroup] \
                    = rate_incidence[agegroup] / self.vars['population' + agegroup] * 1E5

                # Mortality
                for from_label, rate in self.fixed_infection_death_rate_flows:
                    # Under-reporting factor included for those deaths not occurring on treatment
                    if agegroup in from_label:
                        rate_mortality[agegroup] \
                            += self.compartments[from_label] * rate \
                               * self.params['program_prop_death_reporting']
                for from_label, rate in self.var_infection_death_rate_flows:
                    if agegroup in from_label:
                        rate_mortality[agegroup] \
                            += self.compartments[from_label] * self.vars[rate]
                self.vars['mortality' + agegroup] \
                    = rate_mortality[agegroup] / self.vars['population' + agegroup] * 1E5

        # Prevalence
        for agegroup in self.agegroups:
            self.vars['prevalence' + agegroup] = 0.
            for label in self.labels:
                if 'susceptible' not in label and \
                                'latent' not in label and agegroup in label:
                    self.vars['prevalence' + agegroup] \
                        += (self.compartments[label]
                        / self.vars['population' + agegroup] * 1E5)


