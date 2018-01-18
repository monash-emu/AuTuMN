
"""
All TB-specific models should be coded in this file, including the interventions applied to them.

Time unit throughout: years
Compartment unit throughout: patients

Nested inheritance from BaseModel and StratifiedModel in base.py - the former sets some fundamental methods for
creating intercompartmental flows, costs, etc., while the latter sets out the approach to population stratification.
"""

# external imports
from scipy import exp, log
import copy
import warnings
import itertools

# AuTuMN imports
from autumn.base import StratifiedModel, EconomicModel
import tool_kit as t_k


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
        warnings.warn('Proportion parameter not between zero and one, value is %s' % proportion)

    # to avoid errors where the proportion is exactly one (although the function isn't really intended for this):
    if proportion > .99:
        early_proportion = 0.99
    elif proportion < 0.:
        return 0., 0.
    else:
        early_proportion = 1. - exp(log(1. - proportion) * early_period / total_period)
    late_proportion = proportion - early_proportion
    return early_proportion, late_proportion


class ConsolidatedModel(StratifiedModel, EconomicModel):
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

    def __init__(self, scenario=0, inputs=None, gui_inputs=None):
        """
        Instantiation, partly inherited from the lower level model objects through nested inheritance.

        Args:
            scenario: Single number for the scenario to run (with None meaning baseline)
            inputs: Non-GUI inputs from data_processing
            gui_inputs: GUI inputs from Tkinter or JS GUI
        """

        # inherited initialisations
        StratifiedModel.__init__(self)
        EconomicModel.__init__(self)

        # model attributes set from model runner object
        self.inputs = inputs
        self.scenario = scenario

        # start_time can't stay as a model constant, as it must be set for each scenario individually
        self.start_time = inputs.model_constants['start_time']

        # this code just stops the code checker complaining about attributes being undefined in instantiation
        self.compartment_types, self.organ_status, self.strains, self.agegroups, self.mixing, \
            self.vary_detection_by_organ, self.organs_for_detection, self.riskgroups_for_detection, \
            self.vary_detection_by_riskgroup, self.vary_force_infection_by_riskgroup, self.histories, \
            self.relevant_interventions, self.scaleup_fns, self.interventions_to_cost, self.is_lowquality, \
            self.is_amplification, self.is_misassignment, self.is_timevariant_organs, self.country, self.time_step, \
            self.integration_method = [None] * 21
        self.riskgroups = {}

        # model attributes to be set directly to inputs object attributes
        for attribute in ['compartment_types', 'organ_status', 'strains', 'riskgroups', 'agegroups', 'mixing',
                          'vary_detection_by_organ', 'organs_for_detection', 'riskgroups_for_detection',
                          'vary_detection_by_riskgroup', 'vary_force_infection_by_riskgroup', 'histories']:
            setattr(self, attribute, getattr(inputs, attribute))

        # model attributes to set to only the relevant scenario key from an inputs dictionary
        for attribute in ['relevant_interventions', 'scaleup_fns', 'interventions_to_cost']:
            setattr(self, attribute, getattr(inputs, attribute)[scenario])

        # model attributes to be set directly to attributes from the GUI object
        for attribute in ['is_lowquality', 'is_amplification', 'is_misassignment', 'is_timevariant_organs', 'country',
                          'time_step', 'integration_method']:
            setattr(self, attribute, gui_inputs[attribute])

        # set fixed parameters from inputs object
        for key, value in inputs.model_constants.items():
            if type(value) == float:
                self.set_parameter(key, value)

        # susceptibility classes
        self.force_types = ['_fully', '_immune', '_latent']
        if 'int_prop_novel_vaccination' in self.relevant_interventions:
            self.force_types.append('_novelvac')

        # list of infectious compartments
        self.infectious_tags = ['active', 'missed', 'detect', 'treatment', 'lowquality']

        # list out the inappropriate regimens
        self.inappropriate_regimens = []
        for s, strain in enumerate(self.strains):
            for a, as_strain in enumerate(self.strains):
                if s > a:
                    self.inappropriate_regimens.append(strain + '_as' + as_strain[1:])

        # to loop force of infection code over all risk groups if variable, or otherwise to just run once
        self.force_riskgroups = copy.copy(self.riskgroups) if self.vary_force_infection_by_riskgroup else ['']

        # treatment strings
        self.outcomes = ['_success', '_death', '_default']
        self.treatment_stages = ['_infect', '_noninfect']

        # intervention and economics-related initialisations
        if self.eco_drives_epi:
            self.distribute_funding_across_years()

        # list of risk groups affected by ngo activities for detection
        self.contributor_groups = ['_ruralpoor']

        # whether short-course MDR-TB regimen improves outcomes
        self.shortcourse_improves_outcomes = True

        # create time ticker
        self.next_time_point = copy.copy(self.start_time)

    def initialise_compartments(self):
        """
        Initialise all compartments to zero and then populate with the requested values.
        """

        # extract values for compartment initialisation by compartment type
        initial_compartments = {}
        for compartment in self.compartment_types:
            if compartment in self.inputs.model_constants:
                initial_compartments[compartment] = self.params[compartment]

        # initialise to zero
        for strata in itertools.product(self.riskgroups, self.agegroups):
            riskgroup, agegroup = strata
            for history in self.histories:
                end = riskgroup + history + agegroup
                for compartment in self.compartment_types:

                    # replicate susceptible for age and risk groups
                    if 'susceptible' in compartment or 'onipt' in compartment:
                        self.set_compartment(compartment + end, 0.)

                    # replicate latent classes for age groups, risk groups and strains
                    elif 'latent' in compartment:
                        for strain in self.strains:
                            self.set_compartment(compartment + strain + end, 0.)

                    # replicate active classes for age groups, risk groups, strains and organs
                    elif 'active' in compartment or 'missed' in compartment or 'lowquality' in compartment:
                        for other_strata in itertools.product(self.strains, self.organ_status):
                            strain, organ = other_strata
                            self.set_compartment(compartment + organ + strain + end, 0.)

                    # replicate treatment classes for age groups, risk groups, strains, organs and assigned strains
                    else:
                        for other_strata in itertools.product(self.strains, self.organ_status):
                            strain, organ = other_strata
                            if self.is_misassignment:
                                for assigned_strain in self.strains:
                                    self.set_compartment(
                                        compartment + organ + strain + '_as' + assigned_strain[1:] + end, 0.)
                            else:
                                self.set_compartment(compartment + organ + strain + end, 0.)

            # remove the unnecessary fully susceptible treated compartments
            self.remove_compartment('susceptible_fully' + riskgroup + self.histories[-1] + agegroup)

        start_risk_prop = self.find_starting_riskgroup_props() if len(self.riskgroups) > 1 else {'': 1.}
        self.populate_initial_compartments(initial_compartments, start_risk_prop)

    def find_starting_riskgroup_props(self):
        """
        Find starting proportions for risk groups using the functions for their scale-up evaluated at the starting time.
        """

        start_risk_prop = {'_norisk': 1.}
        for riskgroup in self.riskgroups:
            if riskgroup != '_norisk':
                start_risk_prop[riskgroup] \
                    = self.scaleup_fns['riskgroup_prop' + riskgroup](self.inputs.model_constants['start_time'])
                start_risk_prop['_norisk'] -= start_risk_prop[riskgroup]
        return start_risk_prop

    def populate_initial_compartments(self, initial_compartments, start_risk_prop):
        """
        Arbitrarily split equally by age-groups and organ status start with everyone having least resistant strain and
        no treatment history.

        Args:
            initial_compartments: The initial compartment values to set, but without stratifications
            start_risk_prop: Proportions to be allocated by risk group
        """

        for strata in itertools.product(self.compartment_types, self.riskgroups, self.agegroups):
            compartment, riskgroup, agegroup = strata
            if compartment in initial_compartments:
                end = riskgroup + self.histories[0] + agegroup
                if 'susceptible' in compartment:
                    self.set_compartment(
                        compartment + end,
                        initial_compartments[compartment] * start_risk_prop[riskgroup] / len(self.agegroups))
                elif 'latent' in compartment:
                    self.set_compartment(
                        compartment + self.strains[0] + end,
                        initial_compartments[compartment] * start_risk_prop[riskgroup] / len(self.agegroups))
                else:
                    for organ in self.organ_status:
                        self.set_compartment(
                            compartment + organ + self.strains[0] + end,
                            initial_compartments[compartment] * start_risk_prop[riskgroup] / len(self.organ_status)
                            / len(self.agegroups))

    ''' single method to process uncertainty parameters '''

    def process_uncertainty_params(self):
        """
        Perform some simple parameter processing - just for those that are used as uncertainty parameters and so can't
        be processed in the data processing module.
        """

        # find the case fatality of smear-negative and extrapulmonary TB using the relative case fatality
        for organ in self.organ_status[1:]:
            self.params['tb_prop_casefatality_untreated' + organ] \
                = self.params['tb_prop_casefatality_untreated_smearpos'] \
                * self.params['tb_relative_casefatality_untreated_smearneg']

        # calculate the rates of death and recovery from the above parameters
        for organ in self.organ_status:
            self.params['tb_rate_death' + organ] \
                = self.params['tb_prop_casefatality_untreated' + organ] / self.params['tb_timeperiod_activeuntreated']
            self.params['tb_rate_recover' + organ] \
                = (1. - self.params['tb_prop_casefatality_untreated' + organ]) \
                / self.params['tb_timeperiod_activeuntreated']

        # ipt completion
        self.params['rate_ipt_completion'] \
            = 1. / self.params['tb_timeperiod_onipt'] * self.params['int_prop_ipt_effectiveness']
        self.params['rate_ipt_noncompletion'] \
            = 1. / self.params['tb_timeperiod_onipt'] * (1. - self.params['int_prop_ipt_effectiveness'])

    ''' methods that calculate variables to be used in calculating flows
    (Note that all scaleup_fns have already been calculated.) '''

    def calculate_vars(self):
        """
        The master method that calls all the other methods for the calculations of variable rates.
        """

        self.ticker()
        self.calculate_demographic_vars()
        self.calculate_organ_progressions()
        self.calculate_progression_vars()
        self.calculate_detection_vars()
        self.calculate_await_treatment_var()
        self.calculate_treatment_rates()
        if 'agestratified_ipt' in self.relevant_interventions or 'ipt' in self.relevant_interventions:
            self.calculate_ipt_effect()
        self.calculate_force_infection()
        self.calculate_population_sizes()

    def ticker(self):
        """
        Prints time every ten years to indicate progress through integration.
        """

        if self.time > self.next_time_point:
            print(int(self.time))
            self.next_time_point += 10.

    def calculate_demographic_vars(self):
        """
        Short master method to all demographic calculations.
        """

        self.calculate_populations()
        self.calculate_birth_rates_vars()

    def calculate_populations(self):
        """
        Find the total absolute numbers of people in the population by each risk-group and overall.
        """

        population_riskgroups = copy.copy(self.riskgroups)
        if '' not in population_riskgroups:
            population_riskgroups.append('')
        for riskgroup in population_riskgroups:
            self.vars['population' + riskgroup] = 0.
            for label in self.labels:
                if riskgroup in label:
                    self.vars['population' + riskgroup] += self.compartments[label]
        for history in self.histories:
            self.vars['population' + history] = 0.
            for label in self.labels:
                if history in label:
                    self.vars['population' + history] += self.compartments[label]
            self.vars['prop_population' + history] = self.vars['population' + history] / self.vars['population']

    def calculate_birth_rates_vars(self):
        """
        Calculate birth rates into vaccinated and unvaccinated compartments.
        """

        # total births
        self.vars['births_total'] = self.vars['demo_rate_birth'] / 1e3 * self.vars['population']

        # vaccinated and unvaccinated proportions
        vac_props = {'vac': self.vars['int_prop_vaccination']}
        vac_props['unvac'] = 1. - vac_props['vac']
        if 'int_prop_novel_vaccination' in self.relevant_interventions:
            vac_props['novelvac'] = self.vars['int_prop_vaccination'] * self.vars['int_prop_novel_vaccination']
            vac_props['vac'] -= vac_props['novelvac']

        # calculate birth rates
        for riskgroup in self.riskgroups:
            for vac_status in vac_props:
                self.vars['births_' + vac_status + riskgroup] \
                    = vac_props[vac_status] * self.vars['births_total'] * self.target_risk_props[riskgroup][-1]

    def calculate_organ_progressions(self):
        """
        Calculate vars for the proportion of progressions going to each organ status.
        """

        # unstratified (self.organ_status should really have length 0, but length 1 will also run)
        if len(self.organ_status) < 2:
            self.vars['epi_prop'] = 1.

        # stratified into two tiers only (i.e. smear-positive and smear-negative)
        elif len(self.organ_status) == 2:
            self.vars['epi_prop_smearneg'] = 1. - self.vars['epi_prop_smearpos']

        # fully stratified into smear-positive, smear-negative and extrapulmonary
        else:
            self.vars['epi_prop_extrapul'] = 1. - self.vars['epi_prop_smearpos'] - self.vars['epi_prop_smearneg']

    def calculate_progression_vars(self):
        """
        Multiply the previous progression directions by organ status by the total progression rates by riskgroup and
        age group to get the actual flows to implement.
        """

        for strata in itertools.product(self.organ_status, self.agegroups, self.riskgroups, ['_early', '_late']):
            organ, agegroup, riskgroup, timing = strata
            self.vars['tb_rate' + timing + '_progression' + organ + riskgroup + agegroup] \
                = self.vars['epi_prop' + organ] \
                * self.params['tb_rate' + timing + '_progression' + riskgroup + agegroup]

    def calculate_detection_vars(self):
        """
        Master case detection method to collate all the methods relating to case detection.
        """

        if self.vary_detection_by_organ:
            self.calculate_case_detection_by_organ()
            self.adjust_smearneg_detection_for_xpert()

        if 'int_prop_decentralisation' in self.relevant_interventions:
            self.adjust_case_detection_for_decentralisation()
        if 'int_prop_dots_contributor' in self.relevant_interventions and self.vars['int_prop_dots_contributor'] < 1.:
            self.adjust_case_detection_for_dots_contributor()

        self.calculate_detect_missed_vars()
        if self.vary_detection_by_riskgroup:
            self.calculate_acf_rate()
            self.calculate_intensive_screening_rate()
            self.adjust_case_detection_for_acf()
            self.adjust_case_detection_for_intensive_screening()
        if self.is_misassignment:
            self.calculate_assignment_by_strain()
        if self.is_lowquality:
            self.calculate_lowquality_detection_vars()

    def adjust_case_detection_for_decentralisation(self):
        """
        Implement the decentralisation intervention, which narrows the case detection gap between the current values
        and the idealised estimated value.
        Only do so if the current case detection ratio is lower than the idealised detection ratio.
        """

        for organ in self.organs_for_detection:
            self.vars['program_prop_detect' + organ] \
                = t_k.increase_parameter_closer_to_value(self.vars['program_prop_detect' + organ],
                                                         self.params['int_ideal_detection'],
                                                         self.vars['int_prop_decentralisation'])

    def adjust_case_detection_for_dots_contributor(self):
        """
        Adjust detection rates for a program that contributes to basic DOTS activities in all groups. Originally coded
        for the "Open Doors" program in Bulgaria, which contributes a small proportion of all case detection in the
        country. Essentially this is passive case finding as persons with symptoms are encouraged to present, so the
        reported case detection rate for the country is assumed to incorporate this.
        """
        for organ in self.organs_for_detection:
            self.vars['program_prop_detect' + organ] \
                *= 1. - self.params['int_prop_detection_dots_contributor'] \
                * (1. - self.vars['int_prop_dots_contributor'])

    def calculate_case_detection_by_organ(self):
        """
        Method to perform simple weighting on the assumption that the smear-negative and extrapulmonary rates are less
        than the smear-positive rate by a proportion specified in program_prop_snep_relative_algorithm.
        Places a ceiling on these values, to prevent the smear-positive one going too close to (or above) one.
        """

        for parameter in ['_detect', '_algorithm_sensitivity']:

            # weighted increase in smear-positive detection proportion
            self.vars['program_prop' + parameter + '_smearpos'] \
                = min(self.vars['program_prop' + parameter]
                      / (self.vars['epi_prop_smearpos']
                         + self.params['program_prop_snep_relative_algorithm'] * (1. - self.vars['epi_prop_smearpos'])),
                      self.params['tb_prop_detection_algorithm_ceiling'])

            # then set smear-negative and extrapulmonary rates as proportionately lower
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

        if 'int_prop_xpert' in self.relevant_interventions:
            for parameter in ['_detect', '_algorithm_sensitivity']:
                self.vars['program_prop' + parameter + '_smearneg'] \
                    += (self.vars['program_prop' + parameter + '_smearpos'] -
                        self.vars['program_prop' + parameter + '_smearneg']) \
                    * self.params['int_prop_xpert_smearneg_sensitivity'] * self.vars['int_prop_xpert']

    def calculate_detect_missed_vars(self):
        """"
        Calculate rates of detection and failure of detection from the programmatic report of the case detection "rate"
        (which is actually a proportion and referred to as program_prop_detect here).

        Derived by solving the simultaneous equations:

          algorithm sensitivity = detection rate / (detection rate + missed rate)
              - and -
          detection proportion = detection rate
                / (detection rate + spont recover rate + tb death rate + natural death rate)
        """

        organs = copy.copy(self.organs_for_detection)
        if self.vary_detection_by_organ:
            organs.append('')
        for organ in organs:

            # add empty string for use in following calculation of number of missed patients
            riskgroups_to_loop = copy.copy(self.riskgroups_for_detection)
            if '' not in riskgroups_to_loop:
                riskgroups_to_loop.append('')
            for riskgroup in riskgroups_to_loop:

                # calculate detection rate from cdr proportion
                self.vars['program_rate_detect' + organ + riskgroup] \
                    = self.vars['program_prop_detect' + organ] \
                    * (1. / self.params['tb_timeperiod_activeuntreated'] + 1. / self.vars['demo_life_expectancy']) \
                    / (1. - self.vars['program_prop_detect' + organ])

                # adjust detection rates for ngo activities in specific risk-groups
                if 'int_prop_dots_groupcontributor' in self.relevant_interventions \
                        and self.vars['int_prop_dots_groupcontributor'] < 1. and riskgroup in self.contributor_groups:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        *= 1. - self.params['int_prop_detection_ngo' + riskgroup] \
                        * (1. - self.vars['int_prop_dots_groupcontributor'])

                # adjust for awareness raising
                if 'int_prop_awareness_raising' in self.vars:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        *= (self.params['int_multiplier_detection_with_raised_awareness'] - 1.) \
                        * self.vars['int_prop_awareness_raising'] + 1.

            # missed
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

        # loop to cover risk groups and community-wide ACF (empty string)
        riskgroups_to_loop = copy.copy(self.riskgroups)
        if '' not in riskgroups_to_loop:
            riskgroups_to_loop.append('')
        for riskgroup in riskgroups_to_loop:

            # decide whether to use the general detection proportion (as default), otherwise a risk group-specific one
            int_prop_acf_detections_per_round = self.params['int_prop_acf_detections_per_round' + riskgroup] \
                if 'int_prop_acf_detections_per_round' + riskgroup in self.params \
                else self.params['int_prop_acf_detections_per_round']

            # implement ACF by approach and whether CXR first as screening tool
            for acf_approach in itertools.product(['', 'cxr'], ['smear', 'xpert']):
                cxr_prescreen, acf_type = acf_approach
                intervention = 'int_prop_' + cxr_prescreen + acf_type + 'acf' + riskgroup
                if intervention in self.relevant_interventions and '_smearpos' in self.organ_status:

                    # find unadjusted coverage
                    coverage = self.vars[intervention]

                    # adjust effective coverage for screening test, if being used
                    if cxr_prescreen == 'cxr':
                        coverage *= self.params['tb_sensitivity_cxr']

                    # find the additional rate of case finding with ACF for smear-positive cases
                    if 'int_rate_acf_smearpos' + riskgroup not in self.vars:
                        self.vars['int_rate_acf_smearpos' + riskgroup] = 0.
                    self.vars['int_rate_acf_smearpos' + riskgroup] \
                        += coverage * int_prop_acf_detections_per_round / self.params['int_timeperiod_acf_rounds']

                    # find rate for smear-negatives, adjusted for Xpert sensitivity (if smear-negatives in model)
                    if acf_type == 'xpert' and '_smearneg' in self.organ_status:
                        if 'int_rate_acf_smearneg' + riskgroup not in self.vars:
                            self.vars['int_rate_acf_smearneg' + riskgroup] = 0.
                        self.vars['int_rate_acf_smearneg' + riskgroup] \
                            += self.vars['int_rate_acf_smearpos' + riskgroup] \
                            * self.params['int_prop_xpert_smearneg_sensitivity']

    def adjust_case_detection_for_acf(self):
        """
        Add ACF detection rates to previously calculated passive case detection rates, creating vars for case detection
        that are specific for organs.
        """

        for strata in itertools.product(self.organs_for_detection, self.riskgroups):
            organ = strata[0]

            # risk groups
            if 'int_rate_acf' + ''.join(strata) in self.vars:
                self.vars['program_rate_detect' + ''.join(strata)] += self.vars['int_rate_acf' + ''.join(strata)]

            # general community
            if 'int_rate_acf' + organ in self.vars:
                self.vars['program_rate_detect' + ''.join(strata)] += self.vars['int_rate_acf' + organ]

    def calculate_intensive_screening_rate(self):
        """
        Calculates rates of intensive screening from the proportion of programmatic coverage.
        Intensive screening detects smear-positive disease, and some smear-negative disease
        (incorporating a multiplier for the sensitivity of Xpert for smear-negative disease).
        Extrapulmonary disease can't be detected through intensive screening.
        """

        if 'int_prop_intensive_screening' in self.relevant_interventions:
            screened_subgroups = ['_diabetes', '_hiv']  # ultimately to be incorporated into the GUI

            # loop covers risk groups
            for riskgroup in screened_subgroups:
                # the following can't be written as self.organ_status, as it won't work for non-fully-stratified models
                for organ in ['', '_smearpos', '_smearneg', '_extrapul']:
                    self.vars['int_rate_intensive_screening' + organ + riskgroup] = 0.

                for organ in ['_smearpos', '_smearneg']:
                    self.vars['int_rate_intensive_screening' + organ + riskgroup] \
                        += self.vars['int_prop_intensive_screening'] \
                        * self.params['int_prop_attending_clinics' + riskgroup]

                # adjust smear-negative detections for Xpert's sensitivity
                self.vars['int_rate_intensive_screening_smearneg' + riskgroup] \
                    *= self.params['int_prop_xpert_smearneg_sensitivity']

    def adjust_case_detection_for_intensive_screening(self):

        if 'int_prop_intensive_screening' in self.relevant_interventions:
            screened_subgroups = ['_diabetes', '_hiv']
            for strata in itertools.product(self.organs_for_detection, screened_subgroups):
                self.vars['program_rate_detect' + ''.join(strata)] \
                    += self.vars['int_rate_intensive_screening' + ''.join(strata)]

    def calculate_assignment_by_strain(self):
        """
        Calculate the proportions of patients assigned to each strain. (Note that second-line DST availability refers to
        the proportion of those with first-line DST who also have second-line DST available.)
        """

        # with misassignment
        for organ in self.organs_for_detection:
            if len(self.strains) > 1 and self.is_misassignment:
                correct_assign_mdr = self.params['program_prop_clinical_assign_strain']

                # add effect of improve_dst program for culture-positive cases only
                if 'int_prop_firstline_dst' in self.relevant_interventions:
                    correct_assign_mdr += (1. - correct_assign_mdr) * self.vars['int_prop_firstline_dst'] * \
                                          self.params['program_prop' + organ + '_culturable']

                # add effect of Xpert on identification, assuming that independent distribution to conventional DST
                if 'int_prop_xpert' in self.relevant_interventions:
                    correct_assign_mdr += (1. - correct_assign_mdr) * self.vars['int_prop_xpert'] \
                                          * self.params['int_prop_xpert_sensitivity_mdr']

                for riskgroup in self.riskgroups_for_detection:

                    # determine rates of identification/misidentification as each strain
                    self.vars['program_rate_detect' + organ + riskgroup + '_ds_asds'] \
                        = self.vars['program_rate_detect' + organ + riskgroup]
                    self.vars['program_rate_detect' + organ + riskgroup + '_ds_asmdr'] \
                        = 0.
                    self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asds'] \
                        = (1. - correct_assign_mdr) * self.vars['program_rate_detect' + organ + riskgroup]
                    self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asmdr'] \
                        = correct_assign_mdr * self.vars['program_rate_detect' + organ + riskgroup]

                    # if a third strain is present
                    if len(self.strains) > 2:
                        prop_secondline = self.vars['program_prop_secondline_dst']
                        self.vars['program_rate_detect' + organ + riskgroup + '_ds_asxdr'] = 0.
                        self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asxdr'] = 0.
                        self.vars['program_rate_detect' + organ + riskgroup + '_xdr_asds'] \
                            = (1. - correct_assign_mdr) * self.vars['program_rate_detect' + organ + riskgroup]
                        self.vars['program_rate_detect' + organ + riskgroup + '_xdr_asmdr'] \
                            = correct_assign_mdr \
                            * (1. - prop_secondline) * self.vars['program_rate_detect' + organ + riskgroup]
                        self.vars['program_rate_detect' + organ + riskgroup + '_xdr_asxdr'] \
                            = correct_assign_mdr \
                            * prop_secondline * self.vars['program_rate_detect' + organ + riskgroup]

            # without misassignment, everyone correctly allocated by strain
            elif len(self.strains) > 1:
                for riskgroup in self.riskgroups_for_detection:
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
            prop_lowqual *= 1. - self.vars['int_prop_engage_lowquality']

        # note that there is still a program_rate_detect var even if detection is varied by organ and/or risk group
        self.vars['program_rate_enterlowquality'] \
            = self.vars['program_rate_detect'] * prop_lowqual / (1. - prop_lowqual)

    def calculate_await_treatment_var(self):
        """
        Take the reciprocal of the waiting times to calculate the flow rate to start treatment after detection.
        Note that the default behaviour for a single strain model is to use the waiting time for smear-positives.
        Also weight the time period.
        """

        for organ in self.organ_status:

            # adjust smear-negative for Xpert coverage
            if organ == '_smearneg' and 'int_prop_xpert' in self.relevant_interventions:
                time_to_treatment \
                    = self.params['program_timeperiod_await_treatment_smearneg'] * (1. - self.vars['int_prop_xpert']) \
                    + self.params['int_timeperiod_await_treatment_smearneg_xpert'] * self.vars['int_prop_xpert']

            # do other organ stratifications (including smear-negative if Xpert not an intervention)
            else:
                time_to_treatment = self.params['program_timeperiod_await_treatment' + organ]

            # find the rate as the reciprocal of the time to treatment
            self.vars['program_rate_start_treatment' + organ] = 1. / time_to_treatment

    def calculate_treatment_rates(self):
        """
        Master method to coordinate treatment var-related methods.
        """

        self.split_treatment_props_by_riskgroup()
        self.vars['epi_prop_amplification'] = self.params['tb_prop_amplification'] \
            if self.time > self.params['mdr_introduce_time'] else 0.
        if 'int_prop_shortcourse_mdr' in self.relevant_interventions and self.shortcourse_improves_outcomes \
                and '_mdr' in self.strains:
            for history in self.histories:
                self.adjust_treatment_outcomes_shortcourse(history)
        if 'int_prop_dot_groupcontributor' in self.relevant_interventions:
            self.adjust_treatment_outcomes_for_groupcontributor()

        for strain in self.strains:
            self.calculate_treatment_timeperiod_vars(strain)
            for strata in itertools.product(self.riskgroups, self.histories):
                riskgroup, history = strata
                self.adjust_treatment_outcomes_support(riskgroup, strain, history)

        treatment_types = copy.copy(self.strains)
        if self.is_misassignment:
            treatment_types.append('_inappropriate')
        for strain in treatment_types:
            for strata in itertools.product(self.riskgroups, self.histories):
                riskgroup, history = strata
                self.calculate_default_death_props(riskgroup + strain + history)

        for strain in self.strains:
            for strata in itertools.product(self.riskgroups, self.histories):
                riskgroup, history = strata
                self.split_treatment_props_by_stage(strain, strain, strain, strata)
                for stage in self.treatment_stages:
                    self.assign_success_prop_by_treatment_stage(riskgroup + strain + history, stage)
                    self.convert_treatment_props_to_rates(riskgroup, strain, history, stage)
                    if self.is_amplification:
                        self.split_by_amplification(riskgroup + strain + history + '_default' + stage)
                if len(self.strains) > 1 and self.is_misassignment and strain != self.strains[0]:
                    self.calculate_misassigned_outcomes(riskgroup, strain, history)

    def split_treatment_props_by_riskgroup(self):
        """
        Create treatment proportion vars that are specific to the different risk groups.
        The values are initially the same for all risk groups but this may change later with interventions.
        """

        strains_for_treatment = copy.copy(self.strains)
        if self.is_misassignment:
            strains_for_treatment.append('_inappropriate')
        converter = {'_success': 'treatment', '_death': 'nonsuccess'}
        for strata in itertools.product(strains_for_treatment, self.histories, ['_success', '_death']):
            for riskgroup in self.riskgroups:
                self.vars['program_prop_' + converter[strata[-1]] + riskgroup + ''.join(strata)] \
                    = copy.copy(self.vars['program_prop_' + converter[strata[-1]] + ''.join(strata)])

            # delete the var that is not riskgroup-specific
            del self.vars['program_prop_' + converter[strata[-1]] + ''.join(strata)]

    def adjust_treatment_outcomes_shortcourse(self, history):
        """
        Adapt treatment outcomes for short-course regimen. Restricted such that can only improve outcomes by selection
        of the functions used to adjust the treatment outcomes.
        """

        for riskgroup in self.riskgroups:
            self.vars['program_prop_treatment' + riskgroup + '_mdr' + history + '_success'] \
                = t_k.increase_parameter_closer_to_value(
                self.vars['program_prop_treatment' + riskgroup + '_mdr' + history + '_success'],
                self.params['int_prop_treatment_success_shortcoursemdr'],
                self.vars['int_prop_shortcourse_mdr'])

    def adjust_treatment_outcomes_for_groupcontributor(self):
        """
        Adjust treatment success and death for NGO activities. These activities are already running at baseline so we
        need to account for some effect that is already ongoing. The efficacy parameter represents the proportional
        reduction in negative outcomes obtained from the intervention when used at 100% coverage as compared to no
        intervention. The programmatic parameters obtained from the spreadsheets have to be interpreted as being
        affected by the intervention that is already running at some level of coverage.
        """

        # calculate the ratio (1 - efficacy * current_coverage) / (1 - efficacy * baseline_coverage)
        coverage_ratio \
            = (1. - self.params['int_prop_treatment_improvement_contributor']
               * self.vars['int_prop_dot_groupcontributor']) \
            / (1. - self.params['int_prop_treatment_improvement_contributor']
               * self.scaleup_fns['int_prop_dot_groupcontributor'](self.params['reference_time']))

        for strata in itertools.product(self.contributor_groups, self.strains, self.histories):

            # treatment success (which may have become negative in the pre-treatment era)
            self.vars['program_prop_treatment' + ''.join(strata) + '_success'] \
                = max(1. - coverage_ratio * (1. - self.vars['program_prop_treatment' + ''.join(strata) + '_success']),
                      0.)

    def calculate_treatment_timeperiod_vars(self, strain):
        """
        Find the time periods on treatment and in each stage of treatment for the strain being considered. (Note that
        this doesn't need to include inappropriate, because inappropriately treated patients have treated durations that
        come from regimen durations for appropriately treated patients.)
        """

        # find baseline treatment period for the total duration (i.e. '') and for the infectious period
        for treatment_stage in ['', '_infect']:

            # if short-course regimen being implemented
            if strain == '_mdr' and 'int_prop_shortcourse_mdr' in self.relevant_interventions:
                self.vars['tb_timeperiod' + treatment_stage + '_ontreatment_mdr'] \
                    = t_k.decrease_parameter_closer_to_value(
                    self.params['tb_timeperiod' + treatment_stage + '_ontreatment_mdr'],
                    self.params['int_timeperiod_shortcourse_mdr' + treatment_stage],
                    self.vars['int_prop_shortcourse_mdr'])
            else:
                self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                    = self.params['tb_timeperiod' + treatment_stage + '_ontreatment' + strain]

        # simply find non-infectious period as the difference between the infectious period and the total duration
        self.vars['tb_timeperiod_noninfect_ontreatment' + strain] \
            = self.vars['tb_timeperiod_ontreatment' + strain] - self.vars['tb_timeperiod_infect_ontreatment' + strain]

    def adjust_treatment_outcomes_support(self, riskgroup, strain, history):
        """
        Add some extra treatment success if the treatment support intervention is active, either for a specific strain
        or for all outcomes. Also able to select as to whether the improvement is a relative reduction in poor outcomes
        or an improvement towards an idealised value. Note that the idealised values do not differ by treatment history
        but can differ by strain.

        Note that the strain-specific absolute interventions still need ideal values to be specified (at the time of
        writing).
        """

        strain_types = [strain]
        if '' not in self.strains:
            strain_types.append('')
        for strain_type in strain_types:
            if 'int_prop_treatment_support_relative' + strain_type in self.relevant_interventions:
                self.vars['program_prop_treatment' + riskgroup + strain + history + '_success'] \
                    += (1. - self.vars['program_prop_treatment' + riskgroup + strain + history + '_success']) \
                    * self.params['int_prop_treatment_support_improvement' + strain_type] \
                    * self.vars['int_prop_treatment_support_relative' + strain_type]
            elif 'int_prop_treatment_support_absolute' + strain_type in self.relevant_interventions:
                self.vars['program_prop_treatment' + riskgroup + strain + history + '_success'] \
                    = t_k.increase_parameter_closer_to_value(
                    self.vars['program_prop_treatment' + riskgroup + strain + history + '_success'],
                    self.params['program_prop_treatment_success_ideal' + strain_type],
                    self.vars['int_prop_treatment_support_absolute' + strain_type])

    def calculate_default_death_props(self, stratum):
        """
        Calculate the default proportion as the remainder from success and death and warn if numbers don't make sense.
        This could be dangerous, because we are using the same var name for two quantities that are different at
        different stages of the var calculations, but seems to work for now.
        """

        self.vars['program_prop_treatment' + stratum + '_default'] \
            = (1. - self.vars['program_prop_treatment' + stratum + '_success']) \
            * (1. - self.vars['program_prop_nonsuccess' + stratum + '_death'])
        self.vars['program_prop_treatment' + stratum + '_death'] \
            = (1. - self.vars['program_prop_treatment' + stratum + '_success']) \
            * self.vars['program_prop_nonsuccess' + stratum + '_death']

    def split_treatment_props_by_stage(self, regimen_for_outcomes, treated_as, output_string, strata):
        """
        Assign proportions of default and death to early/infectious and late/non-infectious stages of treatment. For an
        appropriately treated patient, the regimen, treatment_type and treated_as inputs are all the same.

        Args:
            regimen_for_outcomes: The regimen whose treatment outcomes are used
            output_string: The full name for the var that needs to come out the end (e.g. _mdr or _dsasmdr)
            treated_as: The regimen they have been assigned to
            strata: Riskgroup and treatment history strata
        """

        riskgroup, history = strata
        for outcome in self.outcomes[1:]:
            outcomes_by_stage = find_outcome_proportions_by_period(
                self.vars['program_prop_treatment' + riskgroup + regimen_for_outcomes + history + outcome],
                self.vars['tb_timeperiod_infect_ontreatment' + treated_as],
                self.vars['tb_timeperiod_ontreatment' + treated_as])
            for s, stage in enumerate(self.treatment_stages):
                self.vars['program_prop_treatment' + riskgroup + output_string + history + outcome + stage] \
                    = outcomes_by_stage[s]

    def assign_success_prop_by_treatment_stage(self, stratum, stage):
        """
        Calculate success proportion by stage as the remainder after death and default accounted for.
        """

        start = 'program_prop_treatment' + stratum
        self.vars[start + '_success' + stage] \
            = 1. - self.vars[start + '_default' + stage] - self.vars[start + '_death' + stage]

    def convert_treatment_props_to_rates(self, riskgroup, strain, history, stage):
        """
        Convert the outcomes proportions by stage of treatment to rates by dividing by the appropriate time period.
        """

        for outcome in self.outcomes:
            end = riskgroup + strain + history + outcome + stage
            self.vars['program_rate_treatment' + end] \
                = self.vars['program_prop_treatment' + end] \
                / self.vars['tb_timeperiod' + stage + '_ontreatment' + strain]

    def split_by_amplification(self, treatment_group):
        """
        Split default according to whether amplification occurs (if not the most resistant strain).
        Previously had a sigmoidal function for amplification proportion, but now thinking that the following switch is
        a better approach because scale-up functions are all calculated in data_processing and we need to be able to
        adjust the time that MDR emerges during model running.
        """

        start = 'program_rate_treatment' + treatment_group
        self.vars[start + '_amplify'] = self.vars[start] * self.vars['epi_prop_amplification']
        self.vars[start + '_noamplify'] = self.vars[start] * (1. - self.vars['epi_prop_amplification'])

    def calculate_misassigned_outcomes(self, riskgroup, strain, history):
        """
        Find treatment outcomes for patients assigned to an incorrect regimen.
        """

        for treated_as in self.strains:

            # if assigned strain is different from the actual strain and regimen is inadequate
            if treated_as != strain and self.strains.index(treated_as) < self.strains.index(strain):
                treatment_type = strain + '_as' + treated_as[1:]
                self.split_treatment_props_by_stage(
                    '_inappropriate', treated_as, treatment_type, [riskgroup] + [history])
                for treatment_stage in self.treatment_stages:
                    self.assign_success_prop_by_treatment_stage(riskgroup + treatment_type + history, treatment_stage)

                    # find rates from proportions (analogous to convert_treatment_props_to_rates)
                    for outcome in self.outcomes:
                        end = riskgroup + treatment_type + history + outcome + treatment_stage
                        self.vars['program_rate_treatment' + end] \
                            = self.vars['program_prop_treatment' + end] \
                            / self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + treated_as]

                    if self.is_amplification:
                        self.split_by_amplification(riskgroup + treatment_type + history + '_default' + treatment_stage)

    def calculate_ipt_effect(self):
        """
        Method to estimate the proportion of infections averted through the IPT program - as the proportion of all cases
        detected (by the high quality sector) multiplied by the proportion of infections in the household (giving the
        proportion of all infections we can target) multiplied by the coverage of the program (in each age-group) and
        the effectiveness of treatment.
        """

        # first calculate the proportion of new infections that are detected and so potentially targeted with IPT
        self.vars['tb_prop_infections_reachable_ipt'] \
            = self.calculate_aggregate_outgoing_proportion('active', 'detect') \
            * self.params['int_prop_infections_in_household'] * self.params['int_prop_ltbi_test_sensitivity']

        # for each age group, calculate proportion of infections averted by IPT program
        for agegroup in self.agegroups:

            # calculate coverage as sum of coverage in age group and overall for entire population, with ceiling of one
            coverage = 0.
            for agegroup_or_entire_population in [agegroup, '']:
                if 'int_prop_ipt' + agegroup_or_entire_population in self.vars:
                    coverage += self.vars['int_prop_ipt' + agegroup_or_entire_population]
            coverage = min(coverage, 1.)

            # calculate infections averted as product of infections of identified cases and coverage
            self.vars['prop_infections_averted_ipt' + agegroup] \
                = self.vars['tb_prop_infections_reachable_ipt'] * coverage

    def calculate_force_infection(self):
        """
        Method to coordinate calculation of the force of infection, calling the various other methods involved.
        """

        for strain in self.strains:
            self.set_initial_force_infection(strain)
            if self.vary_force_infection_by_riskgroup:
                self.adjust_force_infection_for_mixing(strain)
            self.adjust_force_infection_for_immunity(strain)
            self.adjust_force_infection_for_ipt(strain)

    def set_initial_force_infection(self, strain):
        """
        Calculate force of infection independently for each strain, incorporating partial immunity and infectiousness.
        First calculate the effective infectious population (incorporating infectiousness by organ involvement), then
        calculate the raw force of infection, then adjust for various levels of susceptibility.
        """

        # if any modifications to transmission parameter to be made over time
        transmission_modifier = self.vars['transmission_modifier'] if 'transmission_modifier' in self.vars else 1.

        # whether directly calculating the force of infection or a temporary "infectiousness" from which it is derived
        force_string = 'infectiousness' if self.vary_force_infection_by_riskgroup else 'rate_force'

        for riskgroup in self.force_riskgroups:

            # initialise infectiousness vars
            self.vars[force_string + strain + riskgroup] = 0.

            # loop through compartments, skipping on as soon as possible if irrelevant
            for label in self.labels:
                if strain not in label and strain != '':
                    continue
                if riskgroup not in label and riskgroup != '':
                    continue

                # skip on for those in the non-infectious stages of treatment, except if inappropriate
                appropriate_regimen = True
                for regimen in self.inappropriate_regimens:
                    if regimen in label:
                        appropriate_regimen = False
                if 'treatment_noninfect' in label and not appropriate_regimen:
                    continue

                for agegroup in self.agegroups:
                    if agegroup not in label and agegroup != '':
                        continue
                    for organ in self.organ_status:
                        if organ not in label and organ != '':
                            continue
                        if organ == '_extrapul':
                            continue

                        # adjustment for increased infectiousness in riskgroup
                        riskgroup_multiplier = self.params['riskgroup_multiplier_force_infection' + riskgroup] \
                            if 'riskgroup_multiplier_force_infection' + riskgroup in self.params else 1.

                        # increment "infectiousness", the effective number of infectious people in the stratum
                        if t_k.label_intersects_tags(label, self.infectious_tags):
                            self.vars[force_string + strain + riskgroup] \
                                += self.params['tb_n_contact'] \
                                * transmission_modifier * self.params['tb_multiplier_force' + organ] \
                                * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                * self.compartments[label] * riskgroup_multiplier \
                                / self.vars['population' + riskgroup]

    def adjust_force_infection_for_mixing(self, strain):
        """
        Use the mixing matrix to ajdust the force of infection vars for the proportion of contacts received from each
        group. Otherwise, just assign the total infectiousness of the population to be the overall force of infection.
        """

        for to_riskgroup in self.force_riskgroups:
            self.vars['rate_force' + strain + to_riskgroup] = 0.
            for from_riskgroup in self.force_riskgroups:
                self.vars['rate_force' + strain + to_riskgroup] \
                    += self.vars['infectiousness' + strain + from_riskgroup] * self.mixing[to_riskgroup][from_riskgroup]

    def adjust_force_infection_for_immunity(self, strain):
        """
        Find the actual rate of infection for each compartment type depending on the relative immunity/susceptibility
        associated with being in that compartment.
        """

        for strata in itertools.product(self.force_riskgroups, self.force_types, self.agegroups, self.histories):
            riskgroup, force_type, agegroup, history = strata

            immunity_multiplier = self.params['tb_multiplier' + force_type + '_protection']

            # give children greater protection from BCG vaccination
            if t_k.interrogate_age_string(agegroup)[0][1] <= self.params['int_age_bcg_immunity_wane'] \
                    and force_type == '_immune':
                immunity_multiplier *= self.params['int_multiplier_bcg_child_relative_immunity']

            # increase immunity for previously treated
            if history == '_treated':
                immunity_multiplier *= self.params['tb_multiplier_treated_protection']

            # find final rates of infection, except there is no previously treated fully susceptible
            if force_type != '_fully' or (force_type == '_fully' and history == self.histories[0]):
                self.vars['rate_force' + force_type + strain + history + riskgroup + agegroup] \
                    = self.vars['rate_force' + strain + riskgroup] * immunity_multiplier

    def adjust_force_infection_for_ipt(self, strain):
        """
        Adjust the previously calculated force of infection for the use of IPT in contacts. Uses the previously
        calculated proportion of infections averted IPT var to determine how much of the force of infection to assign
        to go to IPT instead and how much to remain as force of infection
        """

        for strata in itertools.product(self.force_types, self.histories, self.force_riskgroups, self.agegroups):
            force_type, history, riskgroup, agegroup = strata
            if force_type != '_fully' or (force_type == '_fully' and history == self.histories[0]):
                stratum = force_type + strain + history + riskgroup + agegroup
                if ('agestratified_ipt' in self.relevant_interventions
                        or 'ipt' in self.relevant_interventions) and strain == self.strains[0]:
                    self.vars['rate_ipt_commencement' + stratum] \
                        = self.vars['rate_force' + stratum] * self.vars['prop_infections_averted_ipt' + agegroup]
                    self.vars['rate_force' + stratum] -= self.vars['rate_ipt_commencement' + stratum]
                else:
                    self.vars['rate_ipt_commencement' + stratum] = 0.

    def calculate_population_sizes(self):
        """
        Calculate the size of the populations to which each intervention is applicable, for use in generating
        cost-coverage curves.
        """

        # treatment support
        strain_types = copy.copy(self.strains)
        if '' not in self.strains:
            strain_types.append('')
        for intervention in ['treatment_support_relative', 'treatment_support_absolute']:
            for strain in strain_types:
                if 'int_prop_' + intervention + strain in self.relevant_interventions:
                    self.vars['popsize_' + intervention + strain] = 0.
                    for compartment in self.compartments:
                        if 'treatment_' in compartment and strain in compartment:
                            self.vars['popsize_' + intervention + strain] += self.compartments[compartment]

        # ambulatory care
        for organ in self.organ_status:
            if 'int_prop_ambulatorycare' + organ in self.relevant_interventions:
                self.vars['popsize_ambulatorycare' + organ] = 0.
                for compartment in self.compartments:
                    if 'treatment_' in compartment and organ in compartment:
                        self.vars['popsize_ambulatorycare' + organ] += self.compartments[compartment]

        # IPT: popsize defined as the household contacts of active cases identified by the high-quality sector
        for agegroup in self.agegroups:
            self.vars['popsize_ipt' + agegroup] = 0.
            for strain in self.strains:
                for from_label, to_label, rate in self.flows_by_type['var_transfer']:
                    if 'latent_early' in to_label and strain in to_label and agegroup in to_label:
                        self.vars['popsize_ipt' + agegroup] += self.compartments[from_label] * self.vars[rate]

        # BCG (so simple that it's almost unnecessary, but needed for loops over int names)
        self.vars['popsize_vaccination'] = self.vars['births_total']

        # Xpert and improve DST - all presentations for assessment for active TB
        for active_tb_presentations_intervention in ['xpert', 'improve_dst']:
            if 'int_prop_' + active_tb_presentations_intervention in self.relevant_interventions:
                self.vars['popsize_' + active_tb_presentations_intervention] = 0.

                for strata in itertools.product(
                        self.agegroups, self.riskgroups, self.strains, self.histories, self.organ_status):
                    agegroup, riskgroup, strain, history, organ = strata
                    if active_tb_presentations_intervention == 'improve_dst' and organ == '_extrapul':
                        continue
                    detection_organ = organ if self.vary_detection_by_organ else ''
                    detection_riskgroup = riskgroup if self.vary_detection_by_riskgroup else ''
                    self.vars['popsize_' + active_tb_presentations_intervention] \
                        += self.vars['program_rate_detect' + detection_organ + detection_riskgroup] \
                        * self.compartments['active' + organ + strain + riskgroup + history + agegroup] \
                        * (self.params['int_number_tests_per_tb_presentation'] + 1.)

        # ACF
        # loop over risk groups, including '', which is no stratification
        for riskgroup in [''] + self.riskgroups:

            # allow proportion of persons actually receiving the screening to vary by risk-group, as per user inputs
            if 'int_prop_population_screened' + riskgroup in self.params:
                int_prop_population_screened = self.params['int_prop_population_screened' + riskgroup]
            else:
                int_prop_population_screened = self.params['int_prop_population_screened']

            # loop over ACF types
            for acf_type in ['_smearacf', '_xpertacf', '_cxrxpertacf']:
                if 'int_prop' + acf_type + riskgroup in self.relevant_interventions:
                    self.vars['popsize' + acf_type + riskgroup] = 0.
                    for compartment in self.compartments:
                        if riskgroup == '' or riskgroup in compartment:
                            self.vars['popsize' + acf_type + riskgroup] \
                                += self.compartments[compartment] / self.params['int_timeperiod_acf_rounds'] \
                                * int_prop_population_screened

        # intensive_screening
        # popsize includes all active TB cases of targeted groups (HIV and diabetes) that attend specific clinics
        if 'int_prop_intensive_screening' in self.relevant_interventions:
            self.vars['popsize_intensive_screening'] = 0.
            screened_subgroups = ['_diabetes', '_hiv']  # may be incorporated into the GUI
            # loop covers risk groups
            for riskgroup in screened_subgroups:
                for compartment in self.compartments:
                    if riskgroup in compartment and 'active' in compartment:
                        self.vars['popsize_intensive_screening'] \
                            += self.compartments[compartment] * self.params['int_prop_attending_clinics' + riskgroup]

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
            if adjust_lowquality:
                self.vars['popsize_engage_lowquality'] *= self.vars['program_prop_lowquality']

        # shortcourse MDR-TB regimen
        if 'int_prop_shortcourse_mdr' in self.relevant_interventions:
            self.vars['popsize_shortcourse_mdr'] = 0.
            for compartment in self.compartments:
                if 'treatment' in compartment and '_mdr' in compartment:
                    self.vars['popsize_shortcourse_mdr'] += self.compartments[compartment]

        # NGO activities
        if 'int_prop_dot_groupcontributor' in self.relevant_interventions:
            # this is the size of the Roma population
            self.vars['popsize_dot_groupcontributor'] = 0.
            for riskgroup in self.contributor_groups:
                self.vars['popsize_dot_groupcontributor'] += self.vars['population' + riskgroup]

        # awareness raising
        if 'int_prop_awareness_raising' in self.relevant_interventions:
            self.vars['popsize_awareness_raising'] = self.vars['population']

    ''' methods that calculate the flows of all the compartments '''

    def set_flows(self):
        """
        Call all the intercompartmental flow setting methods in turn.
        """

        self.set_birth_flows()
        if len(self.agegroups) > 0:
            self.set_ageing_flows()
        self.set_infection_flows()
        self.set_progression_flows()
        self.set_natural_history_flows()
        self.set_fixed_programmatic_flows()
        self.set_variable_programmatic_flows()
        self.set_detection_flows()
        self.set_treatment_flows()
        if 'agestratified_ipt' in self.relevant_interventions or 'ipt' in self.relevant_interventions:
            self.set_ipt_flows()

    def set_birth_flows(self):
        """
        Set birth (or recruitment) flows by vaccination status (including novel vaccination if implemented).
        """

        for riskgroup in self.riskgroups:
            end = riskgroup + self.histories[0] + self.agegroups[0]
            self.set_var_entry_rate_flow('susceptible_fully' + end, 'births_unvac' + riskgroup)
            self.set_var_entry_rate_flow('susceptible_immune' + end, 'births_vac' + riskgroup)
            if 'int_prop_novel_vaccination' in self.relevant_interventions:
                self.set_var_entry_rate_flow('susceptible_novelvac' + end, 'births_novelvac' + riskgroup)

    def set_infection_flows(self):
        """
        Set force of infection flows that were estimated by strain in calculate_force_infection_vars above.
        """

        for strata in itertools.product(
                self.force_types, self.strains, self.riskgroups, self.histories, self.agegroups):
            force_type, strain, riskgroup, history, agegroup, = strata
            force_riskgroup = riskgroup if self.vary_force_infection_by_riskgroup else ''

            # source compartment is split by riskgropup, history and agegroup - plus force type for
            # the susceptibles and strain for the latents
            source_extension = riskgroup + history + agegroup

            # force of infection differs by immunity, strain, history, age group and possibly risk group
            force_extension = force_type + strain + history + force_riskgroup + agegroup

            # destination compartments differ by strain, risk group, history and age group
            latent_compartment = 'latent_early' + strain + riskgroup + history + agegroup

            # on IPT treatment destination compartment differs by risk group, history and age group
            onipt_destination_compartment = 'onipt' + riskgroup + history + agegroup

            # inapplicable situation of previously treated but fully susceptible
            if force_type == '_fully' and history != self.histories[0]:
                continue

            # loop across all strains for infection during latency
            elif force_type == '_latent':
                for from_strain in self.strains:
                    source_compartment = 'latent_late' + from_strain + source_extension
                    self.set_var_transfer_rate_flow(
                        source_compartment, latent_compartment, 'rate_force' + force_extension)
                    self.set_var_transfer_rate_flow(
                        source_compartment, onipt_destination_compartment, 'rate_ipt_commencement' + force_extension)

            # susceptible compartments
            else:
                source_compartment = 'susceptible' + force_type + source_extension
                self.set_var_transfer_rate_flow(source_compartment, latent_compartment, 'rate_force' + force_extension)
                self.set_var_transfer_rate_flow(
                    source_compartment, onipt_destination_compartment, 'rate_ipt_commencement' + force_extension)

    def set_progression_flows(self):
        """
        Set rates of progression from latency to active disease, with rates differing by organ status.
        """

        for strata in itertools.product(self.strains, self.riskgroups, self.histories, self.agegroups):
            strain, riskgroup, history, agegroup = strata

            # stabilisation
            self.set_fixed_transfer_rate_flow('latent_early' + ''.join(strata), 'latent_late' + ''.join(strata),
                                              'tb_rate_stabilise' + riskgroup + agegroup)

            # now smear-pos/smear-neg is always a var, even when a constant function
            for organ in self.organ_status:
                self.set_var_transfer_rate_flow('latent_early' + ''.join(strata), 'active' + organ + ''.join(strata),
                                                'tb_rate_early_progression' + organ + riskgroup + agegroup)
                self.set_var_transfer_rate_flow('latent_late' + ''.join(strata), 'active' + organ + ''.join(strata),
                                                'tb_rate_late_progression' + organ + riskgroup + agegroup)

    def set_natural_history_flows(self):
        """
        Set flows for progression through active disease to either recovery or death.
        """

        # determine the compartments to which natural history flows apply
        active_compartments = ['active', 'missed']
        if self.is_lowquality:
            active_compartments.append('lowquality')
        if not self.is_misassignment:
            active_compartments.append('detect')

        for strata in itertools.product(
                self.strains, self.riskgroups, self.histories, self.agegroups, self.organ_status):
            strain, riskgroup, history, agegroup, organ = strata
            end = ''.join(strata[:-1])
            for compartment in active_compartments:

                # recovery
                self.set_fixed_transfer_rate_flow(compartment + organ + end, 'latent_late' + end, 'tb_rate_recover'
                                                  + organ)

                # death
                self.set_fixed_infection_death_rate_flow(compartment + organ + end, 'tb_rate_death' + organ)

            # detected, with misassignment
            if self.is_misassignment:
                for assigned_strain in self.strains:
                    self.set_fixed_infection_death_rate_flow(
                        'detect' + organ + strain + '_as' + assigned_strain[1:] + riskgroup + history + agegroup,
                        'tb_rate_death' + organ)
                    self.set_fixed_transfer_rate_flow(
                        'detect' + organ + strain + '_as' + assigned_strain[1:] + riskgroup + history + agegroup,
                        'latent_late' + end, 'tb_rate_recover' + organ)

    def set_fixed_programmatic_flows(self):
        """
        Set rates of return to active disease for patients who presented for health care and were missed and for
        patients who were in the low-quality health care sector.
        """

        for strata in itertools.product(
                self.organ_status, self.strains, self.riskgroups, self.histories, self.agegroups):
            end = ''.join(strata)

            # re-start presenting after a missed diagnosis
            self.set_fixed_transfer_rate_flow('missed' + end, 'active' + end, 'program_rate_restart_presenting')

            # giving up on the hopeless low-quality health system
            if self.is_lowquality:
                self.set_fixed_transfer_rate_flow('lowquality' + end, 'active' + end, 'program_rate_leavelowquality')

    def set_detection_flows(self):
        """
        Set previously calculated detection rates (either assuming everyone is correctly identified if misassignment
        not permitted or with proportional misassignment).
        """

        for strata in itertools.product(self.organ_status, self.riskgroups, self.histories, self.agegroups):
            organ, riskgroup, history, agegroup = strata
            for strain_number, strain in enumerate(self.strains):
                detection_riskgroup = riskgroup if self.vary_detection_by_riskgroup else ''
                detection_organ = organ if self.vary_detection_by_organ else ''
                end = organ + strain + riskgroup + history + agegroup

                # with misassignment
                if self.is_misassignment:
                    for assigned_strain_number in range(len(self.strains)):
                        as_assigned_strain = '_as' + self.strains[assigned_strain_number][1:]

                        # if the strain is equally or more resistant than its assignment
                        if strain_number >= assigned_strain_number:
                            self.set_var_transfer_rate_flow(
                                'active' + end,
                                'detect' + organ + strain + as_assigned_strain + riskgroup + history + agegroup,
                                'program_rate_detect' + detection_organ + detection_riskgroup + strain +
                                as_assigned_strain)

                # without misassignment
                else:
                    self.set_var_transfer_rate_flow(
                        'active' + end, 'detect' + end, 'program_rate_detect' + detection_organ + detection_riskgroup)

    def set_variable_programmatic_flows(self):
        """
        Set rate of missed diagnosis (which is variable as the algorithm sensitivity typically will be), rate of
        presentation to low quality health care (which is variable as the extent of this health system typically will
        be) and rate of treatment commencement (which is variable and depends on the diagnostics available).
        """

        # set rate of missed diagnoses and entry to low-quality health care

        for strata in itertools.product(
                self.organ_status, self.strains, self.riskgroups, self.histories, self.agegroups):
            organ, strain, riskgroup, history, agegroup = strata

            detection_organ = organ if self.vary_detection_by_organ else ''
            end = ''.join(strata)

            self.set_var_transfer_rate_flow('active' + end, 'missed' + end, 'program_rate_missed' + detection_organ)

            # treatment commencement, with and without misassignment
            if self.is_misassignment:
                for assigned_strain in self.strains:
                    self.set_var_transfer_rate_flow(
                        'detect' + organ + strain + '_as' + assigned_strain[1:] + riskgroup + history + agegroup,
                        'treatment_infect' + organ + strain + '_as' + assigned_strain[1:] + riskgroup + history +
                        agegroup,
                        'program_rate_start_treatment' + organ)
            else:
                self.set_var_transfer_rate_flow(
                    'detect' + end, 'treatment_infect' + end, 'program_rate_start_treatment' + organ)

            # enter the low quality health care system
            if self.is_lowquality:
                self.set_var_transfer_rate_flow('active' + end, 'lowquality' + end, 'program_rate_enterlowquality')

    def set_treatment_flows(self):
        """
        Set rates of progression through treatment stages - dealing with amplification, as well as misassignment if
        either or both are implemented.
        """

        for strata in itertools.product(self.organ_status, self.riskgroups, self.agegroups, self.histories):
            organ, riskgroup, agegroup, history = strata

            for s, strain in enumerate(self.strains):

                # which strains to loop over for strain assignment
                assignment_strains = self.strains if self.is_misassignment else ['']
                for a, assigned_strain in enumerate(assignment_strains):
                    if self.is_misassignment and s > a:
                        regimen, as_assigned_strain \
                            = strain + '_as' + assigned_strain[1:], '_as' + assigned_strain[1:]
                    elif self.is_misassignment:
                        regimen, as_assigned_strain = assigned_strain, '_as' + assigned_strain[1:]
                    else:
                        regimen, as_assigned_strain = strain, ''

                    # success by treatment stage
                    end = organ + strain + as_assigned_strain + riskgroup + history + agegroup
                    self.set_var_transfer_rate_flow(
                        'treatment_infect' + end, 'treatment_noninfect' + end,
                        'program_rate_treatment' + riskgroup + regimen + history + '_success_infect')
                    self.set_var_transfer_rate_flow(
                        'treatment_noninfect' + end, 'susceptible_immune' + riskgroup + self.histories[-1] + agegroup,
                        'program_rate_treatment' + riskgroup + regimen + history + '_success_noninfect')

                    # death on treatment
                    for treatment_stage in self.treatment_stages:
                        self.set_var_infection_death_rate_flow(
                            'treatment' + treatment_stage + end,
                            'program_rate_treatment' + riskgroup + regimen + history + '_death' + treatment_stage)

                    # default
                    for treatment_stage in self.treatment_stages:

                        # if it's either the most resistant strain available or amplification is not active:
                        if strain == self.strains[-1] or not self.is_amplification:
                            self.set_var_transfer_rate_flow(
                                'treatment' + treatment_stage + end,
                                'active' + organ + strain + riskgroup + history + agegroup,
                                'program_rate_treatment' + riskgroup + regimen + history + '_default' + treatment_stage)

                        # otherwise with amplification
                        else:
                            amplify_to_strain = self.strains[s + 1]
                            self.set_var_transfer_rate_flow(
                                'treatment' + treatment_stage + end,
                                'active' + organ + strain + riskgroup + history + agegroup,
                                'program_rate_treatment' + riskgroup + regimen + history + '_default' + treatment_stage
                                + '_noamplify')
                            self.set_var_transfer_rate_flow(
                                'treatment' + treatment_stage + end,
                                'active' + organ + amplify_to_strain + riskgroup + history + agegroup,
                                'program_rate_treatment' + riskgroup + regimen + history + '_default' + treatment_stage
                                + '_amplify')

    def set_ipt_flows(self):
        """
        Implement IPT-related transitions: completion of treatment and failure to complete treatment.
        """

        for strata in itertools.product(self.riskgroups, self.histories, self.agegroups):
            end = ''.join(strata)

            # treatment completion flows
            self.set_fixed_transfer_rate_flow('onipt' + end, 'susceptible_immune' + end, 'rate_ipt_completion')

            # treatment non-completion flows
            self.set_fixed_transfer_rate_flow('onipt' + end,
                                              'latent_early' + self.strains[0] + end, 'rate_ipt_noncompletion')
