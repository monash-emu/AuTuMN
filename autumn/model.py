
"""
All TB-specific models should be coded in this file, including the interventions applied to them.

Time unit throughout: years
Compartment unit throughout: patients

Nested inheritance from BaseModel and StratifiedModel in base.py - the former sets some fundamental methods for
creating intercompartmental flows, costs, etc., while the latter sets out the approach to population stratification.
"""


from scipy import exp, log
from autumn.base import BaseModel, StratifiedModel
import copy
import tool_kit as t_k
import warnings


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
        warnings.warn('Proportion parameter not between zero and one, value is %s' %proportion)
    # to avoid errors where the proportion is exactly one (although the function isn't really intended for this):
    if proportion > .99:
        early_proportion = 0.99
    elif proportion < 0.:
        return 0., 0.
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

    def __init__(self, scenario=0, inputs=None, gui_inputs=None):
        """
        Instantiation, partly inherited from the lower level model objects through nested inheritance.

        Args:
            scenario: Single number for the scenario to run (with None meaning baseline)
            inputs: Non-GUI inputs from data_processing
            gui_inputs: GUI inputs from Tkinter or JS GUI
        """

        # inherited initialisations
        BaseModel.__init__(self)
        StratifiedModel.__init__(self)

        # model attributes set from model runner object
        self.inputs = inputs

        # model attributes to be set directly to attributes of the inputs object
        for attribute in ['compartment_types', 'organ_status', 'strains', 'riskgroups', 'agegroups',
                          'vary_detection_by_organ', 'organs_for_detection', 'riskgroups_for_detection',
                          'vary_detection_by_riskgroup', 'vary_force_infection_by_riskgroup', 'histories']:
            setattr(self, attribute, getattr(inputs, attribute))

        # model attributes to set to just the relevant scenario key from an inputs dictionary
        for attribute in ['relevant_interventions', 'scaleup_fns', 'interventions_to_cost', 'mixing']:
            setattr(self, attribute, getattr(inputs, attribute)[scenario])

        # start_time can't be given as a model constant, as it must be set for each scenario individually
        self.start_time = inputs.model_constants['start_time']

        # model attributes to be set directly to attributes from the GUI object
        for attribute in ['is_lowquality', 'is_amplification', 'is_misassignment', 'is_timevariant_organs', 'country',
                          'time_step', 'integration_method']:
            setattr(self, attribute, gui_inputs[attribute])

        # set fixed parameters from inputs object
        for key, value in inputs.model_constants.items():
            if type(value) == float: self.set_parameter(key, value)

        # list of infectious compartments
        self.infectious_tags = ['active', 'missed', 'detect', 'treatment_infect', 'lowquality']
        self.initial_compartments = {}

        # treatment outcomes
        self.outcomes = ['_success', '_death', '_default']
        self.treatment_stages = ['_infect', '_noninfect']

        # intervention and economics-related initialisations
        if self.eco_drives_epi: self.distribute_funding_across_years()

        # list of risk groups affected by ngo activities for detection
        self.ngo_groups = ['_ruralpoor']

        # whether short-course MDR-TB regimen improves outcomes
        self.shortcourse_improves_outcomes = False

        # create time ticker
        self.next_time_point = copy.copy(self.start_time)

    def initialise_compartments(self):
        """
        Initialise all compartments to zero and then populate with the requested values.
        """

        # extract values for compartment initialisation by compartment type
        for compartment in self.compartment_types:
            if compartment in self.inputs.model_constants:
                self.initial_compartments[compartment] = self.params[compartment]

        # initialise to zero
        for agegroup in self.agegroups:
            for riskgroup in self.riskgroups:
                for history in self.histories:
                    for compartment in self.compartment_types:

                        # replicate susceptible for age and risk groups
                        if 'susceptible' in compartment or 'onipt' in compartment:
                            self.set_compartment(compartment + riskgroup + history + agegroup, 0.)

                        # replicate latent classes for age groups, risk groups and strains
                        elif 'latent' in compartment:
                            for strain in self.strains:
                                self.set_compartment(compartment + strain + riskgroup + history + agegroup, 0.)

                        # replicate active classes for age groups, risk groups, strains and organs
                        elif 'active' in compartment or 'missed' in compartment or 'lowquality' in compartment:
                            for strain in self.strains:
                                for organ in self.organ_status:
                                    self.set_compartment(compartment + organ + strain + riskgroup + history + agegroup,
                                                         0.)

                        # replicate treatment classes for age groups, risk groups, strains, organs and assigned strains
                        else:
                            for strain in self.strains:
                                for organ in self.organ_status:
                                    if self.is_misassignment:
                                        for assigned_strain in self.strains:
                                            self.set_compartment(
                                                compartment + organ + strain + '_as' + assigned_strain[1:] + riskgroup
                                                + history + agegroup, 0.)
                                    else:
                                        self.set_compartment(
                                            compartment + organ + strain + riskgroup + history + agegroup, 0.)

                # remove the unnecessary fully susceptible treated compartments
                self.remove_compartment('susceptible_fully' + riskgroup + self.histories[-1] + agegroup)

        # find starting proportions for risk groups
        if len(self.riskgroups) > 1:
            start_risk_prop = {'_norisk': 1.}
            for riskgroup in self.riskgroups:
                if riskgroup != '_norisk':
                    start_risk_prop[riskgroup] \
                        = self.scaleup_fns['riskgroup_prop' + riskgroup](self.inputs.model_constants['start_time'])
                    start_risk_prop['_norisk'] -= start_risk_prop[riskgroup]
        else:
            start_risk_prop = {'': 1.}

        # arbitrarily split equally by age-groups and organ status
        # start with everyone having least resistant strain (first in list) and having no treatment history
        for compartment in self.compartment_types:
            if compartment in self.initial_compartments:
                for agegroup in self.agegroups:
                    for riskgroup in self.riskgroups:
                        if 'susceptible' in compartment:
                            self.set_compartment(compartment + riskgroup + self.histories[0] + agegroup,
                                                 self.initial_compartments[compartment] * start_risk_prop[riskgroup]
                                                 / len(self.agegroups))
                        elif 'latent' in compartment:
                            self.set_compartment(
                                compartment + self.strains[0] + riskgroup + self.histories[0] + agegroup,
                                self.initial_compartments[compartment] * start_risk_prop[riskgroup]
                                / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(
                                    compartment + organ + self.strains[0] + riskgroup + self.histories[0] + agegroup,
                                    self.initial_compartments[compartment] * start_risk_prop[riskgroup]
                                    / len(self.organ_status) / len(self.agegroups))

    ''' single method to process uncertainty parameters '''

    def process_uncertainty_params(self):
        """
        Perform some simple parameter processing - just for those that are used as uncertainty parameters and so can't
        be processed in the data processing module.
        """

        # find the case fatality of smear-negative TB using the relative case fatality
        self.params['tb_prop_casefatality_untreated_smearneg'] \
            = self.params['tb_prop_casefatality_untreated_smearpos'] \
              * self.params['tb_relative_casefatality_untreated_smearneg']

        # add the extrapulmonary case fatality (currently not entered from the spreadsheets)
        self.params['tb_prop_casefatality_untreated_extrapul'] = self.params['tb_prop_casefatality_untreated_smearneg']

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
        # parameter values are calculated from the costs, but only in the future
        if self.eco_drives_epi and self.time > self.inputs.model_constants['recent_time']: self.update_vars_from_cost()
        self.calculate_populations()
        self.calculate_birth_rates_vars()
        self.calculate_progression_vars()
        if 'int_prop_decentralisation' in self.relevant_interventions: self.adjust_case_detection_for_decentralisation()
        if 'int_prop_opendoors_activities' in self.relevant_interventions \
                or 'int_prop_ngo_activities' in self.relevant_interventions:
            self.adjust_ipt_for_opendoors()
        if self.vary_detection_by_organ:
            self.calculate_case_detection_by_organ()
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
        if self.is_amplification: self.calculate_amplification_var()
        self.calculate_treatment_rates_vars()
        if 'agestratified_ipt' in self.relevant_interventions or 'ipt' in self.relevant_interventions:
            self.calculate_ipt_effect()
        self.calculate_force_infection_vars()
        self.calculate_population_sizes()

    def ticker(self):
        """
        Prints time every ten years to indicate progress through integration.
        """

        if self.time > self.next_time_point:
            print(int(self.time))
            self.next_time_point += 10.

    def calculate_populations(self):
        """
        Find the total absolute numbers of people in the population by each risk-group and overall.
        """

        for riskgroup in self.riskgroups + ['']:
            self.vars['population' + riskgroup] = 0.
            for label in self.labels:
                if riskgroup in label: self.vars['population' + riskgroup] += self.compartments[label]
        for history in self.histories:
            self.vars['population' + history] = 0.
            for label in self.labels:
                if history in label: self.vars['population' + history] += self.compartments[label]
        for history in self.histories:
            self.vars['prop_population' + history] = self.vars['population' + history] / self.vars['population']

    def calculate_birth_rates_vars(self):
        """
        Calculate birth rates into vaccinated and unvaccinated compartments.
        """

        # calculate total births (also for tracking for for interventions)
        self.vars['births_total'] = self.vars['demo_rate_birth'] / 1e3 * self.vars['population']

        # determine vaccinated and unvaccinated proportions
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

    def calculate_progression_vars(self):
        """
        Calculate vars for the remainder of progressions. The vars for the smear-positive and smear-negative proportions
        have already been calculated, but as all progressions have to go somewhere, we need to calculate the remainder.
        """

        # unstratified (self.organ_status should really have length 0, but length 1 also acceptable)
        if len(self.organ_status) < 2: self.vars['epi_prop'] = 1.

        # stratified into two tiers only (i.e. smear-positive and smear-negative)
        elif len(self.organ_status) == 2: self.vars['epi_prop_smearneg'] = 1. - self.vars['epi_prop_smearpos']

        # fully stratified into smear-positive, smear-negative and extrapulmonary
        else: self.vars['epi_prop_extrapul'] = 1. - self.vars['epi_prop_smearpos'] - self.vars['epi_prop_smearneg']

        # determine variable progression rates
        for organ in self.organ_status:
            for agegroup in self.agegroups:
                for riskgroup in self.riskgroups:
                    for timing in ['_early', '_late']:
                        if riskgroup == '_diabetes':
                            self.vars['tb_rate' + timing + '_progression' + organ + riskgroup + agegroup] \
                                = self.vars['epi_prop' + organ] \
                                  * self.params['tb_rate' + timing + '_progression_norisk' + agegroup] \
                                  * self.params['riskgroup_multiplier_diabetes_progression']
                        else:
                            self.vars['tb_rate' + timing + '_progression' + organ + riskgroup + agegroup] \
                                = self.vars['epi_prop' + organ] \
                                  * self.params['tb_rate' + timing + '_progression' + riskgroup + agegroup]

    def adjust_case_detection_for_decentralisation(self):
        """
        Implement the decentralisation intervention, which narrows the case detection gap between the current values
        and the idealised estimated value.
        Only do so if the current case detection ratio is lower than the idealised detection ratio.
        """

        self.vars['program_prop_detect'] \
            = t_k.increase_parameter_closer_to_value(self.vars['program_prop_detect'],
                                                     self.params['int_ideal_detection'],
                                                     self.vars['int_prop_decentralisation'])

    def adjust_ipt_for_opendoors(self):
        """
        If opendoors programs are stopped, case detection and IPT coverage are reduced.
        This applies to the whole population, as opposed to NGOs activities that apply to specific risk groups.
        """

        if 'int_prop_opendoors_activities' in self.relevant_interventions \
                and self.vars['int_prop_opendoors_activities'] < 1.:
            for agegroup in self.agegroups:
                if 'int_prop_ipt' + agegroup in self.vars:
                    self.vars['int_prop_ipt' + agegroup] \
                        *= 1. - self.params['int_prop_ipt_opendoors']*(1-self.vars['int_prop_opendoors_activities'])

    def calculate_case_detection_by_organ(self):
        """
        Method to perform simple weighting on the assumption that the smear-negative and extra-pulmonary rates are less
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
        if self.vary_detection_by_organ: organs.append('')
        for organ in organs:

            # add empty string for use in following calculation of number of missed patients
            riskgroups_to_loop = copy.copy(self.riskgroups_for_detection)
            if '' not in riskgroups_to_loop: riskgroups_to_loop.append('')
            for riskgroup in riskgroups_to_loop:

                # calculate detection rate from cdr proportion
                self.vars['program_rate_detect' + organ + riskgroup] \
                    = self.vars['program_prop_detect' + organ] \
                      * (1. / self.params['tb_timeperiod_activeuntreated'] + 1. / self.vars['demo_life_expectancy']) \
                      / (1. - self.vars['program_prop_detect' + organ])

                # adjust detection rates for opendoors activities in all groups
                if 'int_prop_opendoors_activities' in self.relevant_interventions \
                        and self.vars['int_prop_opendoors_activities'] < 1.:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        *= 1. - self.params['int_prop_detection_opendoors'] \
                                * (1. - self.vars['int_prop_opendoors_activities'])

                # adjust detection rates for ngo activities in specific risk-groups
                if 'int_prop_ngo_activities' in self.relevant_interventions \
                        and self.vars['int_prop_ngo_activities'] < 1. and riskgroup in self.ngo_groups:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        *= 1. - self.params['int_prop_detection_ngo'] * (1. - self.vars['int_prop_ngo_activities'])

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

        # loop to cover risk groups and community-wide ACF (an empty string)
        riskgroups_to_loop = copy.copy(self.riskgroups)
        if '' not in riskgroups_to_loop: riskgroups_to_loop.append('')
        for riskgroup in riskgroups_to_loop:

            # decide whether to use the general detection proportion (as default), otherwise a risk-group-specific one
            int_prop_acf_detections_per_round = self.params['int_prop_acf_detections_per_round']
            if 'int_prop_acf_detections_per_round' + riskgroup in self.params:
                int_prop_acf_detections_per_round = self.params['int_prop_acf_detections_per_round' + riskgroup]

            # implement ACF by approach and whether CXR first as screening tool
            for acf_type in ['smear', 'xpert']:
                for whether_cxr_screen in ['', 'cxr']:
                    intervention = 'int_prop_' + whether_cxr_screen + acf_type + 'acf' + riskgroup
                    if intervention in self.relevant_interventions and '_smearpos' in self.organ_status:

                        # find unadjusted coverage
                        coverage = self.vars[intervention]

                        # adjust effective coverage for screening test, if being used
                        if whether_cxr_screen == 'cxr': coverage *= self.params['tb_sensitivity_cxr']

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

    def calculate_intensive_screening_rate(self):
        """
        Calculates rates of intensive screening from the proportion of programmatic coverage.
        Intensive screening detects smear-positive disease, and some smear-negative disease
        (incorporating a multiplier for the sensitivity of Xpert for smear-negative disease).
        Extrapulmonary disease can't be detected through intensive screening.
        """

        if 'int_prop_intensive_screening' in self.relevant_interventions:
            screened_subgroups = ['_diabetes', '_hiv']  # may be incorporated into the GUI

            # loop covers risk groups
            for riskgroup in screened_subgroups:
                # the following can't be written as self.organ_status, as it won't work for non-fully-stratified models
                for organ in ['', '_smearpos', '_smearneg', '_extrapul']:
                    self.vars['int_rate_intensive_screening' + organ + riskgroup] = 0.

                for organ in ['_smearpos', '_smearneg']:
                    self.vars['int_rate_intensive_screening' + organ + riskgroup] \
                        += self.vars['int_prop_intensive_screening'] * \
                           self.params['int_prop_attending_clinics' + riskgroup]

                # adjust smear-negative detections for Xpert's sensitivity
                self.vars['int_rate_intensive_screening_smearneg' + riskgroup] \
                    *= self.params['int_prop_xpert_smearneg_sensitivity']

    def adjust_case_detection_for_acf(self):
        """
        Add ACF detection rates to previously calculated passive case detection rates, creating vars for case detection
        that are specific for organs.
        """

        for organ in self.organs_for_detection:
            for riskgroup in self.riskgroups:

                # risk groups
                if 'int_rate_acf' + organ + riskgroup in self.vars:
                    self.vars['program_rate_detect' + organ + riskgroup] \
                        += self.vars['int_rate_acf' + organ + riskgroup]

                # general community
                if 'int_rate_acf' + organ in self.vars:
                    self.vars['program_rate_detect' + organ + riskgroup] += self.vars['int_rate_acf' + organ]

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

        # with misassignment
        for organ in self.organs_for_detection:
            prop_firstline = self.vars['program_prop_firstline_dst']

            # add effect of Xpert on identification, assuming that independent distribution to conventional DST
            if 'int_prop_xpert' in self.relevant_interventions:
                prop_firstline \
                    += (1. - prop_firstline) \
                       * self.vars['int_prop_xpert'] * self.params['int_prop_xpert_sensitivity_mdr']

            # add effect of improve_dst program for culture-positive cases only
            if 'int_prop_improve_dst' in self.relevant_interventions:
                if organ == '_smearpos':
                    prop_firstline += (1. - prop_firstline) * self.vars['int_prop_improve_dst'] * \
                                      self.params['program_prop_smearpos_cultured']
                elif organ == '_smearneg':
                    prop_firstline += (1. - prop_firstline) * self.vars['int_prop_improve_dst'] * \
                                      self.params['program_prop_smearneg_cultured'] * \
                                      self.params['tb_prop_smearneg_culturepos']
                elif organ == '_extrapul':
                    prop_firstline = 0.

            for riskgroup in self.riskgroups_for_detection:
                if len(self.strains) > 1 and self.is_misassignment:
                    # determine rates of identification/misidentification as each strain
                    self.vars['program_rate_detect' + organ + riskgroup + '_ds_asds'] \
                        = self.vars['program_rate_detect' + organ + riskgroup]
                    self.vars['program_rate_detect' + organ + riskgroup + '_ds_asmdr'] = 0.
                    self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asds'] \
                        = (1. - prop_firstline) * self.vars['program_rate_detect' + organ + riskgroup]
                    self.vars['program_rate_detect' + organ + riskgroup + '_mdr_asmdr'] \
                        = prop_firstline * self.vars['program_rate_detect' + organ + riskgroup]

                    # if a third strain is present
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

                # without misassignment, everyone is correctly allocated
                elif len(self.strains) > 1:
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

        # note that there is still a program_rate_detect var even if detection is varied by organ and/or risk group
        self.vars['program_rate_enterlowquality'] \
            = self.vars['program_rate_detect'] * prop_lowqual / (1. - prop_lowqual)

    def calculate_await_treatment_var(self):
        """
        Take the reciprocal of the waiting times to calculate the flow rate to start treatment after detection.
        Note that the default behaviour for a single strain model is to use the waiting time for smear-positives.
        Also weight the time period
        """

        for organ in self.organ_status:

            # adjust smear-negative for Xpert coverage
            if organ == '_smearneg' and 'int_prop_xpert' in self.relevant_interventions:
                time_to_treatment \
                    = self.params['program_timeperiod_await_treatment_smearneg'] \
                      * (1. - self.vars['int_prop_xpert']) \
                      + self.params['int_timeperiod_await_treatment_smearneg_xpert'] * self.vars['int_prop_xpert']

            # do other organ stratifications (including smear-negative if Xpert not an intervention)
            else:
                time_to_treatment = self.params['program_timeperiod_await_treatment' + organ]

            # find the rate as the reciprocal of the time to treatment
            self.vars['program_rate_start_treatment' + organ] = 1. / time_to_treatment

    def calculate_amplification_var(self):
        """
        Previously had a sigmoidal function for amplification proportion, but now thinking that the following switch is
        a better approach because scale-up functions are all calculated in data_processing and we need to be able to
        adjust the time that MDR emerges during model running.
        """

        self.vars['epi_prop_amplification'] = 0. if self.time < self.params['mdr_introduce_time'] \
            else self.params['tb_prop_amplification']

    def calculate_treatment_rates_vars(self):
        """
        Work out rates of progression through treatment by stage of treatment from the proportions provided for success
        and death.
        """

        for strain in self.strains:

            # find baseline treatment period for total duration and for period infectious
            for treatment_stage in ['', '_infect']:
                self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                    = self.params['tb_timeperiod' + treatment_stage + '_ontreatment' + strain]

            # adapt treatment periods for short-course regimen
            if strain == '_mdr' and 'int_prop_shortcourse_mdr' in self.relevant_interventions:
                relative_treatment_duration_mdr \
                    = 1. - self.vars['int_prop_shortcourse_mdr'] \
                           * (1. - self.params['int_prop_shortcourse_mdr_relativeduration'])
                for treatment_stage in ['', '_infect']:
                    self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain] \
                        *= relative_treatment_duration_mdr

                # adapt treatment outcomes for short-course regimen
                if self.shortcourse_improves_outcomes:
                    for outcome in ['_success', '_death']:
                        self.vars['program_prop_treatment' + outcome + '_mdr'] \
                            += (self.params['program_prop_treatment' + outcome + '_shortcoursemdr']
                                - self.vars['program_prop_treatment' + outcome + '_mdr']) \
                               * self.vars['int_prop_shortcourse_mdr']

            # add some extra treatment success if the treatment support intervention is active
            if 'int_prop_treatment_support_relative' in self.relevant_interventions:
                for history in self.histories:
                    self.vars['program_prop_treatment' + history + '_success' + strain] \
                        += (1. - self.vars['program_prop_treatment' + history + '_success' + strain]) \
                           * self.params['int_prop_treatment_support_improvement'] \
                           * self.vars['int_prop_treatment_support_relative']
                    self.vars['program_prop_treatment' + history + '_death' + strain] \
                        -= self.vars['program_prop_treatment' + history + '_death' + strain] \
                           * self.params['int_prop_treatment_support_improvement'] \
                           * self.vars['int_prop_treatment_support_relative']

            elif 'int_prop_treatment_support_absolute' in self.relevant_interventions and strain == self.strains[0]:
                for history in self.histories:
                    self.vars['program_prop_treatment' + history + '_success' + strain] \
                        = t_k.increase_parameter_closer_to_value(
                        self.vars['program_prop_treatment' + history + '_success' + strain],
                        self.vars['program_prop_treatment' + history + '_success' + strain],
                        self.vars['int_prop_treatment_support_absolute'])
                    self.vars['program_prop_treatment' + history + '_death' + strain] \
                        = t_k.decrease_parameter_closer_to_value(
                        self.vars['program_prop_treatment' + history + '_death' + strain],
                        self.params['int_prop_treatment_death_ideal'],
                        self.vars['int_prop_treatment_support_absolute'])

            # subtract some treatment success if ngo activities program has discontinued
            # if 'int_prop_ngo_activities' in self.relevant_interventions:
            #     self.vars['program_prop_treatment_success' + strain] \
            #         -= self.vars['program_prop_treatment_success' + strain] \
            #             * self.params['int_prop_treatment_support_improvement'] \
            #             * (1. - self.vars['int_prop_ngo_activities'])

            #  add some extra treatment success if food vouchers are provided to treated cases
            if 'int_prop_food_voucher' + strain in self.relevant_interventions:
                for history in self.histories:
                    self.vars['program_prop_treatment' + strain + history + '_success'] \
                       += (1. - self.vars['program_prop_treatment' + strain + history + '_success']) \
                          * self.params['int_prop_food_voucher_improvement'] \
                          * self.vars['int_prop_food_voucher' + strain]
                    self.vars['program_prop_treatment' + strain + history + '_death'] \
                        -= self.vars['program_prop_treatment' + strain + history + '_death'] \
                          * self.params['int_prop_food_voucher_improvement'] \
                          * self.vars['int_prop_food_voucher' + strain]

            # calculate the default proportion as the remainder from success and death
            for history in self.histories:
                start = 'program_prop_treatment' + strain + history
                self.vars[start + '_default'] \
                    = max([1. - self.vars[start + '_success'] - self.vars[start + '_death'], 0.])
                sum_of_success_and_death = self.vars[start + '_success'] + self.vars[start + '_death']
                if sum_of_success_and_death > 1.:
                    print('Success and death sum to %s for %s outcome at %s time'
                          %(sum_of_success_and_death, start, self.time))

            # find non-infectious period from infectious and total
            self.vars['tb_timeperiod_noninfect_ontreatment' + strain] \
                = self.vars['tb_timeperiod_ontreatment' + strain] \
                  - self.vars['tb_timeperiod_infect_ontreatment' + strain]

            # find the proportion of deaths/defaults during the infectious and non-infectious stages
            props = {}
            for history in self.histories:
                for outcome in self.outcomes[1:]:
                    props['_infect'], props['_noninfect'] \
                        = find_outcome_proportions_by_period(
                        self.vars['program_prop_treatment' + strain + history + outcome],
                        self.vars['tb_timeperiod_infect_ontreatment' + strain],
                        self.vars['tb_timeperiod_ontreatment' + strain])
                    for treatment_stage in props:
                        self.vars['program_prop_treatment' + strain + history + outcome + treatment_stage] \
                            = props[treatment_stage]

                for treatment_stage in self.treatment_stages:

                    # find the success proportions
                    start = 'program_prop_treatment' + strain + history
                    self.vars[start + '_success' + treatment_stage] \
                        = 1. - self.vars[start + '_default' + treatment_stage] \
                          - self.vars[start + '_death' + treatment_stage]

                    # find the corresponding rates from the proportions
                    for outcome in self.outcomes:
                        end = strain + history + outcome + treatment_stage
                        self.vars['program_rate_treatment' + end] \
                            = self.vars['program_prop_treatment' + end]\
                              / self.vars['tb_timeperiod' + treatment_stage + '_ontreatment' + strain]

                    # split default according to whether amplification occurs (if not the most resistant strain)
                    if self.is_amplification:
                        start = 'program_rate_treatment' + strain + history + '_default' + treatment_stage
                        self.vars[start + '_amplify'] \
                            = self.vars[start] * self.vars['epi_prop_amplification']
                        self.vars[start + '_noamplify'] \
                            = self.vars[start] * (1. - self.vars['epi_prop_amplification'])

            # create new vars for misassigned individuals
            if len(self.strains) > 1 and self.is_misassignment and strain != '_ds':
                for treated_as in self.strains:  # for each strain
                    if treated_as != strain:  # misassigned strain has to be different from the actual strain
                        if self.strains.index(treated_as) < self.strains.index(strain):  # if treated with worse regimen
                            for history in self.histories:

                                # calculate the default proportion as the remainder from success and death
                                start = 'program_prop_treatment_inappropriate' + history
                                self.vars[start + '_default'] \
                                    = 1. - self.vars[start + '_success'] - self.vars[start + '_death']

                                props = {}
                                for outcome in self.outcomes[1:]:
                                    treatment_type = strain + '_as' + treated_as[1:]
                                    props['_infect'], props['_noninfect'] \
                                        = find_outcome_proportions_by_period(
                                        self.vars['program_prop_treatment_inappropriate' + history + outcome],
                                        self.params['tb_timeperiod_infect_ontreatment' + treated_as],
                                        self.params['tb_timeperiod_ontreatment' + treated_as])
                                    for treatment_stage in props:
                                        self.vars['program_prop_treatment' + treatment_type + history + outcome
                                                  + treatment_stage] \
                                            = props[treatment_stage]

                                for treatment_stage in self.treatment_stages:
                                    # find the success proportions
                                    start = 'program_prop_treatment' + treatment_type + history
                                    self.vars[start + '_success' + treatment_stage] \
                                        = 1. - self.vars[start + '_default' + treatment_stage] \
                                          - self.vars[start + '_death' + treatment_stage]

                                    # find the corresponding rates from the proportions
                                    for outcome in self.outcomes:
                                        end = treatment_type + history + outcome + treatment_stage
                                        self.vars['program_rate_treatment' + end] \
                                            = self.vars['program_prop_treatment' + end] \
                                              / self.vars['tb_timeperiod' + treatment_stage + '_ontreatment'
                                                          + treated_as]

                                    # split default according to whether amplification occurs
                                    if self.is_amplification:
                                        start = 'program_rate_treatment' + treatment_type + history + '_default'\
                                                + treatment_stage
                                        self.vars[start + '_amplify'] \
                                            = self.vars[start] * self.vars['epi_prop_amplification']
                                        self.vars[start + '_noamplify'] \
                                            = self.vars[start] * (1. - self.vars['epi_prop_amplification'])

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

            # calculate infections averted as product of infections of identified cases, coverage and effectiveness
            self.vars['prop_infections_averted_ipt' + agegroup] \
                = self.vars['tb_prop_infections_reachable_ipt'] * coverage

    def calculate_force_infection_vars(self):
        """
        Calculate force of infection independently for each strain, incorporating partial immunity and infectiousness.
        First calculate the effective infectious population (incorporating infectiousness by organ involvement), then
        calculate the raw force of infection, then adjust for various levels of susceptibility.
        """

        # to loop over all risk groups if needed, or otherwise to just run once
        force_riskgroups = copy.copy(self.riskgroups) if self.vary_force_infection_by_riskgroup else ['']

        # if any modifications to transmission parameter to be made over time
        transmission_modifier = self.vars['transmission_modifier'] if 'transmission_modifier' in self.vars else 1.

        # susceptibility classes
        force_types = ['_fully', '_immune', '_latent']
        if 'int_prop_novel_vaccination' in self.relevant_interventions: force_types.append('_novelvac')

        # find the effective infectious population for each strain and risk group
        for strain in self.strains:
            for riskgroup in force_riskgroups:

                # initialise infectiousness vars
                self.vars['infectiousness' + strain + riskgroup] = 0.

                # loop through compartments, skipping on as soon as possible if irrelevant
                for label in self.labels:
                    if riskgroup not in label and riskgroup != '': continue
                    if strain not in label and strain != '': continue
                    for agegroup in self.agegroups:
                        if agegroup not in label and agegroup != '': continue
                        for organ in self.organ_status:
                            if (organ not in label and organ != '') or organ == '_extrapul': continue

                            # adjustment for increased infectiousness in riskgroup
                            riskgroup_force_multiplier \
                                = self.params['riskgroup_multiplier_force_infection' + riskgroup] \
                                if 'riskgroup_multiplier_force_infection' + riskgroup in self.params else 1.

                            # increment "infectiousness", the effective number of infectious people in the stratum
                            if t_k.label_intersects_tags(label, self.infectious_tags):
                                self.vars['infectiousness' + strain + riskgroup] \
                                    += self.params['tb_n_contact'] \
                                       * transmission_modifier \
                                       * self.params['tb_multiplier_force' + organ] \
                                       * self.params['tb_multiplier_child_infectiousness' + agegroup] \
                                       * self.compartments[label] \
                                       * riskgroup_force_multiplier \
                                       / self.vars['population' + riskgroup]

            # calculate force of infection using mixing matrix to weight the infectious populations if required
            if self.vary_force_infection_by_riskgroup:
                for to_riskgroup in force_riskgroups:
                    self.vars['rate_force' + strain + to_riskgroup] = 0.
                    for from_riskgroup in force_riskgroups:
                        self.vars['rate_force' + strain + to_riskgroup] \
                            += self.vars['infectiousness' + strain + to_riskgroup] \
                               * self.mixing[to_riskgroup][from_riskgroup]
            else:
                self.vars['rate_force' + strain] = self.vars['infectiousness' + strain]

            # adjust for immunity in various groups, first defining broad immunity categories
            for riskgroup in force_riskgroups:
                for force_type in force_types:
                    immunity_multiplier = self.params['tb_multiplier' + force_type + '_protection']

                    # give children greater protection from BCG vaccination
                    for agegroup in self.agegroups:
                        if t_k.interrogate_age_string(agegroup)[0][1] <= self.params['int_age_bcg_immunity_wane'] \
                                and force_type == '_immune':
                            immunity_multiplier *= self.params['int_multiplier_bcg_child_relative_immunity']

                        # increase immunity for previously treated
                        for history in self.histories:
                            if history == '_treated':
                                immunity_multiplier *= self.params['tb_multiplier_treated_protection']

                            # find forces of infection, except that there is no previously treated fully susceptible
                            if force_type != '_fully' or (force_type == '_fully' and history == self.histories[0]):
                                self.vars['rate_force' + force_type + strain + history + riskgroup + agegroup] \
                                    = self.vars['rate_force' + strain + riskgroup] * immunity_multiplier

        # adjusted for IPT and create IPT flow rate
        for strain in self.strains:
            for riskgroup in force_riskgroups:
                for force_type in force_types:
                    for agegroup in self.agegroups:
                        for history in self.histories:

                            if force_type != '_fully' or (force_type == '_fully' and history == self.histories[0]):

                                # separate infections out into those treated and those not treated for DS-TB infection
                                stratum = force_type + strain + history + riskgroup + agegroup
                                self.vars['rate_ipt_commencement' + stratum] = 0.
                                if ('agestratified_ipt' in self.relevant_interventions
                                    or 'ipt' in self.relevant_interventions) and strain == self.strains[0]:
                                    self.vars['rate_ipt_commencement' + stratum] \
                                        += self.vars['rate_force' + stratum] \
                                           * self.vars['prop_infections_averted_ipt' + agegroup]
                                    self.vars['rate_force' + stratum] \
                                        -= self.vars['rate_ipt_commencement' + stratum]

    def calculate_population_sizes(self):
        """
        Calculate the size of the populations to which each intervention is applicable, for use in generating
        cost-coverage curves.
        """

        # treatment support
        for intervention in ['treatment_support_relative', 'treatment_support_absolute']:
            if 'int_prop_' + intervention in self.relevant_interventions:
                self.vars['popsize_' + intervention] = 0.
                for compartment in self.compartments:
                    if 'treatment_' in compartment:
                        self.vars['popsize_' + intervention] += self.compartments[compartment]

        # ambulatory care
        for organ in self.organ_status:
            if 'int_prop_ambulatorycare' + organ in self.relevant_interventions:
                self.vars['popsize_ambulatorycare' + organ] = 0.
                for compartment in self.compartments:
                    if 'treatment_' in compartment and organ in compartment:
                        self.vars['popsize_ambulatorycare' + organ] += self.compartments[compartment]

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
                        self.vars['popsize_ipt' + agegroup] += self.compartments[from_label] * self.vars[rate]

        # BCG (so simple that it's almost unnecessary, but needed for loops over int names)
        self.vars['popsize_vaccination'] = self.vars['births_total']

        # Xpert and improve DST - all presentations for assessment for active TB
        for active_tb_presentations_intervention in ['xpert', 'improve_dst']:
            if 'int_prop_' + active_tb_presentations_intervention in self.relevant_interventions:
                self.vars['popsize_' + active_tb_presentations_intervention] = 0.
                for agegroup in self.agegroups:
                    for riskgroup in self.riskgroups:
                        for strain in self.strains:
                            for history in self.histories:
                                for organ in self.organ_status:
                                    if active_tb_presentations_intervention == 'improve_dst' and organ == '_extrapul':
                                        continue
                                    if self.vary_detection_by_organ:
                                        detection_organ = organ
                                    else:
                                        detection_organ = ''
                                    if self.vary_detection_by_riskgroup:
                                        detection_riskgroup = riskgroup
                                    else:
                                        detection_riskgroup = ''
                                    self.vars['popsize_' + active_tb_presentations_intervention] \
                                        += self.vars['program_rate_detect' + detection_organ + detection_riskgroup] \
                                           * self.compartments[
                                               'active' + organ + strain + riskgroup + history + agegroup] \
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
            if adjust_lowquality: self.vars['popsize_engage_lowquality'] *= self.vars['program_prop_lowquality']

        # shortcourse MDR-TB regimen
        if 'int_prop_shortcourse_mdr' in self.relevant_interventions:
            self.vars['popsize_shortcourse_mdr'] = 0.
            for compartment in self.compartments:
                if 'treatment' in compartment and '_mdr' in compartment:
                    self.vars['popsize_shortcourse_mdr'] += self.compartments[compartment]

        # open doors activities
        if 'int_prop_opendoors_activities' in self.relevant_interventions:
            # number of diagnosed cases (LTBI + TB) if coverage was 100%
            self.vars['popsize_opendoors_activities'] = 598*2.

        # NGO activities
        if 'int_prop_ngo_activities' in self.relevant_interventions:
            # number of diagnosed cases (LTBI + TB) if coverage was 100%
            self.vars['popsize_ngo_activities'] = 456*2

        # awareness raising
        if 'int_prop_awareness_raising' in self.relevant_interventions:
            self.vars['popsize_awareness_raising'] = self.vars['population']

    ''' methods that calculate the flows of all the compartments '''

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
        if 'agestratified_ipt' in self.relevant_interventions or 'ipt' in self.relevant_interventions:
            self.set_ipt_flows()

    def set_birth_flows(self):
        """
        Set birth (or recruitment) flows by vaccination status (including novel vaccination if implemented).
        """

        for riskgroup in self.riskgroups:
            self.set_var_entry_rate_flow(
                'susceptible_fully' + riskgroup + self.histories[0] + self.agegroups[0],
                'births_unvac' + riskgroup)
            self.set_var_entry_rate_flow(
                'susceptible_immune' + riskgroup + self.histories[0] + self.agegroups[0],
                'births_vac' + riskgroup)
            if 'int_prop_novel_vaccination' in self.relevant_interventions:
                self.set_var_entry_rate_flow(
                    'susceptible_novelvac' + riskgroup + self.histories[0] + self.agegroups[0],
                    'births_novelvac' + riskgroup)

    def set_infection_flows(self):
        """
        Set force of infection flows that were estimated by strain in calculate_force_infection_vars above.
        """

        for strain in self.strains:

            # vary force of infection by risk-group if heterogeneous mixing is incorporated
            for riskgroup in self.riskgroups:
                force_riskgroup = ''
                if self.vary_force_infection_by_riskgroup: force_riskgroup = riskgroup
                for agegroup in self.agegroups:
                    force_types = ['_fully', '_immune', '_latent']
                    if 'int_prop_novel_vaccination' in self.relevant_interventions: force_types.append('_novelvac')
                    for force_type in force_types:
                        for history in self.histories:
                            if force_type == '_immune' or (force_type == '_fully' and history == self.histories[0]):
                                self.set_var_transfer_rate_flow(
                                    'susceptible' + force_type + riskgroup + history + agegroup,
                                    'latent_early' + strain + riskgroup + history + agegroup,
                                    'rate_force' + force_type + strain + history + force_riskgroup + agegroup)
                                self.set_var_transfer_rate_flow(
                                    'susceptible' + force_type + riskgroup + history + agegroup,
                                    'onipt' + riskgroup + history + agegroup,
                                    'rate_ipt_commencement' + force_type + strain + history + riskgroup + agegroup)
                            elif force_type == '_latent':
                                for from_strain in self.strains:
                                    self.set_var_transfer_rate_flow(
                                        'latent_late' + from_strain + riskgroup + history + agegroup,
                                        'latent_early' + strain + riskgroup + history + agegroup,
                                        'rate_force' + force_type + strain + history + force_riskgroup + agegroup)
                                    self.set_var_transfer_rate_flow(
                                        'latent_late' + from_strain + riskgroup + history + agegroup,
                                        'onipt' + riskgroup + history + agegroup,
                                        'rate_ipt_commencement' + force_type + strain + history + riskgroup + agegroup)

    def set_progression_flows(self):
        """
        Set rates of progression from latency to active disease, with rates differing by organ status.
        """

        for agegroup in self.agegroups:
            for history in self.histories:
                for riskgroup in self.riskgroups:
                    for strain in self.strains:

                        # stabilisation
                        self.set_fixed_transfer_rate_flow(
                            'latent_early' + strain + riskgroup + history + agegroup,
                            'latent_late' + strain + riskgroup + history + agegroup,
                            'tb_rate_stabilise' + riskgroup + agegroup)

                        # now smear-pos/smear-neg is always a var, even when a constant function
                        for organ in self.organ_status:
                            self.set_var_transfer_rate_flow(
                                'latent_early' + strain + riskgroup + history + agegroup,
                                'active' + organ + strain + riskgroup + history + agegroup,
                                'tb_rate_early_progression' + organ + riskgroup + agegroup)
                            self.set_var_transfer_rate_flow(
                                'latent_late' + strain + riskgroup + history + agegroup,
                                'active' + organ + strain + riskgroup + history + agegroup,
                                'tb_rate_late_progression' + organ + riskgroup + agegroup)

    def set_natural_history_flows(self):
        """
        Set flows for progression through active disease to either recovery or death.
        """

        # determine the compartments to which natural history flows apply
        active_compartments = ['active', 'missed']
        if self.is_lowquality: active_compartments.append('lowquality')
        if not self.is_misassignment: active_compartments.append('detect')

        # apply flows
        for history in self.histories:
            for agegroup in self.agegroups:
                for riskgroup in self.riskgroups:
                    for strain in self.strains:
                        for organ in self.organ_status:
                            for compartment in active_compartments:

                                # recovery
                                self.set_fixed_transfer_rate_flow(
                                    compartment + organ + strain + riskgroup + history + agegroup,
                                    'latent_late' + strain + riskgroup + history + agegroup,
                                    'tb_rate_recover' + organ)

                                # death
                                self.set_fixed_infection_death_rate_flow(
                                    compartment + organ + strain + riskgroup + history + agegroup,
                                    'tb_rate_death' + organ)

                            # detected, with misassignment
                            if self.is_misassignment:
                                for assigned_strain in self.strains:
                                    self.set_fixed_infection_death_rate_flow(
                                        'detect' + organ + strain + '_as' + assigned_strain[1:] + riskgroup
                                        + history + agegroup,
                                        'tb_rate_death' + organ)
                                    self.set_fixed_transfer_rate_flow(
                                        'detect' + organ + strain + '_as' + assigned_strain[1:] + riskgroup
                                        + history + agegroup,
                                        'latent_late' + strain + riskgroup + history + agegroup,
                                        'tb_rate_recover' + organ)

    def set_fixed_programmatic_flows(self):
        """
        Set rates of return to active disease for patients who presented for health care and were missed and for
        patients who were in the low-quality health care sector.
        """

        for history in self.histories:
            for agegroup in self.agegroups:
                for riskgroup in self.riskgroups:
                    for strain in self.strains:
                        for organ in self.organ_status:

                            # re-start presenting after a missed diagnosis
                            self.set_fixed_transfer_rate_flow(
                                'missed' + organ + strain + riskgroup + history + agegroup,
                                'active' + organ + strain + riskgroup + history + agegroup,
                                'program_rate_restart_presenting')

                            # giving up on the hopeless low-quality health system
                            if self.is_lowquality:
                                self.set_fixed_transfer_rate_flow(
                                    'lowquality' + organ + strain + riskgroup + history + agegroup,
                                    'active' + organ + strain + riskgroup + history + agegroup,
                                    'program_rate_leavelowquality')

    def set_detection_flows(self):
        """
        Set previously calculated detection rates (either assuming everyone is correctly identified if misassignment
        not permitted or with proportional misassignment).
        """

        for agegroup in self.agegroups:
            for history in self.histories:
                for riskgroup in self.riskgroups:
                    riskgroup_for_detection = ''
                    if self.vary_detection_by_riskgroup: riskgroup_for_detection = riskgroup
                    for organ in self.organ_status:
                        organ_for_detection = organ
                        if not self.vary_detection_by_organ:
                            organ_for_detection = ''

                        for strain_number, strain in enumerate(self.strains):

                            # with misassignment
                            if self.is_misassignment:
                                for assigned_strain_number in range(len(self.strains)):
                                    as_assigned_strain = '_as' + self.strains[assigned_strain_number][1:]

                                    # if the strain is equally or more resistant than its assignment
                                    if strain_number >= assigned_strain_number:
                                        self.set_var_transfer_rate_flow(
                                            'active' + organ + strain + riskgroup + history + agegroup,
                                            'detect' + organ + strain + as_assigned_strain + riskgroup
                                            + history + agegroup,
                                            'program_rate_detect' + organ_for_detection + riskgroup_for_detection
                                            + strain + as_assigned_strain)

                            # without misassignment
                            else:
                                self.set_var_transfer_rate_flow(
                                    'active' + organ + strain + riskgroup + history + agegroup,
                                    'detect' + organ + strain + riskgroup + history + agegroup,
                                    'program_rate_detect' + organ_for_detection + riskgroup_for_detection)

    def set_variable_programmatic_flows(self):
        """
        Set rate of missed diagnosis (which is variable as the algorithm sensitivity typically will be), rate of
        presentation to low quality health care (which is variable as the extent of this health system typically will
        be) and rate of treatment commencement (which is variable and depends on the diagnostics available).
        """

        # set rate of missed diagnoses and entry to low-quality health care
        for agegroup in self.agegroups:
            for history in self.histories:
                for riskgroup in self.riskgroups:
                    for strain in self.strains:
                        for organ in self.organ_status:
                            detection_organ = ''
                            if self.vary_detection_by_organ: detection_organ = organ
                            self.set_var_transfer_rate_flow(
                                'active' + organ + strain + riskgroup + history + agegroup,
                                'missed' + organ + strain + riskgroup + history + agegroup,
                                'program_rate_missed' + detection_organ)

                            # treatment commencement, with and without misassignment
                            if self.is_misassignment:
                                for assigned_strain in self.strains:
                                    self.set_var_transfer_rate_flow(
                                        'detect' + organ + strain + '_as' + assigned_strain[1:] + riskgroup
                                        + history + agegroup,
                                        'treatment_infect' + organ + strain + '_as' + assigned_strain[1:]
                                        + riskgroup + history + agegroup,
                                        'program_rate_start_treatment' + organ)
                            else:
                                self.set_var_transfer_rate_flow(
                                    'detect' + organ + strain + riskgroup + history + agegroup,
                                    'treatment_infect' + organ + strain + riskgroup + history + agegroup,
                                    'program_rate_start_treatment' + organ)

                            # enter the low quality health care system
                            if self.is_lowquality:
                                self.set_var_transfer_rate_flow(
                                    'active' + organ + strain + riskgroup + history + agegroup,
                                    'lowquality' + organ + strain + riskgroup + history + agegroup,
                                    'program_rate_enterlowquality')

    def set_treatment_flows(self):
        """
        Set rates of progression through treatment stages - dealing with amplification, as well as misassignment if
        either or both are implemented.
        """

        for agegroup in self.agegroups:
            for history in self.histories:
                for riskgroup in self.riskgroups:
                    for organ in self.organ_status:
                        for strain_number, strain in enumerate(self.strains):

                            # which strains to loop over for strain assignment
                            assignment_strains = ['']
                            if self.is_misassignment: assignment_strains = self.strains
                            for assigned_strain_number, assigned_strain in enumerate(assignment_strains):
                                as_assigned_strain = ''
                                strain_or_inappropriate = strain
                                if self.is_misassignment:
                                    as_assigned_strain = '_as' + assigned_strain[1:]

                                    # which treatment parameters to use - for the strain or for inappropriate treatment
                                    strain_or_inappropriate = assigned_strain
                                    if strain_number > assigned_strain_number:
                                        strain_or_inappropriate = strain + '_as' + assigned_strain[1:]

                                # success by treatment stage
                                end = organ + strain + as_assigned_strain + riskgroup + history + agegroup
                                self.set_var_transfer_rate_flow(
                                    'treatment_infect' + end,
                                    'treatment_noninfect' + end,
                                    'program_rate_treatment' + strain_or_inappropriate + history + '_success_infect')
                                self.set_var_transfer_rate_flow(
                                    'treatment_noninfect' + end,
                                    'susceptible_immune' + riskgroup + self.histories[-1] + agegroup,
                                    'program_rate_treatment' + strain_or_inappropriate + history + '_success_noninfect')

                                # death on treatment
                                for treatment_stage in self.treatment_stages:
                                    self.set_var_infection_death_rate_flow(
                                        'treatment' + treatment_stage + end,
                                        'program_rate_treatment' + strain_or_inappropriate + history + '_death'
                                        + treatment_stage)

                                # default
                                for treatment_stage in self.treatment_stages:

                                    # if it's either the most resistant strain available or amplification is not active:
                                    if strain == self.strains[-1] or not self.is_amplification:
                                        self.set_var_transfer_rate_flow(
                                            'treatment' + treatment_stage + end,
                                            'active' + organ + strain + riskgroup + history + agegroup,
                                            'program_rate_treatment' + strain_or_inappropriate + history + '_default'
                                            + treatment_stage)

                                    # otherwise with amplification
                                    else:
                                        amplify_to_strain = self.strains[strain_number + 1]
                                        self.set_var_transfer_rate_flow(
                                            'treatment' + treatment_stage + end,
                                            'active' + organ + strain + riskgroup + history + agegroup,
                                            'program_rate_treatment' + strain_or_inappropriate + history + '_default'
                                            + treatment_stage + '_noamplify')
                                        self.set_var_transfer_rate_flow(
                                            'treatment' + treatment_stage + end,
                                            'active' + organ + amplify_to_strain + riskgroup + history + agegroup,
                                            'program_rate_treatment' + strain_or_inappropriate + history + '_default'
                                            + treatment_stage + '_amplify')

    def set_ipt_flows(self):
        """
        Implement IPT-related transitions: completion of treatment and failure to complete treatment.
        """

        for agegroup in self.agegroups:
            for history in self.histories:
                for riskgroup in self.riskgroups:

                    # treatment completion flows
                    self.set_fixed_transfer_rate_flow('onipt' + riskgroup + history + agegroup,
                        'susceptible_immune' + riskgroup + history + agegroup, 'rate_ipt_completion')

                    # treatment non-completion flows
                    self.set_fixed_transfer_rate_flow('onipt' + riskgroup + history + agegroup,
                        'latent_early' + self.strains[0] + riskgroup + history + agegroup, 'rate_ipt_noncompletion')

