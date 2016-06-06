# -*- coding: utf-8 -*-


"""

The TB-specific model, or models, should be coded in this file

Time unit throughout: years
Compartment unit throughout: patients

"""

import random
from scipy import exp, log
from autumn.base import BaseModel
from curve import make_sigmoidal_curve, make_two_step_curve, scale_up_function
import numpy
import pylab


def label_intersects_tags(label, tags):
    for tag in tags:
        if tag in label:
            return True
    return False


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
                 n_organ=0,
                 n_strain=0,
                 n_comorbidity=0,
                 is_lowquality=False,
                 is_amplification=False,
                 is_misassignment=False,
                 scenario=None,
                 data=None):

        """
        Args:
            n_organ: whether pulmonary status and smear-positive/smear-negative status
                can be included in the model (which apply to all compartments representing active disease)
                0. No subdivision
                1. All patients are smear-positive pulmonary (avoid)
                2. All patients are pulmonary, but smear status can be selected (i.e. smear-pos/smear-neg)
                3. Full stratification into smear-positive, smear-negative and extra-pulmonary
            n_strain: number of types of drug-resistance included (not strains in the strict phylogenetic sense)
                0. No strains included
                1. All TB is DS-TB (avoid)
                2. DS-TB and MDR-TB
                3. DS-TB, MDR-TB and XDR-TB
                (N.B. this may change in future models, which may include isoniazid mono-resistance, etc.)
            n_comorbidity: number of whole-population stratifications, other than age
                0. No population stratification
                1. Entire population is not at increased risk (avoid)
                2. No additional risk factor or HIV
                3. No additional risk factor, HIV or diabetes
            is_lowquality: Boolean of whether to include detections through the private/low-quality sector
            is_amplification: Boolean of whether to include resistance amplification through treatment default
            is_misassignment: Boolean of whether also to incorporate misclassification of patients with drug-resistance
                    to the wrong strain by the high-quality health system
                Avoid amplification=False but misassignment=True (the model should run with both
                amplification and misassignment, but this combination doesn't make sense)
        """

        BaseModel.__init__(self)

        # Convert inputs to attributes
        self.age_breakpoints = data['attributes']['age_breakpoints']
        self.n_organ = n_organ
        self.n_strain = n_strain
        self.n_comorbidity = n_comorbidity
        self.start_time = data['attributes']['start_time']

        self.is_lowquality = is_lowquality
        self.is_amplification = is_amplification
        self.is_misassignment = is_misassignment

        self.scenario = scenario

        self.data = data

        if self.is_misassignment:
            assert self.is_amplification, "Misassignment requested without amplification"

        # Initialise model compartmental structure and set un-processed parameters
        self.define_model_structure()
        self.initialise_compartments()
        self.set_fixed_parameters()
        self.set_fixed_demo_parameters()
        self.set_fixed_epi_parameters()

        # Treatment outcomes that will be universal to all models
        # Global TB outcomes of "completion" and "cure" can be considered "_success",
        # "death" is "_death" (of course), "failure" and "default" are considered "_default"
        # and "transfer out" is removed from denominator calculations.
        self.outcomes = ["_success", "_death", "_default"]
        self.non_success_outcomes = self.outcomes[1: 3]

        self.make_times(self.data['attributes']['start_time'],
                        self.data['attributes']['scenario_end_time'],
                        self.data['attributes']['time_step'])

    def define_model_structure(self):

        """
        Args:
            All arguments are set through __init__
            Please refer to __init__ method comments above
        """

        # All compartmental disease stages
        self.compartment_types = [
            "susceptible_fully",
            "susceptible_vac",
            "susceptible_treated",
            "latent_early",
            "latent_late",
            "active",
            "detect",
            "missed",
            "treatment_infect",
            "treatment_noninfect"]
        if self.is_lowquality: self.compartment_types += ["lowquality"]

        # Broader stages for calculating outputs later
        self.broad_compartment_types = [
            "susceptible",
            "latent",
            "active",
            "missed",
            "treatment"]
        if self.is_lowquality: self.broad_compartment_types += ["lowquality"]

        # Stages in progression through treatment
        self.treatment_stages = [
            "_infect",
            "_noninfect"]

        # Compartments that contribute to force of infection calculations
        self.infectious_tags = [
            "active",
            "missed",
            "detect",
            "treatment_infect"]
        if self.is_lowquality: self.infectious_tags += ["lowquality"]

        # Select number of organ statuses
        available_organs = [
            "_smearpos",
            "_smearneg",
            "_extrapul"]
        if self.n_organ == 0:
            # Need a list of an empty string to be iterable for methods iterating by organ status
            self.organ_status = [""]
        else:
            self.organ_status = available_organs[:self.n_organ]

        # Select number of strains
        available_strains = [
            "_ds",
            "_mdr",
            "_xdr"]
        if self.n_strain == 0:
            # Need a list of an empty string to be iterable for methods iterating by strain
            self.strains = [""]
        else:
            self.strains = available_strains[:self.n_strain]

        # Select number of risk groups
        available_comorbidities = [
            "_nocomorbs",
            "_hiv",
            "_diabetes"]
        if self.n_comorbidity == 0:
            # Need a list of an empty string to be iterable for methods iterating by risk group
            self.comorbidities = [""]
        else:
            self.comorbidities = available_comorbidities[:self.n_comorbidity]

        self.define_age_structure()

    def initialise_compartments(self, compartment_dict=None):

        # First initialise all compartments to zero
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for compartment in self.compartment_types:

                    # Replicate susceptible for age-groups and comorbidities
                    if "susceptible" in compartment:
                        self.set_compartment(compartment + comorbidity + agegroup, 0.)

                    # Replicate latent classes for age-groups, comorbidities and strains
                    elif "latent" in compartment:
                        for strain in self.strains:
                            self.set_compartment(compartment + strain + comorbidity + agegroup, 0.)

                    # Replicate active classes for age-groups, comorbidities, strains and organ status
                    elif "active" in compartment or "missed" in compartment or "lowquality" in compartment:
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
                                            "_as" + self.strains[assigned_strain_number][1:] +
                                            comorbidity + agegroup, 0.)
                                else:
                                    self.set_compartment(compartment +
                                                         organ + strain + comorbidity + agegroup, 0.)

        # Populate input_compartments
        # Initialise to DS-TB or no strain if single-strain model
        default_start_strain = "_ds"
        if self.strains == [""]: default_start_strain = ""

        # The equal splits may need to be adjusted, but the important thing is not to
        # initialise multiple strains too early, so that MDR-TB doesn't overtake the model
        for compartment in self.compartment_types:
            if compartment in self.data['attributes']['start_compartments']:
                for agegroup in self.agegroups:
                    for comorbidity in self.comorbidities:
                        if "susceptible_fully" in compartment:
                            # Split equally by comorbidities and age-groups
                            self.set_compartment(compartment + comorbidity + agegroup,
                                                 self.data['attributes']['start_compartments'][compartment]
                                                 / len(self.comorbidities)
                                                 / len(self.agegroups))
                        elif "latent" in compartment:
                            # Assign all to DS-TB, split equally by comorbidities and age-groups
                            self.set_compartment(compartment + default_start_strain + comorbidity + agegroup,
                                                 self.data['attributes']['start_compartments'][compartment]
                                                 / len(self.comorbidities)
                                                 / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(compartment +
                                                     organ + default_start_strain + comorbidity + agegroup,
                                                     self.data['attributes']['start_compartments'][compartment]
                                                     / len(self.organ_status)  # Split equally by organ statuses,
                                                     / len(self.comorbidities)  # split equally by comorbidities
                                                     / len(self.agegroups))  # and split equally by age-groups

    def set_fixed_parameters(self):

        """
        Sets parameters that should never be changed in any situation,
        i.e. "by definition" parameters
        """

        fixed_parameters = {
            "epi_proportion_cases":
                1.,  # Proportion infectious if only one organ-status running
            "tb_multiplier_force":
                1.,  # Infectiousness multiplier if only one organ-status running
            "tb_multiplier_force_smearpos":
                1.,  # Proporiton of smear-positive patients infectious
            "tb_multiplier_force_extrapul":
                0.
        }
        for parameter in fixed_parameters:
            self.set_parameter(parameter, fixed_parameters[parameter])

    def set_fixed_demo_parameters(self):

        """
        Temporary code to set birth and death rates to the most recent ones recorded in the
        country-specific spreadsheets being read in.
        """

        self.set_parameter('demo_rate_birth',
                           self.data['birth_rate'][max(self.data['birth_rate'])]
                           / 1e3)
        self.set_parameter('demo_rate_death',
                           1.
                           / self.data['life_expectancy'][max(self.data['life_expectancy'])])

    def set_fixed_epi_parameters(self):

        """
        Sets fixed proportions smear-positive, smear-negative and extra-pulmonary based on the
        most recent proportions available from GTB notifications for the country being simulated.
        """

        ########## Temporary code - needs to be converted to a var

        arbitray_index = 30
        if self.n_organ > 1:
            self.set_parameter('epi_proportion_cases_smearpos',
                               self.data['notifications'][u'prop_new_sp'][arbitray_index])
        if self.n_organ > 2:
            self.set_parameter('epi_proportion_cases_smearneg',
                               self.data['notifications'][u'prop_new_sn'][arbitray_index])
            self.set_parameter('epi_proportion_cases_extrapul',
                               self.data['notifications'][u'prop_new_ep'][arbitray_index])
        elif self.n_organ == 2:
            self.set_parameter('epi_proportion_cases_smearneg',
                               self.data['notifications'][u'prop_new_sn'][arbitray_index]
                               + self.data['notifications'][u'prop_new_ep'][arbitray_index])

    def define_age_structure(self):

        # Work out age-groups from list of breakpoints
        self.agegroups = []

        # If age-group breakpoints are supplied
        if len(self.age_breakpoints) > 0:
            for i in range(len(self.age_breakpoints)):
                if i == 0:

                    # The first age-group
                    self.agegroups += \
                        ["_age0to" + str(self.age_breakpoints[i])]
                else:

                    # Middle age-groups
                    self.agegroups += \
                        ["_age" + str(self.age_breakpoints[i - 1]) +
                         "to" + str(self.age_breakpoints[i])]

            # Last age-group
            self.agegroups += ["_age" + str(self.age_breakpoints[len(self.age_breakpoints) - 1]) + "up"]

        # Otherwise
        else:
            # List consisting of one empty string required
            # for the methods that iterate over strains
            self.agegroups += [""]

    ############################################################
    # General underlying methods for use by other methods

    def find_outcome_proportions_by_period(
            self, proportion, early_period, total_period):

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
            # This is just to avoid warnings appearing where the proportion
            # is one - this function isn't really intended for this situation,
            # but preferable to avoid both errors and warnings.
            early_proportion = 0.5
        else:
            early_proportion \
                = 1. - exp(log(1. - proportion) * early_period / total_period)
        late_proportion = proportion - early_proportion
        return early_proportion, late_proportion

    ##################################################################
    # The master parameter processing and scale-up setting method

    def process_parameters(self):

        """
        Calls all the methods to set parameters and scale-up functions
        The order in which these methods is run is often important
        """

        if len(self.agegroups) > 1: self.find_ageing_rates()

        self.find_natural_history_params()

        self.find_programs_to_run()

        self.find_nontreatment_rates_params()

        self.find_treatment_rates_scaleups()

        self.find_amplification_scaleup()

        self.find_organ_scaleup()

        self.set_population_death_rate("demo_rate_death")

    ##################################################################
    # The methods that process_parameters calls to set parameters and
    # scale-up functions

    def find_ageing_rates(self):

        # Calculate ageing rates as the reciprocal of the width of the age bracket
        for agegroup in self.agegroups:
            if "up" not in agegroup:
                self.set_parameter("ageing_rate" + agegroup, 1. / (
                    float(agegroup[agegroup.find("to") + 2:]) -
                    float(agegroup[agegroup.find("age") + 3: agegroup.find("to")])))

    def find_natural_history_params(self):

        # If extrapulmonary case-fatality not stated, use smear-negative case-fatality
        if "tb_proportion_casefatality_untreated_extrapul" not in self.params:
            self.set_parameter(
                "tb_proportion_casefatality_untreated_extrapul",
                self.params["tb_proportion_casefatality_untreated_smearneg"])

        # Overall progression and stabilisation rates
        self.set_parameter("tb_rate_early_progression",
                           self.params["tb_proportion_early_progression"]
                           / self.params["tb_timeperiod_early_latent"])
        self.set_parameter("tb_rate_stabilise",
                           (1. - self.params["tb_proportion_early_progression"])
                           / self.params["tb_timeperiod_early_latent"])

        # Adjust overall rates by organ status
        for organ in self.organ_status:
            self.set_parameter(
                "tb_rate_early_progression" + organ,
                self.params["tb_proportion_early_progression"]
                / self.params["tb_timeperiod_early_latent"]
                * self.params["epi_proportion_cases" + organ])
            self.set_parameter(
                "tb_rate_late_progression" + organ,
                self.params["tb_rate_late_progression"]
                * self.params["epi_proportion_cases" + organ])
            self.set_parameter(
                "tb_rate_death" + organ,
                self.params["tb_proportion_casefatality_untreated" + organ]
                / self.params["tb_timeperiod_activeuntreated"])
            self.set_parameter(
                "tb_rate_recover" + organ,
                (1 - self.params["tb_proportion_casefatality_untreated" + organ])
                / self.params["tb_timeperiod_activeuntreated"])

    def find_amplification_scaleup(self):

        # Set the amplification scale-up function
        self.set_scaleup_fn("tb_proportion_amplification",
                            scale_up_function([self.start_time,
                                               self.data['miscellaneous']['timepoint_introduce_mdr'] - 5.,
                                               self.data['miscellaneous']['timepoint_introduce_mdr'] + 5.],
                                              [0.,
                                               0.,
                                               self.params['tb_proportion_amplification']],
                                              self.data['attributes'][u'fitting_method'], self.data['attributes']['organ_smoothness'],
                                              0., 1.))

    def find_organ_scaleup(self):

        """
        Work out the scale-up function for progression to disease by organ status.
        This should only need to be done for smear-positive and smear-negative at most,
        as the remaining proportion will be calculated as one minus these proportions.
        """

        # Use lists with the nan-s removed
        year = []
        smearpos = []
        smearneg = []
        for i in range(len(self.data['notifications'][u'prop_new_sp'])):
            if not numpy.isnan(self.data['notifications'][u'prop_new_sp'][i]):
                year += [self.data['notifications'][u'year'][i]]
                smearpos += [self.data['notifications'][u'prop_new_sp'][i]]
                smearneg += [self.data['notifications'][u'prop_new_sn'][i]]

        # Currently setting a scale-up function with an extra point added at the
        # model start time that is equal to the first (non-nan) proportion in the list.
        # (Note that eval(organ_data) turns the string into the variable name.)
        for organ_data in ['smearpos', 'smearneg']:
            self.set_scaleup_fn('tb_proportion_' + organ_data,
                                scale_up_function([self.start_time] + year,
                                                  [eval(organ_data)[0]] + eval(organ_data),
                                                  self.data['attributes'][u'fitting_method'], self.data['attributes']['organ_smoothness'],
                                                  0., 1.))

    def find_programs_to_run(self):

        # Extract relevant programs to run from the data object and truncate as needed
        self.programs = {}
        for program in self.data['programs'].keys():

            # Extract the ones that aren't for cost curve development
            if u'cost' not in program:
                self.programs[program] = {}
                for i in self.data['programs'][program]:
                    if type(i) == int:
                        self.programs[program][i] = self.data['programs'][program][i]

            if 'prop' in program:

                # Convert percentages to proportions
                for i in self.programs[program]:
                    self.programs[program][i] \
                        = self.programs[program][i] / 1E2

    def find_nontreatment_rates_params(self):

        # For the six non-treatment programs
        for program in self.data['programs'].keys():

            if 'prop' in program or 'timeperiod' in program:

                scenario_string = 'scenario_' + str(self.scenario)
                if scenario_string in self.data['programs'][program]:
                    scenario_values = [self.data['attributes'][u'scenario_full_time'],
                                       self.data['programs'][program][scenario_string]]
                    if 'prop' in program:
                        scenario_values[1] /= 1E2
                    scenario_start = self.data['attributes'][u'scenario_start_time']
                else:
                    scenario_values = None
                    scenario_start = None

                # Find scale-up functions

                # Allow a different smoothness parameter for case detection,
                # because abrupt changes in this time-variant parameter lead to major model problems
                # (e.g. negative compartment values)
                if 'detect' in program:
                    self.set_scaleup_fn(program,
                                        scale_up_function(self.programs[program].keys(),
                                                          self.programs[program].values(),
                                                          self.data['attributes'][u'fitting_method'],
                                                          self.data['attributes']['detection_smoothness'],
                                                          0., 1.,
                                                          intervention_end=scenario_values,
                                                          intervention_start_date=scenario_start))
                elif 'timeperiod' in program:
                    self.set_scaleup_fn(program,
                                        scale_up_function(self.programs[program].keys(),
                                                          self.programs[program].values(),
                                                          self.data['attributes'][u'fitting_method'],
                                                          self.data['attributes']['detection_smoothness'],
                                                          0., 1.,
                                                          intervention_end = scenario_values,
                                                          intervention_start_date = scenario_start))
                else:
                    self.set_scaleup_fn(program,
                                    scale_up_function(self.programs[program].keys(),
                                                      self.programs[program].values(),
                                                      self.data['attributes'][u'fitting_method'],
                                                      self.data['attributes']['program_smoothness'],
                                                      0., 1.,
                                                      intervention_end=scenario_values,
                                                      intervention_start_date=scenario_start))

    def find_treatment_rates_scaleups(self):

        """
        Calculate treatment rates from the treatment outcome proportions
        Has to be done for each strain separately and also for those on inappropriate treatment
        """

        # If the model isn't stratified by strain, use DS-TB time-periods for the single strain
        if self.strains == [""]:
            self.params["tb_timeperiod_infect_ontreatment"] \
                = self.params["tb_timeperiod_infect_ontreatment_ds"]
            self.params["tb_timeperiod_treatment"] \
                = self.params["tb_timeperiod_treatment_ds"]

        for strain in self.strains + ["_inappropriate"]:

            # Find the non-infectious periods
            self.set_parameter(
                "tb_timeperiod_noninfect_ontreatment" + strain,
                self.params["tb_timeperiod_treatment" + strain]
                - self.params["tb_timeperiod_infect_ontreatment" + strain])

            # Populate treatment outcomes from previously calculated functions
            self.set_scaleup_fn(
                "program_proportion_success" + strain,
                scale_up_function(self.programs[u'program_prop_treatment_success' + strain].keys(),
                                  self.programs[u'program_prop_treatment_success' + strain].values(),
                                  self.data['attributes'][u'fitting_method'], self.data['attributes']['program_smoothness'],
                                  0., 1.))
            self.set_scaleup_fn(
                "program_proportion_death" + strain,
                scale_up_function(self.programs[u'program_prop_treatment_death' + strain].keys(),
                                  self.programs[u'program_prop_treatment_death' + strain].values(),
                                  self.data['attributes'][u'fitting_method'], self.data['attributes']['program_smoothness'],
                                  0., 1.))

    ##################################################################
    # Methods that calculate variables to be used in calculating flows
    # Note: all scaleup_fns are calculated and put into self.vars before
    # calculate_vars
    # I think we have to put any calculations that are dependent upon vars
    # into this section

    def calculate_vars(self):

        """
        The master method that calls all the other methods for the calculations of
        variable rates
        """

        self.vars["population"] = sum(self.compartments.values())

        self.calculate_birth_rates_vars()

        self.calculate_force_infection_vars()

        self.calculate_progression_vars()

        self.calculate_detect_missed_vars()

        self.calculate_proportionate_detection_vars()

        if self.is_lowquality: self.calculate_lowquality_detection_vars()

        self.calculate_await_treatment_var()

        self.calculate_treatment_rates_vars()

    def calculate_birth_rates_vars(self):

        # Calculate vaccinated and unvaccinated birth rates
        # Rates are dependent upon two variables, i.e. the total population
        # and the scale-up function of BCG vaccination coverage
        self.vars["births_unvac"] = \
            (1. - self.vars["program_prop_vaccination"]) \
            * self.params["demo_rate_birth"] \
            * self.vars["population"] \
            / len(self.comorbidities)
        self.vars["births_vac"] = \
            self.vars["program_prop_vaccination"] \
            * self.params["demo_rate_birth"] \
            * self.vars["population"] \
            / len(self.comorbidities)

    def calculate_force_infection_vars(self):

        # Calculate force of infection by strain,
        # incorporating partial immunity and infectiousness
        for strain in self.strains:

            # Initialise infectious population to zero
            self.vars["infectious_population" + strain] = 0.
            for organ in self.organ_status:
                for label in self.labels:

                    # If we haven't reached the part of the model divided by organ status
                    if organ not in label and organ != "":
                        continue

                    # If we haven't reached the part of the model divided by strain
                    if strain not in label and strain != "":
                        continue

                    # If the compartment is non-infectious
                    if not label_intersects_tags(label, self.infectious_tags):
                        continue
                    self.vars["infectious_population" + strain] += \
                        self.params["tb_multiplier_force" + organ] \
                        * self.compartments[label]

            # Calculate non-immune force of infection
            self.vars["rate_force" + strain] = \
                self.params["tb_n_contact"] \
                * self.vars["infectious_population" + strain] \
                / self.vars["population"]

            # Adjust for immunity
            self.vars["rate_force_vacc" + strain] = \
                self.params["tb_multiplier_bcg_protection"] \
                * self.vars["rate_force" + strain]
            self.vars["rate_force_latent" + strain] = \
                self.params["tb_multiplier_latency_protection"] \
                * self.vars["rate_force" + strain]

    def calculate_progression_vars(self):

        """
        Calculate vars for the remainder of progressions.
        Note that the vars for the smear-positive and smear-negative proportions
        have already been calculated. However, all progressions have to go somewhere,
        so need to calculate the remaining proportions.
        """

        # If unstratified (self.organ_status should have length 0, but will work for length 1)
        if len(self.organ_status) < 2:
            self.vars['tb_proportion'] = 1.

        # Stratified into smear-positive and smear-negative
        elif len(self.organ_status) == 2:
            self.vars['tb_proportion_smearneg'] = \
                1. - self.vars['tb_proportion_smearpos']

        # Fully stratified into smear-positive, smear-negative and extra-pulmonary
        elif len(self.organ_status) > 2:
            self.vars['tb_proportion_extrapul'] = \
                1. - self.vars['tb_proportion_smearpos'] - self.vars['tb_proportion_smearneg']

        # Determine variable progression rates
        for organ in self.organ_status:
            for timing in ['_early', '_late']:
                self.vars['tb_rate' + timing + '_progression' + organ] \
                    = self.vars['tb_proportion' + organ] * self.params['tb_rate' + timing + '_progression']

    def calculate_detect_missed_vars(self):

        """"
        Calculate rates of detection and failure of detection
        from the programmatic report of the case detection "rate"
        (which is actually a proportion and referred to as program_prop_detect here)

        Derived from original formulas of by solving the simultaneous equations:
          algorithm sensitivity = detection rate / (detection rate + missed rate)
          - and -
          detection proportion = detection rate / (detection rate + missed rate + spont recover rate + death rate)
        """

        # Detection
        # Note that all organ types are assumed to have the same untreated active
        # sojourn time, so any organ status can be arbitrarily selected (here the first, or smear-positive)

        # If no division by zero
        if self.vars["program_prop_algorithm_sensitivity"] > 0.:

            # Detections
            self.vars["program_rate_detect"] = \
                self.vars["program_prop_detect"] \
                * (self.params["tb_rate_recover" + self.organ_status[0]] + self.params[
                    "tb_rate_death" + self.organ_status[0]]) \
                / (1. - self.vars["program_prop_detect"] \
                   * (1. + (1. - self.vars["program_prop_algorithm_sensitivity"]) \
                      / self.vars["program_prop_algorithm_sensitivity"]))

            # Missed
            self.vars["program_rate_missed"] = \
                self.vars["program_rate_detect"] \
                * (1. - self.vars["program_prop_algorithm_sensitivity"]) \
                / self.vars["program_prop_algorithm_sensitivity"]

        # Otherwise just assign detection and missed rates to zero
        else:
            self.vars['program_rate_detect'] = 0.
            self.vars['program_rate_missed'] = 0.

        # Repeat for each strain
        for strain in self.strains:
            for programmatic_rate in ["_detect", "_missed"]:
                self.vars["program_rate" + programmatic_rate + strain] = \
                    self.vars["program_rate" + programmatic_rate]

    def calculate_await_treatment_var(self):

        # Just simply take the reciprocal of the time to start on treatment
        self.vars['program_var_rate_start_treatment'] = 1. / self.vars['program_timeperiod_await_treatment']

    def calculate_lowquality_detection_vars(self):

        """
        Calculate rate of entry to low-quality care,
        form the proportion of treatment administered in low-quality sector

        Note that this now means that the case detection proportion only
        applies to those with access to care and so that proportion of all
        cases isn't actually detected
        """

        self.vars["program_rate_enterlowquality"] = \
            self.vars["program_rate_detect"] \
            * self.vars["program_prop_lowquality"] \
            / (1. - self.vars["program_prop_lowquality"])

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

            # DS-TB
            self.vars["program_rate_detect_ds_asds"] = \
                self.vars["program_rate_detect"]
            self.vars["program_rate_detect_ds_asmdr"] = 0.
            self.vars["program_rate_detect_ds_asxdr"] = 0.

            # MDR-TB
            self.vars["program_rate_detect_mdr_asds"] = \
                (1. - self.vars["program_prop_firstline_dst"]) \
                * self.vars["program_rate_detect"]
            self.vars["program_rate_detect_mdr_asmdr"] = \
                self.vars["program_prop_firstline_dst"] \
                * self.vars["program_rate_detect"]
            self.vars["program_rate_detect_mdr_asxdr"] = 0.

            # XDR-TB
            self.vars["program_rate_detect_xdr_asds"] = \
                (1. - self.vars["program_prop_firstline_dst"]) \
                 * self.vars["program_rate_detect"]
            self.vars["program_rate_detect_xdr_asmdr"] = \
                self.vars["program_prop_firstline_dst"] \
                * (1. - self.vars["program_prop_secondline_dst"])\
                * self.vars["program_rate_detect"]
            self.vars["program_rate_detect_xdr_asxdr"] = \
                self.vars["program_prop_firstline_dst"] \
                * self.vars["program_prop_secondline_dst"] \
                * self.vars["program_rate_detect"]

        # Without misassignment:
        else:
            for strain in self.strains:
                self.vars["program_rate_detect" + strain + "_as"+strain[1:]] = \
                    self.vars["program_rate_detect"]

    def calculate_treatment_rates_vars(self):

        for strain in self.strains + ["_inappropriate"]:

            self.vars['program_proportion_default' + strain] \
                = 1. \
                  - self.vars['program_proportion_success' + strain] \
                  - self.vars['program_proportion_death' + strain]

            # Find the proportion of deaths/defaults during the infectious and non-infectious stages
            for outcome in self.non_success_outcomes:
                early_proportion, late_proportion = self.find_outcome_proportions_by_period(
                    self.vars["program_proportion" + outcome + strain],
                    self.params["tb_timeperiod_infect_ontreatment" + strain],
                    self.params["tb_timeperiod_treatment" + strain])
                self.vars["program_proportion" + outcome + "_infect" + strain] = early_proportion
                self.vars["program_proportion" + outcome + "_noninfect" + strain] = late_proportion

            # Find the success proportions
            for treatment_stage in self.treatment_stages:
                self.vars["program_proportion_success" + treatment_stage + strain] = \
                    1. - self.vars["program_proportion_default" + treatment_stage + strain] \
                    - self.vars["program_proportion_death" + treatment_stage + strain]

                # Find the corresponding rates from the proportions
                for outcome in self.outcomes:
                    self.vars["program_rate" + outcome + treatment_stage + strain] = \
                        1. / self.params["tb_timeperiod" + treatment_stage + "_ontreatment" + strain] \
                        * self.vars["program_proportion" + outcome + treatment_stage + strain]
                if self.is_amplification:
                    self.vars["program_rate_default" + treatment_stage + "_amplify" + strain] = \
                        self.vars["program_rate_default" + treatment_stage + strain] \
                        * self.vars["tb_proportion_amplification"]
                    self.vars["program_rate_default" + treatment_stage + "_noamplify" + strain] = \
                        self.vars["program_rate_default" + treatment_stage + strain] \
                        * (1. - self.vars["tb_proportion_amplification"])

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

    def set_birth_flows(self):

        # Set birth flows (currently evenly distributed between comorbidities)
        for comorbidity in self.comorbidities:
            self.set_var_entry_rate_flow(
                "susceptible_fully" + comorbidity + self.agegroups[0], "births_unvac")
            self.set_var_entry_rate_flow(
                "susceptible_vac" + comorbidity + self.agegroups[0], "births_vac")

    def set_ageing_flows(self):

        # Set simple ageing flows for any number of strata
        for label in self.labels:
            for number_agegroup in range(len(self.agegroups)):
                if self.agegroups[number_agegroup] in label and number_agegroup < len(self.agegroups) - 1:
                    self.set_fixed_transfer_rate_flow(
                        label, label[0: label.find("_age")] + self.agegroups[number_agegroup + 1],
                               "ageing_rate" + self.agegroups[number_agegroup])

    def set_infection_flows(self):

        # Set force of infection flows
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:

                    # Fully susceptible
                    self.set_var_transfer_rate_flow(
                        "susceptible_fully" + comorbidity + agegroup,
                        "latent_early" + strain + comorbidity + agegroup,
                        "rate_force" + strain)

                    # Partially immune
                    self.set_var_transfer_rate_flow(
                        "susceptible_vac" + comorbidity + agegroup,
                        "latent_early" + strain + comorbidity + agegroup,
                        "rate_force_vacc" + strain)
                    self.set_var_transfer_rate_flow(
                        "susceptible_treated" + comorbidity + agegroup,
                        "latent_early" + strain + comorbidity + agegroup,
                        "rate_force_vacc" + strain)
                    self.set_var_transfer_rate_flow(
                        "latent_late" + strain + comorbidity + agegroup,
                        "latent_early" + strain + comorbidity + agegroup,
                        "rate_force_latent" + strain)

    def set_progression_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:

                        # Stabilisation
                        self.set_fixed_transfer_rate_flow(
                            "latent_early" + strain + comorbidity + agegroup,
                            "latent_late" + strain + comorbidity + agegroup,
                            "tb_rate_stabilise")

                        for organ in self.organ_status:

                            # Early progression
                            self.set_var_transfer_rate_flow(
                                "latent_early" + strain + comorbidity + agegroup,
                                "active" + organ + strain + comorbidity + agegroup,
                                "tb_rate_early_progression" + organ)

                            # Late progression
                            self.set_var_transfer_rate_flow(
                                "latent_late" + strain + comorbidity + agegroup,
                                "active" + organ + strain + comorbidity + agegroup,
                                "tb_rate_late_progression" + organ)

    def set_natural_history_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Recovery, base compartments
                            self.set_fixed_transfer_rate_flow(
                                "active" + organ + strain + comorbidity + agegroup,
                                "latent_late" + strain + comorbidity + agegroup,
                                "tb_rate_recover" + organ)
                            self.set_fixed_transfer_rate_flow(
                                "missed" + organ + strain + comorbidity + agegroup,
                                "latent_late" + strain + comorbidity + agegroup,
                                "tb_rate_recover" + organ)

                            # Death, base compartments
                            self.set_fixed_infection_death_rate_flow(
                                "active" + organ + strain + comorbidity + agegroup,
                                "tb_rate_death" + organ)
                            self.set_fixed_infection_death_rate_flow(
                                "missed" + organ + strain + comorbidity + agegroup,
                                "tb_rate_death" + organ)

                            # Extra low-quality compartments
                            if self.is_lowquality:
                                self.set_fixed_transfer_rate_flow(
                                    "lowquality" + organ + strain + comorbidity + agegroup,
                                    "latent_late" + strain + comorbidity + agegroup,
                                    "tb_rate_recover" + organ)
                                self.set_fixed_infection_death_rate_flow(
                                    "lowquality" + organ + strain + comorbidity + agegroup,
                                    "tb_rate_death" + organ)

                            # Detected, with misassignment
                            if self.is_misassignment:
                                for assigned_strain in self.strains:
                                    self.set_fixed_infection_death_rate_flow(
                                        "detect" +
                                        organ + strain + "_as" + assigned_strain[1:] + comorbidity + agegroup,
                                        "tb_rate_death" + organ)
                                    self.set_fixed_transfer_rate_flow(
                                        "detect" +
                                        organ + strain + "_as" + assigned_strain[1:] + comorbidity + agegroup,
                                        "latent_late" + strain + comorbidity + agegroup,
                                        "tb_rate_recover" + organ)

                            # Detected, without misassignment
                            else:
                                self.set_fixed_transfer_rate_flow(
                                    "detect" + organ + strain + comorbidity + agegroup,
                                    "latent_late" + strain + comorbidity + agegroup,
                                    "tb_rate_recover" + organ)
                                self.set_fixed_infection_death_rate_flow(
                                    "detect" + organ + strain + comorbidity + agegroup,
                                    "tb_rate_death" + organ)

    def set_fixed_programmatic_flows(self):

        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:

                        # Re-start presenting after a missed diagnosis
                        self.set_fixed_transfer_rate_flow(
                            "missed" + organ + strain + comorbidity + agegroup,
                            "active" + organ + strain + comorbidity + agegroup,
                            "program_rate_restart_presenting")

                        # Give up on the hopeless low-quality health system
                        if self.is_lowquality:
                            self.set_fixed_transfer_rate_flow(
                                "lowquality" + organ + strain + comorbidity + agegroup,
                                "active" + organ + strain + comorbidity + agegroup,
                                "program_rate_leavelowquality")

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
                                as_assigned_strain = "_as" + self.strains[assigned_strain_number][1:]
                                # If the strain is equally or more resistant than its assignment
                                if actual_strain_number >= assigned_strain_number:
                                    self.set_var_transfer_rate_flow(
                                        "active" + organ + strain + comorbidity + agegroup,
                                        "detect" + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        "program_rate_detect" + strain + as_assigned_strain)

                        # Without misassignment - everyone is correctly identified by strain
                        else:
                            self.set_var_transfer_rate_flow(
                                "active" + organ + strain + comorbidity + agegroup,
                                "detect" + organ + strain + comorbidity + agegroup,
                                "program_rate_detect")

    def set_variable_programmatic_flows(self):

        # Set rate of missed diagnoses and entry to low-quality health care
        for agegroup in self.agegroups:
            for comorbidity in self.comorbidities:
                for strain in self.strains:
                    for organ in self.organ_status:
                        self.set_var_transfer_rate_flow(
                            "active" + organ + strain + comorbidity + agegroup,
                            "missed" + organ + strain + comorbidity + agegroup,
                            "program_rate_missed")
                        # Detection, with and without misassignment

                        ### Warning, the following two rate setting functions should actually be
                        # variable (i.e. set_var_transfer_rate_flow, rather than
                        # set_fixed_transfer_rate_flow). However, when you change them to variable,
                        # scipy integration fails. No idea why this should be.
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                # Following line only for models incorporating mis-assignment
                                self.set_fixed_transfer_rate_flow(
                                    "detect" +
                                    organ + strain + "_as" + assigned_strain[1:] + comorbidity + agegroup,
                                    "treatment_infect" +
                                    organ + strain + "_as" + assigned_strain[1:] + comorbidity + agegroup,
                                    "program_rate_start_treatment")
                        else:
                            # Following line is the currently running line (without mis-assignment)
                            self.set_fixed_transfer_rate_flow(
                                "detect" + organ + strain + comorbidity + agegroup,
                                "treatment_infect" + organ + strain + comorbidity + agegroup,
                                "program_rate_start_treatment")

                        if self.is_lowquality:
                            self.set_var_transfer_rate_flow(
                                "active" + organ + strain + comorbidity + agegroup,
                                "lowquality" + organ + strain + comorbidity + agegroup,
                                "program_rate_enterlowquality")

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
                            assignment_strains = [""]
                        for assigned_strain_number in assignment_strains:
                            if self.is_misassignment:
                                as_assigned_strain = "_as" + self.strains[assigned_strain_number][1:]

                                # Which treatment parameters to use - for the strain or for inappropriate treatment
                                if actual_strain_number > assigned_strain_number:
                                    strain_or_inappropriate = "_inappropriate"
                                else:
                                    strain_or_inappropriate = self.strains[assigned_strain_number]

                            else:
                                as_assigned_strain = ""
                                strain_or_inappropriate = self.strains[actual_strain_number]

                            # Success at either treatment stage
                            self.set_var_transfer_rate_flow(
                                "treatment_infect" + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                "treatment_noninfect" + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                "program_rate_success_infect" + strain_or_inappropriate)
                            self.set_var_transfer_rate_flow(
                                "treatment_noninfect" + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                "susceptible_treated" + comorbidity + agegroup,
                                "program_rate_success_noninfect" + strain_or_inappropriate)

                            # Rates of death on treatment
                            for treatment_stage in self.treatment_stages:
                                self.set_var_infection_death_rate_flow(
                                    "treatment" +
                                    treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                    "program_rate_death" + treatment_stage + strain_or_inappropriate)

                            # Default
                            for treatment_stage in self.treatment_stages:

                                # If it's either the most resistant strain available or amplification is not active:
                                if actual_strain_number == len(self.strains) - 1 or not self.is_amplification:
                                    self.set_var_transfer_rate_flow(
                                        "treatment" +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        "active" + organ + strain + comorbidity + agegroup,
                                        "program_rate_default" + treatment_stage + strain_or_inappropriate)

                                # Otherwise
                                else:
                                    amplify_to_strain = self.strains[actual_strain_number + 1]
                                    self.set_var_transfer_rate_flow(
                                        "treatment" +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        "active" + organ + strain + comorbidity + agegroup,
                                        "program_rate_default" + treatment_stage + "_noamplify" + strain_or_inappropriate)
                                    self.set_var_transfer_rate_flow(
                                        "treatment" +
                                        treatment_stage + organ + strain + as_assigned_strain + comorbidity + agegroup,
                                        "active" + organ + amplify_to_strain + comorbidity + agegroup,
                                        "program_rate_default" + treatment_stage + "_amplify" + strain_or_inappropriate)

    ##################################################################
    # Methods that calculate the output vars and diagnostic properties

    def calculate_output_vars(self):

        """
        Method similarly structured to calculate_output_vars,
        just replicated by strains
        """

        # Initialise dictionaries
        rate_incidence = {}
        rate_mortality = {}
        rate_notifications = {}

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
            self.vars["incidence" + strain] \
                = rate_incidence[strain] / self.vars["population"] * 1E5

            # Notifications
            for from_label, to_label, rate in self.var_transfer_rate_flows:
                if 'active' in from_label and 'detect' in to_label and strain in from_label:
                    rate_notifications[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            self.vars["notifications" + strain] \
                = rate_notifications[strain]

            # Mortality
            for from_label, rate in self.fixed_infection_death_rate_flows:
                if strain in from_label:
                    rate_mortality[strain] \
                        += self.compartments[from_label] * rate
            for from_label, rate in self.var_infection_death_rate_flows:
                if strain in from_label:
                    rate_mortality[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            self.vars["mortality" + strain] \
                = rate_mortality[strain] / self.vars["population"] * 1E5 \
                * self.data['miscellaneous'][u'program_proportion_death_reporting']

        # Prevalence
        for strain in self.strains:
            self.vars["prevalence" + strain] = 0.
            for label in self.labels:
                if "susceptible" not in label and \
                                "latent" not in label and strain in label:
                    self.vars["prevalence" + strain] \
                        += (self.compartments[label]
                        / self.vars["population"] * 1E5)

        # Summing MDR and XDR to get the total of all MDRs
        if len(self.strains) > 1:
            rate_incidence["all_mdr_strains"] = 0.
            if len(self.strains) > 1:
                for actual_strain_number in range(len(self.strains)):
                    strain = self.strains[actual_strain_number]
                    if actual_strain_number > 0:
                        rate_incidence["all_mdr_strains"] \
                            += rate_incidence[strain]
            self.vars["all_mdr_strains"] \
                = rate_incidence["all_mdr_strains"] / self.vars["population"] * 1E5
            # Convert to percentage
            self.vars["proportion_mdr"] \
                = self.vars["all_mdr_strains"] / self.vars["incidence"] * 1E2

    def calculate_additional_diagnostics(self):

        """
        Diagnostics calculates data structures after the full simulation has been run
        with calls to output calculations at each time step
        """

        self.broad_compartment_soln, broad_compartment_denominator \
            = self.sum_over_compartments(self.broad_compartment_types)
        self.broad_fraction_soln \
            = self.get_fraction_soln(
            self.broad_compartment_types,
            self.broad_compartment_soln,
            broad_compartment_denominator)

        self.compartment_type_soln, compartment_type_denominator \
            = self.sum_over_compartments(self.compartment_types)
        self.compartment_type_fraction_soln \
            = self.get_fraction_soln(
            self.compartment_types,
            self.compartment_type_soln,
            compartment_type_denominator)

        self.broad_compartment_type_bystrain_soln, broad_compartment_type_bystrain_denominator, \
        self.broad_compartment_types_bystrain \
            = self.sum_over_compartments_bycategory(self.broad_compartment_types, "strain")
        self.broad_compartment_type_bystrain_fraction_soln \
            = self.get_fraction_soln(
            self.broad_compartment_types_bystrain,
            self.broad_compartment_type_bystrain_soln,
            broad_compartment_type_bystrain_denominator)

        self.broad_compartment_type_byorgan_soln, broad_compartment_type_byorgan_denominator, \
        self.broad_compartment_types_byorgan \
            = self.sum_over_compartments_bycategory(self.broad_compartment_types, "organ")
        self.broad_compartment_type_byorgan_fraction_soln \
            = self.get_fraction_soln(
            self.broad_compartment_types_byorgan,
            self.broad_compartment_type_byorgan_soln,
            broad_compartment_type_byorgan_denominator)

        self.compartment_type_bystrain_soln, compartment_type_bystrain_denominator, \
        self.compartment_types_bystrain \
            = self.sum_over_compartments_bycategory(self.compartment_types, "strain")
        self.compartment_type_bystrain_fraction_soln \
            = self.get_fraction_soln(
            self.compartment_types_bystrain,
            self.compartment_type_bystrain_soln,
            compartment_type_bystrain_denominator)

        self.calculate_subgroup_diagnostics()

    def calculate_subgroup_diagnostics(self):

        """
        Calculate fractions and populations within subgroups of the full population
        """
        self.groups = {
            "ever_infected": ["susceptible_treated", "latent", "active", "missed", "lowquality", "detect", "treatment"],
            "infected": ["latent", "active", "missed", "lowquality", "detect", "treatment"],
            "active": ["active", "missed", "detect", "lowquality", "treatment"],
            "infectious": ["active", "missed", "lowquality", "detect", "treatment_infect"],
            "identified": ["detect", "treatment"],
            "treatment": ["treatment_infect", "treatment_noninfect"]}
        for key in self.groups:
            compartment_soln, compartment_denominator \
                = self.sum_over_compartments(self.groups[key])
            setattr(self, key + "_compartment_soln", compartment_soln)
            setattr(self, key + "_compartment_denominator", compartment_denominator)
            setattr(self, key + "_fraction_soln",
                    self.get_fraction_soln(
                        self.groups[key],
                        compartment_soln,
                        compartment_denominator))

    def get_fraction_soln(self, numerator_labels, numerators, denominator):

        """
        General method for calculating the proportion of a subgroup of the population
        in each compartment type
        Args:
            numerator_labels: Labels of numerator compartments
            numerators: Lists of values of each numerator
            denominator: List of values for the denominator

        Returns:
            Fractions of the denominator in each numerator
        """

        fraction = {}

        # Just to avoid warnings, replace any zeros in the denominators with small values
        # (numerators will still be zero, so all fractions should be zero)
        for i in range(len(denominator)):
            if denominator[i] == 0.:
                denominator[i] = 1E-3

        for label in numerator_labels:
            fraction[label] = [
                v / t
                for v, t
                in zip(
                    numerators[label],
                    denominator)]
        return fraction

    def sum_over_compartments(self, compartment_types):

        """
        General method to sum sets of compartments
        Args:
            compartment_types: List of the compartments to be summed over

        Returns:
            summed_soln: Dictionary of lists for the sums of each compartment
            summed_denominator: List of the denominator values
        """
        summed_soln = {}
        summed_denominator \
            = [0] * len(random.sample(self.compartment_soln.items(), 1)[0][1])
        for compartment_type in compartment_types:
            summed_soln[compartment_type] \
                = [0] * len(random.sample(self.compartment_soln.items(), 1)[0][1])
            for label in self.labels:
                if compartment_type in label:
                    summed_soln[compartment_type] = [
                        a + b
                        for a, b
                        in zip(
                            summed_soln[compartment_type],
                            self.compartment_soln[label])]
                    summed_denominator += self.compartment_soln[label]
        return summed_soln, summed_denominator

    def sum_over_compartments_bycategory(self, compartment_types, categories):

        # Waiting for Bosco's input, so won't fully comment yet
        summed_soln = {}
        # HELP BOSCO
        # The following line of code works, but I'm sure this isn't the best approach:
        summed_denominator \
            = [0] * len(random.sample(self.compartment_soln.items(), 1)[0][1])
        compartment_types_bycategory = []
        # HELP BOSCO
        # I think there is probably a more elegant way to do the following, but perhaps not.
        # Also, it could possibly be better generalised. That is, rather than insisting that
        # strain applies to all compartments except for the susceptible, it might be possible
        # to say that strain applies to all compartments except for those that have any
        # strain in their label.
        if categories == "strain":
            working_categories = self.strains
        elif categories == "organ":
            working_categories = self.organ_status
        for compartment_type in compartment_types:
            if (categories == "strain" and "susceptible" in compartment_type) \
                    or (categories == "organ" and
                                ("susceptible" in compartment_type or "latent" in compartment_type)):
                summed_soln[compartment_type] \
                    = [0] * len(random.sample(self.compartment_soln.items(), 1)[0][1])
                for label in self.labels:
                    if compartment_type in label:
                        summed_soln[compartment_type] = [
                            a + b
                            for a, b
                            in zip(
                                summed_soln[compartment_type],
                                self.compartment_soln[label])]
                        summed_denominator += self.compartment_soln[label]
                    if compartment_type in label \
                            and compartment_type not in compartment_types_bycategory:
                        compartment_types_bycategory.append(compartment_type)
            else:
                for working_category in working_categories:
                    compartment_types_bycategory.append(compartment_type + working_category)
                    summed_soln[compartment_type + working_category] \
                        = [0] * len(random.sample(self.compartment_soln.items(), 1)[0][1])
                    for label in self.labels:
                        if compartment_type in label and working_category in label:
                            summed_soln[compartment_type + working_category] = [
                                a + b
                                for a, b
                                in zip(
                                    summed_soln[compartment_type + working_category],
                                    self.compartment_soln[label])]
                            summed_denominator += self.compartment_soln[label]

        return summed_soln, summed_denominator, compartment_types_bycategory

    ##################################################################
    # Methods to call base integration function depending on the type of
    # integration required

    def integrate(self):

        if self.data['attributes'][u'integration'] == u'explicit':
            self.integrate_explicit()
        elif self.data['attributes'][u'integration'] == u'scipy':
            self.integrate_scipy()


