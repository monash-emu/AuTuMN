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
                 age_breakpoints=[],
                 n_organ=0,
                 n_strain=0,
                 n_comorbidity=0,
                 is_lowquality=False,
                 is_amplification=False,
                 is_misassignment=False,
                 country_data=None):

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
        self.age_breakpoints = age_breakpoints
        self.n_organ = n_organ
        self.n_strain = n_strain
        self.n_comorbidity = n_comorbidity

        self.is_lowquality = is_lowquality
        self.is_amplification = is_amplification
        self.is_misassignment = is_misassignment

        self.country_data = country_data

        if self.is_misassignment:
            assert self.is_amplification, "Misassignment requested without amplification"

        # Initialise model compartmental structure and set un-processed parameters
        self.define_model_structure()
        self.initialise_compartments()
        self.set_parameters()

        # Treatment outcomes that will be universal to all models
        # Global TB outcomes of "completion" and "cure" can be considered "_success",
        # "death" is "_death" (of course), "failure" and "default" are considered "_default"
        # and "transfer out" is removed from denominator calculations.
        self.outcomes = ["_success", "_death", "_default"]
        self.non_success_outcomes = self.outcomes[1: 3]

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

        # Some defaults initial compartment defaults if None given as the compartment dictionary
        if compartment_dict is None:
            compartment_dict = {
                "susceptible_fully":
                    2e7,
                "active":
                    3.}

        # Populate input_compartments
        # Initialise to DS-TB or no strain if single-strain model
        default_start_strain = "_ds"
        if self.strains == [""]: default_start_strain = ""

        # The equal splits may need to be adjusted, but the important thing is not to
        # initialise multiple strains too early, so that MDR-TB doesn't overtake the model
        for compartment in self.compartment_types:
            if compartment in compartment_dict:
                for agegroup in self.agegroups:
                    for comorbidity in self.comorbidities:
                        if "susceptible_fully" in compartment:
                            # Split equally by comorbidities and age-groups
                            self.set_compartment(compartment + comorbidity + agegroup,
                                                 compartment_dict[compartment]
                                                 / len(self.comorbidities)
                                                 / len(self.agegroups))
                        elif "latent" in compartment:
                            # Assign all to DS-TB, split equally by comorbidities and age-groups
                            self.set_compartment(compartment + default_start_strain + comorbidity + agegroup,
                                                 compartment_dict[compartment]
                                                 / len(self.comorbidities)
                                                 / len(self.agegroups))
                        else:
                            for organ in self.organ_status:
                                self.set_compartment(compartment +
                                                     organ + default_start_strain + comorbidity + agegroup,
                                                     compartment_dict[compartment]
                                                     / len(self.organ_status)  # Split equally by organ statuses,
                                                     / len(self.comorbidities)  # split equally by comorbidities
                                                     / len(self.agegroups))  # and split equally by age-groups

    def set_parameters(self, paramater_dict=None):

        """
        Sets some default model parameters for testing when required
        These will be set externally through the spreadsheets in all real world applications

        Args:
            paramater_dict: a key-value dictionary where typically key
                              is a string for a param name and value is a float
        """

        # Set defaults for testing
        if paramater_dict is None:
            paramater_dict = {
                "demo_rate_birth":
                    24. / 1e3,
                "demo_rate_death":
                    1. / 69.,
                "epi_proportion_cases_smearpos":
                    (92991. + 6277.) / 243379.,  # Total bacteriologically confirmed
                "epi_proportion_cases_smearneg":
                    139950. / 243379.,  # Clinically diagnosed
                "epi_proportion_cases_extrapul":
                    4161. / 243379.,  # Bacteriologically confirmed
                "epi_proportion_cases":  # If no organ status in model
                    1.,
                "tb_multiplier_force_smearpos":
                    1.,
                "tb_multiplier_force_smearneg":
                    0.24,
                "tb_multiplier_force_extrapul":
                    0.,
                "tb_multiplier_force":
                    1.,
                "tb_n_contact":
                    14.,
                "tb_proportion_early_progression":
                    0.12,
                "tb_timeperiod_early_latent":
                    0.4,
                "tb_rate_late_progression":
                    0.007,
                "tb_proportion_casefatality_untreated_smearpos":
                    0.6,
                "tb_proportion_casefatality_untreated_smearneg":
                    0.2,
                "tb_proportion_casefatality_untreated":
                    0.4,
                "tb_timeperiod_activeuntreated":
                    3.,
                "tb_multiplier_bcg_protection":
                    0.5,
                "program_prop_vac":
                    0.88,
                "program_prop_unvac":
                    1. - 0.88,
                "program_prop_detect":
                    0.8,
                "program_algorithm_sensitivity":
                    0.9,
                "program_rate_start_treatment":
                    26.,
                "tb_timeperiod_treatment_ds":
                    0.5,
                "tb_timeperiod_treatment_mdr":
                    2.,
                "tb_timeperiod_treatment_xdr":
                    3.,
                "tb_timeperiod_treatment_inappropriate":
                    3.,
                "tb_timeperiod_infect_ontreatment_ds":
                    0.035,
                "tb_timeperiod_infect_ontreatment_mdr":
                    1. / 12.,
                "tb_timeperiod_infect_ontreatment_xdr":
                    2. / 12.,
                "tb_timeperiod_infect_ontreatment_inappropriate":
                    2.,
                "program_proportion_success_ds":
                    0.9,
                "program_proportion_success_mdr":
                    0.6,
                "program_proportion_success_xdr":
                    0.4,
                "program_proportion_success_inappropriate":
                    0.25,
                "program_rate_restart_presenting":
                    4.,
                "proportion_amplification":
                    1. / 15.,
                "timepoint_introduce_mdr":
                    1940.,
                "timepoint_introduce_xdr":
                    2050.,
                "treatment_available_date":
                    1940.,
                "dots_start_date":
                    1990,
                "finish_scaleup_date":
                    2010,
                "pretreatment_available_proportion":
                    0.,
                "dots_start_proportion":
                    0.85,
                "program_prop_assign_mdr":
                    0.6,
                "program_prop_assign_xdr":
                    .4,
                "program_prop_lowquality":
                    0.05,
                "program_rate_leavelowquality":
                    2.,
                "program_prop_nonsuccessoutcomes_death":
                    0.25,
                "ageing_rate":
                    1. / 5.
            }

        # Populate parameters into model
        for parameter in paramater_dict:
            self.set_parameter(parameter, paramater_dict[parameter])

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

        if self.n_organ > 0: self.ensure_all_progressions_go_somewhere_params()

        if len(self.agegroups) > 1: self.find_ageing_rates()

        self.find_natural_history_params()

        self.process_country_data()

        self.find_nontreatment_rates_params()

        self.find_treatment_rates_scaleups()

        self.find_amplification_scaleup()

        self.set_population_death_rate("demo_rate_death")

    ##################################################################
    # The methods that process_parameters calls to set parameters and
    # scale-up functions

    def ensure_all_progressions_go_somewhere_params(self):

        """
        If fewer than three organ statuses are available,
        ensure that all detected patients go somewhere
        """

        # One organ status shouldn't really be used, but in case it is
        if len(self.organ_status) == 1:
            self.params["epi_proportion_cases_smearpos"] = 1.
        # Extrapuls go to smearneg (arguably should be the other way round)
        elif len(self.organ_status) == 2:
            self.params["epi_proportion_cases_smearneg"] = \
                self.params["epi_proportion_cases_smearneg"] \
                + self.params["epi_proportion_cases_extrapul"]

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
        self.set_scaleup_fn(
            "tb_proportion_amplification",
            make_two_step_curve(0., 1. / 15., 2. / 15., 1950., 1960., 1970.))

    def prepare_data(self, data, upper_limit_believable, percentage=True, zero_start_point=None):

        """
        Method intended for preparation of country data
        Args:
            data: An element of country_data
            upper_limit_believable: The highest believable value (for example, this should
             always be at most 100. for case detection)
            percentage: Boolean of whether the input is a percentage
            zero_start_point: A historical time-point to go back to zero at if required

        Returns:
            xvalues: List of xvalues (i.e. time-points) for functions to calculate scale-ups
            yvales: List of yvales (i.e. data) for functions to calculate scale-ups
        """

        prepared_data = data
        for i in prepared_data:

            prepared_data[int(i)] = prepared_data.pop(i)

            # e.g. make sure no case detection rates are greater than 100%
            if prepared_data[i] > upper_limit_believable:
                prepared_data[i] = upper_limit_believable

            # Divide through by 100 if they are percentages
            if percentage:
                prepared_data[i] /= 100.

        # Add a zero starting point
        if zero_start_point is not None:
            prepared_data[zero_start_point] = 0.

        xvalues = []
        yvalues = []
        for i in sorted(prepared_data.keys()):
            xvalues += [i]
            yvalues += [prepared_data[i]]

        return xvalues, yvalues

    def process_country_data(self):

        self.case_detection_xvalues, self.case_detection_yvalues = \
            self.prepare_data(self.country_data[u'c_cdr'], 90., True, 1950.)
        self.bcg_vaccination_xvalues, self.bcg_vaccination_yvalues = \
            self.prepare_data(self.country_data['bcg'], 95., True, 1930.)
        self.treatment_success_xvalues, self.treatment_success_yvalues = \
            self.prepare_data(self.country_data[u'c_new_tsr'], 95., True, 1950.)
        self.treatment_default_yvalues = []
        self.treatment_death_yvalues = []
        for i in range(len(self.treatment_success_yvalues)):
            self.treatment_death_yvalues += \
                [(1. - self.treatment_success_yvalues[i]) * self.params["program_prop_nonsuccessoutcomes_death"]]
            self.treatment_default_yvalues += \
                [1. - self.treatment_success_yvalues[i] - self.treatment_death_yvalues[i]]

    def find_nontreatment_rates_params(self):

        # Temporary scale-up functions to be populated from spreadsheets
        # as the next thing we need to do

        # Currently using method 4 in the following scale-up to prevent negative values
        self.set_scaleup_fn("program_prop_vaccination",
                            scale_up_function(self.bcg_vaccination_xvalues,
                                              self.bcg_vaccination_yvalues, method=4))
        self.set_scaleup_fn("program_prop_detect",
                            scale_up_function(self.case_detection_xvalues,
                                              self.case_detection_yvalues))
        self.set_scaleup_fn("program_prop_algorithm_sensitivity",
                            scale_up_function([1920., 1980., 2000.], [0.7, 0.8, 0.9]))
        self.set_scaleup_fn("program_prop_lowquality",
                            scale_up_function([1980., 1990., 2000.], [0.3, 0.4, 0.5]))
        self.set_scaleup_fn("program_prop_firstline_dst",
                            scale_up_function([1980., 1990., 2000.], [0., 0.5, 0.7]))
        self.set_scaleup_fn("program_prop_secondline_dst",
                            scale_up_function([1985., 1995., 2000.], [0., 0.5, 0.7]))

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
                scale_up_function(self.treatment_success_xvalues,
                                  self.treatment_success_yvalues))
            self.set_scaleup_fn(
                "program_proportion_default" + strain,
                scale_up_function(self.treatment_success_xvalues,
                                  self.treatment_default_yvalues))
            self.set_scaleup_fn(
                "program_proportion_death" + strain,
                scale_up_function(self.treatment_success_xvalues,
                                  self.treatment_death_yvalues))

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

        self.calculate_detect_missed_vars()

        self.calculate_proportionate_detection_vars()

        if self.is_lowquality: self.calculate_lowquality_detection_vars()

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
            self.vars["infectious_population" + strain] = 0.
            for organ in self.organ_status:
                for label in self.labels:
                    if organ not in label and organ != "":
                        continue
                    if strain not in label and strain != "":
                        continue
                    if not label_intersects_tags(label, self.infectious_tags):
                        continue
                    self.vars["infectious_population" + strain] += \
                        self.params["tb_multiplier_force" + organ] \
                        * self.compartments[label]
            self.vars["rate_force" + strain] = \
                self.params["tb_n_contact"] \
                * self.vars["infectious_population" + strain] \
                / self.vars["population"]
            self.vars["rate_force_weak" + strain] = \
                self.params["tb_multiplier_bcg_protection"] \
                * self.vars["rate_force" + strain]

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
            * (1. - self. vars["program_prop_algorithm_sensitivity"]) \
            / self.vars["program_prop_algorithm_sensitivity"]

        # Repeat for each strain
        for strain in self.strains:
            for programmatic_rate in ["_detect", "_missed"]:
                self.vars["program_rate" + programmatic_rate + strain] = \
                    self.vars["program_rate" + programmatic_rate]

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
                        "rate_force_weak" + strain)
                    self.set_var_transfer_rate_flow(
                        "susceptible_treated" + comorbidity + agegroup,
                        "latent_early" + strain + comorbidity + agegroup,
                        "rate_force_weak" + strain)
                    self.set_var_transfer_rate_flow(
                        "latent_late" + strain + comorbidity + agegroup,
                        "latent_early" + strain + comorbidity + agegroup,
                        "rate_force_weak" + strain)

    def set_natural_history_flows(self):

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
                            self.set_fixed_transfer_rate_flow(
                                "latent_early" + strain + comorbidity + agegroup,
                                "active" + organ + strain + comorbidity + agegroup,
                                "tb_rate_early_progression" + organ)

                            # Late progression
                            self.set_fixed_transfer_rate_flow(
                                "latent_late" + strain + comorbidity + agegroup,
                                "active" + organ + strain + comorbidity + agegroup,
                                "tb_rate_late_progression" + organ)

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

                        # Detection, with and without misassignment
                        if self.is_misassignment:
                            for assigned_strain in self.strains:
                                self.set_fixed_transfer_rate_flow(
                                    "detect" +
                                    organ + strain + "_as" + assigned_strain[1:] + comorbidity + agegroup,
                                    "treatment_infect" +
                                    organ + strain + "_as" + assigned_strain[1:] + comorbidity + agegroup,
                                    "program_rate_start_treatment")
                        else:
                            self.set_fixed_transfer_rate_flow(
                                "detect" + organ + strain + comorbidity + agegroup,
                                "treatment_infect" + organ + strain + comorbidity + agegroup,
                                "program_rate_start_treatment")

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
                        if self.is_lowquality:
                            self.set_var_transfer_rate_flow(
                                "active" + organ + strain + comorbidity + agegroup,
                                "lowquality" + organ + strain + comorbidity + agegroup,
                                "program_rate_enterlowquality")

    def set_treatment_flows(self):

        """
        Set rates of progression through treatment stages
        Accommodates with and without amplification and with and without misassignment
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
        Outputs are calculated as vars in each time-step
        """

        # Initialise scalars
        rate_incidence = 0.
        rate_mortality = 0.
        rate_notifications = 0.

        # Incidence
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            if 'latent' in from_label and 'active' in to_label:
                rate_incidence += self.compartments[from_label] * rate
        self.vars["incidence"] = \
            rate_incidence \
            / self.vars["population"] * 1E5

        # Notifications
        for from_label, to_label, rate in self.var_transfer_rate_flows:
            if 'active' in from_label and \
                    ('detect' in to_label or 'treatment_infect' in to_label):
                rate_notifications += self.compartments[from_label] * self.vars[rate]
        self.vars["notifications"] = \
            rate_notifications / self.vars["population"] * 1E5

        # Mortality
        for from_label, rate in self.fixed_infection_death_rate_flows:
            rate_mortality \
                += self.compartments[from_label] * rate
        for from_label, rate in self.var_infection_death_rate_flows:
            rate_mortality \
                += self.compartments[from_label] * self.vars[rate]
        self.vars["mortality"] = rate_mortality / self.vars["population"] * 1E5

        # Prevalence
        self.vars["prevalence"] = 0.
        for label in self.labels:
            if "susceptible" not in label and "latent" not in label:
                self.vars["prevalence"] += (
                    self.compartments[label]
                    / self.vars["population"] * 1E5)

        self.calculate_bystrain_output_vars()

    def calculate_bystrain_output_vars(self):

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
            for from_label, to_label, rate in self.fixed_transfer_rate_flows:
                if 'latent' in from_label and 'active' in to_label and strain in to_label:
                    rate_incidence[strain] \
                        += self.compartments[from_label] * rate
            self.vars["incidence" + strain] \
                = rate_incidence[strain] / self.vars["population"] * 1E5

            # Notifications
            for from_label, to_label, rate in self.var_transfer_rate_flows:
                if 'active' in from_label and 'detect' in to_label and strain in from_label:
                    rate_notifications[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            self.vars["notifications" + strain] \
                = rate_notifications[strain] / self.vars["population"] * 1E5

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
                = rate_mortality[strain] / self.vars["population"] * 1E5

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
