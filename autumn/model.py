# -*- coding: utf-8 -*-


"""

Base Population Model to handle different type of models.

Implicit time unit: years

"""

import random

from scipy import exp, log

from autumn.base import BaseModel
from autumn.settings import default, philippines
from curve import make_sigmoidal_curve, make_two_step_curve


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
    """

    def __init__(self,
                 number_of_organs=0,
                 number_of_strains=0,
                 number_of_comorbidities=0,
                 lowquality=False,
                 amplification=False,
                 misassignment=False):

        """
        Args:
            n_organ: whether pulmonary status and smear-positive/smear-negative status
                can be included in the model (which applies to all compartments representing active disease)
                0. No subdivision
                1. All patients are smear-positive pulmonary (avoid selecting this)
                2. All patients are pulmonary, but smear status can be selected (i.e. smear-pos/smear-neg)
                3. Full stratification into smear-positive, smear-negative and extra-pulmonary
            n_strain: number of types of drug-resistance included in the model (not strains in the phylogenetic sense)
                0. No strains included
                1. All TB is DS-TB (avoid selecting this)
                2. DS-TB and MDR-TB
                3. DS-TB, MDR-TB and XDR-TB
                N.B. this may change in future models, which may include isoniazid mono-resistance, etc.
            n_comorbidity: number of whole population stratifications, except for age
                0. No population stratification
                1. Entire population is not at increased risk (avoid selecting this)
                2. No increased risk or HIV
                3. No increased risk, HIV or diabetes
            lowquality: Boolean of whether to include detections through the private sector
            amplification: Boolean of whether to include resistance amplification through treatment default
            misassignment: Boolean of whether also to incorporate misclassification of patients with drug-resistance
                    to the wrong strain by the health system
                Avoid amplification=False but misassignment=True (the model should run with both
                amplificaiton and misassignment, but this combination doesn't make sense)

        """
        BaseModel.__init__(self)

        self.is_lowquality = lowquality
        self.is_amplification = amplification
        self.is_misassignment = misassignment

        self.n_organ = number_of_organs
        self.n_strain = number_of_strains
        self.n_comorbidity = number_of_comorbidities

        self.define_model_structure(
            self.n_organ, self.n_strain, self.n_comorbidity)

        self.initialise_compartments()

        self.set_parameters()

    def define_model_structure(self,
                               number_of_organs,
                               number_of_strains,
                               number_of_comorbidities):

        """
        Args:
            All arguments are set through __init__
            Please refer to __init__ method comments above
        """
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

        self.broad_compartment_types = [
            "susceptible",
            "latent",
            "active",
            "missed",
            "treatment"]

        self.treatment_stages = [
            "_infect",
            "_noninfect"]

        self.infectious_tags = [
            "active",
            "missed",
            "detect",
            "treatment_infect"]

        if number_of_organs == 0:
            # Need an empty list to be iterable for methods iterating by organ status
            self.organ_status = [""]
        else:
            # Select number of organ statuses
            available_organs = [
                "_smearpos",
                "_smearneg",
                "_extrapul"]
            self.organ_status = \
                available_organs[0: number_of_organs]

        if self.is_lowquality == True:  # Add low quality detection compartment
            self.compartment_types \
                = self.compartment_types + ["lowquality"]
            self.broad_compartment_types \
                = self.broad_compartment_types + ["lowquality"]
            self.infectious_tags \
                = self.infectious_tags + ["lowquality"]

        if number_of_strains == 0:
            # Need an empty list to be iterable for methods iterating by strain
            self.strains = [""]
        else:
            # Select number of strains
            available_strains = [
                "_ds",
                "_mdr",
                "_xdr"]
            self.strains\
                = available_strains[0: number_of_strains]

        if number_of_comorbidities == 0:
            # Need an empty list to be iterable for methods iterating by risk group
            self.comorbidities = [""]
        else:
            # Select number of risk groups
            available_comorbidities = [
                "_nocomorbs",
                "_hiv",
                "_diabetes"]
            self.comorbidities\
                = available_comorbidities[0: number_of_comorbidities]

    def initialise_compartments(self, compartment_dict=None):

        # Initialise all compartments to zero
        for compartment in self.compartment_types:
            for comorbidity in self.comorbidities:
                if "susceptible" in compartment:  # Replicate for comorbidities only
                    self.set_compartment(compartment + comorbidity, 0.)
                elif "latent" in compartment:  # Replicate for comorbidities and strains
                    for strain in self.strains:
                        self.set_compartment(compartment + strain + comorbidity, 0.)
                elif "active" in compartment or "missed" in compartment or "lowquality" in compartment:
                    for strain in self.strains:
                        for organ in self.organ_status:
                            self.set_compartment(compartment + organ + strain + comorbidity, 0.)
                else:  # Mis-assignment by strain
                    for strain in self.strains:
                        for organ in self.organ_status:
                            if self.is_misassignment == True:
                                for assigned_strain in self.strains:
                                    self.set_compartment(compartment + organ + strain + "_as" + assigned_strain[1:] + comorbidity,
                                                         0.)
                            else:
                                self.set_compartment(compartment + organ + strain + comorbidity, 0.)

        # Put in values from input_compartments - now initialise to DS-TB only

        # Some useful defaults if None given
        if compartment_dict is None:
            compartment_dict = {
                "susceptible_fully":
                    2e7,
                "active":
                    3.
            }

        default_start_strain = "_ds"
        if self.strains == [""]:
            default_start_strain = ""
        for compartment in self.compartment_types:
            if compartment in compartment_dict:
                if "susceptible" in compartment:
                    for comorbidity in self.comorbidities:
                        self.set_compartment(compartment + comorbidity,
                                             compartment_dict[compartment]
                                             / len(self.comorbidities))
                elif "latent" in compartment:
                    for comorbidity in self.comorbidities:
                        self.set_compartment(compartment + default_start_strain + comorbidity,
                                             compartment_dict[compartment]
                                             / len(self.comorbidities))
                else:
                    for comorbidity in self.comorbidities:
                        for organ in self.organ_status:
                            self.set_compartment(compartment + organ + default_start_strain + comorbidity,
                                                 compartment_dict[compartment]
                                                 / len(self.organ_status)
                                                 / len(self.comorbidities))

    def set_parameters(self, paramater_dict=None):
        """
        Sets useful parameters of the model

        Args:
            paramater_dict: a key-value dictionary where typically key
                              is a string for a param name and value is a float
        """

        # Set up some handy defaults for testing
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
                "program_proportion_detect":
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
                    0.25
            }

        for parameter in paramater_dict:
            self.set_parameter(parameter, paramater_dict[parameter])

    def process_parameters(self):
        self.split_default_death_proportions()

        if self.n_organ > 0:
            self.ensure_all_progressions_go_somewhere()

        self.find_natural_history_flows()

        self.find_detection_rates()

        if self.is_lowquality == True:
            self.find_lowquality_detections()

        if self.n_strain > 0:
            self.find_equal_detection_rates()

        self.find_programmatic_rates()

        self.find_treatment_rates()

    def calculate_birth_rates(self):

        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params["program_prop_unvac"] * self.vars["rate_birth"] \
            / len(self.comorbidities)
        self.vars["births_vac"] = \
            self.params["program_prop_vac"] * self.vars["rate_birth"] \
            / len(self.comorbidities)

    def set_birth_flows(self):

        for comorbidity in self.comorbidities:
            self.set_var_entry_rate_flow(
                "susceptible_fully" + comorbidity, "births_unvac")
            self.set_var_entry_rate_flow(
                "susceptible_vac" + comorbidity, "births_vac")

    def calculate_force_infection(self):

        for strain in self.strains:
            self.vars["infectious_population" + strain] = 0.0
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

    def set_infection_flows(self):

        for comorbidity in self.comorbidities:
            for strain in self.strains:
                self.set_var_transfer_rate_flow(
                    "susceptible_fully" + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force" + strain)
                self.set_var_transfer_rate_flow(
                    "susceptible_vac" + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force_weak" + strain)
                self.set_var_transfer_rate_flow(
                    "susceptible_treated" + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force_weak" + strain)
                self.set_var_transfer_rate_flow(
                    "latent_late" + strain + comorbidity,
                    "latent_early" + strain + comorbidity,
                    "rate_force_weak" + strain)

    def find_natural_history_flows(self):

        # If extrapulmonary case-fatality not stated
        if "tb_proportion_casefatality_untreated_extrapul" not in self.params:
            self.set_parameter(
                "tb_proportion_casefatality_untreated_extrapul",
                self.params["tb_proportion_casefatality_untreated_smearneg"])

        # Progression and stabilisation rates
        self.set_parameter("tb_rate_early_progression",  # Overall
                       self.params["tb_proportion_early_progression"]
                           / self.params["tb_timeperiod_early_latent"])
        self.set_parameter("tb_rate_stabilise",  # Stabilisation rate
                       (1 - self.params["tb_proportion_early_progression"])
                           / self.params["tb_timeperiod_early_latent"])
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

    def ensure_all_progressions_go_somewhere(self):

        # Make sure all progressions go somewhere, regardless of number of organ statuses
        if len(self.organ_status) == 1:
            self.params["epi_proportion_cases_smearpos"] = 1.
        elif len(self.organ_status) == 2:
            self.params["epi_proportion_cases_smearneg"] = \
                self.params["epi_proportion_cases_smearneg"] \
                + self.params["epi_proportion_cases_extrapul"]

    def set_natural_history_flows(self):

        for comorbidity in self.comorbidities:
            for strain in self.strains:
                    self.set_fixed_transfer_rate_flow(
                        "latent_early" + strain + comorbidity,
                        "latent_late" + strain + comorbidity,
                        "tb_rate_stabilise")
                    for organ in self.organ_status:
                        self.set_fixed_transfer_rate_flow(
                            "latent_early" + strain + comorbidity,
                            "active" + organ + strain + comorbidity,
                            "tb_rate_early_progression" + organ)
                        self.set_fixed_transfer_rate_flow(
                            "latent_late" + strain + comorbidity,
                            "active" + organ + strain + comorbidity,
                            "tb_rate_late_progression" + organ)
                        self.set_fixed_transfer_rate_flow(
                            "active" + organ + strain + comorbidity,
                            "latent_late" + strain + comorbidity,
                            "tb_rate_recover" + organ)
                        self.set_fixed_transfer_rate_flow(
                            "missed" + organ + strain + comorbidity,
                            "latent_late" + strain + comorbidity,
                            "tb_rate_recover" + organ)
                        self.set_infection_death_rate_flow(
                            "active" + organ + strain + comorbidity,
                            "tb_rate_death" + organ)
                        self.set_infection_death_rate_flow(
                            "missed" + organ + strain + comorbidity,
                            "tb_rate_death" + organ)
                        if self.is_lowquality == True:
                            self.set_fixed_transfer_rate_flow(
                                "lowquality" + organ + strain + comorbidity,
                                "latent_late" + strain + comorbidity,
                                "tb_rate_recover" + organ)
                            self.set_infection_death_rate_flow(
                                "lowquality" + organ + strain + comorbidity,
                                "tb_rate_death" + organ)
                        if self.is_misassignment == True:
                            for assigned_strain in self.strains:
                                self.set_infection_death_rate_flow(
                                    "detect" + organ + strain + "_as" + assigned_strain[1:] + comorbidity,
                                    "tb_rate_death" + organ)
                                self.set_fixed_transfer_rate_flow(
                                    "detect" + organ + strain + "_as" + assigned_strain[1:] + comorbidity,
                                    "latent_late" + strain + comorbidity,
                                    "tb_rate_recover" + organ)
                        else:
                            self.set_fixed_transfer_rate_flow(
                                "detect" + organ + strain + comorbidity,
                                "latent_late" + strain + comorbidity,
                                "tb_rate_recover" + organ)
                            self.set_infection_death_rate_flow(
                                "detect" + organ + strain + comorbidity,
                                "tb_rate_death" + organ)

    def find_detection_rates(self):

        # Rates of detection and failure of detection
        self.set_parameter(
            "program_rate_detect",
            self.params["program_proportion_detect"]
            * (self.params["tb_rate_recover" + self.organ_status[0]] + self.params["tb_rate_death" + self.organ_status[0]])
            / (1. - self.params["program_proportion_detect"]
               * (1. + (1. - self.params["program_algorithm_sensitivity"])
                  / self.params["program_algorithm_sensitivity"])))

        self.set_parameter(
            "program_rate_missed",
            self.params["program_rate_detect"]
            * (1. - self.params["program_algorithm_sensitivity"])
            / self.params["program_algorithm_sensitivity"]
        )
        # Derived from original formulas of:
        #   algorithm sensitivity = detection rate / (detection rate + missed rate)
        #   - and -
        #   detection proportion = detection rate / (detection rate + missed rate + spont recover rate + death rate)

    def find_programmatic_rates(self):

        destinations_from_active = ["_detect", "_missed"]

        if self.is_lowquality == True:
            destinations_from_active =\
                destinations_from_active + ["_enterlowquality"]

        for destination in destinations_from_active:
            self.set_scaleup_var(
                "program_rate" + destination,
                make_two_step_curve(
                    self.params["pretreatment_available_proportion"] * self.params["program_rate" + destination],
                    self.params["dots_start_proportion"] * self.params["program_rate" + destination],
                    self.params["program_rate" + destination],
                    self.params["treatment_available_date"], self.params["dots_start_date"], self.params["finish_scaleup_date"]))

    def set_programmatic_flows(self):

        for strain in self.strains:
            for organ in self.organ_status:
                for comorbidity in self.comorbidities:
                    self.set_var_transfer_rate_flow(
                        "active" + organ + strain + comorbidity,
                        "detect" + organ + strain + comorbidity,
                        "program_rate_detect")
                    self.set_var_transfer_rate_flow(
                        "active" + organ + strain + comorbidity,
                        "missed" + organ + strain + comorbidity,
                        "program_rate_missed")
                    self.set_fixed_transfer_rate_flow(
                        "detect" + organ + strain + comorbidity,
                        "treatment_infect" + organ + strain + comorbidity,
                        "program_rate_start_treatment")
                    self.set_fixed_transfer_rate_flow(
                        "missed" + organ + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "program_rate_restart_presenting")
                    if self.is_lowquality == True:
                        self.set_var_transfer_rate_flow(
                            "active" + organ + strain + comorbidity,
                            "lowquality" + organ + strain + comorbidity,
                            "program_rate_enterlowquality")
                        self.set_fixed_transfer_rate_flow(
                            "lowquality" + organ + strain + comorbidity,
                            "active" + organ + strain + comorbidity,
                            "program_rate_leavelowquality")

    def split_default_death_proportions(self):

        if self.strains == [""]:
            self.params["program_proportion_success"]\
                = self.params["program_proportion_success_ds"]

        # Temporary code
        # to define default and death proportions
        for strain in self.strains:
            self.params["program_proportion_default" + strain] =\
                (1. - self.params["program_proportion_success" + strain])\
                * (1. - self.params["program_prop_nonsuccessoutcomes_death"])
            self.params["program_proportion_death" + strain] =\
                (1. - self.params["program_proportion_success" + strain])\
                * self.params["program_prop_nonsuccessoutcomes_death"]
        self.params["program_proportion_default_inappropriate"] =\
            (1. - self.params["program_proportion_success_inappropriate"])\
            * (1. - self.params["program_prop_nonsuccessoutcomes_death"])
        self.params["program_proportion_death_inappropriate"] = \
            (1. - self.params["program_proportion_success_inappropriate"])\
            * self.params["program_prop_nonsuccessoutcomes_death"]

    def find_treatment_rates(self):

        if self.strains == [""]:
            self.params["tb_timeperiod_infect_ontreatment"] \
                = self.params["tb_timeperiod_infect_ontreatment_ds"]
            self.params["tb_timeperiod_treatment"] \
                = self.params["tb_timeperiod_treatment_ds"]

        outcomes = ["_success", "_death", "_default"]
        non_success_outcomes = outcomes[1: 3]

        for strain in self.strains + ["_inappropriate"]:
            # Find the non-infectious period
            self.set_parameter(
                "tb_timeperiod_noninfect_ontreatment" + strain,
                self.params["tb_timeperiod_treatment" + strain]
                - self.params["tb_timeperiod_infect_ontreatment" + strain])

            # Find the proportion of deaths/defaults during the infectious and non-infectious stages
            for outcome in non_success_outcomes:
                early_proportion, late_proportion = self.find_flow_proportions_by_period(
                    self.params["program_proportion" + outcome + strain],
                    self.params["tb_timeperiod_infect_ontreatment" + strain],
                    self.params["tb_timeperiod_treatment" + strain])
                self.set_parameter(
                    "program_proportion" + outcome + "_infect" + strain,
                    early_proportion)
                self.set_parameter(
                    "program_proportion" + outcome + "_noninfect" + strain,
                    late_proportion)

            # Find the success proportions
            for treatment_stage in self.treatment_stages:
                self.set_parameter(
                    "program_proportion_success" + treatment_stage + strain,
                    1. - self.params["program_proportion_default" + treatment_stage + strain]
                    - self.params["program_proportion_death" + treatment_stage + strain])
                # Find the corresponding rates from the proportions
                for outcome in outcomes:
                    self.set_parameter(
                        "program_rate" + outcome + treatment_stage + strain,
                        1. / self.params["tb_timeperiod" + treatment_stage + "_ontreatment" + strain]
                        * self.params["program_proportion" + outcome + treatment_stage + strain])

    def set_treatment_flows(self):

        for comorbidity in self.comorbidities:
            for strain in self.strains:
                for organ in self.organ_status:
                    self.set_fixed_transfer_rate_flow(
                        "treatment_infect" + organ + strain + comorbidity,
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "program_rate_success_infect" + strain)
                    self.set_fixed_transfer_rate_flow(
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "susceptible_treated" + comorbidity,
                        "program_rate_success_noninfect" + strain)
                    self.set_infection_death_rate_flow(
                        "treatment_infect" + organ + strain + comorbidity,
                        "program_rate_death_infect" + strain)
                    self.set_infection_death_rate_flow(
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "program_rate_death_noninfect" + strain)
                    self.set_fixed_transfer_rate_flow(
                        "treatment_infect" + organ + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "program_rate_default_infect" + strain)
                    self.set_fixed_transfer_rate_flow(
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "program_rate_default_noninfect" + strain)

    def set_flows(self):

        self.set_birth_flows()

        self.set_infection_flows()

        self.set_natural_history_flows()

        if self.is_misassignment is False:
            self.set_programmatic_flows()
        else:
            self.set_programmatic_flows_with_misassignment()

        if self.is_amplification is False:
            self.set_treatment_flows()
        elif self.is_amplification and self.is_misassignment is False:
            self.set_treatment_flows_with_amplification()
        elif self.is_amplification and self.is_misassignment:
            self.set_treatment_flows_with_misassignment()

        self.set_population_death_rate("demo_rate_death")

    def calculate_additional_diagnostics(self):

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

        self.groups = {
            "ever_infected": ["susceptible_treated", "latent", "active", "missed", "detect", "treatment"],
            "infected": ["latent", "active", "missed", "detect", "treatment"],
            "active": ["active", "missed", "detect", "treatment"],
            "infectious": ["active", "missed", "detect", "treatment_infect"],
            "identified": ["detect", "treatment"],
            "treatment": ["treatment_infect", "treatment_noninfect"]}
        for key in self.groups:
            compartment_soln, compartment_denominator\
                = self.sum_over_compartments(self.groups[key])
            setattr(self, key + "_compartment_soln", compartment_soln)
            setattr(self, key + "_compartment_denominator", compartment_denominator)
            setattr(self, key + "_fraction_soln",
                    self.get_fraction_soln(
                        self.groups[key],
                        compartment_soln,
                        compartment_denominator))

    def find_flow_proportions_by_period(
            self, proportion, early_period, total_period):
        early_proportion\
            = 1. - exp( log(1. - proportion) * early_period / total_period)
        late_proportion\
            = proportion - early_proportion
        return early_proportion, late_proportion

    def calculate_variable_rates(self):

        self.vars["population"] = sum(self.compartments.values())

        self.calculate_birth_rates()

        self.calculate_force_infection()

    def get_fraction_soln(self, numerator_labels, numerators, denominator):
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
        summed_soln = {}
        summed_denominator\
            = [0] * len(random.sample(self.compartment_soln.items(), 1)[0][1])
        for compartment_type in compartment_types:
            summed_soln[compartment_type]\
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
        summed_soln = {}
        # HELP BOSCO
        # The following line of code works, but I'm sure this isn't the best approach:
        summed_denominator\
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
                    or (categories == "organ" and \
                            ("susceptible" in compartment_type or "latent" in compartment_type)):
                summed_soln[compartment_type]\
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
                    summed_soln[compartment_type + working_category]\
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

    def calculate_outputs(self):

        rate_incidence = 0.
        rate_mortality = 0.
        rate_notifications = 0.
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            if 'latent' in from_label and 'active' in to_label:
                rate_incidence += self.compartments[from_label] * rate
        self.vars["incidence"] = \
            rate_incidence \
            / self.vars["population"] * 1E5
        for from_label, to_label, rate in self.var_transfer_rate_flows:
            if 'active' in from_label and\
                    ('detect' in to_label or 'treatment_infect' in to_label):
                rate_notifications += self.compartments[from_label] * self.vars[rate]
        self.vars["notifications"] = \
            rate_notifications / self.vars["population"] * 1E5
        for from_label, rate in self.infection_death_rate_flows:
            rate_mortality \
                += self.compartments[from_label] * rate
        self.vars["mortality"] = \
            rate_mortality \
            / self.vars["population"] * 1E5

        self.vars["prevalence"] = 0.0
        for label in self.labels:
            if "susceptible" not in label and "latent" not in label:
                self.vars["prevalence"] += (
                    self.compartments[label]
                     / self.vars["population"] * 1E5)

    def find_lowquality_detections(self):
        self.set_parameter(
            "program_rate_enterlowquality",
            self.params["program_rate_detect"] \
            * self.params["program_prop_lowquality"] \
            / (1. - self.params["program_prop_lowquality"]))

    def find_equal_detection_rates(self):

        # Set detection rates equal for all strains (probably temporary)
        for strain in self.strains:
            self.set_parameter(
                "program_rate_detect" + strain,
                self.params["program_rate_detect"])
            self.set_parameter(
                "program_rate_missed" + strain,
                self.params["program_rate_missed"])
            self.set_parameter(
                "program_rate_start_treatment" + strain,
                self.params["program_rate_start_treatment"])
            self.set_parameter(
                "program_rate_restart_presenting" + strain,
                self.params["program_rate_restart_presenting"])

    def calculate_outputs_bystrain(self):
        # Now by strain:
        rate_incidence = {}
        rate_mortality = {}
        rate_notifications = {}

        for strain in self.strains:
            rate_incidence[strain] = 0.
            rate_mortality[strain] = 0.
            rate_notifications[strain] = 0.
            for from_label, to_label, rate in self.fixed_transfer_rate_flows:
                if 'latent' in from_label and 'active' in to_label and strain in to_label:
                    rate_incidence[strain] \
                        += self.compartments[from_label] * rate
            for from_label, to_label, rate in self.var_transfer_rate_flows:
                if 'active' in from_label and 'detect' in to_label and strain in from_label:
                    rate_notifications[strain] \
                        += self.compartments[from_label] * self.vars[rate]
            for from_label, rate in self.infection_death_rate_flows:
                if strain in from_label:
                    rate_mortality[strain] \
                        += self.compartments[from_label] * rate
            self.vars["incidence" + strain] \
                = rate_incidence[strain] \
                  / self.vars["population"] * 1E5

            self.vars["mortality" + strain] \
                = rate_mortality[strain] \
                  / self.vars["population"] * 1E5
            self.vars["notifications" + strain] \
                = rate_notifications[strain] \
                  / self.vars["population"] * 1E5

        for strain in self.strains:
            self.vars["prevalence" + strain] = 0.
            for label in self.labels:
                if "susceptible" not in label and "latent" not in label and strain in label:
                    self.vars["prevalence" + strain] += (
                        self.compartments[label]
                        / self.vars["population"] * 1E5)

        rate_incidence["all_mdr_strains"] = 0.
        if len(self.strains) > 1:
            for i in range(len(self.strains)):
                strain = self.strains[i]
                if i > 0:
                    rate_incidence["all_mdr_strains"] \
                        += rate_incidence[strain]
        self.vars["all_mdr_strains"] \
            = rate_incidence["all_mdr_strains"] / self.vars["population"] * 1E5
        self.vars["proportion_mdr"] \
            = self.vars["all_mdr_strains"] / self.vars["incidence"] * 1E2

    def set_treatment_flows_with_amplification(self):

        for comorbidity in self.comorbidities:
            for organ in self.organ_status:
                for i in range(len(self.strains)):
                    strain = self.strains[i]

                    # Set treatment success and death flows (unaffected by amplification)
                    self.set_fixed_transfer_rate_flow(
                        "treatment_infect" + organ + strain + comorbidity,
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "program_rate_success_infect" + strain)
                    self.set_fixed_transfer_rate_flow(
                        "treatment_noninfect" + organ + strain + comorbidity,
                        "susceptible_treated" + comorbidity,
                        "program_rate_success_noninfect" + strain)
                    for treatment_stage in self.treatment_stages:
                        self.set_infection_death_rate_flow(
                            "treatment" + treatment_stage + organ + strain + comorbidity,
                            "program_rate_death" + treatment_stage + strain)

                    # If it's the most resistant strain
                    if i == len(self.strains) - 1:
                        for treatment_stage in self.treatment_stages:
                            self.set_fixed_transfer_rate_flow(
                                "treatment" + treatment_stage + organ + strain + comorbidity,
                                "active" + organ + strain + comorbidity,
                                "program_rate_default" + treatment_stage + strain)
                    # Otherwise, there is a more resistant strain available
                    else:
                        amplify_to_strain = self.strains[i + 1]  # Is the more resistant strain
                        # Split default rates into amplification and non-amplification proportions
                        for treatment_stage in self.treatment_stages:
                            # Calculate amplification and non-amplification target proportions:
                            end_rate_default_noamplify = \
                                self.params["program_rate_default" + treatment_stage + strain]\
                                * (1. - self.params["proportion_amplification"])
                            end_rate_default_amplify = \
                                self.params["program_rate_default" + treatment_stage + strain]\
                                * self.params["proportion_amplification"]
                            # Calculate equivalent functions
                            self.set_scaleup_var(
                                "program_rate_default" + treatment_stage + "_noamplify" + strain,
                                make_sigmoidal_curve(
                                    end_rate_default_noamplify + end_rate_default_amplify,
                                    end_rate_default_noamplify,
                                    self.params["timepoint_introduce" + amplify_to_strain],
                                    self.params["timepoint_introduce" + amplify_to_strain] + 3.))
                            self.set_scaleup_var(
                                "program_rate_default" + treatment_stage + "_amplify" + strain,
                                make_sigmoidal_curve(
                                    0.,
                                    end_rate_default_amplify,
                                    self.params["timepoint_introduce" + amplify_to_strain],
                                    self.params["timepoint_introduce" + amplify_to_strain] + 3.))
                            # Actually set the flows
                            self.set_var_transfer_rate_flow(
                                "treatment" + treatment_stage + organ + strain + comorbidity,
                                "active" + organ + strain + comorbidity,
                                "program_rate_default" + treatment_stage + "_noamplify" + strain)
                            self.set_var_transfer_rate_flow(
                                "treatment" + treatment_stage + organ + strain + comorbidity,
                                "active" + organ + amplify_to_strain + comorbidity,
                                "program_rate_default" + treatment_stage + "_amplify" + strain)

    def set_programmatic_flows_with_misassignment(self):

        for i in range(len(self.strains)):
            strain = self.strains[i]
            for j in range(len(self.strains)):
                assigned_strain = self.strains[j]
                # Chance of being assigned to the strain two levels less resistant (XDR to DS)
                if i == j+2:
                    next_strain = self.strains[i - 1]
                    assignment_probability =\
                        (1. - self.params["program_prop_assign" + next_strain])
                # Chance of being assigned to the next less resistant strain
                # if there are two less resistant strains available (XDR to MDR)
                elif i == 2 and j == 1:
                    next_strain = self.strains[i - 1]
                    assignment_probability =\
                        (1. - self.params["program_prop_assign" + strain]) * self.params["program_prop_assign" + next_strain]
                # Chance of being assigned to the next less resistant strain
                # if the assigned strain is the least resistant one (MDR to DS)
                elif i == j+1 and j == 0:
                    assignment_probability =\
                        (1. - self.params["program_prop_assign" + strain])
                # Chance of being assigned to the correct strain, DS-TB
                elif i == 0 and j == 0:
                    assignment_probability = 1.
                # Chance of being assigned to the correct strain, MDR-TB
                elif i == 1 and j == 1:
                    assignment_probability =\
                        self.params["program_prop_assign" + strain]
                # Chance of being assigned to the correct strain, XDR-TB
                elif i == 2 and j == 2:
                    next_strain = self.strains[i - 1]
                    assignment_probability =\
                        self.params["program_prop_assign" + strain] * self.params["program_prop_assign" + next_strain]
                # Can't be assigned to a more resistant strain than you have (currently)
                elif i < j:
                    assignment_probability = 0.
                # Set the parameter values
                if assignment_probability == 0.:
                    self.set_parameter("program_rate_detect" + strain + "_as" + assigned_strain[1:], assignment_probability)
                    for comorbidity in self.comorbidities:
                        for organ in self.organ_status:
                            self.set_fixed_transfer_rate_flow(
                                "active" + organ + strain + comorbidity,
                                "detect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                                "program_rate_detect" + strain + "_as"+assigned_strain[1:])
                else:
                    self.set_scaleup_var(
                        "program_rate_detect" + strain + "_as"+assigned_strain[1:],
                        make_two_step_curve(
                            self.params["pretreatment_available_proportion"] * self.params["program_rate_detect"] * assignment_probability,
                            self.params["dots_start_proportion"]  * self.params["program_rate_detect"] * assignment_probability,
                            self.params["program_rate_detect"] * assignment_probability,
                            self.params["treatment_available_date"], self.params["dots_start_date"], self.params["finish_scaleup_date"]))
                    for comorbidity in self.comorbidities:
                        for organ in self.organ_status:
                            self.set_var_transfer_rate_flow(
                                "active" + organ + strain + comorbidity,
                                "detect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                                "program_rate_detect" + strain + "_as"+assigned_strain[1:])

        self.set_scaleup_var(
            "program_rate_missed",
            make_two_step_curve(
                self.params["pretreatment_available_proportion"] * self.params["program_rate_missed"],
                self.params["dots_start_proportion"] * self.params["program_rate_missed"],
                self.params["program_rate_missed"],
                self.params["treatment_available_date"], self.params["dots_start_date"], self.params["finish_scaleup_date"]))

        for comorbidity in self.comorbidities:
            for strain in self.strains:
                for organ in self.organ_status:
                    self.set_var_transfer_rate_flow(
                        "active" + organ + strain + comorbidity,
                        "missed" + organ + strain + comorbidity,
                        "program_rate_missed")
                    self.set_fixed_transfer_rate_flow(
                        "missed" + organ + strain + comorbidity,
                        "active" + organ + strain + comorbidity,
                        "program_rate_restart_presenting")
                    for assigned_strain in self.strains:
                        self.set_fixed_transfer_rate_flow(
                            "detect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                            "treatment_infect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                            "program_rate_start_treatment")

    def set_treatment_flows_with_misassignment(self):

        for comorbidity in self.comorbidities:
            for organ in self.organ_status:
                for i in range(len(self.strains)):
                    strain = self.strains[i]
                    for j in range(len(self.strains)):
                        assigned_strain = self.strains[j]

                        # Which treatment parameters to use - for the strain or for inappropriate treatment
                        if i <= j:
                            strain_or_inappropriate = assigned_strain
                        else:
                            strain_or_inappropriate = "_inappropriate"

                        # Set treatment success and death flows (unaffected by amplification)
                        self.set_fixed_transfer_rate_flow(
                            "treatment_infect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                            "treatment_noninfect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                            "program_rate_success_infect" + strain_or_inappropriate)
                        self.set_fixed_transfer_rate_flow(
                            "treatment_noninfect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                            "susceptible_treated" + comorbidity,
                            "program_rate_success_noninfect" + strain_or_inappropriate)
                        for treatment_stage in self.treatment_stages:
                            self.set_infection_death_rate_flow(
                                "treatment" + treatment_stage + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                                "program_rate_death" + treatment_stage + strain_or_inappropriate)

                        # If it's the most resistant strain
                        if i == len(self.strains) - 1:
                            for treatment_stage in self.treatment_stages:
                                self.set_fixed_transfer_rate_flow(
                                    "treatment" + treatment_stage + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                                    "active" + organ + strain + comorbidity,
                                    "program_rate_default" + treatment_stage + strain_or_inappropriate)

                        # Otherwise, there is a more resistant strain available
                        else:
                            amplify_to_strain = self.strains[i + 1]  # Is the more resistant strain
                            # Split default rates into amplification and non-amplification proportions
                            for treatment_stage in self.treatment_stages:
                                # Calculate amplification and non-amplification target proportions:
                                end_rate_default_noamplify = \
                                    self.params["program_rate_default" + treatment_stage + strain] \
                                    * (1 - self.params["proportion_amplification"])
                                end_rate_default_amplify = \
                                    self.params["program_rate_default" + treatment_stage + strain] \
                                    * self.params["proportion_amplification"]
                                # Calculate equivalent functions
                                self.set_scaleup_var(
                                    "program_rate_default" + treatment_stage + "_noamplify" + strain_or_inappropriate,
                                    make_sigmoidal_curve(
                                        end_rate_default_noamplify + end_rate_default_amplify,
                                        end_rate_default_noamplify,
                                        self.params["timepoint_introduce" + amplify_to_strain],
                                        self.params["timepoint_introduce" + amplify_to_strain] + 3.))
                                self.set_scaleup_var(
                                    "program_rate_default" + treatment_stage + "_amplify" + strain_or_inappropriate,
                                    make_sigmoidal_curve(
                                        0.,
                                        end_rate_default_amplify,
                                        self.params["timepoint_introduce" + amplify_to_strain],
                                        self.params["timepoint_introduce" + amplify_to_strain] + 3.))
                                # Set the flows
                                self.set_var_transfer_rate_flow(
                                    "treatment_infect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                                    "active" + organ + strain + comorbidity,
                                    "program_rate_default" + treatment_stage + "_noamplify" + strain_or_inappropriate)
                                self.set_var_transfer_rate_flow(
                                    "treatment_infect" + organ + strain + "_as"+assigned_strain[1:] + comorbidity,
                                    "active" + organ + amplify_to_strain + comorbidity,
                                    "program_rate_default" + treatment_stage + "_amplify" + strain_or_inappropriate)
