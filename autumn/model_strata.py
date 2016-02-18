import os
from scipy.integrate import odeint
from scipy import exp, log
import numpy


from model import BaseModel
from settings import default
from settings import philippines



def add_unique_tuple_to_list(a_list, a_tuple):
    """
    Adds or modifies a list of tuples, compares only the items
    before the last in the tuples, the last value in the tuple
    is assumed to be a value.
    """
    for i, test_tuple in enumerate(a_list):
        if test_tuple[:-1] == a_tuple[:-1]:
            a_list[i] = a_tuple
            break
    else:
        a_list.append(a_tuple)


def label_intersects_tags(label, tags):
    for tag in tags:
        if tag in label:
            return True
    return False




class SingleStrainStratifiedModel(BaseModel):
    """
    This model based on James' thesis
    """

    def __init__(self, input_parameters=None, input_compartments=None):

        BaseModel.__init__(self)

        self.compartment_list = [
            "susceptible_fully",
            "susceptible_vac",
            "susceptible_treated",
            "latent_early",
            "latent_late",
        ]

        self.infection_state_list = [
            "active",
            "detect",
            "missed",
            "treatment_infect",
            "treatment_noninfect"
        ]

        self.pulmonary_stratas = [
            "_smearpos",
            "_smearneg",
            "_extrapul"]

        self.comorbidity_stratas = [
            "_cohiv",
            "_codiabetes",
            "_coother"]

        if input_parameters is None:

            def get(param_set_name, param_name, prob=0.5):
                param_set = globals()[param_set_name]
                param = getattr(param_set, param_name)
                ppf = getattr(param, "ppf")
                return ppf(prob)

            input_parameters = {
                "demo_rate_birth": 20. / 1e3,
                "demo_rate_death": 1. / 65,
                "epi_proportion_cases_smearpos": 0.6,
                "epi_proportion_cases_smearneg": 0.2,
                "epi_proportion_cases_extrapul": 0.2,
                "tb_multiplier_force_smearpos": 1.,
                "tb_multiplier_force_smearneg":
                    get("default", "multiplier_force_smearneg"),
                "tb_multiplier_force_extrapul": 0.,
                "tb_n_contact":
                    get("default", "tb_n_contact"),
                "tb_proportion_early_progression":
                    get("default", "proportion_early_progression"),
                "tb_timeperiod_early_latent":
                    get("default", "timeperiod_early_latent"),
                "tb_rate_late_progression":
                    get("default", "rate_late_progression"),
                "tb_proportion_casefatality_untreated_smearpos":
                    get("default", "proportion_casefatality_active_untreated_smearpos"),
                "tb_proportion_casefatality_untreated_smearneg":
                    get("default", "proportion_casefatality_active_untreated_smearneg"),
                "tb_timeperiod_activeuntreated":
                    get("default", "timeperiod_activeuntreated"),
                "tb_multiplier_bcg_protection":
                    get("default", "multiplier_bcg_protection"),
                "program_prop_vac":
                    get("philippines", "bcg_coverage"),
                "program_prop_unvac":
                    1. - get("philippines", "bcg_coverage"),
                "program_proportion_detect":
                    get("philippines", "bcg_coverage"),
                "program_algorithm_sensitivity":
                    get("philippines", "algorithm_sensitivity"),
                "program_rate_start_treatment":
                    1. / get("philippines", "program_timeperiod_delayto_treatment"),
                "tb_timeperiod_treatment":
                    get("default", "timeperiod_treatment_ds"),
                "tb_timeperiod_infect_ontreatment":
                    get("default", "timeperiod_infect_ontreatment"),
                "program_proportion_default":
                    get("philippines", "proportion_default"),
                "program_proportion_death":
                    get("philippines", "proportion_death"),
                "program_rate_restart_presenting":
                    1. / get("philippines", "timeperiod_norepresentation")
            }

        if input_compartments is None:
            input_compartments = {
                "susceptible_fully": 1e6,
                "active": 3.
            }

        self.set_input(input_parameters, input_compartments)

    def make_strata_label(self, base, stratas):
        return base + "".join(stratas)

    def strata_iterator(self):
        for pulmonary in self.pulmonary_stratas:
            for comorbidity in self.comorbidity_stratas:
                yield pulmonary, comorbidity

    def set_input(self, input_parameters, input_compartments):

        for compartment in self.compartment_list:
            if compartment in input_compartments:
                self.set_compartment(compartment, input_compartments[compartment])
            else:
                self.set_compartment(compartment, 0.)

        fraction = 1. / len(self.pulmonary_stratas) / len(self.comorbidity_stratas)
        for state in self.infection_state_list:
            for pulmonary, comorbidity in self.strata_iterator():
                compartment = self.make_strata_label(state, [pulmonary, comorbidity])
                if state in input_compartments:
                    val = input_compartments[state] * fraction
                    self.set_compartment(compartment , val)
                else:
                    self.set_compartment(compartment , 0.)

        for parameter in input_parameters:
            self.set_param(parameter, input_parameters[parameter])

        self.set_param(
            "tb_rate_stabilise",
            (1 - self.params["tb_proportion_early_progression"])
              / self.params["tb_timeperiod_early_latent"])

        if "tb_proportion_casefatality_untreated_extrapul" not in input_parameters:
            self.set_param(
                "tb_proportion_casefatality_untreated_extrapul",
                input_parameters["tb_proportion_casefatality_untreated_smearneg"])

        self.set_param(
            "program_rate_detect",
            1. / self.params["tb_timeperiod_activeuntreated"]
               / (1. - self.params["program_proportion_detect"]))
        # Formula derived from CDR = (detection rate) / (detection rate and spontaneous resolution rates)

        self.set_param(
            "program_rate_missed",
            self.params["program_rate_detect"]
              * (1. - self.params["program_algorithm_sensitivity"])
              / self.params["program_algorithm_sensitivity"])
        # Formula derived from (algorithm sensitivity) = (detection rate) / (detection rate and miss rate)

        # Code to determines the treatment flow rates from the input parameters
        self.outcomes = ["_success", "_death", "_default"]
        self.nonsuccess_outcomes = self.outcomes[1:3]
        self.treatment_stages = ["_infect", "_noninfect"]
        self.set_param(
            "tb_timeperiod_noninfect_ontreatment",  # Find the non-infectious period
            self.params["tb_timeperiod_treatment"]
              - self.params["tb_timeperiod_infect_ontreatment"])
        for outcome in self.nonsuccess_outcomes:  # Find the proportion of deaths/defaults during infectious stage
            self.set_param(
                "program_proportion" + outcome + "_infect",
                1. - exp( log(1. - self.params["program_proportion" + outcome])
                            * self.params["tb_timeperiod_infect_ontreatment"]
                            / self.params["tb_timeperiod_treatment"]
                        )
            )
        for outcome in self.nonsuccess_outcomes:  # Find the proportion of deaths/defaults during non-infectious stage
            self.set_param(
                "program_proportion" + outcome + "_noninfect",
                self.params["program_proportion" + outcome]
                  - self.params["program_proportion" + outcome + "_infect"])
        for treatment_stage in self.treatment_stages:  # Find the success proportions
            self.set_param(
                "program_proportion_success" + treatment_stage,
                1. - self.params["program_proportion_default" + treatment_stage]
                   - self.params["program_proportion_death" + treatment_stage])
            for outcome in self.outcomes:  # Find the corresponding rates from the proportions
                self.set_param(
                    "program_rate" + outcome + treatment_stage,
                    1. / self.params["tb_timeperiod" + treatment_stage + "_ontreatment"]
                       * self.params["program_proportion" + outcome + treatment_stage])

        for pulmonary in self.pulmonary_stratas:
            self.set_param(
                "tb_rate_earlyprogress" + pulmonary,
                self.params["tb_proportion_early_progression"]
                  / self.params["tb_timeperiod_early_latent"]
                  * self.params["epi_proportion_cases" + pulmonary])
            self.set_param(
                "tb_rate_lateprogress" + pulmonary,
                self.params["tb_rate_late_progression"]
                * self.params["epi_proportion_cases" + pulmonary])
            self.set_param(
                "tb_rate_recover" + pulmonary,
                (1 - self.params["tb_proportion_casefatality_untreated" + pulmonary])
                  / self.params["tb_timeperiod_activeuntreated"])
            self.set_param(
                "tb_demo_rate_death" + pulmonary,
                self.params["tb_proportion_casefatality_untreated" + pulmonary]
                  / self.params["tb_timeperiod_activeuntreated"])

        self.infectious_tags = ["active", "missed", "detect", "treatment_infect"]

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["demo_rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params["program_prop_unvac"] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params["program_prop_vac"] * self.vars["rate_birth"]

        self.vars["infectious_population"] = 0.0
        for state in self.infection_state_list:
            if not label_intersects_tags(state, self.infectious_tags):
                continue
            for pulmonary, comorbidity in self.strata_iterator():
                label = self.make_strata_label(state, [pulmonary, comorbidity])
                self.vars["infectious_population"] += \
                    self.params["tb_multiplier_force" + pulmonary] \
                       * self.compartments[label]

        self.vars["rate_force"] = \
            self.params["tb_n_contact"] \
              * self.vars["infectious_population"] \
              / self.vars["population"]

        self.vars["rate_force_weak"] = \
            self.params["tb_multiplier_bcg_protection"] \
              * self.vars["rate_force"]

    def set_flows(self):
        self.set_var_entry_rate_flow(
            "susceptible_fully", "births_unvac")
        self.set_var_entry_rate_flow(
            "susceptible_vac", "births_vac")

        self.set_var_transfer_rate_flow(
            "susceptible_fully", "latent_early", "rate_force")
        self.set_var_transfer_rate_flow(
            "susceptible_vac", "latent_early", "rate_force_weak")
        self.set_var_transfer_rate_flow(
            "susceptible_treated", "latent_early", "rate_force_weak")
        self.set_var_transfer_rate_flow(
            "latent_late", "latent_early", "rate_force_weak")

        self.set_fixed_transfer_rate_flow(
            "latent_early", "latent_late", "tb_rate_stabilise")

        for stratas in self.strata_iterator():
            pulmonary = stratas[0]
            self.set_fixed_transfer_rate_flow(
                "latent_early",
                self.make_strata_label("active", stratas),
                "tb_rate_earlyprogress" + pulmonary)
            self.set_fixed_transfer_rate_flow(
                "latent_late",
                self.make_strata_label("active", stratas),
                "tb_rate_lateprogress" + pulmonary)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("active", stratas),
                "latent_late",
                "tb_rate_recover" + pulmonary)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("active", stratas),
                self.make_strata_label("detect", stratas),
                "program_rate_detect")
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("active", stratas),
                self.make_strata_label("missed", stratas),
                "program_rate_missed")
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("detect", stratas),
                self.make_strata_label("treatment_infect", stratas),
                "program_rate_start_treatment")
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("missed", stratas),
                self.make_strata_label("active", stratas),
                "program_rate_restart_presenting")
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("missed", stratas),
                "latent_late",
                "tb_rate_recover" + pulmonary)
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_infect", stratas),
                self.make_strata_label("treatment_noninfect", stratas),
                "program_rate_success_infect")
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_infect", stratas),
                self.make_strata_label("active", stratas),
                "program_rate_default_infect")
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_noninfect", stratas),
                self.make_strata_label("active", stratas),
                "program_rate_default_noninfect")
            self.set_fixed_transfer_rate_flow(
                self.make_strata_label("treatment_noninfect", stratas),
                "susceptible_treated",
                "program_rate_success_noninfect")

        # death flows
        self.set_population_death_rate("demo_rate_death")

        for stratas in self.strata_iterator():
            pulmonary = stratas[0]
            self.set_infection_death_rate_flow(
                self.make_strata_label("active", stratas),
                "tb_demo_rate_death" + pulmonary)
            self.set_infection_death_rate_flow(
                self.make_strata_label("detect", stratas),
                "tb_demo_rate_death" + pulmonary)
            self.set_infection_death_rate_flow(
                self.make_strata_label("treatment_infect", stratas),
                "program_rate_death_infect")
            self.set_infection_death_rate_flow(
                self.make_strata_label("treatment_noninfect", stratas),
                "program_rate_death_noninfect")

    def calculate_diagnostic_vars(self):

        rate_incidence = 0.0
        rate_infection = 0.0
        rate_notification = 0.0
        rate_missed = 0.0
        rate_death_ontreatment = 0.0
        rate_default = 0.0
        rate_success = 0.0
        for from_label, to_label, rate in self.fixed_transfer_rate_flows:
            if 'latent' in from_label and 'active' in to_label:
                val = self.compartments[from_label] * rate
                rate_incidence += val
            elif 'susceptible' in from_label and 'latent' in to_label:
                val = self.compartments[from_label] * rate
                rate_infection += val
            elif 'active' in from_label and 'detect' in to_label:
                val = self.compartments[from_label] * rate
                rate_notification += val
            elif 'active' in from_label and 'missed' in to_label:
                val = self.compartments[from_label] * rate
                rate_missed += val
            elif 'treatment' in from_label and 'death' in to_label:
                val = self.compartments[from_label] * rate
                rate_death_ontreatment += val
            elif 'treatment' in from_label and 'active' in to_label:
                val = self.compartments[from_label] * rate
                rate_default += val
            elif 'treatment' in from_label and 'susceptible_treated' in to_label:
                val = self.compartments[from_label] * rate
                rate_success += val

        # Main epidemiological indicators - note that denominator is not individuals
        self.vars["incidence"] = \
              rate_incidence \
            / self.vars["population"] * 1E5

        self.vars["notification"] = \
              rate_notification \
            / self.vars["population"] * 1E5

        self.vars["mortality"] = \
              self.vars["rate_infection_death"] \
            / self.vars["population"] * 1E5

        self.vars["prevalence"] = \
              self.vars["infectious_population"] \
            / self.vars["population"] * 1E5

        """ More commonly termed "annual risk of infection", but really a rate
        and annual is implicit"""
        self.vars["infection"] = \
              rate_infection \
            / self.vars["population"]

        """ Better term may be failed diagnosis, but using missed for
        consistency with the compartment name for now"""
        self.vars["missed"] = \
              rate_missed \
            / self.vars["population"]

        self.vars["death_ontreatment"] = \
              rate_death_ontreatment \
            / self.vars["population"]

        self.vars["default"] = \
              rate_default \
            / self.vars["population"]