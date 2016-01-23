# -*- coding: utf-8 -*-


"""
# Stage 18 model of the Dynamic Transmission Model

- latent early and latent late,
- fail detect
- two stage treatment
- three pulmonary status
"""


from modules.model_jt import BasePopulationSystem
from modules.plotting import plot_fractions
import pylab
import os
from scipy.stats import uniform

import modules.parameter_setting as parameter_setting


class Stage5PopulationSystem(BasePopulationSystem):

    """
    This model based on James' thesis
    """

    def __init__(self, input_parameters):

        BasePopulationSystem.__init__(self)

        self.set_compartment("susceptible_fully", 1e6)
        self.set_compartment("susceptible_vac", 0.)
        self.set_compartment("susceptible_treated", 0.)
        self.set_compartment("latent_early", 0.)
        self.set_compartment("latent_late", 0.)

        self.pulmonary_status = [
            "_smearpos",
            "_smearneg",
            "_extrapul"]

        for parameter in input_parameters:
            self.set_param(
                    parameter, input_parameters[parameter])

        for status in self.pulmonary_status:
            self.set_compartment("active" + status, 1.)
            self.set_compartment("detect" + status, 0.)
            self.set_compartment("missed" + status, 0.)
            self.set_compartment("treatment_infect" + status, 0.)
            self.set_compartment("treatment_noninfect" + status, 0.)

            self.set_param("tb_rate_earlyprogress" + status,
                           self.params["proportion_early_progression"]
                           / self.params["timeperiod_early_latent"]
                           * self.params["proportion_cases" + status])
            self.set_param("tb_rate_lateprogress" + status,
                           self.params["rate_late_progression"]
                           * self.params["proportion_cases" + status])

        self.set_param("tb_rate_stabilise",
                       (1 - self.params["proportion_early_progression"])
                       / self.params["timeperiod_early_latent"])
        self.set_param("tb_rate_recover",
                       (1 - self.params["proportion_casefatality_active_untreated_smearpos"])
                       / self.params["timeperiod_activeuntreated"])
        self.set_param("tb_rate_death",
                       self.params["proportion_casefatality_active_untreated_smearpos"]
                       / self.params["timeperiod_activeuntreated"])

        print(self.params)

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["births"] = \
            self.params["rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params['program_prop_unvac'] * self.vars["births"]
        self.vars["births_vac"] = \
            self.params['program_prop_vac'] * self.vars["births"]

        self.vars["infected_populaton"] = 0.0
        for status in self.pulmonary_status:
            for label in self.labels:
                if status in label and "_noninfect" not in label:
                    self.vars["infected_populaton"] += \
                        self.params["tb_multiplier_force" + status] \
                           * self.compartments[label]

        self.vars["rate_force"] = \
              self.params["tb_n_contact"] \
            * self.vars["infected_populaton"] \
            / self.vars["population"]

        self.vars["rate_force_weak"] = \
            parameter_setting.multiplier_bcg_protection.prior_estimate \
            * self.vars["rate_force"]

    def set_flows(self):
        self.set_var_flow(
            "susceptible_fully", "births_unvac")
        self.set_var_flow(
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

        for status in self.pulmonary_status:
            self.set_fixed_transfer_rate_flow(
                "latent_early",
                "active" + status,
                "tb_rate_earlyprogress" + status)
            self.set_fixed_transfer_rate_flow(
                "latent_late",
                "active" + status,
                "tb_rate_lateprogress" + status)
            self.set_fixed_transfer_rate_flow(
                "active" + status,
                "latent_late",
                "tb_rate_recover")
            self.set_fixed_transfer_rate_flow(
                "active" + status,
                "detect" + status,
                "program_rate_detect")
            self.set_fixed_transfer_rate_flow(
                "active" + status,
                "missed" + status,
                "program_rate_missed")
            self.set_fixed_transfer_rate_flow(
                "detect" + status,
                "treatment_infect" + status,
                "program_rate_start_treatment")
            self.set_fixed_transfer_rate_flow(
                "missed" + status,
                "active" + status,
                "program_rate_giveup_waiting")
            self.set_fixed_transfer_rate_flow(
                "treatment_infect" + status,
                "treatment_noninfect" + status,
                "program_rate_completion_infect")
            self.set_fixed_transfer_rate_flow(
                "treatment_infect" + status,
                "active" + status,
                "program_rate_default_infect")
            self.set_fixed_transfer_rate_flow(
                "treatment_noninfect" + status,
                "active" + status,
                "program_rate_default_noninfect")
            self.set_fixed_transfer_rate_flow(
                "treatment_noninfect" + status,
                "susceptible_treated",
                "program_rate_completion_noninfect")

        # death flows
        self.set_population_death_rate("rate_death")

        for status in self.pulmonary_status:
            self.set_death_rate_flow(
                "active" + status,
                "tb_rate_death")
            self.set_death_rate_flow(
                "detect" + status,
                "tb_rate_death")
            self.set_death_rate_flow(
                "treatment_infect" + status,
                "program_rate_death_infect")
            self.set_death_rate_flow(
                "treatment_noninfect" + status,
                "program_rate_death_noninfect")


if __name__ == "__main__":
    input_parameters = \
        {"proportion_cases_smearpos": 0.6,
         "proportion_cases_smearneg": 0.2,
         "proportion_cases_extrapul": 0.2,
         "tb_multiplier_force_smearpos": 1.,
         "tb_multiplier_force_smearneg":
            parameter_setting.multiplier_force_smearneg.prior_estimate,
         "tb_multiplier_force_extrapul": 0.0,
         "rate_birth": 20. / 1e3,
         "rate_death": 1. / 65,
         "tb_n_contact": 25.,
         "proportion_early_progression":
            parameter_setting.proportion_early_progression.prior_estimate,
         "timeperiod_early_latent":
            parameter_setting.timeperiod_early_latent.prior_estimate,
         "rate_late_progression":
            parameter_setting.rate_late_progression.prior_estimate,
         "proportion_casefatality_active_untreated_smearpos":
            parameter_setting.proportion_casefatality_active_untreated_smearpos.prior_estimate,
         "timeperiod_activeuntreated":
            parameter_setting.timeperiod_activeuntreated.prior_estimate,
         "proportion_casefatality_active_untreated_smearpos":
            parameter_setting.proportion_casefatality_active_untreated_smearpos.prior_estimate,
         "timeperiod_activeuntreated":
            parameter_setting.timeperiod_activeuntreated.prior_estimate,
         "program_prop_vac": .9,
         "program_prop_unvac": .1,
         "program_rate_detect": 0.8,
         "program_rate_missed": 0.2,
         "program_rate_start_treatment": 26.,
         "program_rate_giveup_waiting": 4.,
         "program_rate_completion_infect": 26 * 0.9,
         "program_rate_default_infect": 26 * 0.05,
         "program_rate_death_infect": 26 * 0.05,
         "program_rate_completion_noninfect": 2 * 0.7,
         "program_rate_default_noninfect": 2 * 0.1,
         "program_rate_death_noninfect": 2 * 0.1}



    population = Stage5PopulationSystem(input_parameters)
    population.set_flows()
    population.make_graph('stage5.graph.png')
    population.make_steps(1950., 2000., 20)
    population.integrate_scipy()  # Need to fix for integrate explicit as well

    plot_fractions(population, population.labels[:])
    pylab.savefig('stage5.fraction.png', dpi=300)

    os.system('open -a "Google Chrome" stage5.graph.png stage5.fraction.png')

