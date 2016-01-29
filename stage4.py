# -*- coding: utf-8 -*-


"""
# Stage 18 model of the Dynamic Transmission Model

- latent early and latent late,
- fail detect
- two stage treatment
- three pulmonary status
"""


from autumn.model import BasePopulationSystem
from autumn.plotting import plot_fractions
import pylab
import os



class Stage4PopulationSystem(BasePopulationSystem):

    """
    This model based on James' thesis
    """

    def __init__(self):

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

        self.set_param("proportion_cases_smearpos", 0.6)
        self.set_param("proportion_cases_smearneg", 0.2)
        self.set_param("proportion_cases_extrapul", 0.2)
        ''' Previously all flows that represented developing new TB were replicated
        three times for each status, but should be shared between statuses'''

        self.set_param("tb_multiplier_force_smearpos", 1.)
        self.set_param("tb_multiplier_force_smearneg", .25)
        self.set_param("tb_multiplier_force_extrapul", 0.0)
        for status in self.pulmonary_status:
            self.set_compartment("active" + status, 1.)
            self.set_compartment("detect" + status, 0.)
            self.set_compartment("missed" + status, 0.)
            self.set_compartment("treatment_infect" + status, 0.)
            self.set_compartment("treatment_noninfect" + status, 0.)
            self.set_param("tb_rate_earlyprogress" + status,
                            .2 * self.params["proportion_cases" + status])
            self.set_param("tb_rate_lateprogress" + status,
                            .0005 * self.params["proportion_cases" + status])

        self.set_param("rate_birth", 40. / 1e3)
        self.set_param("rate_death", 1. / 65)

        self.set_param("tb_n_contact", 15.)

        self.set_param("tb_rate_lateprogress", .0005)
        self.set_param("tb_rate_stabilise", .8)
        self.set_param("tb_rate_recover", .5 * .3)
        self.set_param("tb_rate_death", .5 * .3)

        self.set_param("program_prop_vac", .9)
        self.set_param("program_prop_unvac", .1)

        self.set_param("program_rate_detect", 0.8)
        self.set_param("program_rate_missed", 0.2)

        self.set_param("program_rate_start_treatment", 26.)
        self.set_param("program_rate_giveup_waiting", 4.)

        self.set_param("program_rate_completion_infect", 26 * 0.9)
        self.set_param("program_rate_default_infect", 26 * 0.05)
        self.set_param("program_rate_death_infect", 26 * 0.05)

        self.set_param("program_rate_completion_noninfect", 2 * 0.7)
        self.set_param("program_rate_default_noninfect", 2 * 0.1)
        self.set_param("program_rate_death_noninfect", 2 * 0.1)

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_birth"] = \
            self.params["rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params['program_prop_unvac'] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params['program_prop_vac'] * self.vars["rate_birth"]
       
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

        self.vars["rate_force_weak"] = 0.5 * self.vars["rate_force"]

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
            self.set_infection_death_rate_flow(
                "active" + status, 
                "tb_rate_death")
            self.set_infection_death_rate_flow(
                "detect" + status, 
                "tb_rate_death")
            self.set_infection_death_rate_flow(
                "treatment_infect" + status, 
                "program_rate_death_infect")
            self.set_infection_death_rate_flow(
                "treatment_noninfect" + status, 
                "program_rate_death_noninfect")


if __name__ == "__main__":

    population = Stage4PopulationSystem()
    population.set_flows()
    population.make_graph('stage4.graph.png')
    population.make_steps(0, 50, 1.)
    population.integrate_explicit()

    plot_fractions(population, population.labels[:])
    pylab.savefig('stage4.fraction.png', dpi=300)

    os.system('open -a "Google Chrome" stage4.graph.png stage4.fraction.png')

