# -*- coding: utf-8 -*-
"""
Test
- implement Stage 18 model of the Dynamic Transmission Model
"""

from autumn.model import BasePopulationSystem, make_steps
from autumn.plotting import plot_fractions
import pylab
import os

class Stage3PopulationSystem(BasePopulationSystem):

    """
    This is a reasonable useful model incorporating most of James' thesis.
    """

    def __init__(self):

        BasePopulationSystem.__init__(self)

        self.set_compartment("susceptible_unvac", 1e6)
        self.set_compartment("susceptible_vac", 0.)
        self.set_compartment("susceptible_treated", 0.)
        self.set_compartment("latent_early", 0.)
        self.set_compartment("latent_late", 0.)
        self.set_compartment("active", 1.)
        self.set_compartment("detect", 0.)
        self.set_compartment("misdetect", 0.)
        self.set_compartment("treatment_infect", 0.)
        self.set_compartment("treatment_noninfect", 0.)

        self.set_param("rate_birth", 20. / 1e3)
        self.set_param("rate_death", 1. / 65)

        self.set_param("tb_n_contact", 40.)

        self.set_param("tb_rate_early_active", .2)
        self.set_param("tb_rate_late_active", .0005 )
        self.set_param("tb_rate_stabilise", .8)
        self.set_param("tb_rate_recover", .5 * .3)
        self.set_param("tb_rate_death", .5 * .3 )

        self.set_param("program_prop_vac", .9)
        self.set_param("program_prop_unvac", .1)

        self.set_param("program_rate_detect", 0.8)
        self.set_param("program_rate_misdetect", 0.2)

        self.set_param("program_rate_start_treatment", 26.)
        self.set_param("program_rate_sick_of_waiting", 4.)

        self.set_param("program_rate_completion_infect", 26 * 0.9)
        self.set_param("program_rate_death_infect", 26 * 0.05)
        self.set_param("program_rate_default_infect", 26 * 0.05)

        self.set_param("program_rate_completion_noninfect", 2 * 0.7)
        self.set_param("program_rate_default_noninfect", 2 * 0.1)
        self.set_param("program_rate_death_noninfect", 2 * 0.1)

    def calculate_pre_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["rate_force"] = \
              self.params["tb_n_contact"] \
            * self.compartments["active"] \
            / self.vars["population"]
        self.vars["rate_force_weak"] = \
            0.5 * self.vars["rate_force"]

        self.vars["births"] = \
            self.params["rate_birth"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params['program_prop_unvac'] * self.vars["births"]
        self.vars["births_vac"] = \
            self.params['program_prop_vac'] * self.vars["births"]

    def set_flows(self):
        self.set_var_flow(
            "susceptible_unvac", "births_unvac" )
        self.set_var_flow(
            "susceptible_vac", "births_vac" )

        self.set_var_transfer_rate_flow(
            "susceptible_unvac", "latent_early", "rate_force")
        self.set_var_transfer_rate_flow(
            "susceptible_vac", "latent_early", "rate_force_weak")
        self.set_var_transfer_rate_flow(
            "susceptible_treated", "latent_early", "rate_force_weak")

        self.set_fixed_transfer_rate_flow(
            "latent_early", "latent_late", "tb_rate_stabilise")
        self.set_fixed_transfer_rate_flow(
            "latent_early", "active", "tb_rate_early_active")

        self.set_fixed_transfer_rate_flow(
            "latent_late", "active", "tb_rate_late_active")

        self.set_fixed_transfer_rate_flow(
            "active", "latent_late", "tb_rate_recover")
        self.set_fixed_transfer_rate_flow(
            "active", "detect", "program_rate_detect")
        self.set_fixed_transfer_rate_flow(
            "active", "misdetect", "program_rate_misdetect")

        self.set_fixed_transfer_rate_flow(
            "detect", "treatment_infect", "program_rate_start_treatment")

        self.set_fixed_transfer_rate_flow(
            "misdetect", "active", "program_rate_sick_of_waiting")

        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "treatment_noninfect", "program_rate_completion_infect")
        self.set_fixed_transfer_rate_flow(
            "treatment_infect", "active", "program_rate_default_infect")

        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "active", "program_rate_default_noninfect")
        self.set_fixed_transfer_rate_flow(
            "treatment_noninfect", "susceptible_treated", "program_rate_completion_noninfect")

        self.set_normal_death_rate("rate_death")
        self.set_death_rate_flow(
            "active", "tb_rate_death")
        self.set_death_rate_flow(
            "detect", "tb_rate_death")
        self.set_death_rate_flow(
            "treatment_infect", "program_rate_death_infect")
        self.set_death_rate_flow(
            "treatment_noninfect", "program_rate_death_noninfect")


population = Stage3PopulationSystem()
population.set_flows()
population.make_graph('stage3.graph.png')
population.integrate_scipy(make_steps(0, 50, 1))
plot_fractions(population, population.labels)
pylab.savefig('stage3.fraction.png', dpi=300)

os.system('open -a "Google Chrome" stage3.graph.png stage3.fraction.png')





