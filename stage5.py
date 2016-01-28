# -*- coding: utf-8 -*-
"""
Test
- implement Stage 18 model of the Dynamic Transmission Model
"""

from autumn.model import BasePopulationSystem, make_steps

class Model(BasePopulationSystem):

    """
    This is a reasonable useful model incorporating most of James' thesis.
    """

    def __init__(self):

        BasePopulationSystem.__init__(self)

        self.set_compartment("suscept_unvac", 1e6)
        self.set_compartment("suscept_vac", 0.)
        self.set_compartment("suscept_treated", 0.)

        self.strains = ['_drugsus', '_drugres']

        for strain in self.strains:
            self.set_compartment("latent_early" + strain, 0.)
            self.set_compartment("latent_late" + strain, 0.)
            self.set_compartment("active" + strain, 1.)
            self.set_compartment("detect" + strain, 0.)
            self.set_compartment("faildetect" + strain, 0.)
            self.set_compartment("treat" + strain, 0.)
            self.set_compartment("treat_noninfect" + strain, 0.)

        self.set_compartment("misdetect_drugres", 0.)
        self.set_compartment("mistreat_drugres", 0.)

        self.infectous_labels = []
        for strain in self.strains:
            self.infectous_labels.append("latent_early" + strain)
            self.infectous_labels.append("latent_late" + strain)
            self.infectous_labels.append("active" + strain)
            self.infectous_labels.append("detect" + strain)
            self.infectous_labels.append("faildetect" + strain)
            self.infectous_labels.append("treat" + strain)
        self.infectous_labels.append("misdetect_drugres")
        self.infectous_labels.append("mistreat_drugres")

        self.set_param("rate_birth_per_capita", 20. / 1e3)
        self.set_param("rate_death_per_capita", 1. / 65)

        self.set_param("tb_n_contact", 40.)

        self.set_param("tb_rate_early", .2)
        self.set_param("tb_rate_late", .0005 )
        self.set_param("tb_rate_stabilise", .8)
        self.set_param("tb_rate_recover", .5 * .3)
        self.set_param("tb_rate_death", .5 * .3 )

        self.set_param("program_prop_vac", .9)
        self.set_param("program_prop_unvac", .1)

        self.set_param("program_rate_detect_drugsus", 0.8)
        self.set_param("program_rate_faildetect_drugsus", 0.2)

        self.set_param("program_rate_detect_drugres", 0.4)
        self.set_param("program_rate_faildetect_drugres", 0.2)
        self.set_param("program_rate_misdetect_drugres", 0.4)

        self.set_param("program_rate_start", 26.)
        self.set_param("program_rate_sick_of_waiting", 4.)

        self.set_param("program_rate_complete", 26 * 0.9)
        self.set_param("program_rate_death", 26 * 0.05)
        self.set_param("program_rate_default", 26 * 0.05)

        self.set_param("program_rate_complete_noninfect", 2 * 0.7)
        self.set_param("program_rate_default_noninfect", 2 * 0.1)
        self.set_param("program_rate_death_noninfect", 2 * 0.1)

    def calculate_vars(self):
        self.vars["population"] = sum(self.compartments.values())

        self.vars["infections"] = 0.0
        for label in self.infectous_labels:
            self.vars["infections"] += self.compartments[label]

        self.vars["rate_force"] = \
              self.params["tb_n_contact"] \
            * self.vars["infections"] \
            / self.vars["population"]

        self.vars["rate_force_weak"] = \
            0.5 * self.vars["rate_force"]

        self.vars["rate_birth"] = \
            self.params["rate_birth_per_capita"] * self.vars["population"]
        self.vars["births_unvac"] = \
            self.params['program_prop_unvac'] * self.vars["rate_birth"]
        self.vars["births_vac"] = \
            self.params['program_prop_vac'] * self.vars["rate_birth"]

    def set_flows(self):
        self.set_var_flow(
            "suscept_unvac",
            "births_unvac" )
        self.set_var_flow(
            "suscept_vac",
            "births_vac" )

        self.set_normal_death_rate("rate_death_per_capita")

        for strain in self.strains:
            self.set_var_transfer_rate_flow(
                "suscept_unvac",
                "latent_early" + strain,
                "rate_force")
            self.set_var_transfer_rate_flow(
                "suscept_vac",
                "latent_early" + strain,
                "rate_force_weak")
            self.set_var_transfer_rate_flow(
                "suscept_treated",
                "latent_early" + strain,
                "rate_force_weak")

            self.set_fixed_transfer_rate_flow(
                "latent_early" + strain,
                "latent_late" + strain,
                "tb_rate_stabilise")
            self.set_fixed_transfer_rate_flow(
                "latent_early" + strain,
                "active" + strain,
                "tb_rate_early")

            self.set_fixed_transfer_rate_flow(
                "latent_late" + strain,
                "active" + strain,
                "tb_rate_late")

            self.set_fixed_transfer_rate_flow(
                "active" + strain,
                "latent_late" + strain,
                "tb_rate_recover")

            self.set_fixed_transfer_rate_flow(
                "detect" + strain,
                "treat" + strain,
                "program_rate_start")

            self.set_fixed_transfer_rate_flow(
                "faildetect" + strain,
                "active" + strain,
                "program_rate_sick_of_waiting")

            self.set_fixed_transfer_rate_flow(
                "treat" + strain,
                "treat_noninfect" + strain,
                "program_rate_complete")
            self.set_fixed_transfer_rate_flow(
                "treat" + strain,
                "active" + strain,
                "program_rate_default")

            self.set_fixed_transfer_rate_flow(
                "treat_noninfect" + strain,
                "active" + strain,
                "program_rate_default_noninfect")
            self.set_fixed_transfer_rate_flow(
                "treat_noninfect" + strain,
                "suscept_treated",
                "program_rate_complete_noninfect")

            self.set_disease_death_rate_flow(
                "active" + strain,
                "tb_rate_death")
            self.set_disease_death_rate_flow(
                "detect" + strain,
                "tb_rate_death")
            self.set_disease_death_rate_flow(
                "faildetect" + strain,
                "tb_rate_death")
            self.set_disease_death_rate_flow(
                "treat" + strain,
                "program_rate_death")
            self.set_disease_death_rate_flow(
                "treat_noninfect" + strain,
                "program_rate_death_noninfect")

        # drugsus detection
        self.set_fixed_transfer_rate_flow(
            "active_drugsus",
            "detect_drugsus",
            "program_rate_detect_drugsus")
        self.set_fixed_transfer_rate_flow(
            "active_drugsus",
            "faildetect_drugsus",
            "program_rate_faildetect_drugsus")

        # drugres detection
        self.set_fixed_transfer_rate_flow(
            "active_drugres",
            "detect_drugres",
            "program_rate_detect_drugres")
        self.set_fixed_transfer_rate_flow(
            "active_drugres",
            "faildetect_drugres",
            "program_rate_faildetect_drugres")
        self.set_fixed_transfer_rate_flow(
            "active_drugres",
            "misdetect_drugres",
            "program_rate_misdetect_drugres")
        self.set_fixed_transfer_rate_flow(
            "misdetect_drugres",
            "mistreat_drugres",
            "program_rate_start")

        self.set_disease_death_rate_flow(
            "misdetect_drugres",
            "tb_rate_death")
        self.set_disease_death_rate_flow(
            "mistreat_drugres",
            "tb_rate_death")


if __name__ == "__main__":
    import pylab
    import os
    from autumn.plotting import plot_fractions

    population = Model()
    population.set_flows()
#    population.make_graph('stage5.graph.png')
    population.integrate_scipy(make_steps(0, 50, 1))
    plot_fractions(population, population.labels)
    pylab.savefig('stage5.fraction.png', dpi=300)

    os.system('open -a "Google Chrome" stage5.graph.png stage5.fraction.png')





