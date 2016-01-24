# -*- coding: utf-8 -*-


"""
# Stage 18 model of the Dynamic Transmission Model

- latent early and latent late,
- fail detect
- two stage treatment
- three pulmonary status
"""


from modules.model_jt import Stage6PopulationSystem
from modules.plotting import plot_fractions
import pylab
import os
from scipy.stats import uniform

import modules.parameter_setting as parameter_setting

if __name__ == "__main__":
    input_parameters = \
        {"demo_rate_birth": 20. / 1e3,
         "demo_rate_death": 1. / 65,
         "epi_proportion_cases_smearpos": 0.6,
         "epi_proportion_cases_smearneg": 0.2,
         "epi_proportion_cases_extrapul": 0.2,
         "tb_multiplier_force_smearpos": 1.,
         "tb_multiplier_force_smearneg":
            parameter_setting.multiplier_force_smearneg.prior_estimate,
         "tb_multiplier_force_extrapul": 0.0,
         "tb_n_contact": 25.,
         "tb_proportion_early_progression":
            parameter_setting.proportion_early_progression.prior_estimate,
         "tb_timeperiod_early_latent":
            parameter_setting.timeperiod_early_latent.prior_estimate,
         "tb_rate_late_progression":
            parameter_setting.rate_late_progression.prior_estimate,
         "tb_proportion_casefatality_untreated_smearpos":
            parameter_setting.proportion_casefatality_active_untreated_smearpos.prior_estimate,
         "tb_proportion_casefatality_untreated_smearpos":
            parameter_setting.proportion_casefatality_active_untreated_smearpos.prior_estimate,
         "tb_timeperiod_activeuntreated":
            parameter_setting.timeperiod_activeuntreated.prior_estimate,
         "tb_multiplier_bcg_protection":
            parameter_setting.multiplier_bcg_protection.prior_estimate,
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

    input_compartments = {"susceptible_fully": 1e6, "active": 3.}

    population = Stage6PopulationSystem(input_parameters, input_compartments)
    population.set_flows()
    population.make_graph('Stage6.graph.png')
    population.make_steps(1950., 2015., 20)
    population.integrate_scipy()

    plot_fractions(population, population.labels[:])
    pylab.savefig('Stage6.fraction.png', dpi=300)

    os.system('open -a "Google Chrome" Stage6.graph.png Stage6.fraction.png')

