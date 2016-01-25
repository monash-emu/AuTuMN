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
import numpy
from scipy.stats import uniform
import modules.parameter_setting as parameter_setting
import modules.philippines_parameters as philippines_parameters

if __name__ == "__main__":
    parameter_method = "ppf"
    # ppf_value = uniform.rvs(size=10)
    ppf_value = 0.5 * numpy.ones(11)

    input_parameters = \
        {"demo_rate_birth": 20. / 1e3,
         "demo_rate_death": 1. / 65,
         "epi_proportion_cases_smearpos": 0.6,
         "epi_proportion_cases_smearneg": 0.2,
         "epi_proportion_cases_extrapul": 0.2,
         "tb_multiplier_force_smearpos": 1.,
         "tb_multiplier_force_smearneg":
            getattr(parameter_setting.multiplier_force_smearneg, parameter_method)(ppf_value[0]),
         "tb_multiplier_force_extrapul": 0.0,
         "tb_n_contact":
            getattr(parameter_setting.tb_n_contact, parameter_method)(ppf_value[1]),
         "tb_proportion_early_progression":
            getattr(parameter_setting.proportion_early_progression, parameter_method)(ppf_value[2]),
         "tb_timeperiod_early_latent":
            getattr(parameter_setting.timeperiod_early_latent, parameter_method)(ppf_value[3]),
         "tb_rate_late_progression":
            getattr(parameter_setting.rate_late_progression, parameter_method)(ppf_value[4]),
         "tb_proportion_casefatality_untreated_smearpos":
            getattr(parameter_setting.proportion_casefatality_active_untreated_smearpos, parameter_method)(ppf_value[5]),
         "tb_proportion_casefatality_untreated_smearneg":
            getattr(parameter_setting.proportion_casefatality_active_untreated_smearneg, parameter_method)(ppf_value[6]),
         "tb_timeperiod_activeuntreated":
            getattr(parameter_setting.timeperiod_activeuntreated, parameter_method)(ppf_value[7]),
         "tb_multiplier_bcg_protection":
            getattr(parameter_setting.multiplier_bcg_protection, parameter_method)(ppf_value[8]),
         "program_prop_vac":
            getattr(philippines_parameters.bcg_coverage, parameter_method)(ppf_value[9]),
         "program_prop_unvac":
            1 - getattr(philippines_parameters.bcg_coverage, parameter_method)(ppf_value[9]),
         "program_proportion_detect":
            getattr(philippines_parameters.bcg_coverage, parameter_method)(ppf_value[10]),
         "program_algorithm_sensitivity": 0.85,
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

