# -*- coding: utf-8 -*-


"""
# Stage 18 model of the Dynamic Transmission Model

- latent early and latent late,
- fail detect
- two stage treatment
- three pulmonary status
"""


from autumn.model import Stage6PopulationSystem
import autumn.plotting as plotting
import pylab
import os
import numpy
from scipy.stats import uniform

if __name__ == "__main__":


    if not os.path.isdir('stage6'):
        os.makedirs('stage6')

    population = Stage6PopulationSystem()
    population.set_flows()
    population.make_graph(os.path.join('stage6', 'flow_chart.png'))
    population.make_times_with_n_step(1950., 2015., 20)
    population.integrate_explicit()

    groups = {
        "ever_infected": ["suscptible_treated", "latent", "active", "missed", "detect", "treatment"],
        "infected": ["latent", "active", "missed", "detect", "treatment"],
        "active": ["active", "missed", "detect", "treatment"],
        "infectious": ["active", "missed", "detect", "treatment_infect"],
        "identified": ["detect", "treatment"],
        "treatment": ["treatment"]
    }
    for title, tags in groups.items():
        plotting.plot_population_group(population, title, tags)
        pylab.savefig(os.path.join('stage6', '%s.population.png' % title), dpi=300)
        plotting.plot_fraction_group(population, title, tags)
        pylab.savefig(os.path.join('stage6', '%s.fraction.png' % title), dpi=300)

    plotting.plot_fractions(population, population.labels[:])
    pylab.savefig(os.path.join('stage6', 'fraction.png'), dpi=300)
    plotting.plot_populations(population, population.labels[:])
    pylab.savefig(os.path.join('stage6', 'population.png'), dpi=300)

    plotting.plot_vars(population, ['incidence', 'prevalence', 'mortality'])
    pylab.savefig(os.path.join('stage6', 'rates.png'), dpi=300)

    plotting.plot_flows(population, ['latent_early'])
    pylab.savefig(os.path.join('stage6', 'flows.png'), dpi=300)

    import platform
    import os
    import glob
    operating_system = platform.system()
    if 'Windows' in operating_system:
        os.system("start " + " ".join(glob.glob(os.path.join('stage6', '*png'))))
    elif 'Darwin' in operating_system:
        os.system('open ' + os.path.join('stage6', '*.png'))

