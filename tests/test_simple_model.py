
from autumn.model import NaivePopulation, SingleComponentPopluationSystem, make_steps
from autumn.plotting import plot_fractions

import pylab

import os

population = NaivePopulation()
population.integrate_explicit(make_steps(1700, 2050, 0.05))

plot_fractions(population, population.labels)
pylab.savefig('simple.fraction.png', dpi=300)

population.make_graph('simple.graph')

os.system('open simple.fraction.png')