
from autumn.model import SimplePopluationSystem, make_steps
from autumn.plotting import plot_fractions

import pylab

population = SimplePopluationSystem()
population.integrate_explicit(make_steps(1700, 2050, 0.05))

plot_fractions(population, population.labels)
pylab.show()
