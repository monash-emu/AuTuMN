
from autumn.model import SimplePopluationSystem, make_steps
from autumn.plotting import make_time_plots_color

population = SimplePopluationSystem()
population.integrate_explicit(make_steps(1700, 2050, 0.05))
make_time_plots_color(population, population.labels)

