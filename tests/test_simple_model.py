
from model import SimplePopluationSystem, make_steps
from plotting import make_time_plots_color

population = SimplePopluationSystem()
population.integrate(make_steps(1700, 2050, 0.05))
make_time_plots_color(population, population.labels)

