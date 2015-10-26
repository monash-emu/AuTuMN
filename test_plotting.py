from autumn.model import SingleComponentPopluationSystem, make_steps
from autumn.plotting import make_time_plots_color

population = SingleComponentPopluationSystem()
population.integrate_scipy(make_steps(0, 50, 1))
labels = population.labels
make_time_plots_color(population, labels)
make_time_plots_one_panel(population, labels, labels[1:])
