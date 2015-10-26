from autumn.model import SingleComponentPopluationSystem, make_steps
from autumn.plotting import make_time_plots_color, make_time_plots_one_panel, plot_fractions
import pylab

population = SingleComponentPopluationSystem()
population.integrate_scipy(make_steps(0, 50, 1))
labels = population.labels
plot_fractions(population, population.labels[1:])
pylab.show()
make_time_plots_color(population, labels)
make_time_plots_one_panel(population, labels, labels[1:])
