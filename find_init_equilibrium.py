
from autumn.model import SimplePopluationSystem, make_steps
from autumn.plotting import make_time_plots_color

"""
find equilibrium
"""

population = SimplePopluationSystem()
labels = population.labels

for time in make_steps(50, 1000, 10):

    times = make_steps(0, time, 1)
    population.integrate_explicit(times)

    is_converged = True

    labels = population.labels
    population.calculate_fractions()

    n = len(population.times)
    label = "active"
    fraction = population.fractions[label]
    n_step_test = 10
    i = 0
    max_diff = 0
    for i in range(n_step_test):
        if i == 0:
            continue
        j = -i
        diff = (fraction[j + 1] - fraction[j])
        if abs(diff) > max_diff:
            max_diff = diff
        if abs(diff) > 0.5:
            is_converged = False

    print time, max_diff, is_converged



