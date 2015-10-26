
from autumn.model import SimplePopluationSystem, make_steps
from autumn.plotting import plot_fractions
import pylab
"""
find equilibrium
"""

# look back 10 years
equil_cutoff_time = 10

# 1% difference allowed for equilibrium
max_equil_frac_diff = 0.01 

label = "latent"

population = SimplePopluationSystem()
labels = population.labels
for time in make_steps(100, 1000, 10):

    times = make_steps(0, time, 1)
    population.integrate_explicit(times)

    is_converged = True

    labels = population.labels
    population.calculate_fractions()

    fraction = population.fractions[label]

    i = -2
    max_fraction_diff = 0
    time_diff = 0
    while time_diff < equil_cutoff_time:
        i -= 1
        time_diff = abs(times[-1] - times[i])
        frac_diff = (fraction[-1] - fraction[i])
        if abs(frac_diff) > max_fraction_diff:
            max_fraction_diff = frac_diff
        if abs(frac_diff) > max_equil_frac_diff:
            is_converged = False

    print time, max_fraction_diff, is_converged



