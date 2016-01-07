

from autumn.model import make_steps
from autumn.plotting import plot_fractions
import pylab

import stage5
"""

Find first time for population to achieve equilibrium

"""


def find_equilibrium_time(
        population, 
        label, 
        equil_time=10, 
        max_search_time=100,
        test_fraction_diff=0.01):

    """
    Returns to first time that equilibrium conditions are achieved.

    If no equilibrium is found within time, None is returned.
    """
    
    for max_test_time in make_steps(equil_time, max_search_time, 1):

        times = make_steps(0, max_test_time, 1)
        population.integrate_scipy(times)

        is_converged = True

        labels = population.labels
        population.calculate_fractions()

        fraction = population.fractions[label]

        i = -2
        max_fraction_diff = 0
        time_diff = 0
        while time_diff < equil_time:
            i -= 1
            if -i >= len(times):
                is_converged = False
                break
            time_diff = abs(times[-1] - times[i])
            frac_diff = (fraction[-1] - fraction[i])
            if abs(frac_diff) > max_fraction_diff:
                max_fraction_diff = frac_diff
            if abs(frac_diff) > test_fraction_diff:
                is_converged = False

        print ("time=%.f %s=%.1f-/+%.1f%% " % (
            max_test_time, label, fraction[-1]*100, abs(max_fraction_diff*100)))

        if is_converged:
            return max_test_time

    return None


population = stage5.Model()
time = find_equilibrium_time(population, "latent_early_drugsus")
plot_fractions(population, population.labels)
pylab.show()
print ("Found time=%s years" % time)


