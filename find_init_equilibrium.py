
#from autumn.model import SimplePopluationSystem, make_steps
from autumn.model import SingleComponentPopluationSystem, make_steps
from autumn.plotting import plot_fractions
import pylab


"""

Find first time for population to achieve equilibrium

"""


def find_equilibrium_time(
        population, 
        label, 
        equil_cutoff_time=50, 
        max_equil_frac_diff=0.01):

    """
    Returns to first time that equilibrium conditions are achieved.

    If no equilibrium is found within time, None is returned.
    """
    
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

        print ("time=%.f dev=%.1f%% %s=%.1f%%" % (
            time, max_fraction_diff*100, label, fraction[-1]*100))

        if is_converged:
            return time

    return None


#population = SimplePopluationSystem()
population = SingleComponentPopluationSystem()
time = find_equilibrium_time(population, "latent")
plot_fractions(population, population.labels)
pylab.show()
print ("Found time=%s years" % time)
