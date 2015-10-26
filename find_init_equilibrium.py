
from model import SimplePopluationSystem, make_steps
from plotting import make_time_plots_color

"""
find equilibrium
"""

population = SimplePopluationSystem()
labels = population.labels

for time in make_steps(50, 1000, 5):

    times = make_steps(0, time, 1)
    population.integrate(times)

    output = {}
    for label in labels:
        output[label] = population.get_soln(label)[-1]

    total = sum(output.values())
    s = "t=%.f " % time
    for label in labels:
        s += "%s=%.2f%% " % (label, 100*output[label]/float(total))
    print s



