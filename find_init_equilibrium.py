

from model import PopulationSystem, make_steps
import math
import pylab


"""
find equilibrium
"""


def make_time_plots(population, plot_labels, png=None):
    n_row = int(math.ceil(len(plot_labels) / 2.))
    n_col=2

    for i_plot, plot_label in enumerate(plot_labels):
        pylab.subplot(n_row, n_col, i_plot+1)
        pylab.plot(
            population.times, population.get_soln(plot_label), linewidth=2)
        pylab.ylabel(plot_label)
        pylab.xlabel('time')
        pylab.tight_layout()
    
    if png is None:
        pylab.show()
    else:
        pylab.savefig(png)
        


population = PopulationSystem()

population.set_compartment("susceptible", 1e6)
population.set_compartment("latent_early", 0.)
population.set_compartment("latent_late", 0.)
population.set_compartment("active", 3000.)
population.set_compartment("undertreatment", 0.)

population.set_param("rate_pop_birth", 20. / 1e3)
population.set_param("rate_pop_death", 1. / 65)

population.set_param("n_tbfixed_contact", 10.)
population.set_param("rate_tbfixed_earlyprog", .1 / .5)
population.set_param("rate_tbfixed_lateprog", .1 / 100.)
population.set_param("rate_tbfixed_stabilise", .9 / .5)
population.set_param("rate_tbfixed_recover", .6 / 3.)
population.set_param("rate_tbfixed_death", .4 / 3.)

time_treatment = .5
population.set_param("rate_tbprog_detect", 1.)
population.set_param("time_treatment", time_treatment)
population.set_param("rate_tbprog_completion", .9 / time_treatment)
population.set_param("rate_tbprog_default", .05 / time_treatment)
population.set_param("rate_tbprog_death", .05 / time_treatment)

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



