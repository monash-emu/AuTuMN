

from model import PopulationSystem
import math
import pylab


"""

Basic test of plotting model.

To run:

    python test_plot.py

"""


def make_times(start, end, step):
    times = []
    time = start
    while time < end:
        times.append(time)
        time += step
    return times


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
        

def make_time_plots_color(population, plot_labels, png=None):
    n_row = int(math.ceil(len(plot_labels) / 2.))
    n_col=2
    colors = ('r', 'b', 'm', 'g', 'k') 
    for i_plot, plot_label in enumerate(plot_labels):
        pylab.subplot(n_row, n_col, i_plot+1)
        pylab.plot(
            population.times, 
            population.get_soln(plot_label), 
            linewidth=2,
            color=colors[i_plot % len(colors)])
        pylab.ylabel(plot_label)
        pylab.xlabel('time')
        pylab.tight_layout()
    
    if png is None:
        pylab.show()
    else:
        pylab.savefig(png)


def make_time_plots_one_panel(population, plot_labels, png=None):
    line_styles = ['-r', '-b', '-m', '-g', '-k']
    pylab.subplot(211)
    for plot_label, line_style in zip(plot_labels, line_styles):
        pylab.plot(
            population.times, 
            population.get_soln(plot_label), 
            line_style,
            label=plot_label.title(), linewidth=2)
    pylab.xlabel('Time')
    pylab.ylabel('Number of patients')
    pylab.legend(loc=0)
    pylab.show()


population = PopulationSystem()

population.set_compartment("susceptible", 1e6)
population.set_compartment("latent_early", 0.)
population.set_compartment("latent_late", 0.)
population.set_compartment("active", 1.)
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

times = make_times(0, 50, 1)
population.integrate(times)

labels = population.labels

make_time_plots(population, labels)
make_time_plots_color(population, labels)
make_time_plots_one_panel(population, labels)
make_time_plots_one_panel(population, labels[1:])



