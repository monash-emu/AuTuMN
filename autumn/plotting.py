

import math
import pylab


"""

Module for plotting population systems

"""


def make_time_plots_color(
        population, plot_labels, png=None):

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


def make_time_plots_one_panel(
        population, plot_labels0, plot_labels1, png=None):

    line_styles = ['-r', '-b', '-m', '-g', '-k']
    for subplot_index, plot_labels in [(210, plot_labels0), (211, plot_labels1)]:
        pylab.subplot(subplot_index)
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



def plot_fractions(population, plot_labels, png=None):

    line_styles = ['-r', '-b', '-m', '-g', '-k']
    population.calculate_fractions()
    for plot_label, line_style in zip(plot_labels, line_styles):
        pylab.plot(
            population.times, 
            population.fractions[plot_label], 
            line_style,
            label=plot_label.title(), linewidth=2)
    pylab.xlabel('Time')
    pylab.ylabel('Number of patients')
    pylab.legend(loc=0)



