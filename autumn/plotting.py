

import math
import pylab
from matplotlib import pyplot

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
            population.get_compartment_soln(plot_label),
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
                population.get_compartment_soln(plot_label),
                line_style,
                label=plot_label.title(), linewidth=2)
        pylab.xlabel('Time')
        pylab.ylabel('Number of patients')
        pylab.legend(loc=0)
    pylab.show()



def plot_fractions(population, plot_labels, png=None):
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line+color)
    n_style = len(line_styles)
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    for i_plot, plot_label in enumerate(plot_labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            population.steps,
            population.fractions[plot_label],
            line_style,
            label=plot_label, linewidth=2)
    ax.set_xlabel('year')
    ax.set_ylabel('fraction of population')
    ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc=2, borderaxespad=0., prop={'size':8})


def plot_fractions_jt(population, plot_labels, png=None):
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line+color)
    n_style = len(line_styles)
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    fractions_to_plot = population.fractions_active
    fractions_to_plot = population.fractions
    for i_plot, plot_label in enumerate(plot_labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            population.steps,
            fractions_to_plot[plot_label],
            line_style,
            label=plot_label, linewidth=2)
    ax.set_xlabel('year')
    ax.set_ylabel('fraction of population')
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc=2, borderaxespad=0., prop={'size':8})



def plot_populations(population, plot_labels, png=None):
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line+color)
    n_style = len(line_styles)
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot(
        population.steps,
        population.total_population,
        'orange',
        label="total", linewidth=2)
    for i_plot, plot_label in enumerate(plot_labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            population.steps,
            population.populations[plot_label], 
            line_style,
            label=plot_label, linewidth=2)
    ax.set_xlabel('year')
    ax.set_ylabel('population')
    ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc=2, borderaxespad=0., prop={'size':8})


def plot_fraction_subgroups(population, subgroup, png=None):
    labels = []
    for subgroup_tag in subgroup:
        for label in population.labels:
            if subgroup_tag in label and label not in labels:
                labels.append(label)

    steps = population.steps
    n_step = len(steps)

    total_population = []
    for i in range(n_step):
        pops =[population.populations[label][i] for label in labels]
        total_population.append(sum(pops))
    
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line+color)
    n_style = len(line_styles)
    for i_plot, plot_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        vals = population.populations[plot_label]
        vals = [v/t for v, t in zip(vals, total_population)]
        ax.plot(
            population.steps,
            vals, 
            line_style,
            label=plot_label, linewidth=2)
    ax.set_xlabel('year')
    ax.set_ylabel('fraction of population')
    ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc=2, borderaxespad=0., prop={'size':8})


def plot_population_subgroups(population, subgroup, png=None):
    labels = []
    for subgroup_tag in subgroup:
        for label in population.labels:
            if subgroup_tag in label and label not in labels:
                labels.append(label)

    steps = population.steps
    n_step = len(steps)

    total_population = []
    for i in range(n_step):
        pops =[population.populations[label][i] for label in labels]
        total_population.append(sum(pops))
    
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot(
        population.steps,
        total_population,
        'orange',
        label="total", linewidth=2)
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line+color)
    n_style = len(line_styles)
    for i_plot, plot_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        vals = population.populations[plot_label]
        ax.plot(
            population.steps,
            vals, 
            line_style,
            label=plot_label, linewidth=2)
    ax.set_xlabel('year')
    ax.set_ylabel('population')
    ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc=2, borderaxespad=0., prop={'size':8})


def plot_vars(population, labels, png=None):
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line+color)
    n_style = len(line_styles)
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    for i_plot, var_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            population.steps,
            population.get_var_soln(var_label), 
            line_style,
            label=var_label, linewidth=2)
    ax.set_xlabel('year')
    ax.set_ylabel('value')
    ax.legend(
        bbox_to_anchor=(1.05, 1), 
        loc=2, borderaxespad=0., prop={'size':8})


