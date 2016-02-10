

import math
import pylab
from matplotlib import pyplot

"""

Module for plotting population systems

"""


def make_default_line_styles():
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line + color)
    return line_styles


def make_axes_with_room_for_legend():
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    return ax


def set_axes_props(
        ax, xlabel=None, ylabel=None, title=None, is_legend=True):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if is_legend:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc=2, borderaxespad=0., prop={'size':8})
    if title is not None:
        ax.set_title(title)


def plot_fractions(model, labels, png=None):
    ax = make_axes_with_room_for_legend()
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    for i_plot, plot_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            model.times,
            model.fraction_soln[plot_label],
            line_style,
            label=plot_label, linewidth=2)
    set_axes_props(ax, 'year', 'fraction')
    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_populations(model, labels, png=None):
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    ax = make_axes_with_room_for_legend()
    ax.plot(
        model.times,
        model.total_population_soln,
        'orange',
        label="total", linewidth=2)
    for i_plot, plot_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            model.times,
            model.population_soln[plot_label],
            line_style,
            label=plot_label, linewidth=2)
    set_axes_props(ax, 'year', 'population')
    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_fraction_group(model, title, tags, png=None):
    labels = []
    for tag in tags:
        for label in model.labels:
            if tag in label and label not in labels:
                labels.append(label)

    group_population_soln = []
    for i, time in enumerate(model.times):
        pops =[model.population_soln[label][i] for label in labels]
        group_population_soln.append(sum(pops))
    
    ax = make_axes_with_room_for_legend()
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    for i_plot, plot_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        vals = [
            v/t for v, t in
            zip(
                model.population_soln[plot_label],
                group_population_soln)]
        ax.plot(
            model.times,
            vals,
            line_style,
            label=plot_label, linewidth=2)
    set_axes_props(
        ax, 'year', 'fraction of population',
        title + ' fraction')
    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_population_group(model, title, tags, png=None, linestyles=None):
    labels = []
    for tag in tags:
        for label in model.labels:
            if tag in label and label not in labels:
                labels.append(label)

    group_population_soln = []
    for i, time in enumerate(model.times):
        pops =[model.population_soln[label][i] for label in labels]
        group_population_soln.append(sum(pops))
    
    ax = make_axes_with_room_for_legend()
    ax.plot(
        model.times,
        group_population_soln,
        'orange',
        label=title + "_total", linewidth=2)

    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    for i_plot, plot_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            model.times,
            model.population_soln[plot_label],
            line_style,
            label=plot_label, linewidth=2)

    set_axes_props(
        ax, 'year', 'population',title + ' population')
    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_vars(model, labels, png=None):
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    ax = make_axes_with_room_for_legend()
    for i_plot, var_label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            model.times,
            model.get_var_soln(var_label),
            line_style,
            label=var_label, linewidth=2)
    set_axes_props(ax, 'year', 'value')
    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_flows(model, labels, png=None):
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    ax = make_axes_with_room_for_legend()
    for i_plot, label in enumerate(labels):
        line_style = line_styles[i_plot % n_style]
        ax.plot(
            model.times,
            model.get_flow_soln(label),
            line_style,
            label=label, linewidth=2)
    set_axes_props(ax, 'year', 'change / year', 'flows')
    if png is not None:
        pylab.savefig(png, dpi=300)


