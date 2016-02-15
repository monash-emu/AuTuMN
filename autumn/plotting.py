

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

def make_related_line_styles(model, labels):
    colours = {}
    patterns = {}
    compartment_full_names = {}
    for label in labels:
        if "susceptible_fully" in label:
            colours[label] = (0, 0, 0)
        elif "susceptible_vac" in label:
            colours[label] = (0.3, 0.3, 0.3)
        elif "susceptible_treated" in label:
            colours[label] = (0.6, 0.6, 0.6)
        elif "latent_early" in label:
            colours[label] = (0, 0.4, 0.8)
        elif "latent_late" in label:
            colours[label] = (0, 0.2, 0.4)
        elif "active" in label:
            colours[label] = (0.9, 0, 0)
        elif "detect" in label:
            colours[label] = (0, 0.5, 0)
        elif "missed" in label:
            colours[label] = (0.5, 0, 0.5)
        elif "treatment_infect" in label:
            colours[label] = (1, 1, 0)
        elif "treatment_noninfect" in label:
            colours[label] = (1, 0.5, 0)
        if "smearneg" in label:
            patterns[label] = "-."
        elif "extrapul" in label:
            patterns[label] = ":"
        else:
            patterns[label] = "-"
        if "susceptible_fully" in label:
            compartment_full_names[label] = "Fully susceptible"
        elif "susceptible_vac" in label:
            compartment_full_names[label] = "BCG vaccinated, susceptible"
        elif "susceptible_treated" in label:
            compartment_full_names[label] = "Previously treated, susceptible"
        elif "latent_early" in label:
            compartment_full_names[label] = "Early latency"
        elif "latent_late" in label:
            compartment_full_names[label] = "Late latency"
        elif "active" in label:
            compartment_full_names[label] = "Active TB, "
        elif "detect" in label:
            compartment_full_names[label] = "Detected, "
        elif "missed" in label:
            compartment_full_names[label] = "Missed, "
        elif "treatment_infect" in label:
            compartment_full_names[label] = "Infectious under treatment, "
        elif "treatment_noninfect" in label:
            compartment_full_names[label] = "Non-infectious under treatment, "
        else:
            compartment_full_names[label] = compartment_full_names[label]
        if "smearpos" in label:
            compartment_full_names[label] = compartment_full_names[label] + "\n" + "smear-positive"
        elif "smearneg" in label:
            compartment_full_names[label] = compartment_full_names[label] + "\n" + "smear-negative"
        elif "extrapul" in label:
            compartment_full_names[label] = compartment_full_names[label] + "\n" + "extrapulmonary"
    return colours, patterns, compartment_full_names


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


def save_png(png):
    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_fractions(model, labels, png=None):
    ax = make_axes_with_room_for_legend()
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            model.fraction_soln[plot_label],
            line_styles[i_plot % n_style],
            label=plot_label, linewidth=2)
    set_axes_props(ax, 'year', 'fraction')
    save_png(png)


def plot_fractions_jt(model, labels, png=None):
    colours, patterns, compartment_full_names\
        = make_related_line_styles(model, labels)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            model.fraction_soln[plot_label],
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])
    handles, labels = ax.get_legend_handles_labels()
    set_axes_props(ax, 'Year', 'Proportion of population',
        'Total population by subgroups', False)
    ax.legend(handles, axis_labels, bbox_to_anchor=(1.05, 1),
        loc=2, borderaxespad=0., prop={'size': 7})
    save_png(png)


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
        ax.plot(
            model.times,
            model.population_soln[plot_label],
            line_styles[i_plot % n_style],
            label=plot_label, linewidth=2)
    set_axes_props(ax, 'year', 'population')
    save_png(png)


def plot_populations_jt(model, labels, png=None):
    colours, patterns, compartment_full_names\
        = make_related_line_styles(model, labels)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
    total_pop_millions = []
    for i in range(len(model.total_population_soln)):
        total_pop_millions.append(model.total_population_soln[i] / 1e6)
    ax.plot(
        model.times,
        total_pop_millions,
        'k',
        label="total", linewidth=2)
    axis_labels.append("Total population")
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            model.population_soln[plot_label] / 1e6,
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])
    handles, labels = ax.get_legend_handles_labels()
    set_axes_props(ax, 'Year', 'Million persons',
                   'Absolute population', False)
    ax.legend(handles, axis_labels, bbox_to_anchor=(1.05, 1),
              loc=2, borderaxespad=0., prop={'size': 7})
    save_png(png)


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
        vals = [
            v/t for v, t in
            zip(
                model.population_soln[plot_label],
                group_population_soln)]
        ax.plot(
            model.times,
            vals,
            line_styles[i_plot % n_style],
            label=plot_label, linewidth=2)
    set_axes_props(
        ax, 'year', 'fraction of population',
        title + ' fraction')
    save_png(png)


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
        ax.plot(
            model.times,
            model.population_soln[plot_label],
            line_styles[i_plot % n_style],
            label=plot_label, linewidth=2)

    set_axes_props(
        ax, 'year', 'population',title + ' population')
    save_png(png)


def plot_vars(model, labels, png=None):
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    ax = make_axes_with_room_for_legend()
    for i_plot, var_label in enumerate(labels):
        ax.plot(
            model.times,
            model.get_var_soln(var_label),
            line_styles[i_plot % n_style],
            label=var_label, linewidth=2)
    set_axes_props(ax, 'year', 'value')
    save_png(png)


def plot_flows(model, labels, png=None):
    line_styles = make_default_line_styles()
    n_style = len(line_styles)
    ax = make_axes_with_room_for_legend()
    for i_plot, label in enumerate(labels):
        ax.plot(
            model.times,
            model.get_flow_soln(label),
            line_styles[i_plot % n_style],
            label=label, linewidth=2)
    set_axes_props(ax, 'year', 'change / year', 'flows')
    save_png(png)


