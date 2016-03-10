

import math
import pylab
import numpy
from matplotlib import pyplot

"""

Module for plotting population systems

"""


def make_default_line_styles():
    """
    Now inactive
    """
    line_styles = []
    for line in ["-", ":", "-.", "--"]:
        for color in "rbmgk":
            line_styles.append(line + color)
    return line_styles


def get_line_style(label):
    # Unassigned groups remain black
    colour = (0, 0, 0)
    if "susceptible_vac" in label:  # susceptible_unvac remains black
        colour = (0.3, 0.3, 0.3)
    elif "susceptible_treated" in label:
        colour = (0.6, 0.6, 0.6)
    if "latent" in label:  # latent_early remains as for latent
        colour = (0, 0.4, 0.8)
    if "latent_late" in label:
        colour = (0, 0.2, 0.4)
    if "active" in label:
        colour = (0.9, 0, 0)
    elif "detect" in label:
        colour = (0, 0.5, 0)
    elif "missed" in label:
        colour = (0.5, 0, 0.5)
    if "treatment" in label:  # treatment_infect remains as for treatment
        colour = (1, 0.5, 0)
    if "treatment_noninfect" in label:
        colour = (1, 1, 0)

    pattern = "-"  # Default filled line
    if "smearneg" in label:
        pattern = "-."
    elif "extrapul" in label:
        pattern = ":"

    if "susceptible" in label:
        category_full_name = "Susceptible"
    if "susceptible_fully" in label:
        category_full_name = "Fully susceptible"
    elif "susceptible_vac" in label:
        category_full_name = "BCG vaccinated, susceptible"
    elif "susceptible_treated" in label:
        category_full_name = "Previously treated, susceptible"
    if "latent" in label:
        category_full_name = "Latent"
    if "latent_early" in label:
        category_full_name = "Early latent"
    elif "latent_late" in label:
        category_full_name = "Late latent"
    if "active" in label:
        category_full_name = "Active, yet to present"
    elif "detect" in label:
        category_full_name = "Detected"
    elif "missed" in label:
        category_full_name = "Missed"
    if "treatment" in label:
        category_full_name = "Under treatment"
    if "treatment_infect" in label:
        category_full_name = "Infectious under treatment"
    elif "treatment_noninfect" in label:
        category_full_name = "Non-infectious under treatment"

    if "smearpos" in label:
        category_full_name += ", \nsmear-positive"
    elif "smearneg" in label:
        category_full_name += ", \nsmear-negative"
    elif "extrapul" in label:
        category_full_name += ", \nextrapulmonary"

    if "_ds" in label:
        category_full_name += ", \nDS-TB"
        marker = ''
    elif "_mdr" in label:
        category_full_name += ", \nMDR-TB"
        marker = '|'
    elif "_xdr" in label:
        category_full_name += ", \nXDR-TB"
        marker = '.'
    else:
        marker = ''

    return colour, pattern, category_full_name, marker


def make_related_line_styles(labels):
    colours = {}
    patterns = {}
    compartment_full_names = {}
    markers = {}
    for label in labels:
        colour, pattern, compartment_full_name, marker = get_line_style(label)
        colours[label] = colour
        patterns[label] = pattern
        compartment_full_names[label] = compartment_full_name
        markers[label] = marker
    return colours, patterns, compartment_full_names, markers


def make_axes_with_room_for_legend():
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    return ax


def humanise_y_ticks(ax):
    vals = list(ax.get_yticks())
    max_val = max([abs(v) for v in vals])
    if max_val < 1e3:
        return map(str, vals)
    if max_val >= 1e3 and max_val < 1e6:
        labels = ["%.1fK" % (v/1e3) for v in vals]
    elif max_val >= 1e6 and max_val < 1e9:
        labels = ["%.1fM" % (v/1e6) for v in vals]
    elif max_val >= 1e9:
        labels = ["%.1fB" % (v/1e9) for v in vals]
    is_fraction = False
    for label in labels:
        if label[-3:-1] != ".0":
            is_fraction = True
    if not is_fraction:
        labels = [l[:-3] + l[-1] for l in labels]
    ax.set_yticklabels(labels)


def set_axes_props(
        ax, xlabel=None, ylabel=None, title=None, is_legend=True,
        axis_labels=None):
    frame_color = "grey"

    # hide top and right border of plot
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if is_legend:
        if axis_labels:
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(
                handles, 
                axis_labels,
                bbox_to_anchor=(1.05, 1),
                loc=2, 
                borderaxespad=0., 
                frameon=False,
                prop={'size':7})
        else:
            leg = ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc=2, 
                borderaxespad=0., 
                frameon=False,
                prop={'size':7})
        for text in leg.get_texts():
            text.set_color(frame_color)

    if title is not None:
        t = ax.set_title(title)
        t.set_color(frame_color)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(8)

    ax.tick_params(color=frame_color, labelcolor=frame_color)
    for spine in ax.spines.values():
        spine.set_edgecolor(frame_color)
    ax.xaxis.label.set_color(frame_color)
    ax.yaxis.label.set_color(frame_color)

    humanise_y_ticks(ax)


def save_png(png):
    if png is not None:
        pylab.savefig(png, dpi=300)


def plot_populations(model, labels, values, png=None):
    colours, patterns, compartment_full_names, markers\
        = make_related_line_styles(labels)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
    ax.plot(
        model.times,
        model.get_var_soln("population"),
        'k',
        label="total", linewidth=2)
    axis_labels.append("Number of persons")
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            values[plot_label],
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            marker=markers[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])

    title = make_plot_title(model, labels)

    set_axes_props(ax, 'Year', 'Persons',
                   'Population, by ' + title, True,
                   axis_labels)
    save_png(png)


def plot_population_group(model, title, tags, png=None, linestyles=None):
    """
    Inactive?
    """
    subgroup_solns = {}
    for tag in tags:
        labels = [l for l in model.labels if tag in l]
        labels = list(set(labels))
        if len(labels) == 0:
            continue
        sub_group_soln = None
        for l in labels:
            vals = numpy.array(model.population_soln[l])
            if sub_group_soln is None:
                sub_group_soln = vals
            else:
                sub_group_soln = vals + sub_group_soln
        subgroup_solns[tag] = sub_group_soln

    group_soln = sum(subgroup_solns.values())

    ax = make_axes_with_room_for_legend()
    # ax.plot(
    #     model.times,
    #     group_soln,
    #     'k',
    #     label='total ' + title,
    #     linewidth=2)
    for tag, soln in subgroup_solns.items():
        colour, pattern, full_name = get_line_style(tag)
        ax.plot(
            model.times,
            soln,
            linewidth=1,
            color=colour,
            linestyle=pattern,
            label=full_name
        )

    set_axes_props(
        ax, 
        'Year', 
        'Persons',
        'Subgroups within ' + title + ' (absolute)', 
        True)

    save_png(png)


def plot_fractions(model, labels, values, png=None):
    colours, patterns, compartment_full_names, markers\
        = make_related_line_styles(labels)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            values[plot_label],
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            marker=markers[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])
    title = make_plot_title(model, labels)
    set_axes_props(ax, 'Year', 'Proportion of population',
        'Population, ' + title, True, axis_labels)
    save_png(png)


def make_plot_title(model, labels):
    if labels is model.labels:
        title = "by each individual compartment"
    elif labels is model.compartment_types:
        title = "by types of compartments"
    elif labels is model.broad_compartment_types:
        title = "by broad types of compartments"
    elif labels is model.groups["ever_infected"]:
        title = "within ever infected compartments"
    elif labels is model.groups["infected"]:
        title = "within infected compartments"
    elif labels is model.groups["active"]:
        title = "within active disease compartments"
    elif labels is model.groups["infectious"]:
        title = "within infectious compartments"
    elif labels is model.groups["identified"]:
        title = "within identified compartments"
    elif labels is model.groups["treatment"]:
        title = "within treatment compartments"
    else:
        title = "not sure"
    return title


def plot_outputs(model, labels, png=None):
    colours = {}
    full_names = {}
    axis_labels = []
    for label in labels:
        if "incidence" in label:
            colours[label] = (0, 0, 0)
            full_names[label] = "Incidence"
            title = "Main rate outputs"
            yaxis_label = "Per 100,000 per year"
        elif "notification" in label:
            colours[label] = (0, 0, 1)
            full_names[label] = "Notifications"
        elif "mortality" in label:
            colours[label] = (1, 0, 0)
            full_names[label] = "Mortality"
        elif "prevalence" in label:
            colours[label] = (0, 0.5, 0)
            full_names[label] = "Prevalence"
            title = "Main proportion output"
            yaxis_label = "Per 100,000"
    ax = make_axes_with_room_for_legend()

    for i_plot, var_label in enumerate(labels):
        ax.plot(
            model.times,
            model.get_var_soln(var_label),
            color=colours[var_label],
            label=var_label, linewidth=1)
        axis_labels.append(full_names[var_label])
    set_axes_props(ax, 'Year', yaxis_label, title, True,
        axis_labels)
    save_png(png)


def plot_flows(model, labels, png=None):
    colours, patterns, compartment_full_names\
        = make_related_line_styles(labels)
    ax = make_axes_with_room_for_legend()
    axis_labels = []
    for i_plot, plot_label in enumerate(labels):
        ax.plot(
            model.times,
            model.get_flow_soln(plot_label) / 1E3,
            label=plot_label, linewidth=1,
            color=colours[plot_label],
            linestyle=patterns[plot_label])
        axis_labels.append(compartment_full_names[plot_label])
    set_axes_props(ax, 'Year', 'Change per year, thousands',
                   'Aggregate flows in/out of compartment',
                   True, axis_labels)
    save_png(png)


def open_pngs(pngs):
    import platform
    import os
    operating_system = platform.system()
    if 'Windows' in operating_system:
        os.system("start " + " ".join(pngs))
    elif 'Darwin' in operating_system:
        os.system('open ' + " ".join(pngs))


